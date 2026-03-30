"""Two-pass LLM extraction pipeline for format-agnostic ingestion.

Pass 1 (Discovery): Sends the full document to the LLM. Returns a
    lightweight manifest of all knowledge items found — titles, types,
    locations. No full extraction yet.

Pass 2 (Focused Extraction): For each item in the manifest, a targeted
    LLM call extracts structured details. Runs in parallel.

Quality over efficiency — ingestion is infrequent, so spending extra
LLM calls for better extraction is the right trade-off.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime

from app.config import Settings, get_settings
from app.corpus import ChunkRecord
from app.llm import Message, complete
from app.preprocess import preprocess

logger = logging.getLogger(__name__)

_MAX_ITEMS = 25  # Configurable cap on items per document
_MAX_CONCURRENT_PASS2 = 5  # Limit parallel LLM calls


# ── Data types ───────────────────────────────────────────────────────


@dataclass
class DiscoveredItem:
    """A knowledge item found during Pass 1."""

    title: str
    knowledge_type: str  # decision | constraint | principle
    summary: str
    relevant_sections: list[str] = field(default_factory=list)
    has_rejected_alternatives: bool = False
    depends_on_image: bool = False


@dataclass
class ExtractionResult:
    """Result of the full two-pass pipeline."""

    chunks: list[ChunkRecord] = field(default_factory=list)
    items_discovered: int = 0
    items_extracted: int = 0
    items_failed: list[str] = field(default_factory=list)


# ── Public API ───────────────────────────────────────────────────────


async def normalize_document(
    content: str,
    *,
    filename: str = "",
    source_url: str = "",
    source_type: str = "",
    settings: Settings | None = None,
) -> ExtractionResult:
    """Run the two-pass extraction pipeline on a document.

    Args:
        content: Raw document text.
        filename: Filename for provenance.
        source_url: URL of the original document.
        source_type: Provenance label (e.g. "confluence", "rfc").
        settings: Optional settings override.

    Returns:
        ExtractionResult with extracted chunks and summary.
    """
    s = settings or get_settings()

    # Preprocess: raw content → clean markdown
    clean = preprocess(content, filename)

    if not clean.strip():
        return ExtractionResult()

    # Pass 1: discover knowledge items
    manifest = await _pass1_discover(clean, filename, s)

    if not manifest:
        logger.info("Pass 1: no knowledge items found in %s", filename)
        return ExtractionResult()

    # Cap items
    if len(manifest) > _MAX_ITEMS:
        logger.warning(
            "Pass 1 found %d items in %s, capping at %d",
            len(manifest), filename, _MAX_ITEMS,
        )
        manifest = manifest[:_MAX_ITEMS]

    result = ExtractionResult(items_discovered=len(manifest))

    # Build manifest summary for cross-reference context in Pass 2
    manifest_summary = _build_manifest_summary(manifest)

    # Pass 2: extract per item (parallel, rate-limited)
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_PASS2)

    async def _throttled_extract(**kwargs):
        async with semaphore:
            return await _pass2_extract(**kwargs)

    tasks = [
        _throttled_extract(
            item=item,
            document=clean,
            manifest_summary=manifest_summary,
            filename=filename,
            source_url=source_url,
            source_type=source_type,
            item_index=i,
            settings=s,
        )
        for i, item in enumerate(manifest)
    ]

    extraction_results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, res in enumerate(extraction_results):
        if isinstance(res, Exception):
            logger.error("Pass 2 failed for item %d (%s): %s", i, manifest[i].title, res)
            result.items_failed.append(manifest[i].title)
        elif res:
            result.chunks.extend(res)
            result.items_extracted += 1
        else:
            result.items_failed.append(manifest[i].title)

    logger.info(
        "Normalized %s: %d discovered, %d extracted, %d failed, %d chunks",
        filename, result.items_discovered, result.items_extracted,
        len(result.items_failed), len(result.chunks),
    )

    return result


# ── Pass 1 — Discovery ──────────────────────────────────────────────

_PASS1_SYSTEM = """\
You are an architectural knowledge discovery agent. Given a document, identify \
all architectural knowledge items — decisions, constraints, and principles.

Definitions:
- Decision: The team chose X over Y (has alternatives considered/rejected)
- Constraint: X must always be true (a hard rule, often from compliance/security)
- Principle: The team prefers X approach (a standing guideline, may have exceptions)

For each item found, return:
- title: Short descriptive title
- knowledge_type: decision | constraint | principle
- summary: 1-sentence summary of the item
- relevant_sections: List of document section headings where this item's \
  information can be found. A single item's info is often scattered across \
  multiple sections.
- has_rejected_alternatives: true if the document mentions alternatives that \
  were considered and rejected for this item
- depends_on_image: true if the item's evidence is primarily in an image \
  the system couldn't process

IMPORTANT: Only include items explicitly stated in the document. Do not infer \
decisions that might have been made. If unsure, leave it out.

Return ONLY a JSON object. No prose, no code fences:
{
  "items": [
    {
      "title": "string",
      "knowledge_type": "decision | constraint | principle",
      "summary": "string",
      "relevant_sections": ["string"],
      "has_rejected_alternatives": boolean,
      "depends_on_image": boolean
    }
  ]
}

If no architectural knowledge items are found, return: {"items": []}"""


async def _pass1_discover(
    document: str,
    filename: str,
    settings: Settings,
) -> list[DiscoveredItem]:
    """Run Pass 1 discovery on a preprocessed document."""
    result = await complete(
        [
            Message("system", _PASS1_SYSTEM),
            Message("user", f"Document: {filename}\n\n{document}"),
        ],
        model=settings.STAGE2_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
    )

    try:
        raw = result.content.strip()
        parsed = json.loads(raw)
        items_raw = parsed.get("items", []) if isinstance(parsed, dict) else []
    except (json.JSONDecodeError, AttributeError):
        logger.error("Pass 1: failed to parse output: %s", result.content[:200])
        return []

    items: list[DiscoveredItem] = []
    for raw_item in items_raw:
        kt = raw_item.get("knowledge_type", "decision")
        if kt not in ("decision", "constraint", "principle"):
            kt = "decision"

        items.append(DiscoveredItem(
            title=raw_item.get("title", ""),
            knowledge_type=kt,
            summary=raw_item.get("summary", ""),
            relevant_sections=raw_item.get("relevant_sections", []),
            has_rejected_alternatives=raw_item.get("has_rejected_alternatives", False),
            depends_on_image=raw_item.get("depends_on_image", False),
        ))

    return items


# ── Pass 2 — Focused Extraction ─────────────────────────────────────

_PASS2_DECISION_SYSTEM = """\
You are an architectural knowledge extractor. Extract a structured DECISION \
from the document based on the item described below.

A decision is a choice the team made — X over Y. Extract:
- context: Why this decision was needed (problem, requirements, constraints)
- decision: What was decided and how it will be implemented
- consequences: Tradeoffs accepted, operational implications
- rejected_alternatives: What was explicitly considered and ruled out, and WHY. \
  For each alternative: what it was, why it was appealing, why it was rejected. \
  If the document doesn't mention rejected alternatives, return an empty string — \
  do NOT invent them.
- affected_services: Specific service names if mentioned (empty array if project-wide)
- domain: One of: security, compliance, performance, scalability, data_model, operational

Return ONLY JSON. No prose, no code fences:
{
  "context": "string",
  "decision": "string",
  "consequences": "string",
  "rejected_alternatives": "string or empty",
  "affected_services": ["string"],
  "domain": "string"
}"""

_PASS2_CONSTRAINT_SYSTEM = """\
You are an architectural knowledge extractor. Extract a structured CONSTRAINT \
from the document based on the item described below.

A constraint is something that must always be true — a hard rule. Extract:
- context: Where does this constraint come from? What happens if it's violated?
- constraint: The constraint itself — what must be true
- consequences: Implications of this constraint on development and operations
- affected_services: Specific service names if mentioned (empty array if project-wide)
- domain: One of: security, compliance, performance, scalability, data_model, operational

Return ONLY JSON. No prose, no code fences:
{
  "context": "string",
  "constraint": "string",
  "consequences": "string",
  "affected_services": ["string"],
  "domain": "string"
}"""

_PASS2_PRINCIPLE_SYSTEM = """\
You are an architectural knowledge extractor. Extract a structured PRINCIPLE \
from the document based on the item described below.

A principle is a standing preference — the team prefers X approach. Extract:
- context: Why this principle exists, what motivates it
- principle: The principle itself — what the team prefers
- consequences: What this means for development, known exceptions
- affected_services: Specific service names if mentioned (empty array if project-wide)
- domain: One of: security, compliance, performance, scalability, data_model, operational

Return ONLY JSON. No prose, no code fences:
{
  "context": "string",
  "principle": "string",
  "consequences": "string",
  "affected_services": ["string"],
  "domain": "string"
}"""

_PASS2_SYSTEMS = {
    "decision": _PASS2_DECISION_SYSTEM,
    "constraint": _PASS2_CONSTRAINT_SYSTEM,
    "principle": _PASS2_PRINCIPLE_SYSTEM,
}


async def _pass2_extract(
    *,
    item: DiscoveredItem,
    document: str,
    manifest_summary: str,
    filename: str,
    source_url: str,
    source_type: str,
    item_index: int,
    settings: Settings,
    _retry: bool = True,
) -> list[ChunkRecord]:
    """Run Pass 2 extraction for a single discovered item."""
    system_prompt = _PASS2_SYSTEMS.get(item.knowledge_type, _PASS2_DECISION_SYSTEM)

    # Collect relevant sections from the document
    section_context = _extract_relevant_sections(document, item.relevant_sections)

    user_prompt = (
        f"ITEM TO EXTRACT:\n"
        f"Title: {item.title}\n"
        f"Type: {item.knowledge_type}\n"
        f"Summary: {item.summary}\n\n"
        f"RELEVANT DOCUMENT SECTIONS:\n{section_context}\n\n"
        f"OTHER ITEMS IN THIS DOCUMENT (for cross-reference):\n{manifest_summary}"
    )

    result = await complete(
        [
            Message("system", system_prompt),
            Message("user", user_prompt),
        ],
        model=settings.STAGE2_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
    )

    try:
        parsed = json.loads(result.content.strip())
    except (json.JSONDecodeError, AttributeError):
        logger.error("Pass 2: failed to parse output for '%s'", item.title)
        return []

    # Validate and optionally retry
    chunks = _build_chunks_from_extraction(
        parsed=parsed,
        item=item,
        filename=filename,
        source_url=source_url,
        source_type=source_type,
        item_index=item_index,
    )

    validation_ok = _validate_extraction(chunks, item)

    if not validation_ok and _retry:
        logger.info("Pass 2: validation failed for '%s', retrying", item.title)
        return await _pass2_extract(
            item=item,
            document=document,
            manifest_summary=manifest_summary,
            filename=filename,
            source_url=source_url,
            source_type=source_type,
            item_index=item_index,
            settings=settings,
            _retry=False,  # One retry only
        )

    return chunks


def _extract_relevant_sections(document: str, section_names: list[str]) -> str:
    """Extract sections from the document matching the given heading names.

    If no section names match, returns the full document (capped).
    """
    if not section_names:
        return document[:6000]

    # Split document into sections
    parts = []
    current_heading = ""
    current_text = []

    for line in document.split("\n"):
        if line.strip().startswith("#"):
            if current_text:
                parts.append((current_heading, "\n".join(current_text)))
            current_heading = line.strip().lstrip("#").strip()
            current_text = [line]
        else:
            current_text.append(line)

    if current_text:
        parts.append((current_heading, "\n".join(current_text)))

    # Match requested sections (fuzzy: case-insensitive substring match)
    matched: list[str] = []
    for heading, text in parts:
        for requested in section_names:
            if requested.lower() in heading.lower() or heading.lower() in requested.lower():
                matched.append(text)
                break

    if not matched:
        # No sections matched — return full document
        return document[:6000]

    return "\n\n".join(matched)[:6000]


def _build_chunks_from_extraction(
    *,
    parsed: dict,
    item: DiscoveredItem,
    filename: str,
    source_url: str,
    source_type: str,
    item_index: int,
) -> list[ChunkRecord]:
    """Build ChunkRecords from a Pass 2 extraction result."""
    chunks: list[ChunkRecord] = []
    now = datetime.utcnow().isoformat() + "Z"

    stem = filename.replace(".", "_").replace("/", "_") or "doc"
    doc_id = f"norm-{stem}-{item_index + 1}"

    services = parsed.get("affected_services", [])
    if isinstance(services, str):
        services = [services]
    services = [s for s in services if s and s.lower() != "all"]

    domain = parsed.get("domain", "operational")
    kt = item.knowledge_type

    # Context chunk
    context_text = parsed.get("context", "")
    if context_text:
        chunks.append(ChunkRecord(
            id=f"{doc_id}-context",
            text=f"Decision: {item.title}\nSection: Context\n\n{context_text}",
            knowledge_type=kt,
            section_type="context",
            source_type=source_type or "design_doc",
            doc_id=doc_id,
            author=filename,
            affected_services=services,
            domain=domain,
            source_url=source_url,
            source_title=item.title,
            ingested_at=now,
        ))

    # Core assertion chunk (decision / constraint / principle)
    core_key = {
        "decision": "decision",
        "constraint": "constraint",
        "principle": "principle",
    }.get(kt, "decision")
    core_text = parsed.get(core_key, "")

    if core_text:
        chunks.append(ChunkRecord(
            id=f"{doc_id}-decision",
            text=f"Decision: {item.title}\nSection: Decision\n\n{core_text}",
            knowledge_type=kt,
            section_type="decision",
            source_type=source_type or "design_doc",
            doc_id=doc_id,
            author=filename,
            affected_services=services,
            domain=domain,
            source_url=source_url,
            source_title=item.title,
            ingested_at=now,
        ))

    # Consequences chunk
    consequences_text = parsed.get("consequences", "")
    if consequences_text:
        chunks.append(ChunkRecord(
            id=f"{doc_id}-consequences",
            text=f"Decision: {item.title}\nSection: Consequences\n\n{consequences_text}",
            knowledge_type=kt,
            section_type="consequences",
            source_type=source_type or "design_doc",
            doc_id=doc_id,
            author=filename,
            affected_services=services,
            domain=domain,
            source_url=source_url,
            source_title=item.title,
            ingested_at=now,
        ))

    # Rejected alternatives chunk (decisions only)
    rejected_text = parsed.get("rejected_alternatives", "")
    if rejected_text and kt == "decision":
        chunks.append(ChunkRecord(
            id=f"{doc_id}-rejected_alternatives",
            text=f"Decision: {item.title}\nSection: Rejected Alternatives\n\n{rejected_text}",
            knowledge_type="decision",
            section_type="rejected_alternatives",
            source_type=source_type or "design_doc",
            doc_id=doc_id,
            author=filename,
            affected_services=services,
            domain=domain,
            source_url=source_url,
            source_title=item.title,
            ingested_at=now,
        ))

    return chunks


# ── Validation ───────────────────────────────────────────────────────


def _validate_extraction(chunks: list[ChunkRecord], item: DiscoveredItem) -> bool:
    """Validate a Pass 2 extraction result. Returns True if valid."""
    if not chunks:
        return False

    # If Pass 1 said has_rejected_alternatives, verify we got one
    if item.has_rejected_alternatives:
        has_rejected = any(c.section_type == "rejected_alternatives" for c in chunks)
        if not has_rejected:
            logger.warning(
                "Validation: '%s' expected rejected_alternatives but none found",
                item.title,
            )
            return False

    # Check decision chunk content isn't suspiciously short
    # Strip the header (e.g. "Decision: X\nSection: Decision\n\n") before measuring
    decision_chunks = [c for c in chunks if c.section_type == "decision"]
    if decision_chunks:
        content = decision_chunks[0].text.split("\n\n", 1)[-1]
    if decision_chunks and len(content) < 30:
        logger.warning("Validation: '%s' decision chunk suspiciously short", item.title)
        return False

    return True


# ── Helpers ──────────────────────────────────────────────────────────


def _build_manifest_summary(manifest: list[DiscoveredItem]) -> str:
    """Build a compact summary of all discovered items for cross-reference."""
    lines = []
    for item in manifest:
        lines.append(f"- [{item.knowledge_type}] {item.title}: {item.summary}")
    return "\n".join(lines)