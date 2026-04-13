"""Corpus ingestion — ADR files, Confluence pages, Jira epics.

ADR files are parsed deterministically via regex (no LLM call). Each
section (Context, Decision, Consequences, Rejected Alternatives) becomes
a separate corpus chunk.

Confluence pages and Jira epics are routed through the two-pass LLM
extraction pipeline (extract_from_document) for knowledge-type-aware
chunking — decisions, constraints, and principles are extracted and
classified the same way as any other document.

Sources ingested (in order):
1. Local ADR markdown files in /adrs
2. Confluence pages labelled "architecture-decision"
3. Jira epics labelled "arch-decision"
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from app.config import Settings, get_settings
from app.corpus import ChunkRecord, ensure_collection, upsert
from app.extract import extract_from_document

logger = logging.getLogger(__name__)

_ADR_DIR = Path(__file__).resolve().parent.parent / "adrs"


@dataclass
class IngestResults:
    """Summary of an ingestion run."""

    adr: int = 0
    confluence: int = 0
    jira: int = 0
    errors: list[str] = field(default_factory=list)


# ── Public API ───────────────────────────────────────────────────────


async def ingest(settings: Settings | None = None) -> IngestResults:
    """Run the full ingestion pipeline."""
    s = settings or get_settings()
    await ensure_collection(s)

    results = IngestResults()

    # ── 1. Local ADR files ───────────────────────────────────────────
    logger.info("Scanning local ADR files...")
    try:
        adr_chunks = _ingest_local_adrs()
        if adr_chunks:
            await upsert(adr_chunks, s)
            results.adr = len(adr_chunks)
    except Exception as exc:
        logger.error("ADR ingestion failed: %s", exc)
        results.errors.append(f"ADR: {exc}")

    # ── 2. Confluence ────────────────────────────────────────────────
    if s.CONFLUENCE_BASE_URL and s.CONFLUENCE_TOKEN:
        logger.info("Fetching Confluence pages...")
        try:
            cf_chunks = await _ingest_confluence(s)
            if cf_chunks:
                await upsert(cf_chunks, s)
                results.confluence = len(cf_chunks)
        except Exception as exc:
            logger.error("Confluence ingestion failed: %s", exc)
            results.errors.append(f"Confluence: {exc}")
    else:
        logger.info("Confluence not configured — skipping")

    # ── 3. Jira ──────────────────────────────────────────────────────
    if s.JIRA_BASE_URL and s.JIRA_TOKEN:
        logger.info("Fetching Jira epics...")
        try:
            jira_chunks = await _ingest_jira(s)
            if jira_chunks:
                await upsert(jira_chunks, s)
                results.jira = len(jira_chunks)
        except Exception as exc:
            logger.error("Jira ingestion failed: %s", exc)
            results.errors.append(f"Jira: {exc}")
    else:
        logger.info("Jira not configured — skipping")

    logger.info(
        "Ingest complete — adr:%d confluence:%d jira:%d",
        results.adr,
        results.confluence,
        results.jira,
    )
    if results.errors:
        logger.warning("Errors: %s", "; ".join(results.errors))

    return results


# ── Local ADR ingestion ──────────────────────────────────────────────


def _ingest_local_adrs() -> list[ChunkRecord]:
    """Read all .md files from /adrs and convert to corpus chunks."""
    if not _ADR_DIR.exists():
        logger.info("/adrs directory not found — skipping local ADRs")
        return []

    files = sorted(_ADR_DIR.glob("*.md"))
    if not files:
        logger.info("No .md files found in /adrs")
        return []

    logger.info("Found %d ADR files", len(files))
    chunks: list[ChunkRecord] = []

    for file_path in files:
        raw = file_path.read_text(encoding="utf-8")
        try:
            chunks.extend(_parse_adr(raw, file_path.stem))
        except ValueError as exc:
            logger.warning("Skipping %s: %s", file_path.name, exc)

    return chunks


def _parse_adr(raw: str, filename: str) -> list[ChunkRecord]:
    """Parse a single ADR markdown file into section-level chunks."""
    frontmatter, body = _extract_frontmatter(raw)

    adr_id = frontmatter.get("id", filename)
    status = frontmatter.get("status", "active")
    title = frontmatter.get("title", adr_id)
    domain = frontmatter.get("constraint_type", "operational")
    author = frontmatter.get("author", "unknown")
    adr_date = frontmatter.get("date", "")

    services_raw = frontmatter.get("services", [])
    if isinstance(services_raw, str):
        services = [services_raw]
    else:
        services = list(services_raw)

    # Split body into sections on ## headings
    sections = _split_sections(body)

    if not sections:
        raise ValueError("no sections found — check ADR format")

    chunks: list[ChunkRecord] = []

    for heading, text in sections:
        if not text.strip():
            continue

        section_type = _classify_section_type(heading)

        # Prepend title + heading so the chunk is self-contained
        chunk_text = f"ADR: {title}\nSection: {heading}\n\n{text}"

        chunks.append(
            ChunkRecord(
                id=f"{adr_id}-{section_type}",
                text=chunk_text,
                knowledge_type="decision",
                source_type="adr",
                doc_id=adr_id,
                section_type=section_type,
                affected_services=services,
                date=adr_date,
                status=status,
                domain=domain,
                author=author,
                source_title=title,
            )
        )

    if not chunks:
        raise ValueError("no sections found — check ADR format")

    return chunks


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Split markdown body into (heading, content) pairs on ## headings."""
    parts = re.split(r"^(##\s+.+)$", body, flags=re.MULTILINE)

    sections: list[tuple[str, str]] = []
    current_heading = "Overview"

    for part in parts:
        stripped = part.strip()
        if stripped.startswith("## "):
            current_heading = stripped.removeprefix("## ").strip()
        elif stripped:
            sections.append((current_heading, stripped))

    return sections


def _classify_section_type(heading: str) -> str:
    """Map an ADR heading to a canonical section_type."""
    h = heading.lower()
    if any(kw in h for kw in ("context", "background", "problem")):
        return "context"
    if any(kw in h for kw in ("decision", "chosen")):
        return "decision"
    if any(kw in h for kw in ("consequence", "implication", "result")):
        return "consequences"
    if any(kw in h for kw in ("reject", "alternative", "considered")):
        return "rejected_alternatives"
    return "context"


def _extract_frontmatter(raw: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and body from a markdown file."""
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, flags=re.DOTALL)
    if not match:
        return {}, raw

    frontmatter: dict = {}
    for line in match.group(1).splitlines():
        colon_idx = line.find(":")
        if colon_idx == -1:
            continue

        key = line[:colon_idx].strip()
        value = line[colon_idx + 1:].strip()

        # Parse simple YAML arrays: [a, b, c]
        if value.startswith("[") and value.endswith("]"):
            value = [s.strip() for s in value[1:-1].split(",") if s.strip()]

        frontmatter[key] = value

    return frontmatter, match.group(2).strip()


# ── Confluence ingestion ─────────────────────────────────────────────


async def _ingest_confluence(settings: Settings) -> list[ChunkRecord]:
    """Fetch Confluence pages labelled 'architecture-decision' and extract knowledge.

    Routes each page through the two-pass extraction pipeline for
    proper knowledge-type classification (decision/constraint/principle)
    and section-level chunking.
    """
    space_key = settings.CONFLUENCE_SPACE_KEY
    if not space_key:
        logger.info("CONFLUENCE_SPACE_KEY not set — skipping")
        return []

    url = (
        f"{settings.CONFLUENCE_BASE_URL}/wiki/rest/api/content"
        f"?spaceKey={space_key}&label=architecture-decision"
        f"&expand=body.storage,version&limit=50"
    )

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            url,
            auth=(settings.CONFLUENCE_USER_EMAIL, settings.CONFLUENCE_TOKEN),
            headers={
                "Accept": "application/json",
            },
        )

    if resp.status_code != 200:
        raise RuntimeError(f"Confluence API {resp.status_code}")

    data = resp.json()
    all_chunks: list[ChunkRecord] = []

    for page in data.get("results", []):
        html = page.get("body", {}).get("storage", {}).get("value", "")
        if not html.strip():
            continue

        page_id = page["id"]
        page_title = page.get("title", "")
        page_author = (
            page.get("version", {}).get("by", {}).get("displayName", "unknown")
        )
        page_date = (
            page.get("version", {}).get("when", "")[:10]
        )
        source_url = (
            f"{settings.CONFLUENCE_BASE_URL}/wiki/pages/viewpage.action"
            f"?pageId={page_id}"
        )

        logger.info("Extracting from Confluence page: %s (%s)", page_title, page_id)

        result = await extract_from_document(
            html,
            filename=f"confluence-{page_id}-{page_title}",
            source_url=source_url,
            source_type="confluence",
            author=page_author,
            date=page_date,
            settings=settings,
        )

        if not result.chunks:
            logger.warning(
                "No knowledge items extracted from Confluence page '%s' (%s). "
                "The page has the 'architecture-decision' label but the extraction "
                "pipeline found no decisions, constraints, or principles.",
                page_title, page_id,
            )
            continue

        all_chunks.extend(result.chunks)
        logger.info(
            "Confluence page '%s': %d items, %d chunks",
            page_title, result.items_extracted, len(result.chunks),
        )

    return all_chunks


# ── Jira ingestion ───────────────────────────────────────────────────


async def _ingest_jira(settings: Settings) -> list[ChunkRecord]:
    """Fetch Jira epics labelled 'arch-decision' and extract knowledge.

    Routes each epic through the two-pass extraction pipeline for
    proper knowledge-type classification and section-level chunking.
    """
    jql = "issuetype = Epic AND labels = \"arch-decision\" ORDER BY created DESC"

    url = (
        f"{settings.JIRA_BASE_URL}/rest/api/3/search"
        f"?jql={httpx.URL('', params={'jql': jql}).params.get('jql')}"
        f"&maxResults=50&fields=summary,description,assignee,created"
    )

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            url,
            auth=(settings.JIRA_USER_EMAIL, settings.JIRA_TOKEN),
            headers={
                "Accept": "application/json",
            },
        )

    if resp.status_code != 200:
        raise RuntimeError(f"Jira API {resp.status_code}")

    data = resp.json()
    all_chunks: list[ChunkRecord] = []

    for issue in data.get("issues", []):
        fields = issue.get("fields", {})
        description = _extract_jira_text(fields.get("description"))
        if not description:
            continue

        issue_key = issue["key"]
        summary = fields.get("summary", "")
        assignee = (fields.get("assignee") or {}).get("displayName", "unknown")
        created = (fields.get("created") or "")[:10]
        source_url = f"{settings.JIRA_BASE_URL}/browse/{issue_key}"

        # Prepend summary as title so the LLM has full context
        content = f"# {summary}\n\n{description}"

        logger.info("Extracting from Jira epic: %s (%s)", summary, issue_key)

        result = await extract_from_document(
            content,
            filename=f"jira-{issue_key}",
            source_url=source_url,
            source_type="jira",
            author=assignee,
            date=created,
            settings=settings,
        )

        if not result.chunks:
            logger.warning(
                "No knowledge items extracted from Jira epic '%s' (%s).",
                summary, issue_key,
            )
            continue

        all_chunks.extend(result.chunks)
        logger.info(
            "Jira epic '%s': %d items, %d chunks",
            summary, result.items_extracted, len(result.chunks),
        )

    return all_chunks


# ── Helpers ──────────────────────────────────────────────────────────


def _extract_jira_text(doc) -> str:
    """Extract plain text from a Jira Atlassian Document Format node."""
    if not doc:
        return ""
    if isinstance(doc, str):
        return doc

    texts: list[str] = []

    def walk(node: dict) -> None:
        if node.get("type") == "text":
            texts.append(node.get("text", ""))
        for child in node.get("content", []):
            walk(child)

    walk(doc)
    return " ".join(texts).strip()