"""Rules file bridge: CLAUDE.md / .cursorrules / AGENTS.md to corpus.

Reads an AI coding rules file, uses an LLM to extract architectural
knowledge items (decisions, constraints, and principles) from the
free-form text, and converts them into structured ChunkRecords for
the corpus. Ignores code style, commands, and setup instructions
that don't represent architectural knowledge.

This bridges the gap between "teams that already have rules files"
and "arch-conscience needs structured knowledge items."
"""

import json
import logging
from pathlib import Path

from app.config import Settings, get_settings
from app.corpus import ChunkRecord
from app.llm import Message, complete

logger = logging.getLogger(__name__)

_EXTRACTION_SYSTEM = """\
You are an architectural knowledge extractor. Given the contents of an AI coding \
rules file (CLAUDE.md, .cursorrules, AGENTS.md, rules.md, or similar), your job \
is to identify and extract architectural knowledge items: decisions, constraints, \
and principles.

Definitions:
- Decision: The team chose X over Y. There are alternatives that were considered \
  and rejected. Example: "We use PostgreSQL, not DynamoDB, because the team lacks \
  DynamoDB operational experience."
- Constraint: Something that must always be true — a hard rule, often from \
  compliance or security. No alternatives — it's non-negotiable. Example: "All PII \
  must be encrypted at rest using AES-256. No exceptions."
- Principle: A standing preference or guideline. May have exceptions. Example: \
  "Prefer composition over inheritance" or "Keep business logic out of HTTP handlers."

How to classify:
- If the text mentions alternatives that were considered/rejected → decision
- If the text says "must", "never", "no exceptions", "always", or cites compliance/legal → constraint
- If the text says "prefer", "when possible", "guideline", "by default" → principle
- If unsure between decision and constraint, choose decision
- If unsure between decision and principle, check if alternatives are mentioned — \
  if yes, it's a decision; if it's just a preference, it's a principle

What IS architectural knowledge:
- Which technologies, databases, or frameworks to use (and which NOT to use)
- How services communicate with each other
- Authentication and authorization patterns
- Data storage and access patterns
- Service boundaries and ownership
- Infrastructure choices with architectural implications
- Explicit constraints or prohibitions on approaches
- Standing engineering principles and preferences
- Tribal knowledge: implicit decisions embedded in warnings or known issues

What is NOT architectural knowledge (ignore these):
- Code formatting and style (indentation, naming conventions, export style)
- Linting and tooling configuration
- CLI commands and setup instructions
- Testing framework preferences (unless they have architectural impact)
- Git conventions (commit messages, branch naming)
- General coding patterns that don't affect architecture

For each item, extract:
- title: A short descriptive title
- knowledge_type: decision | constraint | principle
- content: The core assertion — what was decided, what must be true, or what is preferred
- context: Why it exists (if stated or inferable). For constraints, include what \
  happens if violated. For principles, include the motivation.
- rejected: What was rejected or prohibited (decisions only — leave empty for \
  constraints and principles)
- services: Which services this applies to (use an empty array [] if project-wide)
- domain: One of: security, compliance, performance, scalability, data_model, operational

Return ONLY a JSON object with an "items" key containing an array. No prose, no code fences:
{
  "items": [
    {
      "title": "string",
      "knowledge_type": "decision | constraint | principle",
      "content": "string",
      "context": "string or empty",
      "rejected": "string or empty",
      "services": ["string"],
      "domain": "string"
    }
  ]
}

For the "services" field: identify specific service names if mentioned or inferable.
If the item is truly project-wide, use an empty array [].

If no architectural knowledge items are found, return: {"items": []}"""


async def extract_decisions_from_rules(
    content: str,
    source_file: str = "rules.md",
    settings: Settings | None = None,
) -> list[ChunkRecord]:
    """Extract architectural knowledge from a rules file.

    Args:
        content: The raw text content of the rules file.
        source_file: Filename for provenance tracking.
        settings: Optional settings override.

    Returns:
        List of ChunkRecords ready for corpus upsert.
    """
    s = settings or get_settings()

    result = await complete(
        [
            Message("system", _EXTRACTION_SYSTEM),
            Message("user", f"Rules file: {source_file}\n\n{content}"),
        ],
        model=s.STAGE2_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
    )

    try:
        raw = result.content.strip()
        parsed = json.loads(raw)

        # Handle {"items": [...]}, {"decisions": [...]}, and bare [...]
        if isinstance(parsed, dict):
            items = parsed.get("items", parsed.get("decisions", []))
        elif isinstance(parsed, list):
            items = parsed
        else:
            items = []

    except (json.JSONDecodeError, AttributeError):
        logger.error("Failed to parse LLM output: %s", result.content[:200])
        return []

    if not items:
        logger.info("No architectural knowledge found in %s", source_file)
        return []

    chunks: list[ChunkRecord] = []
    source_stem = Path(source_file).stem.lstrip(".").replace(".", "_") or source_file

    for i, d in enumerate(items):
        title = d.get("title", f"Item {i + 1}")
        knowledge_type = d.get("knowledge_type", "decision")
        if knowledge_type not in ("decision", "constraint", "principle"):
            knowledge_type = "decision"

        content_text = d.get("content", d.get("decision", ""))
        context_text = d.get("context", "")
        rejected_text = d.get("rejected", "")
        services = d.get("services", ["all"])
        # Accept both "domain" (new) and "constraint_type" (old) from LLM output
        domain = d.get("domain", d.get("constraint_type", "operational"))

        if isinstance(services, str):
            services = [services]
        # Filter out "all" placeholder - empty list means project-wide
        services = [s for s in services if s and s.lower() != "all"]

        doc_id = f"rules-{source_stem}-{i + 1}"

        # Build the core chunk — label adapts to knowledge type
        type_label = {
            "decision": "Rule",
            "constraint": "Constraint",
            "principle": "Principle",
        }.get(knowledge_type, "Rule")

        core_parts = [f"{type_label}: {title}", "Section: Decision", ""]
        if context_text:
            core_parts.append(context_text)
            core_parts.append("")
        core_parts.append(content_text)

        chunks.append(
            ChunkRecord(
                id=f"{doc_id}-decision",
                text="\n".join(core_parts),
                knowledge_type=knowledge_type,
                source_type="rules_file",
                doc_id=doc_id,
                section_type="decision",
                affected_services=services,
                status="active",
                domain=domain,
                author=source_file,
                source_title=title,
            )
        )

        # If there are rejected alternatives, create a separate chunk
        # (only meaningful for decisions)
        if rejected_text and knowledge_type == "decision":
            rejected_chunk_text = (
                f"{type_label}: {title}\n"
                f"Section: Rejected Alternatives\n\n"
                f"{rejected_text}"
            )
            chunks.append(
                ChunkRecord(
                    id=f"{doc_id}-rejected_alternatives",
                    text=rejected_chunk_text,
                    knowledge_type="decision",
                    source_type="rules_file",
                    doc_id=doc_id,
                    section_type="rejected_alternatives",
                    affected_services=services,
                    status="active",
                    domain=domain,
                    author=source_file,
                    source_title=title,
                )
            )

        # Context chunk for items that have substantial context
        if context_text and len(context_text) > 30:
            context_chunk_text = (
                f"{type_label}: {title}\n"
                f"Section: Context\n\n"
                f"{context_text}"
            )
            chunks.append(
                ChunkRecord(
                    id=f"{doc_id}-context",
                    text=context_chunk_text,
                    knowledge_type=knowledge_type,
                    source_type="rules_file",
                    doc_id=doc_id,
                    section_type="context",
                    affected_services=services,
                    status="active",
                    domain=domain,
                    author=source_file,
                    source_title=title,
                )
            )

    logger.info(
        "Extracted %d items (%d chunks) from %s",
        len(items),
        len(chunks),
        source_file,
    )

    return chunks


async def ingest_rules_file(
    file_path: str,
    settings: Settings | None = None,
) -> list[ChunkRecord]:
    """Read a rules file from disk and extract architectural knowledge.

    Args:
        file_path: Path to the rules file.
        settings: Optional settings override.

    Returns:
        List of ChunkRecords extracted.
    """
    path = Path(file_path)
    if not path.exists():
        logger.error("Rules file not found: %s", file_path)
        return []

    content = path.read_text(encoding="utf-8")
    filename = path.name

    return await extract_decisions_from_rules(
        content=content,
        source_file=filename,
        settings=settings,
    )


# Well-known rules file names to search for
KNOWN_RULES_FILES = [
    "CLAUDE.md",
    ".cursorrules",
    "AGENTS.md",
    "rules.md",
    "implementation.md",
    "architecture.md",
    ".github/copilot-instructions.md",
    ".windsurfrules",
    "CODEX.md",
    "GEMINI.md",
]


async def discover_and_ingest(
    project_root: str = ".",
    settings: Settings | None = None,
) -> list[ChunkRecord]:
    """Discover and ingest all known rules files from a project root.

    Args:
        project_root: Path to the project root directory.
        settings: Optional settings override.

    Returns:
        All ChunkRecords extracted across all discovered files.
    """
    root = Path(project_root)
    all_chunks: list[ChunkRecord] = []

    for filename in KNOWN_RULES_FILES:
        path = root / filename
        if path.exists():
            logger.info("Found rules file: %s", path)
            chunks = await ingest_rules_file(str(path), settings)
            all_chunks.extend(chunks)

    if not all_chunks:
        logger.info("No rules files found in %s", project_root)

    return all_chunks