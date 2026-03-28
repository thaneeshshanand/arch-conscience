"""Rules file bridge: CLAUDE.md / .cursorrules / AGENTS.md to corpus.

Reads an AI coding rules file, uses an LLM to extract architectural
decisions from the free-form text, and converts them into structured
ChunkRecords for the corpus. Ignores code style, commands, and setup
instructions that don't represent architectural decisions.

This bridges the gap between "teams that already have rules files"
and "arch-conscience needs structured ADRs."
"""

import json
import logging
from pathlib import Path

from app.config import Settings, get_settings
from app.corpus import ChunkRecord
from app.llm import Message, complete

logger = logging.getLogger(__name__)

_EXTRACTION_SYSTEM = """\
You are an architectural decision extractor. Given the contents of an AI coding \
rules file (CLAUDE.md, .cursorrules, AGENTS.md, rules.md, or similar), your job \
is to identify and extract ONLY the architectural decisions.

An architectural decision is a choice about:
- Which technologies, databases, or frameworks to use (and which NOT to use)
- How services communicate with each other
- Authentication and authorization patterns
- Data storage and access patterns
- Service boundaries and ownership
- Infrastructure choices with architectural implications
- Explicit constraints or prohibitions on approaches

NOT architectural decisions (ignore these):
- Code formatting and style (indentation, naming conventions, export style)
- Linting and tooling configuration
- CLI commands and setup instructions
- Testing framework preferences (unless they have architectural impact)
- Git conventions (commit messages, branch naming)
- General coding patterns (functional vs class, arrow functions)

For each architectural decision you find, extract:
- title: A short descriptive title
- decision: What was decided
- context: Why it was decided (if stated or inferable)
- rejected: What was rejected or prohibited (if stated or inferable)
- services: Which services this applies to (use "all" if project-wide)
- constraint_type: One of: security, compliance, performance, scalability, data_model, operational

Also extract tribal knowledge that represents implicit architectural decisions:
- "The payments module uses a legacy REST client, don't refactor it yet" is an implicit decision
- "Auth middleware runs before rate limiting, order matters" is an implicit constraint

Return ONLY a JSON object with a "decisions" key containing an array. No prose, no code fences:
{
  "decisions": [
    {
      "title": "string",
      "decision": "string",
      "context": "string or empty",
      "rejected": "string or empty",
      "services": ["string"],
      "constraint_type": "string"
    }
  ]
}

For the "services" field: identify specific service names if mentioned or inferable
(e.g. "auth-service", "payments-service", "api-gateway"). If the decision is truly
project-wide and no specific services can be identified, use an empty array [].

If no architectural decisions are found, return: {"decisions": []}"""


async def extract_decisions_from_rules(
    content: str,
    source_file: str = "rules.md",
    settings: Settings | None = None,
) -> list[ChunkRecord]:
    """Extract architectural decisions from a rules file.

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

        # Handle both {"decisions": [...]} and bare [...]
        if isinstance(parsed, dict):
            decisions = parsed.get("decisions", [])
        elif isinstance(parsed, list):
            decisions = parsed
        else:
            decisions = []

    except (json.JSONDecodeError, AttributeError):
        logger.error("Failed to parse LLM output: %s", result.content[:200])
        return []

    if not decisions:
        logger.info("No architectural decisions found in %s", source_file)
        return []

    chunks: list[ChunkRecord] = []
    source_stem = Path(source_file).stem.lstrip(".").replace(".", "_") or source_file

    for i, d in enumerate(decisions):
        title = d.get("title", f"Decision {i + 1}")
        decision_text = d.get("decision", "")
        context_text = d.get("context", "")
        rejected_text = d.get("rejected", "")
        services = d.get("services", ["all"])
        constraint_type = d.get("constraint_type", "operational")

        if isinstance(services, str):
            services = [services]

        doc_id = f"rules-{source_stem}-{i + 1}"

        # Build a decision chunk (mirrors ADR format: "ADR: <title>\nSection: <heading>")
        decision_parts = [f"Rule: {title}", "Section: Decision", ""]
        if context_text:
            decision_parts.append(context_text)
            decision_parts.append("")
        decision_parts.append(decision_text)

        chunks.append(
            ChunkRecord(
                id=f"{doc_id}-decision",
                text="\n".join(decision_parts),
                source_type="rules_file",
                doc_id=doc_id,
                section_type="decision",
                affected_services=services,
                status="active",
                constraint_type=constraint_type,
                author=source_file,
            )
        )

        # If there are rejected alternatives, create a separate chunk
        if rejected_text:
            rejected_chunk_text = (
                f"Rule: {title}\n"
                f"Section: Rejected Alternatives\n\n"
                f"{rejected_text}"
            )
            chunks.append(
                ChunkRecord(
                    id=f"{doc_id}-rejected_alternatives",
                    text=rejected_chunk_text,
                    source_type="rules_file",
                    doc_id=doc_id,
                    section_type="rejected_alternatives",
                    affected_services=services,
                    status="active",
                    constraint_type=constraint_type,
                    author=source_file,
                )
            )

    logger.info(
        "Extracted %d decisions (%d chunks) from %s",
        len(decisions),
        len(chunks),
        source_file,
    )

    return chunks


async def ingest_rules_file(
    file_path: str,
    settings: Settings | None = None,
) -> list[ChunkRecord]:
    """Read a rules file from disk and extract architectural decisions.

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