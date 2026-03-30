"""Document format detection and routing.

Three ingestion paths. detect_format() routes to the right handler:

1. Has YAML frontmatter with id/status/services → ADR regex parser (existing)
2. Known rules filename (CLAUDE.md, .cursorrules, etc.) → Rules bridge (existing)
3. Everything else → Two-pass normalizer (new)

All three paths produce list[ChunkRecord] in the same schema.
"""

import re
from enum import Enum

from app.rules_bridge import KNOWN_RULES_FILES


class DocumentFormat(Enum):
    """Detected document format determining the ingestion path."""
    ADR = "adr"
    RULES_FILE = "rules_file"
    GENERIC = "generic"


def detect_format(content: str, filename: str = "") -> DocumentFormat:
    """Detect document format and return the appropriate ingestion path.

    Args:
        content: Raw document text.
        filename: Filename (e.g. "CLAUDE.md", "adr-001.md").

    Returns:
        DocumentFormat indicating which parser to use.
    """
    # Path 1: ADR with YAML frontmatter
    if _has_adr_frontmatter(content):
        return DocumentFormat.ADR

    # Path 2: Known rules file
    if _is_known_rules_file(filename):
        return DocumentFormat.RULES_FILE

    # Path 3: Everything else
    return DocumentFormat.GENERIC


def _has_adr_frontmatter(content: str) -> bool:
    """Check if content has YAML frontmatter with ADR-like fields.

    Requires frontmatter to contain at least 'id' or 'title', plus
    one of 'status' or 'services' to distinguish from generic YAML.
    """
    match = re.match(r"^---\n(.*?)\n---", content, flags=re.DOTALL)
    if not match:
        return False

    frontmatter = match.group(1).lower()

    has_id_or_title = "id:" in frontmatter or "title:" in frontmatter
    has_adr_field = "status:" in frontmatter or "services:" in frontmatter

    return has_id_or_title and has_adr_field


def _is_known_rules_file(filename: str) -> bool:
    """Check if filename matches a known AI coding rules file."""
    if not filename:
        return False

    # Normalize: strip path, compare basename
    basename = filename.split("/")[-1].split("\\")[-1]

    # Check against known list (case-insensitive for safety)
    known_basenames = {f.split("/")[-1].lower() for f in KNOWN_RULES_FILES}

    return basename.lower() in known_basenames