"""Document preprocessor — raw content to clean markdown.

Deterministic (no LLM call). Converts raw document content into clean
markdown suitable for the two-pass extraction pipeline.

Uses markdownify for HTML→markdown conversion with table preservation.
Custom image converter produces [Image: alt_text] placeholders.

Handles:
- HTML → markdown (headings, tables, lists, bold, italic, links)
- Image tags → placeholders [Image: alt_text]
- Encoding normalization (UTF-8, BOM, line endings)
- Headingless documents → synthetic numbered sections
"""

import re

from markdownify import MarkdownConverter


# ── Custom converter ─────────────────────────────────────────────────


class _ArchConscienceConverter(MarkdownConverter):
    """Custom markdownify converter with image placeholders."""

    def convert_img(self, el, text, parent_tags):
        """Convert <img> tags to [Image: alt_text] placeholders."""
        alt = el.get("alt", "")
        if alt:
            return f"[Image: {alt}]"

        src = el.get("src", "")
        if src:
            filename = src.split("/")[-1].split("?")[0]
            return f"[Image: {filename}]"

        return "[Image]"


def _html_to_markdown(html: str) -> str:
    """Convert HTML to markdown using markdownify."""
    return _ArchConscienceConverter(
        heading_style="ATX",
        escape_asterisks=False,
        escape_underscores=False,
        table_infer_header=True,
    ).convert(html)


# ── Public API ───────────────────────────────────────────────────────


def preprocess(content: str, filename: str = "") -> str:
    """Convert raw document content to clean markdown.

    Args:
        content: Raw document text (HTML, markdown, or plain text).
        filename: Optional filename hint for format detection.

    Returns:
        Clean markdown string ready for LLM extraction.
    """
    text = _normalize_encoding(content)

    # Detect if content is HTML-heavy and convert
    if _is_html(text):
        text = _html_to_markdown(text)

    # Clean up whitespace
    text = _normalize_whitespace(text)

    # If no headings found, create synthetic sections
    if not _has_headings(text):
        text = _add_synthetic_headings(text)

    return text.strip()


# ── Encoding normalization ───────────────────────────────────────────


def _normalize_encoding(text: str) -> str:
    """Normalize encoding: strip BOM, normalize line endings."""
    # Strip UTF-8 BOM
    if text.startswith("\ufeff"):
        text = text[1:]

    # Normalize line endings to \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    return text


# ── HTML detection ───────────────────────────────────────────────────


def _is_html(text: str) -> bool:
    """Detect if content is primarily HTML."""
    tag_count = len(re.findall(
        r"<(?:p|div|table|tr|td|th|h[1-6]|ul|ol|li|span|br|img)\b",
        text,
        re.IGNORECASE,
    ))
    return tag_count > 3


# ── Heading detection and synthetic sections ─────────────────────────


def _has_headings(text: str) -> bool:
    """Check if markdown text contains any headings."""
    return bool(re.search(r"^#{1,6}\s+\S", text, re.MULTILINE))


def _add_synthetic_headings(text: str) -> str:
    """Add synthetic section headings to headingless documents.

    Splits on paragraph groups (double newlines) and assigns
    numbered section identifiers.
    """
    paragraphs = re.split(r"\n{2,}", text.strip())

    if len(paragraphs) <= 1:
        return text

    sections: list[str] = []
    section_num = 1
    current_group: list[str] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        current_group.append(para)

        # Group roughly 2-3 paragraphs per section
        if len(current_group) >= 2:
            sections.append(f"## Section {section_num}\n\n" + "\n\n".join(current_group))
            current_group = []
            section_num += 1

    # Flush remaining paragraphs
    if current_group:
        sections.append(f"## Section {section_num}\n\n" + "\n\n".join(current_group))

    return "\n\n".join(sections)


# ── Whitespace normalization ─────────────────────────────────────────


def _normalize_whitespace(text: str) -> str:
    """Clean up excessive whitespace without destroying structure."""
    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    return text