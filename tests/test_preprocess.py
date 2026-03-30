"""Preprocessor tests — HTML conversion, tables, images, headings.

Run with:
    pytest tests/test_preprocess.py -v
"""

from app.preprocess import preprocess, _html_to_markdown, _normalize_encoding


class TestEncodingNormalization:
    """UTF-8 BOM, line endings."""

    def test_strips_utf8_bom(self):
        assert _normalize_encoding("\ufeffHello") == "Hello"

    def test_normalizes_crlf(self):
        assert _normalize_encoding("line1\r\nline2") == "line1\nline2"

    def test_normalizes_cr(self):
        assert _normalize_encoding("line1\rline2") == "line1\nline2"


class TestHtmlToMarkdown:
    """HTML → markdown conversion via markdownify."""

    def test_converts_headings(self):
        html = "<h1>Title</h1><h2>Subtitle</h2><h3>Section</h3>"
        md = _html_to_markdown(html)
        assert "# Title" in md
        assert "## Subtitle" in md
        assert "### Section" in md

    def test_converts_bold_and_italic(self):
        html = "<p><strong>bold</strong> and <em>italic</em></p>"
        md = _html_to_markdown(html)
        assert "**bold**" in md
        assert "*italic*" in md

    def test_converts_links(self):
        html = '<a href="https://example.com">click here</a>'
        md = _html_to_markdown(html)
        assert "[click here](https://example.com)" in md

    def test_converts_lists(self):
        html = "<ul><li>first</li><li>second</li></ul>"
        md = _html_to_markdown(html)
        assert "first" in md
        assert "second" in md

    def test_replaces_images_with_alt(self):
        html = '<img src="diagram.png" alt="Architecture diagram" />'
        md = _html_to_markdown(html)
        assert "[Image: Architecture diagram]" in md

    def test_replaces_images_without_alt(self):
        html = '<img src="/uploads/flow.png" />'
        md = _html_to_markdown(html)
        assert "[Image: flow.png]" in md

    def test_replaces_images_no_attrs(self):
        html = '<img />'
        md = _html_to_markdown(html)
        assert "[Image]" in md

    def test_strips_remaining_tags(self):
        html = "<div class='wrapper'><span>content</span></div>"
        md = _html_to_markdown(html)
        assert "content" in md


class TestTableConversion:
    """HTML tables → markdown tables."""

    def test_simple_table_with_headers(self):
        html = (
            "<table>"
            "<tr><th>Name</th><th>Value</th></tr>"
            "<tr><td>Alpha</td><td>1</td></tr>"
            "<tr><td>Beta</td><td>2</td></tr>"
            "</table>"
        )
        md = _html_to_markdown(html)
        assert "Name" in md
        assert "Value" in md
        assert "Alpha" in md
        assert "Beta" in md
        # Should have pipe-separated table
        assert "|" in md

    def test_table_without_headers(self):
        """Tables with only td still produce markdown table."""
        html = (
            "<table>"
            "<tr><td>A</td><td>B</td></tr>"
            "<tr><td>C</td><td>D</td></tr>"
            "</table>"
        )
        md = _html_to_markdown(html)
        assert "A" in md
        assert "D" in md
        assert "|" in md


class TestSyntheticHeadings:
    """Headingless document fallback."""

    def test_markdown_with_headings_unchanged(self):
        """Documents with headings pass through without synthetic sections."""
        md = "## Existing Heading\n\nSome content here."
        result = preprocess(md)
        assert "Section 1" not in result
        assert "## Existing Heading" in result

    def test_headingless_gets_synthetic_sections(self):
        """Plain text without headings gets numbered section headings."""
        text = "First paragraph of content.\n\nSecond paragraph here.\n\nThird paragraph follows.\n\nFourth paragraph ends."
        result = preprocess(text)
        assert "## Section 1" in result
        assert "## Section 2" in result

    def test_single_paragraph_unchanged(self):
        """Single paragraph doesn't get synthetic headings."""
        text = "Just one paragraph with no breaks."
        result = preprocess(text)
        assert "## Section" not in result


class TestPreprocessIntegration:
    """Full pipeline: raw content → clean markdown."""

    def test_html_document_fully_converted(self):
        """Full HTML document converts to clean markdown."""
        html = (
            "<h1>Architecture</h1>"
            "<p>We decided to use <strong>PostgreSQL</strong> for all data.</p>"
            "<h2>Alternatives</h2>"
            "<table><tr><th>Option</th><th>Reason rejected</th></tr>"
            "<tr><td>MongoDB</td><td>No ACID</td></tr></table>"
            '<img src="arch.png" alt="System architecture" />'
        )
        result = preprocess(html)

        assert "# Architecture" in result
        assert "**PostgreSQL**" in result
        assert "## Alternatives" in result
        assert "Option" in result
        assert "MongoDB" in result
        assert "[Image: System architecture]" in result

    def test_already_clean_markdown_passes_through(self):
        """Clean markdown passes through with minimal changes."""
        md = "## Decision\n\nUse JWT for authentication.\n\n## Rejected\n\nSession cookies rejected."
        result = preprocess(md)
        assert "## Decision" in result
        assert "## Rejected" in result
        assert "Use JWT for authentication." in result

    def test_bom_and_crlf_cleaned(self):
        """BOM and CRLF are normalized."""
        text = "\ufeff## Title\r\n\r\nContent here.\r\n"
        result = preprocess(text)
        assert not result.startswith("\ufeff")
        assert "\r" not in result
        assert "## Title" in result