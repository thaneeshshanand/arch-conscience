"""Ingest tests — Confluence and Jira ingestion via format-agnostic pipeline.

Tests the upgraded ingestion paths that route Confluence pages and
Jira epics through extract_from_document for knowledge-type-aware
chunking instead of dumb size-based splitting.

Run with:
    pytest tests/test_ingest.py -v
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import Settings
from app.corpus import ChunkRecord
from app.extract import ExtractionResult


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def confluence_settings(test_settings):
    """Settings with Confluence credentials configured."""
    return Settings(
        OPENAI_API_KEY="sk-test",
        GITHUB_TOKEN="ghp-test",
        GITHUB_WEBHOOK_SECRET="secret",
        QDRANT_URL="http://localhost:6333",
        CONFLUENCE_BASE_URL="https://myorg.atlassian.net",
        CONFLUENCE_TOKEN="cf-token-test",
        CONFLUENCE_SPACE_KEY="ENG",
        STAGE2_MODEL="gpt-4o",
    )


@pytest.fixture
def jira_settings(test_settings):
    """Settings with Jira credentials configured."""
    return Settings(
        OPENAI_API_KEY="sk-test",
        GITHUB_TOKEN="ghp-test",
        GITHUB_WEBHOOK_SECRET="secret",
        QDRANT_URL="http://localhost:6333",
        JIRA_BASE_URL="https://myorg.atlassian.net",
        JIRA_TOKEN="jira-token-test",
        STAGE2_MODEL="gpt-4o",
    )


@pytest.fixture
def confluence_api_response():
    """Realistic Confluence REST API response with two pages."""
    return {
        "results": [
            {
                "id": "12345",
                "title": "Authentication Architecture",
                "body": {
                    "storage": {
                        "value": (
                            "<h2>Decision</h2>"
                            "<p>Use JWT for stateless auth. Session cookies "
                            "were rejected due to GDPR and scaling concerns.</p>"
                            "<h2>Constraints</h2>"
                            "<p>All PII must be encrypted at rest.</p>"
                        ),
                    },
                },
                "version": {
                    "by": {"displayName": "Alice Engineer"},
                    "when": "2025-06-15T10:30:00.000Z",
                },
            },
            {
                "id": "67890",
                "title": "Data Layer Patterns",
                "body": {
                    "storage": {
                        "value": (
                            "<h2>Overview</h2>"
                            "<p>PostgreSQL for OLTP. Redis for caching only.</p>"
                        ),
                    },
                },
                "version": {
                    "by": {"displayName": "Bob Architect"},
                    "when": "2025-08-01T14:00:00.000Z",
                },
            },
        ],
    }


@pytest.fixture
def jira_api_response():
    """Realistic Jira REST API response with one epic."""
    return {
        "issues": [
            {
                "key": "ARCH-42",
                "fields": {
                    "summary": "Migrate auth to OAuth2",
                    "description": {
                        "type": "doc",
                        "content": [
                            {
                                "type": "paragraph",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": (
                                            "Replace custom JWT auth with OAuth2. "
                                            "Evaluated Auth0, Keycloak, and custom implementation. "
                                            "Chose Keycloak for self-hosted control."
                                        ),
                                    },
                                ],
                            },
                        ],
                    },
                    "assignee": {"displayName": "Charlie Dev"},
                    "created": "2025-09-10T08:00:00.000Z",
                },
            },
        ],
    }


@pytest.fixture
def extraction_result_auth():
    """Mock extraction result for an auth-related page."""
    return ExtractionResult(
        chunks=[
            ChunkRecord(
                id="norm-confluence-12345-1-decision",
                text="Decision: Use JWT\nSection: Decision\n\nUse JWT for stateless auth.",
                knowledge_type="decision",
                section_type="decision",
                source_type="confluence",
                doc_id="norm-confluence-12345-1",
                domain="security",
                source_url="https://myorg.atlassian.net/wiki/pages/viewpage.action?pageId=12345",
                source_title="Use JWT for stateless auth",
            ),
            ChunkRecord(
                id="norm-confluence-12345-2-decision",
                text="Decision: PII encryption\nSection: Decision\n\nAll PII encrypted at rest.",
                knowledge_type="constraint",
                section_type="decision",
                source_type="confluence",
                doc_id="norm-confluence-12345-2",
                domain="compliance",
                source_url="https://myorg.atlassian.net/wiki/pages/viewpage.action?pageId=12345",
                source_title="PII encryption requirement",
            ),
        ],
        items_discovered=2,
        items_extracted=2,
    )


@pytest.fixture
def extraction_result_data():
    """Mock extraction result for a data layer page."""
    return ExtractionResult(
        chunks=[
            ChunkRecord(
                id="norm-confluence-67890-1-decision",
                text="Decision: PostgreSQL for OLTP\nSection: Decision\n\nUse PostgreSQL.",
                knowledge_type="decision",
                section_type="decision",
                source_type="confluence",
                doc_id="norm-confluence-67890-1",
                domain="data_model",
                source_url="https://myorg.atlassian.net/wiki/pages/viewpage.action?pageId=67890",
                source_title="PostgreSQL for OLTP",
            ),
        ],
        items_discovered=1,
        items_extracted=1,
    )


@pytest.fixture
def extraction_result_jira():
    """Mock extraction result for a Jira epic."""
    return ExtractionResult(
        chunks=[
            ChunkRecord(
                id="norm-jira-ARCH-42-1-decision",
                text="Decision: Migrate to OAuth2\nSection: Decision\n\nUse Keycloak.",
                knowledge_type="decision",
                section_type="decision",
                source_type="jira",
                doc_id="norm-jira-ARCH-42-1",
                domain="security",
                source_url="https://myorg.atlassian.net/browse/ARCH-42",
                source_title="Migrate auth to OAuth2",
            ),
        ],
        items_discovered=1,
        items_extracted=1,
    )


def _mock_httpx_get(response_data, status_code=200):
    """Build a patched httpx.AsyncClient that returns the given response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = response_data

    client_mock = AsyncMock()
    client_mock.get = AsyncMock(return_value=mock_resp)

    cls_mock = MagicMock()
    cls_mock.return_value.__aenter__ = AsyncMock(return_value=client_mock)
    cls_mock.return_value.__aexit__ = AsyncMock(return_value=None)
    return cls_mock, client_mock


# ── Confluence tests ──────────────────────────────────────────────────


class TestConfluenceIngestion:

    @pytest.mark.asyncio
    async def test_routes_pages_through_extraction_pipeline(
        self, confluence_settings, confluence_api_response,
        extraction_result_auth, extraction_result_data,
    ):
        """Each Confluence page is routed through extract_from_document."""
        from app.ingest import _ingest_confluence

        cls_mock, _ = _mock_httpx_get(confluence_api_response)
        call_count = 0

        async def mock_extract(content, *, filename="", source_url="", source_type="", author="", date="", settings=None):
            nonlocal call_count
            call_count += 1
            if "12345" in filename:
                return extraction_result_auth
            return extraction_result_data

        with patch("app.ingest.httpx.AsyncClient", cls_mock), \
             patch("app.ingest.extract_from_document", side_effect=mock_extract) as mock_ext:
            chunks = await _ingest_confluence(confluence_settings)

        assert call_count == 2
        assert len(chunks) == 3  # 2 from auth page + 1 from data page

    @pytest.mark.asyncio
    async def test_passes_correct_metadata_to_pipeline(
        self, confluence_settings, confluence_api_response, extraction_result_auth,
    ):
        """source_url, source_type, author, and date are passed correctly."""
        from app.ingest import _ingest_confluence

        # Only return the first page
        confluence_api_response["results"] = [confluence_api_response["results"][0]]
        cls_mock, _ = _mock_httpx_get(confluence_api_response)

        captured_kwargs = {}

        async def mock_extract(content, **kwargs):
            captured_kwargs.update(kwargs)
            return extraction_result_auth

        with patch("app.ingest.httpx.AsyncClient", cls_mock), \
             patch("app.ingest.extract_from_document", side_effect=mock_extract):
            await _ingest_confluence(confluence_settings)

        assert captured_kwargs["source_url"] == (
            "https://myorg.atlassian.net/wiki/pages/viewpage.action?pageId=12345"
        )
        assert captured_kwargs["source_type"] == "confluence"
        assert captured_kwargs["author"] == "Alice Engineer"
        assert captured_kwargs["date"] == "2025-06-15"
        assert "12345" in captured_kwargs["filename"]
        assert "Authentication Architecture" in captured_kwargs["filename"]

    @pytest.mark.asyncio
    async def test_passes_html_content_directly(
        self, confluence_settings, confluence_api_response, extraction_result_auth,
    ):
        """Raw HTML from Confluence is passed to extract_from_document (which preprocesses it)."""
        from app.ingest import _ingest_confluence

        confluence_api_response["results"] = [confluence_api_response["results"][0]]
        cls_mock, _ = _mock_httpx_get(confluence_api_response)

        captured_content = None

        async def mock_extract(content, **kwargs):
            nonlocal captured_content
            captured_content = content
            return extraction_result_auth

        with patch("app.ingest.httpx.AsyncClient", cls_mock), \
             patch("app.ingest.extract_from_document", side_effect=mock_extract):
            await _ingest_confluence(confluence_settings)

        # Should receive the raw HTML, not preprocessed markdown
        assert "<h2>" in captured_content
        assert "<p>" in captured_content

    @pytest.mark.asyncio
    async def test_expands_version_for_author_and_date(
        self, confluence_settings,
    ):
        """API request includes version expansion for author/date metadata."""
        from app.ingest import _ingest_confluence

        cls_mock, client_mock = _mock_httpx_get({"results": []})

        with patch("app.ingest.httpx.AsyncClient", cls_mock):
            await _ingest_confluence(confluence_settings)

        call_url = client_mock.get.call_args[0][0]
        assert "expand=body.storage,version" in call_url

    @pytest.mark.asyncio
    async def test_skips_empty_pages(
        self, confluence_settings, extraction_result_auth,
    ):
        """Pages with empty HTML body are skipped."""
        from app.ingest import _ingest_confluence

        response = {
            "results": [
                {
                    "id": "111",
                    "title": "Empty Page",
                    "body": {"storage": {"value": ""}},
                    "version": {"by": {"displayName": "X"}, "when": "2025-01-01"},
                },
                {
                    "id": "222",
                    "title": "Whitespace Page",
                    "body": {"storage": {"value": "   "}},
                    "version": {"by": {"displayName": "X"}, "when": "2025-01-01"},
                },
            ],
        }
        cls_mock, _ = _mock_httpx_get(response)

        with patch("app.ingest.httpx.AsyncClient", cls_mock), \
             patch("app.ingest.extract_from_document", new_callable=AsyncMock) as mock_ext:
            chunks = await _ingest_confluence(confluence_settings)

        mock_ext.assert_not_called()
        assert chunks == []

    @pytest.mark.asyncio
    async def test_continues_when_page_extraction_finds_nothing(
        self, confluence_settings, confluence_api_response, extraction_result_data,
    ):
        """Pages that yield no knowledge items are logged but don't block others."""
        from app.ingest import _ingest_confluence

        cls_mock, _ = _mock_httpx_get(confluence_api_response)

        async def mock_extract(content, **kwargs):
            if "12345" in kwargs.get("filename", ""):
                return ExtractionResult()  # empty — no items found
            return extraction_result_data

        with patch("app.ingest.httpx.AsyncClient", cls_mock), \
             patch("app.ingest.extract_from_document", side_effect=mock_extract):
            chunks = await _ingest_confluence(confluence_settings)

        # Only the second page's chunks come through
        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_skips_when_space_key_not_set(self, test_settings):
        """Returns empty when CONFLUENCE_SPACE_KEY is not configured."""
        from app.ingest import _ingest_confluence

        settings = Settings(
            OPENAI_API_KEY="sk-test",
            GITHUB_TOKEN="ghp-test",
            GITHUB_WEBHOOK_SECRET="secret",
            QDRANT_URL="http://localhost:6333",
            CONFLUENCE_BASE_URL="https://myorg.atlassian.net",
            CONFLUENCE_TOKEN="token",
            CONFLUENCE_SPACE_KEY="",
        )

        chunks = await _ingest_confluence(settings)
        assert chunks == []

    @pytest.mark.asyncio
    async def test_raises_on_api_error(self, confluence_settings):
        """API errors bubble up as RuntimeError."""
        from app.ingest import _ingest_confluence

        cls_mock, _ = _mock_httpx_get({}, status_code=403)

        with patch("app.ingest.httpx.AsyncClient", cls_mock):
            with pytest.raises(RuntimeError, match="Confluence API 403"):
                await _ingest_confluence(confluence_settings)


# ── Jira tests ────────────────────────────────────────────────────────


class TestJiraIngestion:

    @pytest.mark.asyncio
    async def test_routes_epics_through_extraction_pipeline(
        self, jira_settings, jira_api_response, extraction_result_jira,
    ):
        """Each Jira epic is routed through extract_from_document."""
        from app.ingest import _ingest_jira

        cls_mock, _ = _mock_httpx_get(jira_api_response)

        with patch("app.ingest.httpx.AsyncClient", cls_mock), \
             patch("app.ingest.extract_from_document", new_callable=AsyncMock,
                   return_value=extraction_result_jira) as mock_ext:
            chunks = await _ingest_jira(jira_settings)

        mock_ext.assert_called_once()
        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_passes_correct_metadata(
        self, jira_settings, jira_api_response, extraction_result_jira,
    ):
        """source_url, source_type, author, and date are passed correctly."""
        from app.ingest import _ingest_jira

        cls_mock, _ = _mock_httpx_get(jira_api_response)
        captured_kwargs = {}

        async def mock_extract(content, **kwargs):
            captured_kwargs.update(kwargs)
            return extraction_result_jira

        with patch("app.ingest.httpx.AsyncClient", cls_mock), \
             patch("app.ingest.extract_from_document", side_effect=mock_extract):
            await _ingest_jira(jira_settings)

        assert captured_kwargs["source_url"] == "https://myorg.atlassian.net/browse/ARCH-42"
        assert captured_kwargs["source_type"] == "jira"
        assert captured_kwargs["author"] == "Charlie Dev"
        assert captured_kwargs["date"] == "2025-09-10"
        assert "ARCH-42" in captured_kwargs["filename"]

    @pytest.mark.asyncio
    async def test_prepends_summary_as_title(
        self, jira_settings, jira_api_response, extraction_result_jira,
    ):
        """Epic summary is prepended as a heading to give the LLM context."""
        from app.ingest import _ingest_jira

        cls_mock, _ = _mock_httpx_get(jira_api_response)
        captured_content = None

        async def mock_extract(content, **kwargs):
            nonlocal captured_content
            captured_content = content
            return extraction_result_jira

        with patch("app.ingest.httpx.AsyncClient", cls_mock), \
             patch("app.ingest.extract_from_document", side_effect=mock_extract):
            await _ingest_jira(jira_settings)

        assert captured_content.startswith("# Migrate auth to OAuth2")
        assert "Chose Keycloak" in captured_content

    @pytest.mark.asyncio
    async def test_skips_epics_with_no_description(self, jira_settings):
        """Epics without a description are skipped."""
        from app.ingest import _ingest_jira

        response = {
            "issues": [
                {
                    "key": "ARCH-99",
                    "fields": {
                        "summary": "Empty epic",
                        "description": None,
                        "assignee": None,
                        "created": "2025-01-01",
                    },
                },
            ],
        }
        cls_mock, _ = _mock_httpx_get(response)

        with patch("app.ingest.httpx.AsyncClient", cls_mock), \
             patch("app.ingest.extract_from_document", new_callable=AsyncMock) as mock_ext:
            chunks = await _ingest_jira(jira_settings)

        mock_ext.assert_not_called()
        assert chunks == []

    @pytest.mark.asyncio
    async def test_handles_string_description(
        self, jira_settings, extraction_result_jira,
    ):
        """Jira descriptions that are plain strings (not ADF) still work."""
        from app.ingest import _ingest_jira

        response = {
            "issues": [
                {
                    "key": "ARCH-50",
                    "fields": {
                        "summary": "Simple epic",
                        "description": "Use PostgreSQL for all transactional data.",
                        "assignee": {"displayName": "Dev"},
                        "created": "2025-05-01",
                    },
                },
            ],
        }
        cls_mock, _ = _mock_httpx_get(response)
        captured_content = None

        async def mock_extract(content, **kwargs):
            nonlocal captured_content
            captured_content = content
            return extraction_result_jira

        with patch("app.ingest.httpx.AsyncClient", cls_mock), \
             patch("app.ingest.extract_from_document", side_effect=mock_extract):
            await _ingest_jira(jira_settings)

        assert "# Simple epic" in captured_content
        assert "Use PostgreSQL" in captured_content

    @pytest.mark.asyncio
    async def test_raises_on_api_error(self, jira_settings):
        """API errors bubble up as RuntimeError."""
        from app.ingest import _ingest_jira

        cls_mock, _ = _mock_httpx_get({}, status_code=401)

        with patch("app.ingest.httpx.AsyncClient", cls_mock):
            with pytest.raises(RuntimeError, match="Jira API 401"):
                await _ingest_jira(jira_settings)


# ── Full pipeline integration ─────────────────────────────────────────


class TestIngestPipeline:

    @pytest.mark.asyncio
    async def test_confluence_chunks_counted_correctly(
        self, confluence_settings, confluence_api_response,
        extraction_result_auth, extraction_result_data,
    ):
        """ingest() reports the total chunk count for Confluence."""
        from app.ingest import ingest

        cls_mock, _ = _mock_httpx_get(confluence_api_response)

        async def mock_extract(content, **kwargs):
            if "12345" in kwargs.get("filename", ""):
                return extraction_result_auth
            return extraction_result_data

        with patch("app.ingest.httpx.AsyncClient", cls_mock), \
             patch("app.ingest.extract_from_document", side_effect=mock_extract), \
             patch("app.ingest.ensure_collection", new_callable=AsyncMock), \
             patch("app.ingest.upsert", new_callable=AsyncMock):
            results = await ingest(confluence_settings)

        assert results.confluence == 3  # 2 + 1 chunks across two pages
        assert results.errors == []