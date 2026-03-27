"""MCP server tests — verifies get_architectural_context behavior.

Tests all four input combinations and conflict analysis logic.
Mocks the corpus layer so no Qdrant or LLM calls are needed.

Run with:
    pytest tests/test_mcp.py -v
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.corpus import ChunkRecord, ScoredChunk


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def adr001_chunks():
    """Scored chunks simulating ADR-001 retrieval."""
    base = dict(
        source_type="ADR",
        doc_id="adr-001",
        affected_services=["auth-service", "api-gateway"],
        decision_date="2024-03-15",
        status="active",
        constraint_type="security",
        author="thaneesh",
        linked_adr_ids=[],
    )

    return [
        ScoredChunk(
            chunk=ChunkRecord(
                id="adr-001-rejected_alternatives",
                text=(
                    "ADR: Use JWT for stateless authentication\n"
                    "Section: Rejected Alternatives\n\n"
                    "Session cookies were rejected. They require a centralised "
                    "session store and conflict with GDPR requirements."
                ),
                section_type="rejected_alternatives",
                **base,
            ),
            score=0.596,
            point_id="uuid-1",
        ),
        ScoredChunk(
            chunk=ChunkRecord(
                id="adr-001-decision",
                text=(
                    "ADR: Use JWT for stateless authentication\n"
                    "Section: Decision\n\n"
                    "Use signed JWTs for all authentication. No server-side "
                    "session state."
                ),
                section_type="decision",
                **base,
            ),
            score=0.517,
            point_id="uuid-2",
        ),
        ScoredChunk(
            chunk=ChunkRecord(
                id="adr-001-context",
                text=(
                    "ADR: Use JWT for stateless authentication\n"
                    "Section: Context\n\n"
                    "The auth service needs stateless authentication."
                ),
                section_type="context",
                **base,
            ),
            score=0.497,
            point_id="uuid-3",
        ),
        ScoredChunk(
            chunk=ChunkRecord(
                id="adr-001-consequences",
                text=(
                    "ADR: Use JWT for stateless authentication\n"
                    "Section: Consequences\n\n"
                    "All services validate tokens independently."
                ),
                section_type="consequences",
                **base,
            ),
            score=0.475,
            point_id="uuid-4",
        ),
    ]


@pytest.fixture
def decision_only_chunks():
    """Chunks with no rejected_alternatives section."""
    base = dict(
        source_type="ADR",
        doc_id="adr-002",
        affected_services=["payments-service"],
        decision_date="2024-06-01",
        status="active",
        constraint_type="data_model",
        author="thaneesh",
        linked_adr_ids=[],
    )

    return [
        ScoredChunk(
            chunk=ChunkRecord(
                id="adr-002-decision",
                text=(
                    "ADR: Use event sourcing for payments\n"
                    "Section: Decision\n\n"
                    "All payment state changes are stored as events."
                ),
                section_type="decision",
                **base,
            ),
            score=0.520,
            point_id="uuid-5",
        ),
    ]


# ── Tests ────────────────────────────────────────────────────────────


class TestGetArchitecturalContext:
    """Tests for the unified get_architectural_context tool."""

    @pytest.mark.asyncio
    async def test_service_only_returns_context(self, adr001_chunks, test_settings):
        """Service without approach returns ADR context with instructions."""
        from app.mcp_server import get_architectural_context

        with patch("app.mcp_server.get_settings", return_value=test_settings), \
             patch("app.mcp_server.ensure_collection", new_callable=AsyncMock), \
             patch("app.mcp_server.query", new_callable=AsyncMock, return_value=adr001_chunks):

            result = json.loads(await get_architectural_context(service="auth-service"))

        assert result["service"] == "auth-service"
        assert result["decisions_found"] == 4
        assert "approach" not in result
        assert "verdict" not in result
        assert "instructions" in result
        assert "rejected_alternatives" in result["adr_summary"]["adr-001"]

    @pytest.mark.asyncio
    async def test_service_and_approach_returns_conflict(self, adr001_chunks, test_settings):
        """Service + approach returns conflict analysis when rejected alt found."""
        from app.mcp_server import get_architectural_context

        with patch("app.mcp_server.get_settings", return_value=test_settings), \
             patch("app.mcp_server.ensure_collection", new_callable=AsyncMock), \
             patch("app.mcp_server.query", new_callable=AsyncMock, return_value=adr001_chunks):

            result = json.loads(await get_architectural_context(
                service="auth-service",
                approach="Use session cookies with Redis",
            ))

        assert result["service"] == "auth-service"
        assert result["approach"] == "Use session cookies with Redis"
        assert result["verdict"] == "potential_conflict"
        assert result["conflicts_found"] == 1
        assert "instructions" not in result

    @pytest.mark.asyncio
    async def test_approach_only_searches_broadly(self, adr001_chunks, test_settings):
        """Approach without service searches across all services."""
        from app.mcp_server import get_architectural_context

        mock_query = AsyncMock(return_value=adr001_chunks)

        with patch("app.mcp_server.get_settings", return_value=test_settings), \
             patch("app.mcp_server.ensure_collection", new_callable=AsyncMock), \
             patch("app.mcp_server.query", mock_query):

            result = json.loads(await get_architectural_context(
                approach="Use session cookies",
            ))

        # Verify query was called with services=None (no filter)
        call_kwargs = mock_query.call_args[1]
        assert call_kwargs["services"] is None
        assert result["service"] == "all"
        assert result["verdict"] == "potential_conflict"

    @pytest.mark.asyncio
    async def test_no_args_returns_all_decisions(self, adr001_chunks, test_settings):
        """No arguments returns summary of all active decisions."""
        from app.mcp_server import get_architectural_context

        mock_query = AsyncMock(return_value=adr001_chunks)

        with patch("app.mcp_server.get_settings", return_value=test_settings), \
             patch("app.mcp_server.ensure_collection", new_callable=AsyncMock), \
             patch("app.mcp_server.query", mock_query):

            result = json.loads(await get_architectural_context())

        assert result["service"] == "all"
        assert result["decisions_found"] == 4
        assert "instructions" in result
        assert "adr-001" in result["adr_summary"]

    @pytest.mark.asyncio
    async def test_no_chunks_returns_empty_message(self, test_settings):
        """Empty corpus returns helpful guidance message."""
        from app.mcp_server import get_architectural_context

        with patch("app.mcp_server.get_settings", return_value=test_settings), \
             patch("app.mcp_server.ensure_collection", new_callable=AsyncMock), \
             patch("app.mcp_server.query", new_callable=AsyncMock, return_value=[]), \
             patch("app.mcp_server.stats", new_callable=AsyncMock, return_value={"total_chunks": 0}):

            result = json.loads(await get_architectural_context(service="payments-service"))

        assert result["decisions_found"] == 0
        assert "payments-service" in result["message"]
        assert result["corpus_stats"]["total_chunks"] == 0

    @pytest.mark.asyncio
    async def test_review_recommended_when_no_rejected_alt(self, decision_only_chunks, test_settings):
        """Approach with decision chunks but no rejected alt returns review_recommended."""
        from app.mcp_server import get_architectural_context

        with patch("app.mcp_server.get_settings", return_value=test_settings), \
             patch("app.mcp_server.ensure_collection", new_callable=AsyncMock), \
             patch("app.mcp_server.query", new_callable=AsyncMock, return_value=decision_only_chunks):

            result = json.loads(await get_architectural_context(
                service="payments-service",
                approach="Store payment state in a relational table",
            ))

        assert result["verdict"] == "review_recommended"
        assert result["conflicts_found"] == 0


class TestConflictAnalysis:
    """Tests for the _analyze_conflicts helper."""

    def test_potential_conflict_with_rejected_alt(self, adr001_chunks):
        from app.mcp_server import _analyze_conflicts

        result = _analyze_conflicts(adr001_chunks)
        assert result["verdict"] == "potential_conflict"
        assert result["conflicts_found"] == 1

    def test_review_recommended_with_decisions_only(self, decision_only_chunks):
        from app.mcp_server import _analyze_conflicts

        result = _analyze_conflicts(decision_only_chunks)
        assert result["verdict"] == "review_recommended"
        assert result["conflicts_found"] == 0

    def test_context_available_with_context_only(self):
        from app.mcp_server import _analyze_conflicts

        context_chunk = ScoredChunk(
            chunk=ChunkRecord(
                id="adr-003-context",
                text="Some background context.",
                source_type="ADR",
                doc_id="adr-003",
                section_type="context",
                affected_services=["some-service"],
            ),
            score=0.5,
            point_id="uuid-6",
        )

        result = _analyze_conflicts([context_chunk])
        assert result["verdict"] == "context_available"
        assert result["conflicts_found"] == 0