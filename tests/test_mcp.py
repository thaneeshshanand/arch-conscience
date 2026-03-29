"""MCP server tests — verifies get_architectural_context and draft_adr.

Tests all input combinations, conflict analysis logic, and ADR drafting.
Mocks the corpus and LLM layers so no external calls are needed.

Run with:
    pytest tests/test_mcp.py -v
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.corpus import ChunkRecord, ScoredChunk
from app.llm.base import CompletionResult


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def adr001_chunks():
    """Scored chunks simulating ADR-001 retrieval."""
    base = dict(
        source_type="ADR",
        doc_id="adr-001",
        affected_services=["auth-service", "api-gateway"],
        date="2024-03-15",
        status="active",
        domain="security",
        author="thaneesh",
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
        date="2024-06-01",
        status="active",
        domain="data_model",
        author="thaneesh",
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


class TestDraftAdr:
    """Tests for the draft_adr MCP tool."""

    @pytest.fixture
    def mock_adr_markdown(self):
        return (
            "---\n"
            "id: adr-20260327\n"
            "title: Use event sourcing for payment state\n"
            "status: proposed\n"
            "date: 2026-03-27\n"
            "services: [payments-service]\n"
            "constraint_type: compliance\n"
            "author: thaneesh\n"
            "---\n\n"
            "## Context\n\nThe payments service needs audit compliance.\n\n"
            "## Decision\n\nUse event sourcing with append-only store.\n\n"
            "## Consequences\n\nFull audit trail. Increased read complexity.\n\n"
            "## Rejected Alternatives\n\nCRUD with audit log table rejected."
        )

    @pytest.mark.asyncio
    async def test_draft_adr_basic(self, test_settings, mock_adr_markdown):
        """draft_adr returns a structured ADR with next_steps."""
        from app.mcp_server import draft_adr

        with patch("app.mcp_server.get_settings", return_value=test_settings), \
             patch("app.mcp_server.ensure_collection", new_callable=AsyncMock), \
             patch("app.mcp_server.query", new_callable=AsyncMock, return_value=[]), \
             patch("app.adr_drafter.complete", new_callable=AsyncMock,
                   return_value=CompletionResult(content=mock_adr_markdown, model="gpt-4o")):

            result = json.loads(await draft_adr(
                title="Use event sourcing for payment state",
                services="payments-service",
                context="Need audit compliance for 50k transactions/day.",
                constraint_type="compliance",
                author="thaneesh",
            ))

        assert result["title"] == "Use event sourcing for payment state"
        assert result["services"] == ["payments-service"]
        assert result["status"] == "proposed"
        assert "## Context" in result["draft"]
        assert "## Rejected Alternatives" in result["draft"]
        assert "next_steps" in result

    @pytest.mark.asyncio
    async def test_draft_adr_accepts_singular_service(self, test_settings, mock_adr_markdown):
        """draft_adr accepts 'service' (singular) as an alternative to 'services'."""
        from app.mcp_server import draft_adr

        with patch("app.mcp_server.get_settings", return_value=test_settings), \
             patch("app.mcp_server.ensure_collection", new_callable=AsyncMock), \
             patch("app.mcp_server.query", new_callable=AsyncMock, return_value=[]), \
             patch("app.adr_drafter.complete", new_callable=AsyncMock,
                   return_value=CompletionResult(content=mock_adr_markdown, model="gpt-4o")):

            result = json.loads(await draft_adr(
                title="Use event sourcing",
                service="payments-service",
                context="Audit compliance needed.",
            ))

        assert result["services"] == ["payments-service"]

    @pytest.mark.asyncio
    async def test_draft_adr_accepts_decision_param(self, test_settings, mock_adr_markdown):
        """draft_adr accepts 'decision' as an alternative to 'approach'."""
        from app.mcp_server import draft_adr

        mock_drafter = AsyncMock(return_value=CompletionResult(content=mock_adr_markdown, model="gpt-4o"))

        with patch("app.mcp_server.get_settings", return_value=test_settings), \
             patch("app.mcp_server.ensure_collection", new_callable=AsyncMock), \
             patch("app.mcp_server.query", new_callable=AsyncMock, return_value=[]), \
             patch("app.adr_drafter.complete", mock_drafter):

            await draft_adr(
                title="Use event sourcing",
                services="payments-service",
                context="Audit compliance.",
                decision="Append-only event store",
            )

        # Verify the decision was passed through to the drafter
        call_args = mock_drafter.call_args
        user_msg = call_args[0][0][1].content
        assert "Append-only event store" in user_msg

    @pytest.mark.asyncio
    async def test_draft_adr_accepts_alternatives_param(self, test_settings, mock_adr_markdown):
        """draft_adr accepts 'alternatives' as alternative to 'alternatives_considered'."""
        from app.mcp_server import draft_adr

        mock_drafter = AsyncMock(return_value=CompletionResult(content=mock_adr_markdown, model="gpt-4o"))

        with patch("app.mcp_server.get_settings", return_value=test_settings), \
             patch("app.mcp_server.ensure_collection", new_callable=AsyncMock), \
             patch("app.mcp_server.query", new_callable=AsyncMock, return_value=[]), \
             patch("app.adr_drafter.complete", mock_drafter):

            await draft_adr(
                title="Use event sourcing",
                services="payments-service",
                context="Audit compliance.",
                alternatives="CRUD with audit log rejected due to incomplete history.",
            )

        call_args = mock_drafter.call_args
        user_msg = call_args[0][0][1].content
        assert "CRUD with audit log" in user_msg

    @pytest.mark.asyncio
    async def test_draft_adr_includes_related_decisions(self, test_settings, adr001_chunks, mock_adr_markdown):
        """draft_adr queries corpus for related decisions and passes them to drafter."""
        from app.mcp_server import draft_adr

        mock_drafter = AsyncMock(return_value=CompletionResult(content=mock_adr_markdown, model="gpt-4o"))

        with patch("app.mcp_server.get_settings", return_value=test_settings), \
             patch("app.mcp_server.ensure_collection", new_callable=AsyncMock), \
             patch("app.mcp_server.query", new_callable=AsyncMock, return_value=adr001_chunks), \
             patch("app.adr_drafter.complete", mock_drafter):

            await draft_adr(
                title="Add OAuth to auth-service",
                services="auth-service",
                context="Need third-party login support.",
            )

        call_args = mock_drafter.call_args
        user_msg = call_args[0][0][1].content
        assert "adr-001" in user_msg