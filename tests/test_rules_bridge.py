"""Rules bridge tests — extraction, chunking, and MCP tool.

Run with:
    pytest tests/test_rules_bridge.py -v
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.llm.base import CompletionResult


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def llm_extraction_response():
    """LLM response with mixed architectural decisions."""
    return json.dumps({
        "decisions": [
            {
                "title": "Use JWT for authentication",
                "decision": "All services use signed JWTs. No server-side session state.",
                "context": "Need stateless auth for horizontal scaling.",
                "rejected": "Session cookies rejected due to centralised store and GDPR conflict.",
                "services": ["auth-service", "api-gateway"],
                "domain": "security",
            },
            {
                "title": "Event-driven inter-service communication",
                "decision": "Services communicate via RabbitMQ events.",
                "context": "Decoupling services for independent deployment.",
                "rejected": "Direct HTTP calls between services.",
                "services": [],
                "domain": "scalability",
            },
            {
                "title": "Legacy payments SOAP client",
                "decision": "Do not refactor the payments SOAP client until bank migrates to REST.",
                "context": "",
                "rejected": "",
                "services": ["payments"],
                "domain": "operational",
            },
        ]
    })


@pytest.fixture
def llm_empty_response():
    """LLM response when no architectural decisions found."""
    return json.dumps({"decisions": []})


@pytest.fixture
def llm_all_services_response():
    """LLM response that uses 'all' for services."""
    return json.dumps({
        "decisions": [
            {
                "title": "PostgreSQL for OLTP",
                "decision": "Use PostgreSQL for all transactional data.",
                "context": "",
                "rejected": "MongoDB rejected for transactional workloads.",
                "services": ["all"],
                "domain": "data_model",
            },
        ]
    })


# ── Tests ────────────────────────────────────────────────────────────


class TestExtractDecisions:
    """Tests for extract_decisions_from_rules."""

    @pytest.mark.asyncio
    async def test_extracts_decisions_with_correct_chunk_format(
        self, test_settings, llm_extraction_response
    ):
        """Extracted chunks follow the Rule: <title> / Section: <type> format."""
        from app.rules_bridge import extract_decisions_from_rules

        with patch(
            "app.rules_bridge.complete",
            new_callable=AsyncMock,
            return_value=CompletionResult(content=llm_extraction_response, model="gpt-4o"),
        ):
            chunks = await extract_decisions_from_rules(
                content="# Architecture\n- Use JWT, never session cookies",
                source_file="CLAUDE.md",
                settings=test_settings,
            )

        # 3 decisions: 2 with rejected (4 chunks each pair) + 1 without (1 chunk)
        assert len(chunks) == 5

        # Check format of first decision chunk
        jwt_decision = [c for c in chunks if "JWT" in c.text and c.section_type == "decision"][0]
        assert jwt_decision.text.startswith("Rule: Use JWT for authentication")
        assert "Section: Decision" in jwt_decision.text
        assert jwt_decision.source_type == "rules_file"
        assert jwt_decision.status == "active"
        assert jwt_decision.knowledge_type == "decision"

    @pytest.mark.asyncio
    async def test_creates_separate_rejected_alternatives_chunks(
        self, test_settings, llm_extraction_response
    ):
        """Decisions with rejected text get a separate rejected_alternatives chunk."""
        from app.rules_bridge import extract_decisions_from_rules

        with patch(
            "app.rules_bridge.complete",
            new_callable=AsyncMock,
            return_value=CompletionResult(content=llm_extraction_response, model="gpt-4o"),
        ):
            chunks = await extract_decisions_from_rules(
                content="some rules",
                source_file="rules.md",
                settings=test_settings,
            )

        rejected_chunks = [c for c in chunks if c.section_type == "rejected_alternatives"]
        assert len(rejected_chunks) == 2  # JWT and RabbitMQ have rejected text

        jwt_rejected = [c for c in rejected_chunks if "JWT" in c.text][0]
        assert "Section: Rejected Alternatives" in jwt_rejected.text
        assert "Session cookies" in jwt_rejected.text

    @pytest.mark.asyncio
    async def test_no_rejected_chunk_when_empty(
        self, test_settings, llm_extraction_response
    ):
        """Decisions without rejected text don't create a rejected_alternatives chunk."""
        from app.rules_bridge import extract_decisions_from_rules

        with patch(
            "app.rules_bridge.complete",
            new_callable=AsyncMock,
            return_value=CompletionResult(content=llm_extraction_response, model="gpt-4o"),
        ):
            chunks = await extract_decisions_from_rules(
                content="some rules",
                source_file="rules.md",
                settings=test_settings,
            )

        # Legacy payments has no rejected text
        payments_chunks = [c for c in chunks if "SOAP" in c.text or "payments" in c.text.lower()]
        assert len(payments_chunks) == 1
        assert payments_chunks[0].section_type == "decision"

    @pytest.mark.asyncio
    async def test_services_preserved_correctly(
        self, test_settings, llm_extraction_response
    ):
        """Service names are correctly assigned to chunks."""
        from app.rules_bridge import extract_decisions_from_rules

        with patch(
            "app.rules_bridge.complete",
            new_callable=AsyncMock,
            return_value=CompletionResult(content=llm_extraction_response, model="gpt-4o"),
        ):
            chunks = await extract_decisions_from_rules(
                content="some rules",
                source_file="rules.md",
                settings=test_settings,
            )

        jwt_chunks = [c for c in chunks if "JWT" in c.text]
        for c in jwt_chunks:
            assert c.affected_services == ["auth-service", "api-gateway"]

        rabbitmq_chunks = [c for c in chunks if "RabbitMQ" in c.text]
        for c in rabbitmq_chunks:
            assert c.affected_services == []  # project-wide

    @pytest.mark.asyncio
    async def test_all_services_converted_to_empty_list(
        self, test_settings, llm_all_services_response
    ):
        """Services=['all'] is converted to empty list for project-wide decisions."""
        from app.rules_bridge import extract_decisions_from_rules

        with patch(
            "app.rules_bridge.complete",
            new_callable=AsyncMock,
            return_value=CompletionResult(content=llm_all_services_response, model="gpt-4o"),
        ):
            chunks = await extract_decisions_from_rules(
                content="some rules",
                source_file="rules.md",
                settings=test_settings,
            )

        for c in chunks:
            assert c.affected_services == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_decisions(
        self, test_settings, llm_empty_response
    ):
        """File with only code style returns zero chunks."""
        from app.rules_bridge import extract_decisions_from_rules

        with patch(
            "app.rules_bridge.complete",
            new_callable=AsyncMock,
            return_value=CompletionResult(content=llm_empty_response, model="gpt-4o"),
        ):
            chunks = await extract_decisions_from_rules(
                content="# Code Style\n- Use 2-space indent\n- Named exports",
                source_file=".cursorrules",
                settings=test_settings,
            )

        assert chunks == []

    @pytest.mark.asyncio
    async def test_handles_malformed_llm_output(self, test_settings):
        """Gracefully returns empty list on unparseable LLM output."""
        from app.rules_bridge import extract_decisions_from_rules

        with patch(
            "app.rules_bridge.complete",
            new_callable=AsyncMock,
            return_value=CompletionResult(content="not json at all", model="gpt-4o"),
        ):
            chunks = await extract_decisions_from_rules(
                content="some rules",
                source_file="rules.md",
                settings=test_settings,
            )

        assert chunks == []

    @pytest.mark.asyncio
    async def test_dotfile_stem_handled_correctly(
        self, test_settings, llm_extraction_response
    ):
        """Dotfiles like .cursorrules produce clean doc_ids."""
        from app.rules_bridge import extract_decisions_from_rules

        with patch(
            "app.rules_bridge.complete",
            new_callable=AsyncMock,
            return_value=CompletionResult(content=llm_extraction_response, model="gpt-4o"),
        ):
            chunks = await extract_decisions_from_rules(
                content="some rules",
                source_file=".cursorrules",
                settings=test_settings,
            )

        for c in chunks:
            assert c.doc_id.startswith("rules-cursorrules-")
            assert ".." not in c.doc_id

    @pytest.mark.asyncio
    async def test_domain_preserved(
        self, test_settings, llm_extraction_response
    ):
        """Domain values from LLM output are correctly assigned."""
        from app.rules_bridge import extract_decisions_from_rules

        with patch(
            "app.rules_bridge.complete",
            new_callable=AsyncMock,
            return_value=CompletionResult(content=llm_extraction_response, model="gpt-4o"),
        ):
            chunks = await extract_decisions_from_rules(
                content="some rules",
                source_file="rules.md",
                settings=test_settings,
            )

        jwt_chunk = [c for c in chunks if "JWT" in c.text and c.section_type == "decision"][0]
        assert jwt_chunk.domain == "security"

        rabbitmq_chunk = [c for c in chunks if "RabbitMQ" in c.text and c.section_type == "decision"][0]
        assert rabbitmq_chunk.domain == "scalability"


class TestIngestRulesTool:
    """Tests for the ingest_rules_file MCP tool."""

    @pytest.mark.asyncio
    async def test_tool_extracts_and_upserts(
        self, test_settings, llm_extraction_response
    ):
        """MCP tool extracts decisions and upserts them to corpus."""
        from app.mcp_server import ingest_rules_file

        with patch("app.mcp_server.get_settings", return_value=test_settings), \
             patch("app.mcp_server.ensure_collection", new_callable=AsyncMock), \
             patch(
                 "app.rules_bridge.complete",
                 new_callable=AsyncMock,
                 return_value=CompletionResult(content=llm_extraction_response, model="gpt-4o"),
             ), \
             patch("app.corpus.upsert", new_callable=AsyncMock) as mock_upsert:

            result = json.loads(await ingest_rules_file(
                content="# Architecture\n- Use JWT\n- Use RabbitMQ",
                filename="CLAUDE.md",
            ))

        assert result["decisions_extracted"] == 3
        assert result["chunks_indexed"] == 5
        assert mock_upsert.called

    @pytest.mark.asyncio
    async def test_tool_returns_zero_when_no_decisions(
        self, test_settings, llm_empty_response
    ):
        """MCP tool returns helpful message when no decisions found."""
        from app.mcp_server import ingest_rules_file

        with patch("app.mcp_server.get_settings", return_value=test_settings), \
             patch("app.mcp_server.ensure_collection", new_callable=AsyncMock), \
             patch(
                 "app.rules_bridge.complete",
                 new_callable=AsyncMock,
                 return_value=CompletionResult(content=llm_empty_response, model="gpt-4o"),
             ):

            result = json.loads(await ingest_rules_file(
                content="# Code Style\n- 2 spaces\n- Named exports",
                filename=".cursorrules",
            ))

        assert result["decisions_extracted"] == 0
        assert result["chunks_indexed"] == 0
        assert "No architectural decisions" in result["message"]