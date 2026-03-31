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
    """LLM response with mixed architectural knowledge types."""
    return json.dumps({
        "items": [
            {
                "title": "Use JWT for authentication",
                "knowledge_type": "decision",
                "content": "All services use signed JWTs. No server-side session state.",
                "context": "Need stateless auth for horizontal scaling.",
                "rejected": "Session cookies rejected due to centralised store and GDPR conflict.",
                "services": ["auth-service", "api-gateway"],
                "domain": "security",
            },
            {
                "title": "PII encryption requirement",
                "knowledge_type": "constraint",
                "content": "All PII must be encrypted at rest using AES-256. No exceptions.",
                "context": "Compliance requirement from Legal. Violation triggers audit failure.",
                "rejected": "",
                "services": [],
                "domain": "compliance",
            },
            {
                "title": "Business logic out of HTTP handlers",
                "knowledge_type": "principle",
                "content": "Keep business logic out of HTTP handlers. Handlers should only parse requests, call a service layer function, and format responses.",
                "context": "Improves testability and keeps handlers thin.",
                "rejected": "",
                "services": [],
                "domain": "operational",
            },
            {
                "title": "Event-driven inter-service communication",
                "knowledge_type": "decision",
                "content": "Services communicate via RabbitMQ events.",
                "context": "Decoupling services for independent deployment.",
                "rejected": "Direct HTTP calls between services.",
                "services": [],
                "domain": "scalability",
            },
            {
                "title": "Legacy payments SOAP client",
                "knowledge_type": "decision",
                "content": "Do not refactor the payments SOAP client until bank migrates to REST.",
                "context": "",
                "rejected": "",
                "services": ["payments"],
                "domain": "operational",
            },
        ]
    })


@pytest.fixture
def llm_empty_response():
    """LLM response when no architectural knowledge found."""
    return json.dumps({"items": []})


@pytest.fixture
def llm_all_services_response():
    """LLM response that uses 'all' for services."""
    return json.dumps({
        "items": [
            {
                "title": "PostgreSQL for OLTP",
                "knowledge_type": "decision",
                "content": "Use PostgreSQL for all transactional data.",
                "context": "",
                "rejected": "MongoDB rejected for transactional workloads.",
                "services": ["all"],
                "domain": "data_model",
            },
        ]
    })


@pytest.fixture
def llm_legacy_format_response():
    """LLM response in the old format (backward compatibility)."""
    return json.dumps({
        "decisions": [
            {
                "title": "Use PostgreSQL",
                "decision": "Use PostgreSQL for all data.",
                "context": "Need ACID.",
                "rejected": "MongoDB rejected.",
                "services": [],
                "domain": "data_model",
            },
        ]
    })


# ── Tests ────────────────────────────────────────────────────────────


class TestExtractDecisions:
    """Tests for extract_decisions_from_rules."""

    @pytest.mark.asyncio
    async def test_extracts_all_three_knowledge_types(
        self, test_settings, llm_extraction_response
    ):
        """Extraction produces chunks with correct knowledge_type for each item."""
        from app.rules_bridge import extract_decisions_from_rules

        with patch(
            "app.rules_bridge.complete",
            new_callable=AsyncMock,
            return_value=CompletionResult(content=llm_extraction_response, model="gpt-4o"),
        ):
            chunks = await extract_decisions_from_rules(
                content="# Architecture\n- Use JWT\n- PII encrypted\n- Logic in service layer",
                source_file="CLAUDE.md",
                settings=test_settings,
            )

        # Check knowledge types
        decision_chunks = [c for c in chunks if c.knowledge_type == "decision"]
        constraint_chunks = [c for c in chunks if c.knowledge_type == "constraint"]
        principle_chunks = [c for c in chunks if c.knowledge_type == "principle"]

        assert len(decision_chunks) > 0
        assert len(constraint_chunks) > 0
        assert len(principle_chunks) > 0

    @pytest.mark.asyncio
    async def test_constraint_uses_correct_label(
        self, test_settings, llm_extraction_response
    ):
        """Constraint chunks use 'Constraint:' prefix, not 'Rule:'."""
        from app.rules_bridge import extract_decisions_from_rules

        with patch(
            "app.rules_bridge.complete",
            new_callable=AsyncMock,
            return_value=CompletionResult(content=llm_extraction_response, model="gpt-4o"),
        ):
            chunks = await extract_decisions_from_rules(
                content="some rules",
                source_file="CLAUDE.md",
                settings=test_settings,
            )

        pii_chunks = [c for c in chunks if "PII" in c.source_title]
        assert len(pii_chunks) > 0
        pii_decision = [c for c in pii_chunks if c.section_type == "decision"][0]
        assert pii_decision.text.startswith("Constraint: PII encryption requirement")
        assert pii_decision.knowledge_type == "constraint"

    @pytest.mark.asyncio
    async def test_principle_uses_correct_label(
        self, test_settings, llm_extraction_response
    ):
        """Principle chunks use 'Principle:' prefix."""
        from app.rules_bridge import extract_decisions_from_rules

        with patch(
            "app.rules_bridge.complete",
            new_callable=AsyncMock,
            return_value=CompletionResult(content=llm_extraction_response, model="gpt-4o"),
        ):
            chunks = await extract_decisions_from_rules(
                content="some rules",
                source_file="CLAUDE.md",
                settings=test_settings,
            )

        handler_chunks = [c for c in chunks if "handler" in c.source_title.lower()]
        assert len(handler_chunks) > 0
        handler_decision = [c for c in handler_chunks if c.section_type == "decision"][0]
        assert handler_decision.text.startswith("Principle: Business logic out of HTTP handlers")
        assert handler_decision.knowledge_type == "principle"

    @pytest.mark.asyncio
    async def test_constraint_has_no_rejected_alternatives(
        self, test_settings, llm_extraction_response
    ):
        """Constraints never produce rejected_alternatives chunks."""
        from app.rules_bridge import extract_decisions_from_rules

        with patch(
            "app.rules_bridge.complete",
            new_callable=AsyncMock,
            return_value=CompletionResult(content=llm_extraction_response, model="gpt-4o"),
        ):
            chunks = await extract_decisions_from_rules(
                content="some rules",
                source_file="CLAUDE.md",
                settings=test_settings,
            )

        pii_chunks = [c for c in chunks if "PII" in c.source_title]
        assert not any(c.section_type == "rejected_alternatives" for c in pii_chunks)

    @pytest.mark.asyncio
    async def test_principle_has_no_rejected_alternatives(
        self, test_settings, llm_extraction_response
    ):
        """Principles never produce rejected_alternatives chunks."""
        from app.rules_bridge import extract_decisions_from_rules

        with patch(
            "app.rules_bridge.complete",
            new_callable=AsyncMock,
            return_value=CompletionResult(content=llm_extraction_response, model="gpt-4o"),
        ):
            chunks = await extract_decisions_from_rules(
                content="some rules",
                source_file="CLAUDE.md",
                settings=test_settings,
            )

        handler_chunks = [c for c in chunks if "handler" in c.source_title.lower()]
        assert not any(c.section_type == "rejected_alternatives" for c in handler_chunks)

    @pytest.mark.asyncio
    async def test_decision_still_gets_rejected_alternatives(
        self, test_settings, llm_extraction_response
    ):
        """Decisions with rejected text still get rejected_alternatives chunks."""
        from app.rules_bridge import extract_decisions_from_rules

        with patch(
            "app.rules_bridge.complete",
            new_callable=AsyncMock,
            return_value=CompletionResult(content=llm_extraction_response, model="gpt-4o"),
        ):
            chunks = await extract_decisions_from_rules(
                content="some rules",
                source_file="CLAUDE.md",
                settings=test_settings,
            )

        jwt_rejected = [c for c in chunks if "JWT" in c.source_title and c.section_type == "rejected_alternatives"]
        assert len(jwt_rejected) == 1
        assert "Session cookies" in jwt_rejected[0].text

    @pytest.mark.asyncio
    async def test_context_chunks_created_for_substantial_context(
        self, test_settings, llm_extraction_response
    ):
        """Items with context > 30 chars get a separate context chunk."""
        from app.rules_bridge import extract_decisions_from_rules

        with patch(
            "app.rules_bridge.complete",
            new_callable=AsyncMock,
            return_value=CompletionResult(content=llm_extraction_response, model="gpt-4o"),
        ):
            chunks = await extract_decisions_from_rules(
                content="some rules",
                source_file="CLAUDE.md",
                settings=test_settings,
            )

        # PII constraint has substantial context ("Compliance requirement from Legal...")
        pii_context = [c for c in chunks if "PII" in c.source_title and c.section_type == "context"]
        assert len(pii_context) == 1
        assert "Compliance" in pii_context[0].text

    @pytest.mark.asyncio
    async def test_no_context_chunk_for_short_context(
        self, test_settings, llm_extraction_response
    ):
        """Items with empty or short context don't get a context chunk."""
        from app.rules_bridge import extract_decisions_from_rules

        with patch(
            "app.rules_bridge.complete",
            new_callable=AsyncMock,
            return_value=CompletionResult(content=llm_extraction_response, model="gpt-4o"),
        ):
            chunks = await extract_decisions_from_rules(
                content="some rules",
                source_file="CLAUDE.md",
                settings=test_settings,
            )

        # Legacy SOAP client has empty context
        soap_chunks = [c for c in chunks if "SOAP" in c.source_title]
        assert not any(c.section_type == "context" for c in soap_chunks)

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

        jwt_chunks = [c for c in chunks if "JWT" in c.source_title]
        for c in jwt_chunks:
            assert c.affected_services == ["auth-service", "api-gateway"]

        rabbitmq_chunks = [c for c in chunks if "RabbitMQ" in c.text]
        for c in rabbitmq_chunks:
            assert c.affected_services == []  # project-wide

    @pytest.mark.asyncio
    async def test_all_services_converted_to_empty_list(
        self, test_settings, llm_all_services_response
    ):
        """Services=['all'] is converted to empty list for project-wide items."""
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
    async def test_returns_empty_when_no_items(
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
    async def test_backward_compat_with_legacy_format(
        self, test_settings, llm_legacy_format_response
    ):
        """Old format with 'decisions' key and 'decision' field still works."""
        from app.rules_bridge import extract_decisions_from_rules

        with patch(
            "app.rules_bridge.complete",
            new_callable=AsyncMock,
            return_value=CompletionResult(content=llm_legacy_format_response, model="gpt-4o"),
        ):
            chunks = await extract_decisions_from_rules(
                content="some rules",
                source_file="rules.md",
                settings=test_settings,
            )

        assert len(chunks) > 0
        pg_chunk = [c for c in chunks if "PostgreSQL" in c.source_title][0]
        assert "Use PostgreSQL" in pg_chunk.text

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

        jwt_chunk = [c for c in chunks if "JWT" in c.source_title and c.section_type == "decision"][0]
        assert jwt_chunk.domain == "security"

        pii_chunk = [c for c in chunks if "PII" in c.source_title and c.section_type == "decision"][0]
        assert pii_chunk.domain == "compliance"

    @pytest.mark.asyncio
    async def test_invalid_knowledge_type_defaults_to_decision(
        self, test_settings
    ):
        """Unknown knowledge_type falls back to 'decision'."""
        from app.rules_bridge import extract_decisions_from_rules

        response = json.dumps({
            "items": [{
                "title": "Some item",
                "knowledge_type": "unknown_type",
                "content": "Something.",
                "context": "",
                "rejected": "",
                "services": [],
                "domain": "operational",
            }]
        })

        with patch(
            "app.rules_bridge.complete",
            new_callable=AsyncMock,
            return_value=CompletionResult(content=response, model="gpt-4o"),
        ):
            chunks = await extract_decisions_from_rules(
                content="some rules",
                source_file="rules.md",
                settings=test_settings,
            )

        assert chunks[0].knowledge_type == "decision"


class TestIngestDocumentWithRules:
    """Tests for the ingest_document MCP tool with rules files."""

    @pytest.mark.asyncio
    async def test_tool_extracts_and_upserts_rules(
        self, test_settings, llm_extraction_response
    ):
        """ingest_document detects rules file and extracts items."""
        from app.mcp_server import ingest_document

        with patch("app.mcp_server.get_settings", return_value=test_settings), \
             patch("app.mcp_server.ensure_collection", new_callable=AsyncMock), \
             patch(
                 "app.rules_bridge.complete",
                 new_callable=AsyncMock,
                 return_value=CompletionResult(content=llm_extraction_response, model="gpt-4o"),
             ), \
             patch("app.mcp_server.upsert", new_callable=AsyncMock) as mock_upsert, \
             patch("app.mcp_server.find_overlapping", new_callable=AsyncMock, return_value=[]):

            result = json.loads(await ingest_document(
                content="# Architecture\n- Use JWT\n- PII encrypted\n- Logic in service layer",
                filename="CLAUDE.md",
            ))

        assert result["format_detected"] == "rules_file"
        assert result["items_extracted"] == 5
        assert mock_upsert.called

    @pytest.mark.asyncio
    async def test_tool_returns_zero_when_no_items(
        self, test_settings, llm_empty_response
    ):
        """ingest_document returns helpful message when no items found."""
        from app.mcp_server import ingest_document

        with patch("app.mcp_server.get_settings", return_value=test_settings), \
             patch("app.mcp_server.ensure_collection", new_callable=AsyncMock), \
             patch(
                 "app.rules_bridge.complete",
                 new_callable=AsyncMock,
                 return_value=CompletionResult(content=llm_empty_response, model="gpt-4o"),
             ):

            result = json.loads(await ingest_document(
                content="# Code Style\n- 2 spaces\n- Named exports",
                filename=".cursorrules",
            ))

        assert result["items_extracted"] == 0
        assert result["chunks_indexed"] == 0
        assert "No architectural knowledge" in result["message"]