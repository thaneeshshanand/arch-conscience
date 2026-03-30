"""Normalizer tests — two-pass extraction pipeline.

Mocks LLM calls to verify chunk output, validation, and retry logic.

Run with:
    pytest tests/test_normalize.py -v
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.llm.base import CompletionResult
from app.normalize import (
    DiscoveredItem,
    normalize_document,
    _validate_extraction,
    _build_chunks_from_extraction,
    _extract_relevant_sections,
)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def pass1_response():
    """Pass 1 discovers two items: one decision, one constraint."""
    return json.dumps({
        "items": [
            {
                "title": "Use PostgreSQL for payment ledger",
                "knowledge_type": "decision",
                "summary": "Chose Postgres over DynamoDB for ACID guarantees",
                "relevant_sections": ["Database requirements", "Data layer"],
                "has_rejected_alternatives": True,
                "depends_on_image": False,
            },
            {
                "title": "All API calls through gateway",
                "knowledge_type": "constraint",
                "summary": "Services must not call third-party APIs directly",
                "relevant_sections": ["API patterns"],
                "has_rejected_alternatives": False,
                "depends_on_image": False,
            },
        ]
    })


@pytest.fixture
def pass2_decision_response():
    """Pass 2 extraction for a decision with rejected alternatives."""
    return json.dumps({
        "context": "Payment ledger needs ACID guarantees for financial data.",
        "decision": "Use PostgreSQL with strict isolation for the payment ledger.",
        "consequences": "Higher operational cost. Need DBA expertise.",
        "rejected_alternatives": "DynamoDB rejected due to lack of cross-item transactions.",
        "affected_services": ["payments-service"],
        "domain": "data_model",
    })


@pytest.fixture
def pass2_constraint_response():
    """Pass 2 extraction for a constraint."""
    return json.dumps({
        "context": "Security audit requires centralized API monitoring.",
        "constraint": "All external API calls must go through the API gateway.",
        "consequences": "Services cannot make direct HTTP calls to third-party APIs.",
        "affected_services": [],
        "domain": "security",
    })


@pytest.fixture
def pass1_empty_response():
    return json.dumps({"items": []})


@pytest.fixture
def sample_document():
    return (
        "## Database requirements\n\n"
        "The payment ledger needs ACID guarantees.\n\n"
        "## Data layer\n\n"
        "We chose PostgreSQL for all transactional data. "
        "DynamoDB was considered but lacks cross-item transactions.\n\n"
        "## API patterns\n\n"
        "All external API calls must go through the API gateway. "
        "Services must not call third-party APIs directly.\n"
    )


# ── Tests ────────────────────────────────────────────────────────────


class TestNormalizeDocument:
    """Full pipeline tests."""

    @pytest.mark.asyncio
    async def test_extracts_decision_and_constraint(
        self, test_settings, sample_document,
        pass1_response, pass2_decision_response, pass2_constraint_response,
    ):
        """Pipeline discovers 2 items, extracts both, produces correct chunks."""
        call_count = 0

        async def mock_complete(messages, *, model, temperature=0, max_tokens=4096, response_format=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return CompletionResult(content=pass1_response, model=model)
            elif call_count == 2:
                return CompletionResult(content=pass2_decision_response, model=model)
            else:
                return CompletionResult(content=pass2_constraint_response, model=model)

        with patch("app.normalize.complete", side_effect=mock_complete):
            result = await normalize_document(
                sample_document,
                filename="design.md",
                source_type="design_doc",
                settings=test_settings,
            )

        assert result.items_discovered == 2
        assert result.items_extracted == 2
        assert result.items_failed == []

        # Decision: context + decision + consequences + rejected_alternatives = 4
        # Constraint: context + decision + consequences = 3
        assert len(result.chunks) == 7

        # Verify decision chunks
        pg_chunks = [c for c in result.chunks if "PostgreSQL" in c.text]
        assert any(c.section_type == "rejected_alternatives" for c in pg_chunks)
        assert any(c.section_type == "decision" for c in pg_chunks)

        pg_decision = [c for c in pg_chunks if c.section_type == "decision"][0]
        assert pg_decision.knowledge_type == "decision"
        assert pg_decision.domain == "data_model"
        assert pg_decision.affected_services == ["payments-service"]
        assert pg_decision.source_type == "design_doc"

        # Verify constraint chunks
        gw_chunks = [c for c in result.chunks if "gateway" in c.text.lower()]
        assert any(c.section_type == "decision" for c in gw_chunks)
        gw_decision = [c for c in gw_chunks if c.section_type == "decision"][0]
        assert gw_decision.knowledge_type == "constraint"
        assert gw_decision.domain == "security"

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_items_discovered(
        self, test_settings, pass1_empty_response,
    ):
        """Empty document returns no chunks."""
        with patch(
            "app.normalize.complete",
            new_callable=AsyncMock,
            return_value=CompletionResult(content=pass1_empty_response, model="gpt-4o"),
        ):
            result = await normalize_document(
                "# Code Style\n\nUse 2-space indent.",
                filename="style.md",
                settings=test_settings,
            )

        assert result.items_discovered == 0
        assert result.chunks == []

    @pytest.mark.asyncio
    async def test_handles_pass2_failure_gracefully(
        self, test_settings, pass1_response,
    ):
        """If Pass 2 fails for an item, other items are still extracted."""
        pass1_done = False

        async def mock_complete(messages, *, model, temperature=0, max_tokens=4096, response_format=None):
            nonlocal pass1_done
            if not pass1_done:
                pass1_done = True
                return CompletionResult(content=pass1_response, model=model)

            # Route by item title: PostgreSQL item fails, gateway item succeeds
            user_msg = messages[-1].content if messages else ""
            if "Title: Use PostgreSQL" in user_msg:
                return CompletionResult(content="not json", model=model)
            else:
                return CompletionResult(content=json.dumps({
                    "context": "Security requirement.",
                    "constraint": "All calls through gateway.",
                    "consequences": "No direct calls.",
                    "affected_services": [],
                    "domain": "security",
                }), model=model)

        with patch("app.normalize.complete", side_effect=mock_complete):
            result = await normalize_document(
                "## Arch\n\nSome content about decisions.",
                filename="doc.md",
                settings=test_settings,
            )

        assert result.items_discovered == 2
        assert result.items_extracted == 1
        assert len(result.items_failed) == 1

    @pytest.mark.asyncio
    async def test_empty_content_returns_empty(self, test_settings):
        """Blank content returns immediately."""
        result = await normalize_document("", filename="empty.md", settings=test_settings)
        assert result.chunks == []
        assert result.items_discovered == 0


class TestBuildChunks:
    """Tests for _build_chunks_from_extraction."""

    def test_decision_produces_four_chunks(self):
        """Decision with rejected alternatives produces 4 chunks."""
        item = DiscoveredItem(
            title="Use JWT",
            knowledge_type="decision",
            summary="Chose JWT",
            has_rejected_alternatives=True,
        )
        parsed = {
            "context": "Need stateless auth.",
            "decision": "Use signed JWTs.",
            "consequences": "Token revocation needed.",
            "rejected_alternatives": "Session cookies rejected.",
            "affected_services": ["auth-service"],
            "domain": "security",
        }
        chunks = _build_chunks_from_extraction(
            parsed=parsed, item=item, filename="doc.md",
            source_url="", source_type="design_doc", item_index=0,
        )
        assert len(chunks) == 4
        types = {c.section_type for c in chunks}
        assert types == {"context", "decision", "consequences", "rejected_alternatives"}

    def test_constraint_produces_three_chunks(self):
        """Constraint produces 3 chunks (no rejected_alternatives)."""
        item = DiscoveredItem(
            title="HTTPS only",
            knowledge_type="constraint",
            summary="All traffic over HTTPS",
        )
        parsed = {
            "context": "Security policy.",
            "constraint": "All traffic must use HTTPS.",
            "consequences": "Need cert management.",
            "affected_services": [],
            "domain": "security",
        }
        chunks = _build_chunks_from_extraction(
            parsed=parsed, item=item, filename="doc.md",
            source_url="", source_type="rfc", item_index=0,
        )
        assert len(chunks) == 3
        assert not any(c.section_type == "rejected_alternatives" for c in chunks)

    def test_empty_sections_skipped(self):
        """Empty text sections don't produce chunks."""
        item = DiscoveredItem(title="X", knowledge_type="decision", summary="X")
        parsed = {
            "context": "",
            "decision": "Do X.",
            "consequences": "",
            "rejected_alternatives": "",
            "affected_services": [],
            "domain": "operational",
        }
        chunks = _build_chunks_from_extraction(
            parsed=parsed, item=item, filename="doc.md",
            source_url="", source_type="", item_index=0,
        )
        assert len(chunks) == 1
        assert chunks[0].section_type == "decision"

    def test_source_metadata_set(self):
        """Source URL, title, and ingested_at are set."""
        item = DiscoveredItem(title="My Decision", knowledge_type="decision", summary="S")
        parsed = {
            "context": "Why.",
            "decision": "What.",
            "consequences": "",
            "rejected_alternatives": "",
            "affected_services": [],
            "domain": "operational",
        }
        chunks = _build_chunks_from_extraction(
            parsed=parsed, item=item, filename="doc.md",
            source_url="https://wiki.example.com/page/123",
            source_type="confluence", item_index=0,
        )
        for c in chunks:
            assert c.source_url == "https://wiki.example.com/page/123"
            assert c.source_title == "My Decision"
            assert c.ingested_at != ""
            assert c.source_type == "confluence"


class TestValidation:
    """Tests for _validate_extraction."""

    def test_valid_extraction_passes(self):
        from app.corpus import ChunkRecord
        item = DiscoveredItem(title="X", knowledge_type="decision", summary="S")
        chunks = [ChunkRecord(
            id="test-decision",
            text="Decision: X\nSection: Decision\n\nThe team decided to use approach X for the service layer because it provides better scalability.",
            section_type="decision", knowledge_type="decision",
        )]
        assert _validate_extraction(chunks, item) is True

    def test_empty_chunks_fails(self):
        item = DiscoveredItem(title="X", knowledge_type="decision", summary="S")
        assert _validate_extraction([], item) is False

    def test_missing_rejected_alt_fails(self):
        from app.corpus import ChunkRecord
        item = DiscoveredItem(
            title="X", knowledge_type="decision", summary="S",
            has_rejected_alternatives=True,
        )
        chunks = [ChunkRecord(
            id="test-decision", text="Decision: X\nSection: Decision\n\nSome decision text here that is long enough.",
            section_type="decision", knowledge_type="decision",
        )]
        assert _validate_extraction(chunks, item) is False

    def test_short_decision_fails(self):
        from app.corpus import ChunkRecord
        item = DiscoveredItem(title="X", knowledge_type="decision", summary="S")
        chunks = [ChunkRecord(
            id="test-decision", text="Decision: X\nSection: Decision\n\nShort.",
            section_type="decision", knowledge_type="decision",
        )]
        assert _validate_extraction(chunks, item) is False


class TestExtractRelevantSections:
    """Tests for section extraction helper."""

    def test_matches_sections_by_name(self):
        doc = "## Overview\n\nIntro text.\n\n## Database\n\nUse Postgres.\n\n## Testing\n\nUnit tests."
        result = _extract_relevant_sections(doc, ["Database"])
        assert "Use Postgres" in result
        assert "Unit tests" not in result

    def test_fuzzy_matching(self):
        doc = "## Database requirements\n\nUse Postgres.\n\n## Testing\n\nUnit tests."
        result = _extract_relevant_sections(doc, ["Database"])
        assert "Use Postgres" in result

    def test_no_match_returns_full_doc(self):
        doc = "## Overview\n\nContent here."
        result = _extract_relevant_sections(doc, ["Nonexistent section"])
        assert "Content here" in result

    def test_empty_sections_returns_full_doc(self):
        doc = "Some content."
        result = _extract_relevant_sections(doc, [])
        assert "Some content" in result