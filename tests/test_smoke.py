"""Smoke tests — config loads and Qdrant is reachable.

Run with:
    pytest tests/test_smoke.py -v
"""

import pytest

from app.config import Settings, get_settings


class TestConfig:
    """Settings instantiation and validation."""

    def test_settings_loads_with_defaults(self):
        """Settings can be instantiated with no env vars (all defaults)."""
        s = Settings(
            OPENAI_API_KEY="sk-test",
            GITHUB_TOKEN="ghp-test",
            GITHUB_WEBHOOK_SECRET="secret",
            QDRANT_URL="http://localhost:6333",
        )
        assert s.STAGE1_MODEL == "gpt-4o-mini"
        assert s.STAGE2_MODEL == "gpt-4o"
        assert s.EMBEDDING_MODEL == "text-embedding-3-large"
        assert s.EMBEDDING_DIM == 3072
        assert s.CONFIDENCE_THRESHOLD == 0.7
        assert s.STAGE1_THRESHOLD == 0.5
        assert s.QDRANT_COLLECTION == "arch_decisions"
        assert s.WEBHOOK_PORT == 3456

    def test_validate_required_passes_with_all_keys(self):
        """validate_required() succeeds when all required vars are set."""
        s = Settings(
            OPENAI_API_KEY="sk-test",
            GITHUB_TOKEN="ghp-test",
            GITHUB_WEBHOOK_SECRET="secret",
            QDRANT_URL="http://localhost:6333",
        )
        s.validate_required()  # should not raise

    def test_validate_required_fails_missing_github_token(self):
        """validate_required() raises when GITHUB_TOKEN is missing."""
        s = Settings(
            OPENAI_API_KEY="sk-test",
            GITHUB_TOKEN="",
            GITHUB_WEBHOOK_SECRET="secret",
            QDRANT_URL="http://localhost:6333",
        )
        with pytest.raises(ValueError, match="GITHUB_TOKEN"):
            s.validate_required()

    def test_validate_required_fails_missing_openai_key(self):
        """validate_required() raises when OpenAI key is missing
        but model strings require it."""
        s = Settings(
            OPENAI_API_KEY="",
            GITHUB_TOKEN="ghp-test",
            GITHUB_WEBHOOK_SECRET="secret",
            QDRANT_URL="http://localhost:6333",
            STAGE2_MODEL="gpt-4o",
        )
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            s.validate_required()

    def test_validate_required_fails_missing_anthropic_key(self):
        """validate_required() raises when Anthropic key is missing
        but a model string routes to Anthropic."""
        s = Settings(
            OPENAI_API_KEY="sk-test",
            ANTHROPIC_API_KEY="",
            GITHUB_TOKEN="ghp-test",
            GITHUB_WEBHOOK_SECRET="secret",
            QDRANT_URL="http://localhost:6333",
            STAGE2_MODEL="anthropic/claude-sonnet-4-20250514",
        )
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            s.validate_required()

    def test_validate_required_collects_all_errors(self):
        """validate_required() lists all missing vars, not just the first."""
        s = Settings(
            OPENAI_API_KEY="",
            GITHUB_TOKEN="",
            GITHUB_WEBHOOK_SECRET="",
            QDRANT_URL="http://localhost:6333",
        )
        with pytest.raises(ValueError) as exc_info:
            s.validate_required()
        msg = str(exc_info.value)
        assert "GITHUB_TOKEN" in msg
        assert "GITHUB_WEBHOOK_SECRET" in msg
        assert "OPENAI_API_KEY" in msg

    def test_service_map_parses_valid_json(self):
        """service_map property parses valid JSON."""
        s = Settings(
            SERVICE_MAP='{"services/auth":"auth-service","services/pay":"pay-service"}',
        )
        assert s.service_map == {
            "services/auth": "auth-service",
            "services/pay": "pay-service",
        }

    def test_service_map_returns_empty_on_invalid_json(self):
        """service_map property returns {} on invalid JSON."""
        s = Settings(SERVICE_MAP="not json")
        assert s.service_map == {}

    def test_anthropic_model_detection(self):
        """Models prefixed with 'anthropic/' require ANTHROPIC_API_KEY."""
        s = Settings(
            OPENAI_API_KEY="sk-test",
            ANTHROPIC_API_KEY="sk-ant-test",
            GITHUB_TOKEN="ghp-test",
            GITHUB_WEBHOOK_SECRET="secret",
            QDRANT_URL="http://localhost:6333",
            STAGE2_MODEL="anthropic/claude-sonnet-4-20250514",
        )
        s.validate_required()  # should not raise — both keys present


class TestQdrantConnection:
    """Qdrant reachability (requires a running Qdrant instance)."""

    @pytest.mark.asyncio
    async def test_qdrant_reachable(self):
        """Qdrant responds to a collections list request.

        Requires Qdrant running at QDRANT_URL (default localhost:6333).
        Skip with: pytest -m "not integration"
        """
        pytest.importorskip("qdrant_client")
        from qdrant_client import AsyncQdrantClient

        s = get_settings()
        client = AsyncQdrantClient(
            url=s.QDRANT_URL,
            api_key=s.QDRANT_API_KEY or None,
        )

        try:
            collections = await client.get_collections()
            assert collections is not None
        finally:
            await client.close()