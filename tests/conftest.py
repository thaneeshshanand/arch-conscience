"""Shared pytest configuration and fixtures."""

import pytest


@pytest.fixture
def test_settings():
    """Return a Settings instance with safe test defaults.

    No real API keys — use this for unit tests that don't
    hit external services.
    """
    from app.config import Settings

    return Settings(
        OPENAI_API_KEY="sk-test-fake",
        ANTHROPIC_API_KEY="",
        GITHUB_TOKEN="ghp-test-fake",
        GITHUB_WEBHOOK_SECRET="test-secret",
        QDRANT_URL="http://localhost:6333",
        QDRANT_API_KEY="",
        QDRANT_COLLECTION="arch_decisions_test",
        STAGE1_MODEL="gpt-4o-mini",
        STAGE2_MODEL="gpt-4o",
        EMBEDDING_MODEL="text-embedding-3-large",
        EMBEDDING_DIM=3072,
        CONFIDENCE_THRESHOLD=0.7,
        STAGE1_THRESHOLD=0.5,
        ALERT_CHANNEL="telegram",
        TELEGRAM_BOT_TOKEN="",
        TELEGRAM_CHAT_ID="",
        WEBHOOK_PORT=3456,
    )