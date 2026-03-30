"""Notify tests — Slack channel routing and message formatting.

Run with:
    pytest tests/test_notify.py -v
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import Settings
from app.detect import DetectionResult
from app.notify import _format_slack_message, _resolve_slack_channels


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def slack_settings():
    return Settings(
        OPENAI_API_KEY="sk-test",
        GITHUB_TOKEN="ghp-test",
        GITHUB_WEBHOOK_SECRET="secret",
        QDRANT_URL="http://localhost:6333",
        ALERT_CHANNEL="slack",
        SLACK_WEBHOOK_URL="https://hooks.slack.com/default",
        SLACK_CHANNEL_MAP=(
            '{"auth-service": "https://hooks.slack.com/auth", '
            '"payments-service": "https://hooks.slack.com/payments"}'
        ),
    )


@pytest.fixture
def gap_result():
    return DetectionResult(
        gap_detected=True,
        confidence=0.95,
        severity="high",
        violated_adr_id="adr-001",
        constraint_type="security",
        rejected_alt_reintroduced=True,
        change_summary="Adds session cookies",
        reasoning="Contradicts ADR-001",
        alert_headline="PR #42 may reintroduce session cookies",
        alert_body="ADR-001 mandates JWT. This PR adds session cookies.",
    )


def _make_mock_client(status_code: int = 200):
    """Build a patched httpx.AsyncClient that captures POST calls."""
    post_mock = AsyncMock(return_value=MagicMock(status_code=status_code, text="ok"))
    client_mock = AsyncMock()
    client_mock.post = post_mock
    cls_mock = MagicMock()
    cls_mock.return_value.__aenter__ = AsyncMock(return_value=client_mock)
    cls_mock.return_value.__aexit__ = AsyncMock(return_value=None)
    return cls_mock, post_mock


# ── Channel resolution ────────────────────────────────────────────────


class TestResolveSlackChannels:
    def test_mapped_service_returns_its_channel(self, slack_settings):
        urls = _resolve_slack_channels(["auth-service"], slack_settings)
        assert urls == {"https://hooks.slack.com/auth"}

    def test_unmapped_service_falls_back_to_default(self, slack_settings):
        urls = _resolve_slack_channels(["unknown-service"], slack_settings)
        assert urls == {"https://hooks.slack.com/default"}

    def test_multiple_services_return_multiple_channels(self, slack_settings):
        urls = _resolve_slack_channels(["auth-service", "payments-service"], slack_settings)
        assert urls == {
            "https://hooks.slack.com/auth",
            "https://hooks.slack.com/payments",
        }

    def test_deduplicates_channels_when_services_share_one(self):
        settings = Settings(
            OPENAI_API_KEY="sk-test",
            GITHUB_TOKEN="ghp-test",
            GITHUB_WEBHOOK_SECRET="secret",
            QDRANT_URL="http://localhost:6333",
            SLACK_CHANNEL_MAP=(
                '{"auth-service": "https://hooks.slack.com/shared", '
                '"api-gateway": "https://hooks.slack.com/shared"}'
            ),
        )
        urls = _resolve_slack_channels(["auth-service", "api-gateway"], settings)
        assert len(urls) == 1

    def test_empty_when_nothing_configured(self):
        settings = Settings(
            OPENAI_API_KEY="sk-test",
            GITHUB_TOKEN="ghp-test",
            GITHUB_WEBHOOK_SECRET="secret",
            QDRANT_URL="http://localhost:6333",
            SLACK_WEBHOOK_URL="",
            SLACK_CHANNEL_MAP="{}",
        )
        urls = _resolve_slack_channels(["auth-service"], settings)
        assert urls == set()

    def test_empty_services_list_falls_back_to_default(self, slack_settings):
        urls = _resolve_slack_channels([], slack_settings)
        assert urls == {"https://hooks.slack.com/default"}

    def test_invalid_channel_map_json_returns_empty_then_default(self):
        settings = Settings(
            OPENAI_API_KEY="sk-test",
            GITHUB_TOKEN="ghp-test",
            GITHUB_WEBHOOK_SECRET="secret",
            QDRANT_URL="http://localhost:6333",
            SLACK_WEBHOOK_URL="https://hooks.slack.com/default",
            SLACK_CHANNEL_MAP="not json",
        )
        urls = _resolve_slack_channels(["auth-service"], settings)
        assert urls == {"https://hooks.slack.com/default"}


# ── HTTP dispatch ─────────────────────────────────────────────────────


class TestSendSlack:
    @pytest.mark.asyncio
    async def test_posts_to_mapped_channel(self, slack_settings, gap_result):
        from app.notify import dispatch

        cls_mock, post_mock = _make_mock_client()
        with patch("app.notify.httpx.AsyncClient", cls_mock):
            await dispatch(
                result=gap_result,
                engineer="dev",
                pr_url="https://github.com/org/repo/pull/1",
                affected_services=["auth-service"],
                settings=slack_settings,
            )

        post_mock.assert_called_once()
        assert post_mock.call_args[0][0] == "https://hooks.slack.com/auth"

    @pytest.mark.asyncio
    async def test_posts_to_multiple_channels(self, slack_settings, gap_result):
        from app.notify import dispatch

        cls_mock, post_mock = _make_mock_client()
        with patch("app.notify.httpx.AsyncClient", cls_mock):
            await dispatch(
                result=gap_result,
                engineer="dev",
                pr_url="https://github.com/org/repo/pull/1",
                affected_services=["auth-service", "payments-service"],
                settings=slack_settings,
            )

        assert post_mock.call_count == 2
        posted_urls = {c[0][0] for c in post_mock.call_args_list}
        assert posted_urls == {
            "https://hooks.slack.com/auth",
            "https://hooks.slack.com/payments",
        }

    @pytest.mark.asyncio
    async def test_no_http_call_when_no_webhook_configured(self, gap_result):
        from app.notify import dispatch

        settings = Settings(
            OPENAI_API_KEY="sk-test",
            GITHUB_TOKEN="ghp-test",
            GITHUB_WEBHOOK_SECRET="secret",
            QDRANT_URL="http://localhost:6333",
            ALERT_CHANNEL="slack",
            SLACK_WEBHOOK_URL="",
            SLACK_CHANNEL_MAP="{}",
        )
        cls_mock, post_mock = _make_mock_client()
        with patch("app.notify.httpx.AsyncClient", cls_mock):
            await dispatch(
                result=gap_result,
                engineer="dev",
                pr_url="https://github.com/org/repo/pull/1",
                affected_services=["auth-service"],
                settings=settings,
            )

        post_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_on_non_200_response(self, slack_settings, gap_result):
        from app.notify import dispatch

        cls_mock, _ = _make_mock_client(status_code=400)
        with patch("app.notify.httpx.AsyncClient", cls_mock):
            with pytest.raises(RuntimeError, match="Slack webhook error 400"):
                await dispatch(
                    result=gap_result,
                    engineer="dev",
                    pr_url="https://github.com/org/repo/pull/1",
                    affected_services=["auth-service"],
                    settings=slack_settings,
                )

    @pytest.mark.asyncio
    async def test_payload_includes_blocks(self, slack_settings, gap_result):
        from app.notify import dispatch

        cls_mock, post_mock = _make_mock_client()
        with patch("app.notify.httpx.AsyncClient", cls_mock):
            await dispatch(
                result=gap_result,
                engineer="dev",
                pr_url="https://github.com/org/repo/pull/1",
                affected_services=["auth-service"],
                settings=slack_settings,
            )

        payload = post_mock.call_args[1]["json"]
        assert "text" in payload
        assert "blocks" in payload
        assert payload["blocks"][0]["type"] == "section"


# ── Message formatting ────────────────────────────────────────────────


class TestFormatSlackMessage:
    def test_uses_mrkdwn_link(self, gap_result):
        msg = _format_slack_message(
            result=gap_result,
            engineer="dev",
            pr_url="https://github.com/org/repo/pull/1",
        )
        assert "<https://github.com/org/repo/pull/1|View PR>" in msg

    def test_bolds_fields(self, gap_result):
        msg = _format_slack_message(
            result=gap_result,
            engineer="dev",
            pr_url="https://github.com/org/repo/pull/1",
        )
        assert "*PR:*" in msg
        assert "*Severity:*" in msg

    def test_includes_severity_emoji(self, gap_result):
        msg = _format_slack_message(
            result=gap_result,
            engineer="dev",
            pr_url="https://github.com/org/repo/pull/1",
        )
        assert "🔴" in msg

    def test_includes_rejected_alt_warning(self, gap_result):
        msg = _format_slack_message(
            result=gap_result,
            engineer="dev",
            pr_url="https://github.com/org/repo/pull/1",
        )
        assert "explicitly rejected" in msg

    def test_includes_corpus_conflict_note(self):
        result = DetectionResult(
            gap_detected=True,
            severity="medium",
            alert_headline="Conflict",
            alert_body="Body.",
            corpus_conflict=True,
        )
        msg = _format_slack_message(
            result=result,
            engineer="dev",
            pr_url="https://github.com/org/repo/pull/1",
        )
        assert "conflicting guidance" in msg

    def test_no_crash_on_none_fields(self):
        result = DetectionResult(
            gap_detected=False,
            severity=None,
            alert_headline=None,
            alert_body=None,
        )
        msg = _format_slack_message(
            result=result,
            engineer="dev",
            pr_url="https://github.com/org/repo/pull/1",
        )
        assert "@dev" in msg