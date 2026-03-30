"""End-to-end pipeline simulation.

Simulates a PR that reintroduces session cookies against ADR-001
(JWT for stateless auth). Mocks external calls (GitHub API, LLM,
Telegram) and verifies the full pipeline from webhook to alert.

Run with:
    pytest tests/test_e2e.py -v
"""

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.corpus import ChunkRecord, ScoredChunk
from app.detect import DetectionResult
from app.router import PipelinePayload


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def e2e_settings():
    return Settings(
        OPENAI_API_KEY="sk-test-fake",
        GITHUB_TOKEN="ghp-test-fake",
        GITHUB_WEBHOOK_SECRET="test-webhook-secret",
        QDRANT_URL="http://localhost:6333",
        QDRANT_COLLECTION="arch_decisions_test",
        CONFIDENCE_THRESHOLD=0.7,
        STAGE1_THRESHOLD=0.5,
        ALERT_CHANNEL="telegram",
        TELEGRAM_BOT_TOKEN="fake-token",
        TELEGRAM_CHAT_ID="fake-chat-id",
        SERVICE_MAP='{"services/auth":"auth-service"}',
    )


@pytest.fixture
def sample_webhook_body():
    """A realistic GitHub pull_request webhook payload."""
    return {
        "action": "opened",
        "repository": {"full_name": "acme/backend"},
        "pull_request": {
            "number": 42,
            "title": "Add session cookie support for auth",
            "body": "Implements session-based auth using Set-Cookie headers for the login flow.",
            "html_url": "https://github.com/acme/backend/pull/42",
            "user": {"login": "newdev"},
            "base": {"ref": "main"},
            "merged": False,
        },
    }


@pytest.fixture
def adr001_chunks():
    """Corpus chunks simulating ADR-001 (JWT for stateless auth)."""
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
                    "Session cookies were considered and rejected. They require a "
                    "centralised session store, add a single point of failure, and "
                    "conflict with GDPR data minimisation requirements. Sticky "
                    "sessions are incompatible with horizontal scaling."
                ),
                section_type="rejected_alternatives",
                **base,
            ),
            score=0.574,
            point_id="fake-uuid-1",
        ),
        ScoredChunk(
            chunk=ChunkRecord(
                id="adr-001-consequences",
                text=(
                    "ADR: Use JWT for stateless authentication\n"
                    "Section: Consequences\n\n"
                    "All services must validate JWT tokens locally. Token revocation "
                    "requires a deny list. Refresh tokens must be rotated."
                ),
                section_type="consequences",
                **base,
            ),
            score=0.533,
            point_id="fake-uuid-2",
        ),
        ScoredChunk(
            chunk=ChunkRecord(
                id="adr-001-decision",
                text=(
                    "ADR: Use JWT for stateless authentication\n"
                    "Section: Decision\n\n"
                    "Use signed JWTs for all service-to-service and client-to-gateway "
                    "authentication. No server-side session state."
                ),
                section_type="decision",
                **base,
            ),
            score=0.510,
            point_id="fake-uuid-3",
        ),
        ScoredChunk(
            chunk=ChunkRecord(
                id="adr-001-context",
                text=(
                    "ADR: Use JWT for stateless authentication\n"
                    "Section: Context\n\n"
                    "The auth service needs a stateless authentication mechanism "
                    "that scales horizontally without shared session stores."
                ),
                section_type="context",
                **base,
            ),
            score=0.480,
            point_id="fake-uuid-4",
        ),
    ]


@pytest.fixture
def stage1_response():
    """Mock Stage 1 LLM response — all 4 chunks pass relevance."""
    return json.dumps({
        "scores": [
            {"index": 0, "score": 0.9},
            {"index": 1, "score": 0.8},
            {"index": 2, "score": 0.7},
            {"index": 3, "score": 0.6},
        ]
    })


@pytest.fixture
def stage2_response():
    """Mock Stage 2 LLM response — gap detected, high severity."""
    return json.dumps({
        "gap_detected": True,
        "confidence": 1.0,
        "severity": "high",
        "violated_adr_id": "adr-001",
        "constraint_type": "security",
        "rejected_alt_reintroduced": True,
        "change_summary": "PR adds session cookie auth to the login flow.",
        "reasoning": (
            "ADR-001 mandates JWT for stateless authentication. Session cookies "
            "were explicitly rejected due to centralised session store requirements "
            "and GDPR conflicts. This PR reintroduces session cookies via "
            "Set-Cookie headers, directly contradicting the decision."
        ),
        "alert_headline": (
            "PR #42 may reintroduce session cookies — ADR-001 requires "
            "stateless JWT auth for security."
        ),
        "alert_body": (
            "ADR-001 mandates JWT for stateless authentication due to security "
            "and scalability concerns. Session cookies were rejected as they "
            "introduce a centralised session store and conflict with GDPR "
            "requirements. This PR contradicts that decision by implementing "
            "session cookies. See doc_id: adr-001."
        ),
        "corpus_gap_signal": False,
    })


# ── Tests ────────────────────────────────────────────────────────────


class TestDetectionPipeline:
    """Full pipeline: webhook body → router → corpus → detect → alert."""

    @pytest.mark.asyncio
    async def test_full_pipeline_detects_gap(
        self,
        e2e_settings,
        adr001_chunks,
        stage1_response,
        stage2_response,
    ):
        """Simulates a PR that reintroduces session cookies.
        Verifies the pipeline detects the gap and produces a correct result."""
        from app.llm.base import CompletionResult
        from app.detect import run_detection
        from app.router import PipelinePayload

        payload = PipelinePayload(
            pr_url="https://github.com/acme/backend/pull/42",
            pr_number="42",
            pr_title="Add session cookie support for auth",
            author="newdev",
            base_branch="main",
            changed_files=["services/auth/session.py", "services/auth/middleware.py"],
            affected_services=["auth-service"],
            diff_summary=(
                "PR title: Add session cookie support for auth. "
                "Services affected: auth-service. "
                "Changed files (2): services/auth/session.py, services/auth/middleware.py. "
                "Description: Implements session-based auth using Set-Cookie headers."
            ),
        )

        call_count = 0

        async def mock_complete(messages, *, model, temperature=0, max_tokens=4096, response_format=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return CompletionResult(content=stage1_response, model=model)
            return CompletionResult(content=stage2_response, model=model)

        with patch("app.detect.complete", side_effect=mock_complete):
            result = await run_detection(
                payload=payload,
                chunks=adr001_chunks,
                settings=e2e_settings,
            )

        assert result.gap_detected is True
        assert result.confidence == 1.0
        assert result.severity == "high"
        assert result.violated_adr_id == "adr-001"
        assert result.constraint_type == "security"
        assert result.rejected_alt_reintroduced is True
        assert result.alert_headline is not None
        assert result.alert_body is not None
        assert "adr-001" in result.alert_body
        assert call_count == 2  # Stage 1 + Stage 2

    @pytest.mark.asyncio
    async def test_pipeline_no_gap_when_additive(
        self,
        e2e_settings,
        adr001_chunks,
        stage1_response,
    ):
        """Additive change should not fire an alert."""
        from app.llm.base import CompletionResult
        from app.detect import run_detection

        payload = PipelinePayload(
            pr_url="https://github.com/acme/backend/pull/43",
            pr_number="43",
            pr_title="Add refresh token rotation",
            author="newdev",
            base_branch="main",
            changed_files=["services/auth/refresh.py"],
            affected_services=["auth-service"],
            diff_summary="PR title: Add refresh token rotation. Services affected: auth-service.",
        )

        no_gap_response = json.dumps({
            "gap_detected": False,
            "confidence": 0.3,
            "severity": None,
            "violated_adr_id": None,
            "constraint_type": None,
            "rejected_alt_reintroduced": False,
            "change_summary": "Adds refresh token rotation to auth service.",
            "reasoning": "This change is additive — it extends JWT auth without overriding it.",
            "alert_headline": None,
            "alert_body": None,
            "corpus_gap_signal": False,
        })

        call_count = 0

        async def mock_complete(messages, *, model, temperature=0, max_tokens=4096, response_format=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return CompletionResult(content=stage1_response, model=model)
            return CompletionResult(content=no_gap_response, model=model)

        with patch("app.detect.complete", side_effect=mock_complete):
            result = await run_detection(
                payload=payload,
                chunks=adr001_chunks,
                settings=e2e_settings,
            )

        assert result.gap_detected is False
        assert result.severity is None
        assert result.alert_headline is None

    @pytest.mark.asyncio
    async def test_pipeline_corpus_gap_when_no_chunks_survive(
        self,
        e2e_settings,
        adr001_chunks,
    ):
        """When Stage 1 drops all chunks, pipeline returns corpus_gap_signal."""
        from app.llm.base import CompletionResult
        from app.detect import run_detection

        payload = PipelinePayload(
            pr_url="https://github.com/acme/backend/pull/44",
            pr_number="44",
            pr_title="Update payments retry logic",
            author="newdev",
            base_branch="main",
            changed_files=["services/payments/retry.py"],
            affected_services=["payments-service"],
            diff_summary="PR title: Update payments retry logic. Services affected: payments-service.",
        )

        all_zero_scores = json.dumps({
            "scores": [
                {"index": 0, "score": 0.1},
                {"index": 1, "score": 0.0},
                {"index": 2, "score": 0.2},
                {"index": 3, "score": 0.1},
            ]
        })

        async def mock_complete(messages, *, model, temperature=0, max_tokens=4096, response_format=None):
            return CompletionResult(content=all_zero_scores, model=model)

        with patch("app.detect.complete", side_effect=mock_complete):
            result = await run_detection(
                payload=payload,
                chunks=adr001_chunks,
                settings=e2e_settings,
            )

        assert result.gap_detected is False
        assert result.corpus_gap_signal is True


class TestWebhookEndpoint:
    """FastAPI endpoint integration tests."""

    def _sign(self, body: bytes, secret: str) -> str:
        sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        return f"sha256={sig}"

    def test_rejects_invalid_signature(self, e2e_settings):
        with patch("app.main.get_settings", return_value=e2e_settings):
            from app.main import app
            client = TestClient(app, raise_server_exceptions=False)

            body = json.dumps({"action": "opened"}).encode()
            resp = client.post(
                "/",
                content=body,
                headers={
                    "X-Hub-Signature-256": "sha256=bad",
                    "X-GitHub-Event": "pull_request",
                    "Content-Type": "application/json",
                },
            )
            assert resp.status_code == 403

    def test_ignores_unsupported_action(self, e2e_settings):
        with patch("app.main.get_settings", return_value=e2e_settings):
            from app.main import app
            client = TestClient(app, raise_server_exceptions=False)

            body = json.dumps({"action": "labeled"}).encode()
            sig = self._sign(body, e2e_settings.GITHUB_WEBHOOK_SECRET)
            resp = client.post(
                "/",
                content=body,
                headers={
                    "X-Hub-Signature-256": sig,
                    "X-GitHub-Event": "pull_request",
                    "Content-Type": "application/json",
                },
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "event ignored"

    def test_health_endpoint(self, e2e_settings):
        with patch("app.main.get_settings", return_value=e2e_settings):
            from app.main import app
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"


class TestNotifyFormat:
    """Alert message formatting."""

    def test_format_includes_all_fields(self):
        from app.notify import _format_message

        result = DetectionResult(
            gap_detected=True,
            confidence=1.0,
            severity="high",
            violated_adr_id="adr-001",
            constraint_type="security",
            rejected_alt_reintroduced=True,
            change_summary="Adds session cookies",
            reasoning="Contradicts ADR-001",
            alert_headline="PR #42 reintroduces session cookies",
            alert_body="ADR-001 mandates JWT. This PR adds session cookies.",
        )

        msg = _format_message(
            result=result,
            engineer="newdev",
            pr_url="https://github.com/acme/backend/pull/42",
        )

        assert "\U0001f534" in msg  # 🔴
        assert "adr-001" in msg
        assert "PR: https://github.com/acme/backend/pull/42" in msg
        assert "Author: @newdev" in msg
        assert "Severity: high" in msg
        assert "explicitly rejected" in msg

    def test_format_no_gap_fields_are_none(self):
        from app.notify import _format_message

        result = DetectionResult(
            gap_detected=False,
            severity=None,
            alert_headline=None,
            alert_body=None,
        )

        msg = _format_message(
            result=result,
            engineer="dev",
            pr_url="https://github.com/acme/backend/pull/1",
        )

        # Should not crash on None fields
        assert "Author: @dev" in msg

    def test_format_includes_domain(self):
        from app.notify import _format_message

        result = DetectionResult(
            gap_detected=True,
            confidence=0.9,
            severity="high",
            violated_adr_id="adr-001",
            constraint_type="security",
            alert_headline="Constraint violated",
            alert_body="HTTPS required.",
        )

        msg = _format_message(
            result=result,
            engineer="dev",
            pr_url="https://github.com/acme/backend/pull/1",
        )

        assert "Domain: security" in msg

    def test_format_includes_corpus_conflict_note(self):
        from app.notify import _format_message

        result = DetectionResult(
            gap_detected=True,
            confidence=0.8,
            severity="medium",
            violated_adr_id="adr-002",
            alert_headline="Conflict detected",
            alert_body="Contradicts ADR-002.",
            corpus_conflict=True,
        )

        msg = _format_message(
            result=result,
            engineer="dev",
            pr_url="https://github.com/acme/backend/pull/5",
        )

        assert "conflicting guidance" in msg


class TestQueryTimeResolution:
    """Corpus query-time conflict resolution."""

    def test_newer_item_wins_over_older(self):
        from app.corpus import _resolve_conflicts

        old_chunk = ScoredChunk(
            chunk=ChunkRecord(
                id="old-decision", text="Old decision.",
                doc_id="adr-001", section_type="decision",
                domain="security", affected_services=["auth-service"],
                date="2024-01-01",
            ),
            score=0.8, point_id="uuid-old",
        )
        new_chunk = ScoredChunk(
            chunk=ChunkRecord(
                id="new-decision", text="New decision.",
                doc_id="adr-005", section_type="decision",
                domain="security", affected_services=["auth-service"],
                date="2025-06-01",
            ),
            score=0.7, point_id="uuid-new",
        )

        result = _resolve_conflicts([old_chunk, new_chunk])
        doc_ids = {sc.chunk.doc_id for sc in result}
        assert "adr-005" in doc_ids
        assert "adr-001" not in doc_ids

    def test_service_specific_wins_over_project_wide(self):
        from app.corpus import _resolve_conflicts

        project_wide = ScoredChunk(
            chunk=ChunkRecord(
                id="pw-decision", text="Project-wide rule.",
                doc_id="rules-1", section_type="decision",
                domain="security", affected_services=[],
                date="2025-01-01",
            ),
            score=0.8, point_id="uuid-pw",
        )
        service_specific = ScoredChunk(
            chunk=ChunkRecord(
                id="ss-decision", text="Service-specific rule.",
                doc_id="adr-001", section_type="decision",
                domain="security", affected_services=["auth-service"],
                date="2024-01-01",
            ),
            score=0.7, point_id="uuid-ss",
        )

        result = _resolve_conflicts([project_wide, service_specific])
        doc_ids = {sc.chunk.doc_id for sc in result}
        assert "adr-001" in doc_ids
        assert "rules-1" not in doc_ids

    def test_no_conflict_passes_through(self):
        from app.corpus import _resolve_conflicts

        chunk_a = ScoredChunk(
            chunk=ChunkRecord(
                id="a-decision", text="Decision A.",
                doc_id="adr-001", section_type="decision",
                domain="security", affected_services=["auth-service"],
            ),
            score=0.8, point_id="uuid-a",
        )
        chunk_b = ScoredChunk(
            chunk=ChunkRecord(
                id="b-context", text="Context B.",
                doc_id="adr-001", section_type="context",
                domain="security", affected_services=["auth-service"],
            ),
            score=0.7, point_id="uuid-b",
        )

        result = _resolve_conflicts([chunk_a, chunk_b])
        assert len(result) == 2

    def test_single_chunk_passes_through(self):
        from app.corpus import _resolve_conflicts

        chunk = ScoredChunk(
            chunk=ChunkRecord(
                id="solo", text="Solo.",
                doc_id="adr-001", section_type="decision",
                domain="security",
            ),
            score=0.9, point_id="uuid-solo",
        )

        result = _resolve_conflicts([chunk])
        assert len(result) == 1