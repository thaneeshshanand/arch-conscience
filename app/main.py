"""FastAPI webhook server for arch-conscience.

Receives GitHub pull_request webhooks, verifies HMAC signatures,
and runs the two-stage detection pipeline. Replaces both server.js
and webhook.js from the Node.js version.

Start with:
    uvicorn app.main:app --host 0.0.0.0 --port $WEBHOOK_PORT
"""

import hashlib
import hmac
import logging
import sys
from contextlib import asynccontextmanager

# ── Logging setup (stdout for Railway) ───────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    stream=sys.stdout,
)

from fastapi import FastAPI, Header, HTTPException, Request

from app.config import get_settings
from app.corpus import ensure_collection, query
from app.detect import run_detection
from app.gap_log import create_gap_entry, log_gap
from app.mcp_server import mcp as mcp_server
from app.notify import dispatch
from app.router import build_payload

logger = logging.getLogger(__name__)

_SUPPORTED_ACTIONS = {"opened", "synchronize", "closed"}


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup: validate config, ensure Qdrant collection, start MCP session manager."""
    settings = get_settings()
    settings.validate_required()
    await ensure_collection(settings)
    logger.info("arch-conscience ready on port %d", settings.WEBHOOK_PORT)

    # Start MCP session manager for streamable HTTP transport
    async with mcp_server.session_manager.run():
        yield


app = FastAPI(title="arch-conscience", lifespan=lifespan)

# Mount MCP server at /mcp — agents connect via streamable HTTP
mcp_app = mcp_server.streamable_http_app()
app.mount("/mcp", mcp_app)


@app.post("/")
async def webhook(
    request: Request,
    x_hub_signature_256: str | None = Header(default=None),
    x_github_event: str | None = Header(default=None),
):
    """GitHub webhook endpoint.

    Accepts pull_request events, verifies the HMAC signature,
    and runs the detection pipeline when appropriate.
    """
    settings = get_settings()
    raw_body = await request.body()

    logger.info("Event: %s", x_github_event)

    # ── Signature verification ───────────────────────────────────────
    if not _verify_signature(raw_body, x_hub_signature_256, settings.GITHUB_WEBHOOK_SECRET):
        raise HTTPException(status_code=403, detail="Invalid signature")

    gh: dict = await request.json()
    action = gh.get("action")

    # ── Event filtering ──────────────────────────────────────────────
    if action not in _SUPPORTED_ACTIONS:
        return {"status": "event ignored"}

    is_merge = action == "closed" and gh.get("pull_request", {}).get("merged") is True
    is_review = action in ("opened", "synchronize")

    if not is_merge and not is_review:
        return {"status": "event ignored"}

    # ── Build pipeline payload ───────────────────────────────────────
    payload = await build_payload(gh, settings)

    if not payload.affected_services:
        return {"status": "no tracked services touched"}

    # ── Retrieve corpus chunks ───────────────────────────────────────
    chunks = await query(
        text=payload.diff_summary,
        services=payload.affected_services,
        top_k=8,
        status_filter="active",
        settings=settings,
    )

    if not chunks:
        log_gap(create_gap_entry(
            gap_type="no_chunks_found",
            services=payload.affected_services,
            pr_url=payload.pr_url,
            diff_summary=payload.diff_summary,
        ))
        return {"status": "no relevant decisions found — corpus gap logged"}

    # ── Run detection pipeline ───────────────────────────────────────
    result = await run_detection(
        payload=payload,
        chunks=chunks,
        settings=settings,
    )

    if result.corpus_gap_signal:
        log_gap(create_gap_entry(
            gap_type="sparse_retrieval",
            services=payload.affected_services,
            pr_url=payload.pr_url,
        ))

    if not result.gap_detected or result.confidence < settings.CONFIDENCE_THRESHOLD:
        return {"status": "no gap detected"}

    # ── Dispatch alert ───────────────────────────────────────────────
    await dispatch(
        result=result,
        engineer=payload.author,
        pr_url=payload.pr_url,
        affected_services=payload.affected_services,
        settings=settings,
    )

    return {
        "status": "alert dispatched",
        "severity": result.severity,
        "violated_adr_id": result.violated_adr_id,
    }


@app.get("/health")
async def health():
    """Health check for Railway / load balancer probes."""
    return {"status": "ok"}


def _verify_signature(body: bytes, signature: str | None, secret: str) -> bool:
    """Verify GitHub's X-Hub-Signature-256 HMAC signature."""
    if not signature:
        return False

    expected = "sha256=" + hmac.new(
        secret.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(signature, expected)