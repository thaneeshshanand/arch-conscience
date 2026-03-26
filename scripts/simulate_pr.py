"""Simulate a PR that reintroduces session cookies against ADR-001.

Skips GitHub API calls and webhook signature verification.
Runs the real pipeline: corpus retrieval → Stage 1 → Stage 2 → Telegram alert.

Usage:
    python -m scripts.simulate_pr
"""

import asyncio
import logging

from app.config import get_settings
from app.corpus import ensure_collection, query
from app.detect import run_detection
from app.gap_log import create_gap_entry, log_gap
from app.notify import dispatch
from app.router import PipelinePayload

logging.basicConfig(level=logging.INFO, format="%(message)s")


async def main():
    settings = get_settings()
    settings.validate_required()
    await ensure_collection(settings)

    # ── Simulate a PR payload (no GitHub API call) ───────────────────
    payload = PipelinePayload(
        pr_url="https://github.com/acme/backend/pull/42",
        pr_number="42",
        pr_title="Add session cookie support for auth",
        author="newdev",
        base_branch="main",
        changed_files=[
            "services/auth/session.py",
            "services/auth/middleware.py",
        ],
        affected_services=["auth-service"],
        diff_summary=(
            "PR title: Add session cookie support for auth. "
            "Services affected: auth-service. "
            "Changed files (2): services/auth/session.py, services/auth/middleware.py. "
            "Description: Implements session-based authentication using Set-Cookie "
            "headers for the login flow. Adds a Redis-backed session store for "
            "persisting user sessions across requests."
        ),
    )

    print(f"Simulating PR: {payload.pr_title}")
    print(f"Services: {payload.affected_services}")
    print()

    # ── Retrieve corpus chunks ───────────────────────────────────────
    print("Querying corpus...")
    chunks = await query(
        text=payload.diff_summary,
        services=payload.affected_services,
        top_k=8,
        status_filter="active",
        settings=settings,
    )

    if not chunks:
        print("No chunks retrieved — corpus may be empty. Run ingest first:")
        print("  python -m scripts.run_ingest")
        log_gap(create_gap_entry(
            gap_type="no_chunks_found",
            services=payload.affected_services,
            pr_url=payload.pr_url,
            diff_summary=payload.diff_summary,
        ))
        return

    print(f"Retrieved {len(chunks)} chunks:")
    for sc in chunks:
        print(f"  [{sc.score:.3f}] {sc.chunk.doc_id} — {sc.chunk.section_type}")
    print()

    # ── Run detection pipeline ───────────────────────────────────────
    print("Running detection pipeline...")
    result = await run_detection(
        payload=payload,
        chunks=chunks,
        settings=settings,
    )

    print()
    print(f"gap_detected:              {result.gap_detected}")
    print(f"confidence:                {result.confidence}")
    print(f"severity:                  {result.severity}")
    print(f"violated_adr_id:           {result.violated_adr_id}")
    print(f"rejected_alt_reintroduced: {result.rejected_alt_reintroduced}")
    print(f"corpus_gap_signal:         {result.corpus_gap_signal}")
    print()

    if result.corpus_gap_signal:
        log_gap(create_gap_entry(
            gap_type="sparse_retrieval",
            services=payload.affected_services,
            pr_url=payload.pr_url,
        ))

    if not result.gap_detected or result.confidence < settings.CONFIDENCE_THRESHOLD:
        print("No gap detected — no alert sent.")
        return

    # ── Dispatch alert ───────────────────────────────────────────────
    print("Dispatching alert...")
    await dispatch(
        result=result,
        engineer=payload.author,
        pr_url=payload.pr_url,
        settings=settings,
    )

    print("✅ Alert dispatched — check your Telegram bot!")


if __name__ == "__main__":
    asyncio.run(main())