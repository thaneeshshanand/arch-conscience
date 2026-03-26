"""Two-stage architectural gap detection pipeline.

Stage 1: Relevance filter (cheap model) — scores each retrieved chunk
         against the diff payload, drops chunks below threshold.
Gate:    If no chunks survive Stage 1, return early with
         corpus_gap_signal=true.
Stage 2: Gap detection (capable model) — reasons over surviving chunks
         using a structured chain-of-thought prompt. Returns JSON.
"""

import json
import logging
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.corpus import ScoredChunk
from app.llm import Message, complete
from app.router import PipelinePayload

logger = logging.getLogger(__name__)


# ── Result type ──────────────────────────────────────────────────────


@dataclass(slots=True)
class DetectionResult:
    """Structured output from the detection pipeline."""

    gap_detected: bool = False
    confidence: float = 0.0
    severity: str | None = None
    violated_adr_id: str | None = None
    constraint_type: str | None = None
    rejected_alt_reintroduced: bool = False
    change_summary: str = ""
    reasoning: str = ""
    alert_headline: str | None = None
    alert_body: str | None = None
    corpus_gap_signal: bool = False


# ── Public API ───────────────────────────────────────────────────────


async def run_detection(
    *,
    payload: PipelinePayload,
    chunks: list[ScoredChunk],
    settings: Settings | None = None,
) -> DetectionResult:
    """Run the full two-stage pipeline.

    Args:
        payload: Structured PR payload from router.
        chunks: Retrieved corpus chunks from corpus.query().
        settings: Optional settings override (for tests).

    Returns:
        DetectionResult with gap analysis.
    """
    s = settings or get_settings()

    # ── Stage 1: relevance filter ────────────────────────────────────
    relevant = await _filter_relevant(payload=payload, chunks=chunks, settings=s)

    if not relevant:
        logger.info("Stage 1: no relevant chunks survived — corpus gap")
        return DetectionResult(
            corpus_gap_signal=True,
            change_summary=payload.diff_summary,
        )

    logger.info("Stage 1: %d/%d chunks passed", len(relevant), len(chunks))

    # ── Stage 2: gap detection ───────────────────────────────────────
    result = await _detect_gap(payload=payload, chunks=relevant, settings=s)

    logger.info(
        "Stage 2: gap=%s confidence=%s severity=%s",
        result.gap_detected,
        result.confidence,
        result.severity,
    )

    return result


# ── Stage 1 — Relevance filter ──────────────────────────────────────

_STAGE1_SYSTEM = """You are a relevance classifier for architectural decision records (ADRs). \
Given a code change summary and a list of architectural decision chunks, score each chunk \
0.0–1.0 for how directly relevant it is to the change.

Scoring guide:
- 0.8–1.0: The chunk directly governs the same service AND the same architectural concern \
(e.g. auth mechanism, data storage pattern) as the change.
- 0.5–0.7: The chunk governs the same service but a related (not identical) concern, \
or the same concern in a closely related service.
- 0.2–0.4: Tangential — same domain but different concern.
- 0.0–0.1: Unrelated.

A "Rejected Alternatives" section that describes the exact approach being introduced in \
the code change is maximally relevant (0.9–1.0).

Return only JSON."""


async def _filter_relevant(
    *,
    payload: PipelinePayload,
    chunks: list[ScoredChunk],
    settings: Settings,
) -> list[ScoredChunk]:
    """Score each chunk for relevance and drop those below threshold."""
    user_prompt = _build_stage1_prompt(payload, chunks)

    result = await complete(
        [
            Message("system", _STAGE1_SYSTEM),
            Message("user", user_prompt),
        ],
        model=settings.STAGE1_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
    )

    try:
        scores = json.loads(result.content).get("scores", [])
        logger.info("Stage 1 scores: %s", json.dumps(scores))
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Stage 1: failed to parse scores — passing all chunks")
        return chunks

    threshold = settings.STAGE1_THRESHOLD

    return [
        chunk
        for i, chunk in enumerate(chunks)
        if (scores[i].get("score", 0) if i < len(scores) else 0) >= threshold
    ]


def _build_stage1_prompt(
    payload: PipelinePayload,
    chunks: list[ScoredChunk],
) -> str:
    chunk_list = "\n\n".join(
        f"[{i}]\n"
        f"source_type: {c.chunk.source_type}\n"
        f"doc_id: {c.chunk.doc_id}\n"
        f"section_type: {c.chunk.section_type}\n"
        f"affected_services: {', '.join(c.chunk.affected_services)}\n"
        f"constraint_type: {c.chunk.constraint_type}\n"
        f"text: {c.chunk.text[:300]}"
        for i, c in enumerate(chunks)
    )

    return (
        f"CODE CHANGE SUMMARY:\n{payload.diff_summary}\n\n"
        f"RETRIEVED CHUNKS:\n{chunk_list}\n\n"
        "Return JSON in this exact shape:\n"
        "{\n"
        '  "scores": [\n'
        '    { "index": 0, "score": 0.0 },\n'
        "    ...one entry per chunk, in order...\n"
        "  ]\n"
        "}"
    )


# ── Stage 2 — Gap detection ─────────────────────────────────────────

# Inline so detect.py is self-contained and the prompt stays
# co-located with the parsing logic that depends on its output schema.

_STAGE2_SYSTEM = """\
You are an architectural conscience for a software engineering team. Your sole job is to determine whether an incoming code change directly contradicts or silently invalidates a documented architectural decision.

You are not a code reviewer. You are not a style enforcer. You do not comment on code quality, test coverage, naming conventions, or implementation details. You fire only when a change breaks a promise the team made to itself.

## Reasoning protocol — follow this order exactly

### Step 1 — Summarise the change intent
In 2–3 sentences, state in plain English what the code change does and what architectural concern it touches (auth, data storage, service communication, etc.). Do not quote code. Do not evaluate quality.

### Step 2 — Filter active decisions
Discard any chunk where status is not active. Discard any chunk where none of the affected_services overlap with the services touched by the change. Note what you kept and why.

### Step 3 — Match decisions to change
For each remaining chunk, state whether the change is:
- Unrelated — touches the same domain but does not interact with this decision
- Additive — extends or builds on this decision without overriding it
- Contradicting — removes, replaces, or bypasses a constraint this decision establishes
- Reintroducing — reimplements an approach explicitly listed in a rejected_alternatives section

### Step 4 — Apply the false positive guard
Before concluding a gap exists, ask: does the change remove or override an existing constraint, or does it merely add something new alongside existing patterns? If the change is purely additive, there is no gap. When in doubt, default to no gap.

### Step 5 — Determine severity
Apply these rules in order (first match wins):
- constraint_type is security or compliance → high
- rejected_alt_reintroduced is true → high
- constraint_type is data_model or scalability → medium
- All other contradictions → low

### Step 6 — Produce output
Output a single JSON object. No prose before or after it. No markdown fences.

## Output schema
{
  "gap_detected": boolean,
  "confidence": float (0.0–1.0),
  "severity": "low" | "medium" | "high" | null,
  "violated_adr_id": string | null,
  "constraint_type": string | null,
  "rejected_alt_reintroduced": boolean,
  "change_summary": string,
  "reasoning": string,
  "alert_headline": string | null,
  "alert_body": string | null,
  "corpus_gap_signal": boolean
}

Field notes:
- confidence reflects certainty that a genuine contradiction exists. If uncertain, set below 0.7 and set gap_detected to false.
- reasoning is 3–5 sentences max. Name the specific constraint violated and explain precisely how the change breaks it.
- alert_headline is ≤120 characters, plain English. Example: "PR #447 may reintroduce session cookies — ADR-12 requires stateless JWT auth for GDPR compliance." Null if no gap.
- alert_body is 2–3 sentences: what the ADR decided, why, and what in this PR contradicts it. Include the doc_id. Null if no gap.
- corpus_gap_signal is true when the change touches a concern for which no relevant active decisions were found.

## Hard rules
1. Never flag additive changes.
2. Never flag superseded decisions.
3. Never reason from code quality.
4. Never produce partial JSON. If confidence < 0.7, set gap_detected to false.
5. One violation per output — report highest severity, mention others in reasoning."""


async def _detect_gap(
    *,
    payload: PipelinePayload,
    chunks: list[ScoredChunk],
    settings: Settings,
) -> DetectionResult:
    """Run the Stage 2 chain-of-thought gap detection."""
    user_prompt = _build_stage2_prompt(payload, chunks)

    result = await complete(
        [
            Message("system", _STAGE2_SYSTEM),
            Message("user", user_prompt),
        ],
        model=settings.STAGE2_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
    )

    try:
        parsed = json.loads(result.content)
        return _normalise(parsed, fallback_summary=payload.diff_summary)
    except (json.JSONDecodeError, AttributeError):
        logger.error("Stage 2: failed to parse JSON output: %s", result.content[:200])
        return DetectionResult(change_summary=payload.diff_summary)


def _build_stage2_prompt(
    payload: PipelinePayload,
    chunks: list[ScoredChunk],
) -> str:
    chunk_list = "\n\n".join(
        "---\n"
        f"source_type: {c.chunk.source_type}\n"
        f"doc_id: {c.chunk.doc_id}\n"
        f"section_type: {c.chunk.section_type}\n"
        f"affected_services: {', '.join(c.chunk.affected_services)}\n"
        f"constraint_type: {c.chunk.constraint_type}\n"
        f"status: {c.chunk.status}\n"
        f"decision_date: {c.chunk.decision_date}\n"
        f"author: {c.chunk.author}\n"
        f"text: {c.chunk.text}"
        for c in chunks
    )

    return (
        f"CODE CHANGE SUMMARY:\n{payload.diff_summary}\n\n"
        f"PR URL: {payload.pr_url}\n"
        f"Author: {payload.author}\n"
        f"Base branch: {payload.base_branch}\n\n"
        f"RETRIEVED DECISION CHUNKS:\n{chunk_list}"
    )


# ── Helpers ──────────────────────────────────────────────────────────


def _normalise(raw: dict, fallback_summary: str = "") -> DetectionResult:
    """Ensure every expected field is present, even if the model omitted some."""
    return DetectionResult(
        gap_detected=raw.get("gap_detected", False),
        confidence=raw.get("confidence", 0.0),
        severity=raw.get("severity"),
        violated_adr_id=raw.get("violated_adr_id"),
        constraint_type=raw.get("constraint_type"),
        rejected_alt_reintroduced=raw.get("rejected_alt_reintroduced", False),
        change_summary=raw.get("change_summary", fallback_summary),
        reasoning=raw.get("reasoning", ""),
        alert_headline=raw.get("alert_headline"),
        alert_body=raw.get("alert_body"),
        corpus_gap_signal=raw.get("corpus_gap_signal", False),
    )