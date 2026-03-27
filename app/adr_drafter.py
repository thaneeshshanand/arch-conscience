"""ADR draft generator — first AaaS feature.

Given context about a decision (what, why, what was considered),
generates a structured ADR draft in the project's standard format.
Related existing ADRs are passed in by the caller — this module
has no dependency on the corpus layer.

This is the smallest useful version. Future iterations will
automatically gather context from Jira, Slack, PR discussions,
and codebase analysis.
"""

import logging
from datetime import date

from app.config import Settings, get_settings
from app.llm import Message, complete

logger = logging.getLogger(__name__)

_DRAFTER_SYSTEM = """\
You are an architectural decision record (ADR) author for a software engineering team. \
Your job is to produce a well-structured ADR in Markdown format based on the context \
the engineer provides.

## Output format

Produce ONLY the ADR markdown. No preamble, no commentary, no code fences. The output \
must follow this exact structure:

---
id: {adr_id}
title: {title}
status: proposed
date: {date}
services: [{services}]
constraint_type: {constraint_type}
author: {author}
---

## Context

Why this decision is needed. What problem or requirement drives it. Include technical \
and business context. Reference specific constraints (scale, compliance, team expertise, \
cost) that influenced the decision.

## Decision

What was decided and how it will be implemented. Be specific about technologies, \
patterns, and interfaces. This section should be concrete enough that an engineer \
can act on it.

## Consequences

What this means going forward. Include both positive outcomes and tradeoffs accepted. \
Mention operational implications, migration needs, and what other teams or services \
are affected.

## Rejected Alternatives

What was explicitly considered and ruled out, and WHY. This is the most important \
section — it prevents future engineers from reintroducing approaches the team has \
already evaluated. For each alternative, state:
1. What the alternative was
2. Why it was considered (its appeal)
3. Why it was rejected (the specific reasons)

## Rules

- Write in third person ("The team decided..." not "We decided...")
- Be specific — name technologies, protocols, and patterns
- Every rejected alternative must have a concrete rejection reason
- Reference related ADRs when they exist (use doc_id)
- The Context section must include enough detail that someone unfamiliar \
  with the project understands why this decision matters
- constraint_type must be one of: security, compliance, performance, \
  scalability, data_model, operational
- If the engineer hasn't provided enough information for a section, \
  write a placeholder with [TODO: ...] indicating what's needed"""


async def draft_adr(
    *,
    title: str,
    services: list[str],
    context: str,
    approach: str = "",
    alternatives_considered: str = "",
    related_decisions: str = "",
    constraint_type: str = "operational",
    author: str = "unknown",
    adr_id: str = "",
    settings: Settings | None = None,
) -> str:
    """Generate a structured ADR draft.

    Args:
        title: Short decision title.
        services: Services affected by this decision.
        context: Why this decision is needed — problem, requirements, constraints.
        approach: The decided approach. If empty, the LLM will infer or mark TODO.
        alternatives_considered: What other options were evaluated and why rejected.
        related_decisions: Pre-formatted string of related existing ADRs from
                           the corpus. Passed in by the caller (e.g. MCP tool).
        constraint_type: One of: security, compliance, performance,
                         scalability, data_model, operational.
        author: Who is authoring this decision.
        adr_id: ADR identifier. Auto-generated if empty.
        settings: Optional settings override.

    Returns:
        The complete ADR markdown as a string.
    """
    s = settings or get_settings()

    today = date.today().isoformat()
    generated_id = adr_id or f"adr-{today.replace('-', '')}"

    prompt_parts = [
        "Generate an ADR with the following details:",
        "",
        f"ADR ID: {generated_id}",
        f"Title: {title}",
        f"Date: {today}",
        f"Services: {', '.join(services)}",
        f"Constraint type: {constraint_type}",
        f"Author: {author}",
        "",
        "CONTEXT PROVIDED BY THE ENGINEER:",
        context,
    ]

    if approach:
        prompt_parts.extend(["", "DECIDED APPROACH:", approach])

    if alternatives_considered:
        prompt_parts.extend(["", "ALTERNATIVES CONSIDERED:", alternatives_considered])

    if related_decisions:
        prompt_parts.extend([
            "",
            "EXISTING RELATED DECISIONS (reference these where relevant):",
            related_decisions,
        ])

    result = await complete(
        [
            Message("system", _DRAFTER_SYSTEM),
            Message("user", "\n".join(prompt_parts)),
        ],
        model=s.STAGE2_MODEL,
        temperature=0.2,
    )

    return result.content