"""arch-conscience MCP server.

Exposes architectural decisions to AI coding agents (Cursor, Claude Code,
Copilot, etc.) via the Model Context Protocol. Agents call the
get_architectural_context tool before generating code to ensure
compliance with documented ADRs.

Start with:
    python -m app.mcp_server                          # stdio (for IDE integration)
    python -m app.mcp_server --transport http          # streamable HTTP (for remote)

Configure in Claude Desktop / Cursor / Claude Code:
    {
      "mcpServers": {
        "arch-conscience": {
          "command": "python",
          "args": ["-m", "app.mcp_server"],
          "cwd": "/path/to/arch-conscience"
        }
      }
    }
"""

import json
import logging
import os
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from app.config import get_settings
from app.corpus import ensure_collection, query, stats
from app.adr_drafter import draft_adr as _draft_adr
from app.rules_bridge import extract_decisions_from_rules

logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="arch-conscience",
    stateless_http=True,
    json_response=True,
    streamable_http_path="/",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    ),
    instructions=(
        "You have access to this project's architectural decision records (ADRs). "
        "Before generating or modifying code, call get_architectural_context with "
        "the affected service name and/or your proposed approach. This ensures your "
        "code complies with documented decisions and avoids reintroducing patterns "
        "the team has explicitly rejected. "
        "When an engineer makes a significant architectural decision — choosing a "
        "database, defining an API pattern, selecting an auth mechanism — call "
        "draft_adr to generate a structured ADR for team review."
    ),
)


def _format_chunk(chunk, score: float) -> dict[str, Any]:
    """Format a scored chunk for tool output."""
    return {
        "adr_id": chunk.doc_id,
        "section": chunk.section_type,
        "knowledge_type": chunk.knowledge_type,
        "services": chunk.affected_services,
        "domain": chunk.domain,
        "status": chunk.status,
        "relevance_score": round(score, 3),
        "text": chunk.text,
    }


def _analyze_conflicts(chunks) -> dict[str, Any]:
    """Analyze retrieved chunks for potential conflicts with an approach."""
    rejected = [sc for sc in chunks if sc.chunk.section_type == "rejected_alternatives"]
    decisions = [sc for sc in chunks if sc.chunk.section_type == "decision"]

    if rejected:
        return {
            "verdict": "potential_conflict",
            "conflicts_found": len(rejected),
            "message": (
                "WARNING: Retrieved ADR sections include rejected alternatives "
                "that may match your proposed approach. Review carefully — if "
                "your approach reintroduces a pattern that was explicitly rejected, "
                "it will be flagged when you open a PR. Consider an alternative "
                "approach or discuss with the team if the original decision "
                "should be revisited."
            ),
        }

    if decisions:
        return {
            "verdict": "review_recommended",
            "conflicts_found": 0,
            "message": (
                "Active architectural decisions exist in this area. Review them "
                "to ensure your approach is additive (extending existing patterns) "
                "rather than contradicting them."
            ),
        }

    return {
        "verdict": "context_available",
        "conflicts_found": 0,
        "message": (
            "Background context found for this area. No obvious conflicts "
            "detected, but review the retrieved sections for relevant constraints."
        ),
    }


@mcp.tool()
async def get_architectural_context(
    service: str = "",
    approach: str = "",
) -> str:
    """Get architectural decisions and check for conflicts before writing code.

    Call this BEFORE generating or modifying code. The response adapts
    based on what you provide:

    - service only → returns all active ADR context for that service
    - service + approach → returns context + conflict analysis
    - approach only → searches broadly for conflicts across all services
    - neither → returns a summary of all active decisions

    Args:
        service: The service being modified (e.g. "auth-service",
                 "payments-service"). Must match names in SERVICE_MAP.
        approach: Optional description of your proposed approach.
                  Be specific — e.g. "Use session cookies with a Redis
                  session store for authentication" rather than just
                  "change auth". When provided, the response includes
                  conflict analysis against rejected alternatives.
    """
    settings = get_settings()
    await ensure_collection(settings)

    # Build query text from available inputs
    if service and approach:
        query_text = f"Service: {service}. Proposed approach: {approach}"
    elif service:
        query_text = f"Architectural decisions for {service}"
    elif approach:
        query_text = approach
    else:
        query_text = "all architectural decisions"

    services = [service] if service else None

    chunks = await query(
        text=query_text,
        services=services,
        top_k=8,
        status_filter="active",
        settings=settings,
    )

    # ── No results ───────────────────────────────────────────────────
    if not chunks:
        corpus_stats = await stats(settings)
        return json.dumps({
            "service": service or "all",
            "approach": approach or None,
            "decisions_found": 0,
            "corpus_stats": corpus_stats,
            "message": (
                f"No architectural decisions found"
                + (f" for '{service}'" if service else "")
                + ". This may mean decisions exist but aren't documented yet. "
                "Proceed with caution and consider documenting any significant "
                "architectural choices you make."
            ),
        }, indent=2)

    # ── Format results ───────────────────────────────────────────────
    results = [_format_chunk(sc.chunk, sc.score) for sc in chunks]

    response: dict[str, Any] = {
        "service": service or "all",
        "decisions_found": len(results),
        "decisions": results,
    }

    # ── Conflict analysis (when approach is provided) ────────────────
    if approach:
        response["approach"] = approach
        response.update(_analyze_conflicts(chunks))
    else:
        response["instructions"] = (
            "Review these architectural decisions before generating code. "
            "Pay special attention to 'rejected_alternatives' sections — "
            "these describe approaches that were explicitly ruled out. "
            "Do NOT generate code that reintroduces a rejected approach."
        )

    # ── Group by ADR for summary ─────────────────────────────────────
    adrs: dict[str, list[str]] = {}
    for sc in chunks:
        doc_id = sc.chunk.doc_id
        if doc_id not in adrs:
            adrs[doc_id] = []
        if sc.chunk.section_type not in adrs[doc_id]:
            adrs[doc_id].append(sc.chunk.section_type)

    response["adr_summary"] = {
        adr_id: sorted(sections) for adr_id, sections in adrs.items()
    }

    return json.dumps(response, indent=2)


@mcp.tool()
async def draft_adr(
    title: str,
    services: str = "",
    service: str = "",
    context: str = "",
    approach: str = "",
    alternatives_considered: str = "",
    alternatives: str = "",
    decision: str = "",
    consequences: str = "",
    constraint_type: str = "operational",
    author: str = "unknown",
    adr_id: str = "",
) -> str:
    """Draft a structured Architecture Decision Record for team review.

    Call this when a significant architectural decision is being made —
    choosing a database, defining an API pattern, selecting an auth
    mechanism, establishing a service boundary, etc. The generated ADR
    follows the project's standard format with Context, Decision,
    Consequences, and Rejected Alternatives sections.

    The more context you provide, the better the draft. Include the
    problem being solved, constraints (scale, compliance, team expertise,
    cost), and any alternatives that were discussed.

    Args:
        title: Short decision title (e.g. "Use PostgreSQL for payment ledger").
        services: Comma-separated service names affected by this decision
                  (e.g. "payments-service, billing-service").
        service: Alternative to services — single service name.
        context: Why this decision is needed — the problem, requirements,
                 and constraints driving it. Be detailed.
        approach: The decided approach, if known.
        alternatives_considered: What other options were evaluated and why
                                 they were rejected. Free-form text.
        alternatives: Alternative to alternatives_considered.
        decision: The decided approach (alternative to approach).
        consequences: Known consequences and tradeoffs of the decision.
        constraint_type: One of: security, compliance, performance,
                         scalability, data_model, operational.
        author: Who is authoring this decision.
        adr_id: ADR identifier (e.g. "adr-005"). Auto-generated if empty.
    """
    settings = get_settings()

    # Accept both singular and plural, merge
    services_str = services or service
    service_list = [s.strip() for s in services_str.split(",") if s.strip()]

    # Accept both approach and decision
    decided_approach = approach or decision

    # Accept both alternatives_considered and alternatives
    alts = alternatives_considered or alternatives

    # Merge consequences into context if provided
    full_context = context
    if consequences:
        full_context += f"\n\nKnown consequences and tradeoffs: {consequences}"

    # Gather related existing decisions using the same corpus query
    # that get_architectural_context uses
    await ensure_collection(settings)
    related_decisions = ""
    for svc in service_list:
        chunks = await query(
            text=f"Architectural decisions for {svc}",
            services=[svc],
            top_k=4,
            status_filter="active",
            settings=settings,
        )
        if chunks:
            related_decisions += f"\nExisting decisions for {svc}:\n"
            for sc in chunks:
                related_decisions += (
                    f"- [{sc.chunk.doc_id}] {sc.chunk.section_type}: "
                    f"{sc.chunk.text[:200]}\n"
                )

    adr_markdown = await _draft_adr(
        title=title,
        services=service_list,
        context=full_context,
        approach=decided_approach,
        alternatives_considered=alts,
        related_decisions=related_decisions,
        constraint_type=constraint_type,
        author=author,
        adr_id=adr_id,
        settings=settings,
    )

    return json.dumps({
        "adr_id": adr_id or "auto-generated",
        "title": title,
        "services": service_list,
        "status": "proposed",
        "draft": adr_markdown,
        "next_steps": (
            "Review this draft with your team. Edit as needed, then save it "
            "to the /adrs directory and run 'python -m scripts.run_ingest' "
            "to add it to the corpus. Once ingested, arch-conscience will "
            "enforce this decision on future PRs and code generation."
        ),
    }, indent=2)


@mcp.tool()
async def ingest_rules_file(
    content: str,
    filename: str = "rules.md",
) -> str:
    """Extract architectural decisions from a rules file and add them to the corpus.

    Call this when a team has an existing CLAUDE.md, .cursorrules, AGENTS.md,
    rules.md, or similar file containing architectural rules mixed with
    code style and setup instructions. This tool extracts only the
    architectural decisions and indexes them for enforcement.

    The tool uses an LLM to distinguish architectural decisions
    (database choices, auth patterns, service communication rules) from
    code style rules (indentation, naming conventions) and setup
    commands. Only architectural decisions are added to the corpus.

    Args:
        content: The full text content of the rules file.
        filename: Name of the file for provenance tracking
                  (e.g. "CLAUDE.md", ".cursorrules").
    """
    settings = get_settings()
    await ensure_collection(settings)

    from app.corpus import upsert

    chunks = await extract_decisions_from_rules(
        content=content,
        source_file=filename,
        settings=settings,
    )

    if not chunks:
        return json.dumps({
            "filename": filename,
            "decisions_extracted": 0,
            "chunks_indexed": 0,
            "message": (
                "No architectural decisions found in this file. "
                "The file may contain only code style rules, commands, "
                "or setup instructions."
            ),
        }, indent=2)

    await upsert(chunks, settings)

    # Summarize what was extracted
    decisions = {}
    for c in chunks:
        if c.doc_id not in decisions:
            decisions[c.doc_id] = {
                "title": c.text.split("\n")[0].replace("Rule: ", ""),
                "sections": [],
                "services": c.affected_services,
                "domain": c.domain,
            }
        decisions[c.doc_id]["sections"].append(c.section_type)

    return json.dumps({
        "filename": filename,
        "decisions_extracted": len(decisions),
        "chunks_indexed": len(chunks),
        "decisions": list(decisions.values()),
        "message": (
            f"Extracted {len(decisions)} architectural decisions from {filename} "
            f"and indexed {len(chunks)} chunks into the corpus. These decisions "
            "are now active and will be enforced on future code generation "
            "and PR reviews."
        ),
    }, indent=2)


@mcp.resource("arch-conscience://status")
async def get_status() -> str:
    """Current corpus status — total chunks, collection info."""
    settings = get_settings()
    corpus_stats = await stats(settings)
    return json.dumps(corpus_stats, indent=2)


if __name__ == "__main__":
    transport = "stdio"
    if "--transport" in sys.argv:
        idx = sys.argv.index("--transport")
        if idx + 1 < len(sys.argv):
            transport = sys.argv[idx + 1]

    if transport == "http":
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")