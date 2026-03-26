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
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from app.config import get_settings
from app.corpus import ensure_collection, query, stats

logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="arch-conscience",
    instructions=(
        "You have access to this project's architectural decision records (ADRs). "
        "Before generating or modifying code, call get_architectural_context with "
        "the affected service name and/or your proposed approach. This ensures your "
        "code complies with documented decisions and avoids reintroducing patterns "
        "the team has explicitly rejected."
    ),
)


def _format_chunk(chunk, score: float) -> dict[str, Any]:
    """Format a scored chunk for tool output."""
    return {
        "adr_id": chunk.doc_id,
        "section": chunk.section_type,
        "services": chunk.affected_services,
        "constraint_type": chunk.constraint_type,
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