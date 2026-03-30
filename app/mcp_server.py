"""arch-conscience MCP server.

Exposes architectural knowledge to AI coding agents (Cursor, Claude Code,
Copilot, etc.) via the Model Context Protocol. Agents call the
get_architectural_context tool before generating code to ensure
compliance with documented decisions, constraints, and principles.

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
from mcp.server.transport_security import TransportSecuritySettings

from app.config import get_settings
from app.corpus import ensure_collection, find_overlapping, query, stats, update_payload, upsert
from app.adr_drafter import draft_adr as _draft_adr
from app.format_detect import DocumentFormat, detect_format
from app.extract import extract_from_document
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
        "You have access to this project's architectural knowledge base — "
        "decisions, constraints, and principles. "
        "Before generating or modifying code, call get_architectural_context with "
        "the affected service name and/or your proposed approach. This ensures your "
        "code complies with documented decisions and avoids reintroducing patterns "
        "the team has explicitly rejected. "
        "When an engineer makes a significant architectural decision — choosing a "
        "database, defining an API pattern, selecting an auth mechanism — call "
        "draft_adr to generate a structured ADR for team review. "
        "To add knowledge from existing documents (Confluence pages, RFCs, design "
        "docs, rules files), call ingest_document with the document content. "
        "To resolve conflicts or update item status, call update_item_status."
    ),
)


def _format_chunk(chunk, score: float) -> dict[str, Any]:
    """Format a scored chunk for tool output."""
    result = {
        "adr_id": chunk.doc_id,
        "section": chunk.section_type,
        "knowledge_type": chunk.knowledge_type,
        "services": chunk.affected_services,
        "domain": chunk.domain,
        "status": chunk.status,
        "relevance_score": round(score, 3),
        "text": chunk.text,
    }
    if chunk.source_url:
        result["source_url"] = chunk.source_url
    if chunk.source_title:
        result["source_title"] = chunk.source_title
    return result


def _analyze_conflicts(chunks) -> dict[str, Any]:
    """Analyze retrieved chunks for potential conflicts with an approach."""
    rejected = [sc for sc in chunks if sc.chunk.section_type == "rejected_alternatives"]
    decisions = [sc for sc in chunks if sc.chunk.section_type == "decision"]

    if rejected:
        return {
            "verdict": "potential_conflict",
            "conflicts_found": len(rejected),
            "message": (
                "WARNING: Retrieved sections include rejected alternatives "
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

    - service only → returns all active context for that service
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
            "Review these architectural items before generating code. "
            "Behavior depends on knowledge_type: "
            "constraint → refuse to generate violating code (hard stop). "
            "decision with rejected_alternatives → refuse and explain why "
            "the alternative was rejected. "
            "principle → proceed but flag the deviation to the engineer. "
            "Do NOT generate code that reintroduces a rejected approach."
        )

    # ── Group by doc_id for summary ──────────────────────────────────
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
async def ingest_document(
    content: str,
    filename: str = "",
    source_url: str = "",
    source_type: str = "",
) -> str:
    """Ingest any document into the architectural knowledge corpus.

    Auto-detects the document format and routes to the appropriate
    handler:
    - ADR with YAML frontmatter → ADR parser (fast, deterministic)
    - Known rules file (CLAUDE.md, .cursorrules, etc.) → Rules bridge
    - Everything else → Two-pass LLM normalizer

    The response includes extracted items and any conflicts with
    existing corpus items.

    Args:
        content: The full text content of the document.
        filename: Name of the file for format detection and provenance
                  (e.g. "CLAUDE.md", "design-doc.md", "rfc-042.md").
        source_url: URL of the original document for provenance tracking
                    (e.g. "https://wiki.example.com/page/123").
        source_type: Provenance label (e.g. "confluence", "rfc",
                     "design_doc"). Auto-detected from content if empty.
                     Does NOT override format detection routing.
    """
    settings = get_settings()
    await ensure_collection(settings)

    # Detect format and route
    fmt = detect_format(content, filename)
    logger.info("Detected format for '%s': %s", filename, fmt.value)

    if fmt == DocumentFormat.ADR:
        chunks = _ingest_adr(content, filename)
    elif fmt == DocumentFormat.RULES_FILE:
        chunks = await _ingest_rules(content, filename, settings)
    else:
        chunks = await _ingest_generic(content, filename, source_url, source_type, settings)

    if not chunks:
        return json.dumps({
            "filename": filename,
            "format_detected": fmt.value,
            "items_extracted": 0,
            "chunks_indexed": 0,
            "conflicts": [],
            "message": (
                "No architectural knowledge found in this document. "
                "The document may contain only code style rules, commands, "
                "or non-architectural content."
            ),
        }, indent=2)

    # Check for conflicts with existing corpus items
    conflicts = await _detect_conflicts(chunks, settings)

    # Upsert to corpus
    await upsert(chunks, settings)

    # Summarize
    items: dict[str, dict] = {}
    for c in chunks:
        if c.doc_id not in items:
            items[c.doc_id] = {
                "title": c.source_title or c.text.split("\n")[0],
                "knowledge_type": c.knowledge_type,
                "sections": [],
                "services": c.affected_services,
                "domain": c.domain,
            }
        items[c.doc_id]["sections"].append(c.section_type)

    return json.dumps({
        "filename": filename,
        "format_detected": fmt.value,
        "items_extracted": len(items),
        "chunks_indexed": len(chunks),
        "items": list(items.values()),
        "conflicts": conflicts,
        "message": (
            f"Extracted {len(items)} items from {filename or 'document'} "
            f"and indexed {len(chunks)} chunks into the corpus. "
            + (
                f"Found {len(conflicts)} potential conflict(s) with existing items. "
                "Review conflicts and use update_item_status to resolve."
                if conflicts else
                "No conflicts with existing items detected."
            )
        ),
    }, indent=2)


@mcp.tool()
async def update_item_status(
    doc_id: str,
    new_status: str,
    reason: str = "",
) -> str:
    """Update the status of an architectural knowledge item.

    Use this for conflict resolution and lifecycle management.
    Updates the status field on ALL chunks matching the given doc_id.

    Valid statuses:
    - active: Currently enforced
    - superseded: Replaced by a newer item
    - deprecated: No longer relevant
    - proposed: Under review, not yet enforced

    Args:
        doc_id: Document ID of the item (e.g. "adr-001", "norm-design_md-1").
        new_status: New status value (active | superseded | deprecated | proposed).
        reason: Optional reason for the change (logged for audit trail).
    """
    valid_statuses = {"active", "superseded", "deprecated", "proposed"}
    if new_status not in valid_statuses:
        return json.dumps({
            "error": f"Invalid status '{new_status}'. Must be one of: {', '.join(sorted(valid_statuses))}",
        }, indent=2)

    settings = get_settings()
    await ensure_collection(settings)

    updated = await update_payload(doc_id, {"status": new_status}, settings)

    if updated == 0:
        return json.dumps({
            "doc_id": doc_id,
            "updated": 0,
            "message": f"No chunks found with doc_id '{doc_id}'.",
        }, indent=2)

    logger.info(
        "Status updated: %s → %s (%d chunks)%s",
        doc_id, new_status, updated,
        f" — reason: {reason}" if reason else "",
    )

    return json.dumps({
        "doc_id": doc_id,
        "new_status": new_status,
        "chunks_updated": updated,
        "reason": reason,
        "message": f"Updated {updated} chunks for '{doc_id}' to status '{new_status}'.",
    }, indent=2)


@mcp.resource("arch-conscience://status")
async def get_status() -> str:
    """Current corpus status — total chunks, collection info."""
    settings = get_settings()
    corpus_stats = await stats(settings)
    return json.dumps(corpus_stats, indent=2)


# ── Ingestion helpers (per format) ───────────────────────────────────


def _ingest_adr(content: str, filename: str) -> list:
    """Ingest an ADR file using the existing regex parser."""
    from app.ingest import _parse_adr
    from pathlib import Path

    stem = Path(filename).stem if filename else "adr"
    try:
        return _parse_adr(content, stem)
    except ValueError as exc:
        logger.warning("ADR parse failed for %s: %s", filename, exc)
        return []


async def _ingest_rules(content: str, filename: str, settings) -> list:
    """Ingest a rules file using the existing rules bridge."""
    return await extract_decisions_from_rules(
        content=content,
        source_file=filename,
        settings=settings,
    )


async def _ingest_generic(
    content: str, filename: str, source_url: str, source_type: str, settings,
) -> list:
    """Ingest a generic document using the two-pass normalizer."""
    result = await extract_from_document(
        content,
        filename=filename,
        source_url=source_url,
        source_type=source_type,
        settings=settings,
    )
    return result.chunks


async def _detect_conflicts(chunks, settings) -> list[dict]:
    """Check extracted chunks for conflicts with existing corpus items.

    Programmatic overlap check — no LLM call. Groups by domain +
    affected_services and queries for existing active items.
    """
    conflicts: list[dict] = []
    checked: set[tuple] = set()

    for chunk in chunks:
        if chunk.section_type != "decision":
            continue

        key = (chunk.domain, tuple(sorted(chunk.affected_services)))
        if key in checked:
            continue
        checked.add(key)

        overlapping = await find_overlapping(
            domain=chunk.domain,
            affected_services=chunk.affected_services,
            settings=settings,
        )

        for existing in overlapping:
            # Don't flag conflict with self (re-ingestion)
            if existing.chunk.doc_id == chunk.doc_id:
                continue

            new_date = chunk.date or ""
            existing_date = existing.chunk.date or ""

            if new_date > existing_date:
                suggestion = (
                    f"New item is newer. Consider superseding '{existing.chunk.doc_id}' "
                    f"with update_item_status(doc_id='{existing.chunk.doc_id}', "
                    f"new_status='superseded')."
                )
            elif existing_date > new_date:
                suggestion = (
                    f"Existing item '{existing.chunk.doc_id}' is newer and may already "
                    f"govern this area. Review whether the new item adds value."
                )
            else:
                suggestion = "Review both items — dates are equal or unknown."

            conflicts.append({
                "existing_doc_id": existing.chunk.doc_id,
                "existing_title": existing.chunk.source_title or existing.chunk.doc_id,
                "domain": chunk.domain,
                "overlapping_services": chunk.affected_services or ["project-wide"],
                "suggestion": suggestion,
            })

    return conflicts


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