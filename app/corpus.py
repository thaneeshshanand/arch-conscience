"""Qdrant corpus wrapper — embed, upsert, hybrid query.

Mirrors the Node.js src/corpus.js. Dense vectors (Cosine) with BM25
sparse vector infrastructure configured for future hybrid retrieval.
Payload fields indexed for service and status filtering at query time.
"""

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    IsEmptyCondition,
    MatchAny,
    MatchValue,
    PayloadField,
    PayloadSchemaType,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)

from app.config import Settings, get_settings
from app.llm import embed as llm_embed

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

_SCORE_THRESHOLD = 0.35  # drop low-quality matches before Stage 1
_TEXT_TRUNCATE = 8000  # stay within embedding token budget


# ── Data types ───────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """A single corpus chunk, ready for upsert or returned from query.

    16 fields total: 13 auto-populated, 3 enrichment (empty defaults).
    See DESIGN-format-agnostic-ingestion.md for full schema rationale.
    """

    id: str
    text: str

    # ── Auto-populated: core (extracted by LLM or source system) ─────

    knowledge_type: str = "decision"  # decision | constraint | principle
    section_type: str = ""  # context | decision | consequences | rejected_alternatives
    source_type: str = ""  # adr | design_doc | rfc | rules_file | confluence | jira | notion | runbook | informal
    doc_id: str = ""
    author: str = "unknown"
    affected_services: list[str] = field(default_factory=list)
    domain: str = "operational"  # security | compliance | performance | scalability | data_model | operational
    status: str = "active"
    date: str = ""  # when the knowledge was established

    # ── Auto-populated: provenance (from source system, no human input)

    source_url: str = ""
    source_title: str = ""
    ingested_at: str = ""

    # ── Enrichment: optional (empty defaults, teams add over time) ───

    owners: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    source_last_modified: str = ""


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A ChunkRecord returned from query, with retrieval score."""

    chunk: ChunkRecord
    score: float
    point_id: str


# ── Client singleton ─────────────────────────────────────────────────

_client: AsyncQdrantClient | None = None


def _get_client(settings: Settings | None = None) -> AsyncQdrantClient:
    """Return a cached async Qdrant client."""
    global _client
    if _client is None:
        s = settings or get_settings()
        _client = AsyncQdrantClient(
            url=s.QDRANT_URL,
            api_key=s.QDRANT_API_KEY or None,
        )
    return _client


# ── Embedding helper ─────────────────────────────────────────────────


async def _embed(text: str, settings: Settings | None = None) -> list[float]:
    """Embed a single text string via LiteLLM."""
    s = settings or get_settings()
    truncated = text[:_TEXT_TRUNCATE]
    vectors = await llm_embed(
        [truncated],
        model=s.EMBEDDING_MODEL,
        dimensions=s.EMBEDDING_DIM,
    )
    return vectors[0]


# ── Collection management ────────────────────────────────────────────


async def ensure_collection(settings: Settings | None = None) -> None:
    """Create the Qdrant collection if it doesn't exist.

    Sets up dense vectors (Cosine), BM25 sparse vector config,
    and payload indexes for filtering. Safe to call on every startup.
    """
    s = settings or get_settings()
    client = _get_client(s)
    collection = s.QDRANT_COLLECTION

    collections = await client.get_collections()
    if any(c.name == collection for c in collections.collections):
        return

    await client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(
            size=s.EMBEDDING_DIM,
            distance=Distance.COSINE,
        ),
        sparse_vectors_config={
            "bm25": SparseVectorParams(
                modifier="idf",
            ),
        },
    )

    # Index payload fields used as filters at query time
    for field_name, schema in [
        ("knowledge_type", PayloadSchemaType.KEYWORD),
        ("affected_services", PayloadSchemaType.KEYWORD),
        ("status", PayloadSchemaType.KEYWORD),
        ("domain", PayloadSchemaType.KEYWORD),
        ("section_type", PayloadSchemaType.KEYWORD),
        ("doc_id", PayloadSchemaType.KEYWORD),
    ]:
        await client.create_payload_index(
            collection_name=collection,
            field_name=field_name,
            field_schema=schema,
        )

    logger.info('Collection "%s" created', collection)


# ── Upsert ───────────────────────────────────────────────────────────


def _stable_id(chunk_id: str) -> str:
    """Convert a string chunk ID to a deterministic UUID string.

    Qdrant accepts UUIDs or unsigned ints. UUID5 gives us a
    deterministic, collision-resistant ID from any string —
    cleaner than the 32-bit hash in the Node.js version.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def _now_iso() -> str:
    """Return current UTC time as ISO string."""
    return datetime.utcnow().isoformat() + "Z"


async def upsert(chunks: list[ChunkRecord], settings: Settings | None = None) -> int:
    """Embed and upsert a batch of chunks into the corpus.

    Returns the number of points upserted.
    """
    if not chunks:
        return 0

    s = settings or get_settings()
    client = _get_client(s)

    points: list[PointStruct] = []

    for chunk in chunks:
        vector = await _embed(chunk.text, s)

        points.append(
            PointStruct(
                id=_stable_id(chunk.id),
                vector=vector,
                payload={
                    "text": chunk.text,
                    "knowledge_type": chunk.knowledge_type,
                    "section_type": chunk.section_type,
                    "source_type": chunk.source_type,
                    "doc_id": chunk.doc_id,
                    "author": chunk.author,
                    "affected_services": chunk.affected_services,
                    "domain": chunk.domain,
                    "status": chunk.status,
                    "date": chunk.date,
                    "source_url": chunk.source_url,
                    "source_title": chunk.source_title,
                    "ingested_at": chunk.ingested_at or _now_iso(),
                    "owners": chunk.owners,
                    "tags": chunk.tags,
                    "source_last_modified": chunk.source_last_modified,
                },
            )
        )

    await client.upsert(
        collection_name=s.QDRANT_COLLECTION,
        wait=True,
        points=points,
    )

    logger.info("Upserted %d chunks", len(points))
    return len(points)


# ── Query ────────────────────────────────────────────────────────────


async def query(
    *,
    text: str,
    services: list[str] | None = None,
    top_k: int = 8,
    status_filter: str = "active",
    settings: Settings | None = None,
) -> list[ScoredChunk]:
    """Semantic search over the corpus, filtered by service and status.

    Currently dense-only — BM25 sparse reranking is configured in the
    collection and can be wired in here when ready.

    Args:
        text: Query string (typically diffSummary from router).
        services: Filter to chunks affecting these services.
        top_k: Max chunks to return.
        status_filter: Only return chunks with this status.

    Returns:
        Scored chunks ordered by descending similarity.
    """
    s = settings or get_settings()
    client = _get_client(s)

    vector = await _embed(text, s)

    # Build filter — always require active status
    must_clauses: list[FieldCondition] = [
        FieldCondition(key="status", match=MatchValue(value=status_filter)),
    ]

    if services:
        # Return chunks that match the specific services OR are
        # project-wide (empty affected_services). This ensures that
        # decisions from rules files, Confluence, and Jira that apply
        # to all services are always included in service-specific queries.
        query_filter = Filter(
            must=must_clauses,
            should=[
                FieldCondition(key="affected_services", match=MatchAny(any=services)),
                IsEmptyCondition(is_empty=PayloadField(key="affected_services")),
            ],
        )
    else:
        query_filter = Filter(must=must_clauses)

    response = await client.query_points(
        collection_name=s.QDRANT_COLLECTION,
        query=vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
        score_threshold=_SCORE_THRESHOLD,
    )

    scored: list[ScoredChunk] = []
    for r in response.points:
        p = r.payload or {}
        chunk = ChunkRecord(
            id=p.get("doc_id", "") + "-" + p.get("section_type", ""),
            text=p.get("text", ""),
            knowledge_type=p.get("knowledge_type", "decision"),
            section_type=p.get("section_type", ""),
            source_type=p.get("source_type", ""),
            doc_id=p.get("doc_id", ""),
            author=p.get("author", "unknown"),
            affected_services=p.get("affected_services", []),
            domain=p.get("domain", "operational"),
            status=p.get("status", "active"),
            date=p.get("date", ""),
            source_url=p.get("source_url", ""),
            source_title=p.get("source_title", ""),
            ingested_at=p.get("ingested_at", ""),
            owners=p.get("owners", []),
            tags=p.get("tags", []),
            source_last_modified=p.get("source_last_modified", ""),
        )
        scored.append(ScoredChunk(chunk=chunk, score=r.score, point_id=str(r.id)))

    return _resolve_conflicts(scored)


# ── Query-time conflict resolution ───────────────────────────────


def _resolve_conflicts(chunks: list[ScoredChunk]) -> list[ScoredChunk]:
    """Resolve conflicts in retrieved chunks by preferring newer items.

    When the same domain + affected_services area has chunks from
    multiple doc_ids, prefer the newer one (by date field).
    Service-specific items take precedence over project-wide items
    for that service.

    Project-wide items still appear in results — they just don't
    win over a service-specific item in the same domain.

    Resolution is per-area: a doc_id superseded in one area keeps
    its chunks in other areas where it's not in conflict.
    """
    if len(chunks) <= 1:
        return chunks

    # Group chunks by overlap area: (domain, frozenset of services)
    area_groups: dict[tuple[str, frozenset[str]], list[ScoredChunk]] = defaultdict(list)

    for sc in chunks:
        domain = sc.chunk.domain
        services = frozenset(sc.chunk.affected_services)
        area_groups[(domain, services)].append(sc)

    # Collect individual chunks to remove (not doc_ids)
    chunks_to_remove: set[str] = set()  # point_ids to remove

    # Rule 1: Within each area, if multiple doc_ids, keep the newest
    for (domain, services), group in area_groups.items():
        doc_ids_in_group = {sc.chunk.doc_id for sc in group}
        if len(doc_ids_in_group) <= 1:
            continue

        # Find the newest doc_id by date
        newest_date = ""
        newest_doc_id = ""
        for sc in group:
            chunk_date = sc.chunk.date or ""
            if chunk_date > newest_date:
                newest_date = chunk_date
                newest_doc_id = sc.chunk.doc_id

        # Mark chunks from older doc_ids in THIS area only
        if newest_doc_id:
            for sc in group:
                if sc.chunk.doc_id != newest_doc_id:
                    chunks_to_remove.add(sc.point_id)

    # Rule 2: Service-specific items take precedence over project-wide
    # in the same domain
    service_specific_domains: set[str] = set()
    for (domain, services), group in area_groups.items():
        if services:  # non-empty = service-specific
            service_specific_domains.add(domain)

    for (domain, services), group in area_groups.items():
        if not services and domain in service_specific_domains:
            for sc in group:
                chunks_to_remove.add(sc.point_id)

    if not chunks_to_remove:
        return chunks

    resolved = [sc for sc in chunks if sc.point_id not in chunks_to_remove]

    # Safety net: if resolution removed everything, return originals
    return resolved if resolved else chunks


# ── Update ───────────────────────────────────────────────────────


async def update_payload(
    doc_id: str,
    payload_updates: dict,
    settings: Settings | None = None,
) -> int:
    """Update payload fields on all chunks matching a doc_id.

    Uses Qdrant's set_payload API for in-place updates without
    re-embedding. Primary use: status updates for conflict resolution.

    Args:
        doc_id: Document ID to match (e.g. "adr-001").
        payload_updates: Dict of field names to new values.
        settings: Optional settings override.

    Returns:
        Number of points updated.
    """
    s = settings or get_settings()
    client = _get_client(s)

    response = await client.scroll(
        collection_name=s.QDRANT_COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))],
        ),
        limit=100,
        with_payload=False,
    )

    points = response[0]
    if not points:
        return 0

    point_ids = [p.id for p in points]

    await client.set_payload(
        collection_name=s.QDRANT_COLLECTION,
        payload=payload_updates,
        points=point_ids,
    )

    logger.info("Updated %d points for doc_id=%s: %s", len(point_ids), doc_id, payload_updates)
    return len(point_ids)


async def find_overlapping(
    domain: str,
    affected_services: list[str],
    status: str = "active",
    settings: Settings | None = None,
) -> list[ScoredChunk]:
    """Find existing active items with overlapping domain and services.

    Used for conflict detection at ingestion time. Returns decision-type
    chunks that share the same domain and overlap in affected_services.

    Args:
        domain: Domain to check (e.g. "security").
        affected_services: Services to check overlap with.
        status: Status filter (default "active").
        settings: Optional settings override.

    Returns:
        List of overlapping chunks as ScoredChunks (score=1.0).
    """
    s = settings or get_settings()
    client = _get_client(s)

    must_clauses = [
        FieldCondition(key="status", match=MatchValue(value=status)),
        FieldCondition(key="domain", match=MatchValue(value=domain)),
        FieldCondition(key="section_type", match=MatchValue(value="decision")),
    ]

    if affected_services:
        should_clauses = [
            FieldCondition(key="affected_services", match=MatchAny(any=affected_services)),
            IsEmptyCondition(is_empty=PayloadField(key="affected_services")),
        ]
        scroll_filter = Filter(must=must_clauses, should=should_clauses)
    else:
        scroll_filter = Filter(must=must_clauses)

    response = await client.scroll(
        collection_name=s.QDRANT_COLLECTION,
        scroll_filter=scroll_filter,
        limit=50,
        with_payload=True,
    )

    scored: list[ScoredChunk] = []
    for r in response[0]:
        p = r.payload or {}
        chunk = ChunkRecord(
            id=p.get("doc_id", "") + "-" + p.get("section_type", ""),
            text=p.get("text", ""),
            knowledge_type=p.get("knowledge_type", "decision"),
            section_type=p.get("section_type", ""),
            source_type=p.get("source_type", ""),
            doc_id=p.get("doc_id", ""),
            author=p.get("author", "unknown"),
            affected_services=p.get("affected_services", []),
            domain=p.get("domain", "operational"),
            status=p.get("status", "active"),
            date=p.get("date", ""),
            source_url=p.get("source_url", ""),
            source_title=p.get("source_title", ""),
            ingested_at=p.get("ingested_at", ""),
            owners=p.get("owners", []),
            tags=p.get("tags", []),
            source_last_modified=p.get("source_last_modified", ""),
        )
        scored.append(ScoredChunk(chunk=chunk, score=1.0, point_id=str(r.id)))

    return scored


# ── Stats ────────────────────────────────────────────────────────────


async def stats(settings: Settings | None = None) -> dict:
    """Return basic corpus statistics for the /status endpoint."""
    s = settings or get_settings()
    client = _get_client(s)
    info = await client.get_collection(s.QDRANT_COLLECTION)
    return {
        "total_chunks": info.points_count,
        "collection": s.QDRANT_COLLECTION,
        "embedding_model": s.EMBEDDING_MODEL,
        "status": info.status,
    }