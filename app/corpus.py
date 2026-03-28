"""Qdrant corpus wrapper — embed, upsert, hybrid query.

Mirrors the Node.js src/corpus.js. Dense vectors (Cosine) with BM25
sparse vector infrastructure configured for future hybrid retrieval.
Payload fields indexed for service and status filtering at query time.
"""

import logging
import uuid
from dataclasses import dataclass, field

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
    """A single corpus chunk, ready for upsert or returned from query."""

    id: str
    text: str
    source_type: str  # ADR | confluence | jira
    doc_id: str
    section_type: str  # context | decision | consequences | rejected_alternatives
    affected_services: list[str] = field(default_factory=list)
    decision_date: str = ""
    status: str = "active"
    constraint_type: str = "operational"
    author: str = "unknown"
    linked_adr_ids: list[str] = field(default_factory=list)


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
        ("affected_services", PayloadSchemaType.KEYWORD),
        ("status", PayloadSchemaType.KEYWORD),
        ("constraint_type", PayloadSchemaType.KEYWORD),
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
                    "source_type": chunk.source_type,
                    "doc_id": chunk.doc_id,
                    "section_type": chunk.section_type,
                    "affected_services": chunk.affected_services,
                    "decision_date": chunk.decision_date,
                    "status": chunk.status,
                    "constraint_type": chunk.constraint_type,
                    "author": chunk.author,
                    "linked_adr_ids": chunk.linked_adr_ids,
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
            source_type=p.get("source_type", ""),
            doc_id=p.get("doc_id", ""),
            section_type=p.get("section_type", ""),
            affected_services=p.get("affected_services", []),
            decision_date=p.get("decision_date", ""),
            status=p.get("status", "active"),
            constraint_type=p.get("constraint_type", "operational"),
            author=p.get("author", "unknown"),
            linked_adr_ids=p.get("linked_adr_ids", []),
        )
        scored.append(ScoredChunk(chunk=chunk, score=r.score, point_id=str(r.id)))

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