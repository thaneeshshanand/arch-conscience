"""Post-bulk-migration corpus health check.

Scans the corpus for potential conflicts: active items with overlapping
domain + affected_services from different doc_ids. Purely programmatic —
no LLM call. Presents a report and suggests resolution actions.

Usage:
    python -m scripts.check_health
    python scripts/check_health.py
"""

import asyncio
import logging
from collections import defaultdict

from app.config import get_settings
from app.corpus import ensure_collection, stats

logging.basicConfig(level=logging.INFO, format="%(message)s")


async def main():
    settings = get_settings()
    settings.validate_required()
    await ensure_collection(settings)

    corpus_stats = await stats(settings)
    total = corpus_stats.get("total_chunks", 0)
    print(f"Corpus: {corpus_stats['collection']} — {total} chunks")
    print()

    if total == 0:
        print("Corpus is empty. Nothing to check.")
        return

    # Scroll all decision chunks to find overlaps
    from qdrant_client.http.models import (
        FieldCondition, Filter, MatchValue,
    )
    from app.corpus import _get_client

    client = _get_client(settings)

    # Fetch all active decision-type chunks
    response = await client.scroll(
        collection_name=settings.QDRANT_COLLECTION,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="status", match=MatchValue(value="active")),
                FieldCondition(key="section_type", match=MatchValue(value="decision")),
            ],
        ),
        limit=500,
        with_payload=True,
    )

    points = response[0]

    if not points:
        print("No active decision chunks found.")
        return

    print(f"Found {len(points)} active decision chunks.")
    print()

    # Group by (domain, affected_services) overlap area
    overlap_groups: dict[str, list[dict]] = defaultdict(list)

    for point in points:
        p = point.payload or {}
        domain = p.get("domain", "operational")
        services = tuple(sorted(p.get("affected_services", [])))
        doc_id = p.get("doc_id", "")
        title = p.get("source_title", doc_id)
        date = p.get("date", "")
        knowledge_type = p.get("knowledge_type", "decision")

        # Key: domain + service set
        key = f"{domain}:{','.join(services) or 'project-wide'}"
        overlap_groups[key].append({
            "doc_id": doc_id,
            "title": title,
            "date": date,
            "domain": domain,
            "services": list(services),
            "knowledge_type": knowledge_type,
        })

    # Find groups with multiple doc_ids (potential conflicts)
    conflicts_found = 0

    for key, items in sorted(overlap_groups.items()):
        doc_ids = {item["doc_id"] for item in items}
        if len(doc_ids) <= 1:
            continue

        conflicts_found += 1
        domain, services = key.split(":", 1)

        print(f"⚠️  Overlap area: domain={domain}, services={services}")
        print(f"   {len(doc_ids)} items govern this area:")

        # Sort by date (newest first)
        sorted_items = sorted(items, key=lambda x: x["date"] or "", reverse=True)
        for item in sorted_items:
            date_str = f" ({item['date']})" if item["date"] else ""
            print(f"   - [{item['doc_id']}] {item['title']}{date_str} [{item['knowledge_type']}]")

        # Suggest resolution
        newest = sorted_items[0]
        older = sorted_items[1:]
        if newest["date"] and all(o["date"] and o["date"] < newest["date"] for o in older):
            print(f"   → Suggestion: '{newest['doc_id']}' is newest. Consider superseding older items:")
            for o in older:
                print(f"     update_item_status(doc_id='{o['doc_id']}', new_status='superseded')")
        else:
            print("   → Review manually: dates are equal or missing.")

        print()

    if conflicts_found == 0:
        print("✅ No overlapping items found. Corpus is clean.")
    else:
        print(f"Found {conflicts_found} overlap area(s) to review.")

    # Summary stats
    print()
    print("── Summary ─────────────────────────────")

    kt_counts: dict[str, int] = defaultdict(int)
    domain_counts: dict[str, int] = defaultdict(int)
    doc_ids: set[str] = set()

    for point in points:
        p = point.payload or {}
        kt_counts[p.get("knowledge_type", "decision")] += 1
        domain_counts[p.get("domain", "operational")] += 1
        doc_ids.add(p.get("doc_id", ""))

    print(f"Active decision chunks: {len(points)}")
    print(f"Unique items (doc_ids): {len(doc_ids)}")
    print(f"By knowledge type: {dict(kt_counts)}")
    print(f"By domain: {dict(domain_counts)}")


if __name__ == "__main__":
    asyncio.run(main())