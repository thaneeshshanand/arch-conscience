"""CLI script to ingest rules files into the corpus.

Discovers and ingests CLAUDE.md, .cursorrules, AGENTS.md, rules.md,
and similar files from a project directory. Extracts architectural
knowledge items (decisions, constraints, principles) using an LLM
and indexes them for enforcement.

Usage:
    # Discover all known rules files in current directory
    python -m scripts.ingest_rules

    # Ingest a specific file
    python -m scripts.ingest_rules --file /path/to/CLAUDE.md

    # Discover in a specific project directory
    python -m scripts.ingest_rules --project /path/to/project
"""

import argparse
import asyncio
import logging
from collections import Counter

from app.config import get_settings
from app.corpus import ensure_collection, upsert
from app.rules_bridge import discover_and_ingest, ingest_rules_file

logging.basicConfig(level=logging.INFO, format="%(message)s")


async def main():
    parser = argparse.ArgumentParser(
        description="Ingest AI coding rules files into the arch-conscience corpus."
    )
    parser.add_argument(
        "--file",
        help="Path to a specific rules file to ingest.",
    )
    parser.add_argument(
        "--project",
        default=".",
        help="Project root directory to discover rules files in (default: current dir).",
    )
    args = parser.parse_args()

    settings = get_settings()
    settings.validate_required()
    await ensure_collection(settings)

    if args.file:
        print(f"Ingesting rules file: {args.file}")
        chunks = await ingest_rules_file(args.file, settings)
    else:
        print(f"Discovering rules files in: {args.project}")
        chunks = await discover_and_ingest(args.project, settings)

    if not chunks:
        print("\nNo architectural knowledge found.")
        return

    await upsert(chunks, settings)

    # Summarize
    doc_ids = set()
    type_counts: Counter[str] = Counter()
    for c in chunks:
        doc_ids.add(c.doc_id)
        if c.section_type == "decision":
            type_counts[c.knowledge_type] += 1

    print(f"\nDone. Extracted {len(doc_ids)} items, indexed {len(chunks)} chunks.")
    print(f"By type: {', '.join(f'{count} {kt}(s)' for kt, count in type_counts.most_common())}")
    print("\nItems found:")
    for c in chunks:
        if c.section_type == "decision":
            title = c.source_title or c.text.split("\n")[0]
            services = ", ".join(c.affected_services) if c.affected_services else "project-wide"
            print(f"  [{c.knowledge_type}] {title} [{c.domain}] -> {services}")


if __name__ == "__main__":
    asyncio.run(main())