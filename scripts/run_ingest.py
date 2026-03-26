"""CLI script to ingest ADRs into Qdrant.

Usage:
    python -m scripts.run_ingest
"""

import asyncio
import logging

from app.config import get_settings
from app.ingest import ingest

logging.basicConfig(level=logging.INFO, format="%(message)s")


async def main():
    settings = get_settings()
    settings.validate_required()
    results = await ingest(settings)
    print(f"\nDone — adr:{results.adr} confluence:{results.confluence} jira:{results.jira}")
    if results.errors:
        print(f"Errors: {'; '.join(results.errors)}")


if __name__ == "__main__":
    asyncio.run(main())