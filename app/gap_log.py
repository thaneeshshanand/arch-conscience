"""Corpus gap signal logger — JSONL append log.

When retrieval finds no relevant decisions for a changed service,
this module logs the event as a gap signal. The gap log is input to
a future self-healing pipeline that surfaces undocumented architectural
decisions and prompts teams to write missing ADRs.

Each line is a self-contained JSON object — easy to tail, grep,
and eventually feed into the self-healing pipeline.
"""

import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_LOG_PATH = Path(__file__).resolve().parent.parent / "gap.log"


@dataclass(frozen=True, slots=True)
class GapEntry:
    """A single corpus gap signal."""

    type: str  # "no_chunks_found" | "sparse_retrieval"
    services: list[str]
    pr_url: str
    ts: str  # ISO timestamp
    diff_summary: str = ""


@dataclass(frozen=True, slots=True)
class UndocumentedService:
    """Summary of a service with no corpus coverage."""

    service: str
    count: int
    last_seen: str


def log_gap(entry: GapEntry) -> None:
    """Append a gap signal to the JSONL log file.

    Non-fatal — never lets logging break the main pipeline.
    """
    line = json.dumps({
        "type": entry.type,
        "services": entry.services,
        "pr_url": entry.pr_url,
        "diff_summary": entry.diff_summary,
        "ts": entry.ts,
    })

    try:
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        logger.info(
            "Logged %s for services: %s",
            entry.type,
            ", ".join(entry.services),
        )
    except OSError as exc:
        logger.error("Failed to write gap log: %s", exc)


def read_gap_log() -> list[dict]:
    """Read all gap log entries. Returns empty list if file doesn't exist."""
    if not _LOG_PATH.exists():
        return []

    entries: list[dict] = []
    for line in _LOG_PATH.read_text(encoding="utf-8").strip().splitlines():
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return entries


def undocumented_services() -> list[UndocumentedService]:
    """Summarise services that appear in gap signals but have no corpus coverage.

    Returns services sorted by gap count (descending).
    """
    entries = read_gap_log()
    counts: Counter[str] = Counter()
    last_seen: dict[str, str] = {}

    for entry in entries:
        ts = entry.get("ts", "")
        for svc in entry.get("services", []):
            counts[svc] += 1
            if ts > last_seen.get(svc, ""):
                last_seen[svc] = ts

    return [
        UndocumentedService(service=svc, count=count, last_seen=last_seen[svc])
        for svc, count in counts.most_common()
    ]


def create_gap_entry(
    *,
    gap_type: str,
    services: list[str],
    pr_url: str,
    diff_summary: str = "",
) -> GapEntry:
    """Convenience factory — stamps the current time automatically."""
    return GapEntry(
        type=gap_type,
        services=services,
        pr_url=pr_url,
        diff_summary=diff_summary,
        ts=datetime.utcnow().isoformat() + "Z",
    )