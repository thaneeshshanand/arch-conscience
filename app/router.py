"""GitHub webhook payload → pipeline payload.

Transforms a raw GitHub pull_request webhook body into a structured
PipelinePayload that detect.py and corpus.py understand.
"""

import logging
from dataclasses import dataclass, field

import httpx

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PipelinePayload:
    """Structured payload passed through the detection pipeline."""

    pr_url: str
    pr_number: str
    pr_title: str
    author: str
    base_branch: str
    changed_files: list[str] = field(default_factory=list)
    affected_services: list[str] = field(default_factory=list)
    diff_summary: str = ""


async def build_payload(
    gh: dict,
    settings: Settings | None = None,
) -> PipelinePayload:
    """Build a PipelinePayload from a raw GitHub webhook body.

    Fetches the list of changed files from the GitHub API,
    derives affected services via SERVICE_MAP, and builds
    a plain-English diff summary for corpus retrieval.
    """
    s = settings or get_settings()
    pr = gh["pull_request"]

    changed_files = await _fetch_changed_files(
        repo_full_name=gh["repository"]["full_name"],
        pr_number=pr["number"],
        settings=s,
    )

    affected_services = _derive_services(changed_files, s.service_map)

    diff_summary = _build_diff_summary(
        title=pr["title"],
        body=pr.get("body") or "",
        changed_files=changed_files,
        affected_services=affected_services,
    )

    return PipelinePayload(
        pr_url=pr["html_url"],
        pr_number=str(pr["number"]),
        pr_title=pr["title"],
        author=pr["user"]["login"],
        base_branch=pr["base"]["ref"],
        changed_files=changed_files,
        affected_services=affected_services,
        diff_summary=diff_summary,
    )


async def _fetch_changed_files(
    repo_full_name: str,
    pr_number: int,
    settings: Settings,
) -> list[str]:
    """Fetch changed file paths from the GitHub API. Handles pagination."""
    files: list[str] = []
    page = 1

    async with httpx.AsyncClient() as client:
        while True:
            url = (
                f"https://api.github.com/repos/{repo_full_name}"
                f"/pulls/{pr_number}/files?per_page=100&page={page}"
            )

            resp = await client.get(
                url,
                headers={
                    "Authorization": f"Bearer {settings.GITHUB_TOKEN}",
                    "Accept": "application/vnd.github.v3+json",
                },
            )

            if resp.status_code != 200:
                logger.error(
                    "GitHub API error %d fetching changed files",
                    resp.status_code,
                )
                raise RuntimeError(
                    f"GitHub API error {resp.status_code} fetching changed files"
                )

            batch = resp.json()
            if not batch:
                break

            files.extend(f["filename"] for f in batch)

            if len(batch) < 100:
                break
            page += 1

    return files


def _derive_services(
    changed_files: list[str],
    service_map: dict[str, str],
) -> list[str]:
    """Map changed file paths to service names.

    Uses SERVICE_MAP for explicit prefix matching. Falls back to the
    top-level directory name (suitable for monorepos organised as
    services/<name>/...).
    """
    found: set[str] = set()

    for file_path in changed_files:
        matched = False

        for prefix, service_name in service_map.items():
            if file_path.startswith(prefix):
                found.add(service_name)
                matched = True
                break

        if not matched:
            parts = file_path.split("/")
            if len(parts) > 1:
                found.add(parts[0])

    return sorted(found)


def _build_diff_summary(
    title: str,
    body: str,
    changed_files: list[str],
    affected_services: list[str],
) -> str:
    """Build a plain-English summary used as the semantic query text.

    Not shown to engineers — only used as the embedding query vector.
    Richer context goes into the Stage 2 prompt directly.
    """
    service_list = ", ".join(affected_services) if affected_services else "unknown service"

    # Truncate PR body to avoid blowing the embedding token budget
    body_excerpt = (body[:400].replace("\n", " ").strip()) if body else "no description provided"

    file_count = len(changed_files)
    file_list = ", ".join(changed_files[:8])
    if file_count > 8:
        file_list += f" and {file_count - 8} more"

    return ". ".join([
        f"PR title: {title}",
        f"Services affected: {service_list}",
        f"Changed files ({file_count}): {file_list}",
        f"Description: {body_excerpt}",
    ])