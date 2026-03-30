"""Alert formatter and channel dispatcher.

Formats a DetectionResult into a human-readable message and sends it
to the responsible engineer via the configured channel (Telegram or Slack).

Slack routing: per-service channel lookup via SLACK_CHANNEL_MAP. If a
service maps to a channel, that channel receives the alert. If multiple
services map to different channels, all receive it. Falls back to
SLACK_WEBHOOK_URL when no service-specific channel is found.
"""

import logging

import httpx

from app.config import Settings, get_settings
from app.detect import DetectionResult

logger = logging.getLogger(__name__)

_SEVERITY_EMOJI = {
    "high": "\U0001f534",    # 🔴
    "medium": "\U0001f7e1",  # 🟡
    "low": "\U0001f535",     # 🔵
}


async def dispatch(
    *,
    result: DetectionResult,
    engineer: str,
    pr_url: str,
    affected_services: list[str] | None = None,
    settings: Settings | None = None,
) -> None:
    """Format and send an alert for a detected architectural gap.

    Args:
        result: Detection pipeline output.
        engineer: GitHub username of the PR author.
        pr_url: Full PR URL.
        affected_services: Services touched by the PR. Used for Slack
                           channel routing via SLACK_CHANNEL_MAP.
        settings: Optional settings override (for tests).

    Raises:
        RuntimeError: If the configured channel fails to send.
    """
    s = settings or get_settings()
    channel = s.ALERT_CHANNEL

    logger.info(
        "Dispatching %s alert to %s via %s",
        result.severity,
        engineer,
        channel,
    )
    logger.info("Headline: %s", result.alert_headline)

    if channel == "telegram":
        message = _format_message(result=result, engineer=engineer, pr_url=pr_url)
        await _send_telegram(message, s)
    elif channel == "slack":
        message = _format_slack_message(result=result, engineer=engineer, pr_url=pr_url)
        await _send_slack(message, affected_services or [], s)
    else:
        message = _format_message(result=result, engineer=engineer, pr_url=pr_url)
        logger.warning(
            "Unknown alert channel '%s' — printing alert:\n%s",
            channel,
            message,
        )


# ── Telegram ──────────────────────────────────────────────────────────


def _format_message(
    *,
    result: DetectionResult,
    engineer: str,
    pr_url: str,
) -> str:
    """Build the Telegram alert message text."""
    emoji = _SEVERITY_EMOJI.get(result.severity or "", "\u26aa")  # ⚪
    adr_ref = f" ({result.violated_adr_id})" if result.violated_adr_id else ""

    lines = [
        f"{emoji} Architectural gap detected{adr_ref}",
        "",
        result.alert_headline,
        "",
        result.alert_body,
        "",
        f"PR: {pr_url}",
        f"Author: @{engineer}",
        f"Severity: {result.severity}",
    ]

    if result.constraint_type:
        lines.append(f"Domain: {result.constraint_type}")

    if result.rejected_alt_reintroduced:
        lines.append("\u26a0\ufe0f This approach was explicitly rejected in the ADR.")

    if result.corpus_conflict:
        lines.append(
            "\u26a0\ufe0f Note: The corpus contains conflicting guidance in this "
            "area. Review related items and resolve with update_item_status."
        )

    return "\n".join(line for line in lines if line is not None)


async def _send_telegram(message: str, settings: Settings) -> None:
    """Send via Telegram Bot API."""
    token = settings.TELEGRAM_BOT_TOKEN
    chat_id = settings.TELEGRAM_CHAT_ID

    if not token or not chat_id:
        logger.warning(
            "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — logging only"
        )
        logger.info(message)
        return

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": message,
                "disable_web_page_preview": True,
            },
        )

    if resp.status_code != 200:
        raise RuntimeError(f"Telegram API error {resp.status_code}: {resp.text}")

    logger.info("Telegram alert sent to chat %s", chat_id)


# ── Slack ─────────────────────────────────────────────────────────────


def _format_slack_message(
    *,
    result: DetectionResult,
    engineer: str,
    pr_url: str,
) -> str:
    """Build the Slack alert message in mrkdwn format."""
    emoji = _SEVERITY_EMOJI.get(result.severity or "", "\u26aa")
    adr_ref = f" ({result.violated_adr_id})" if result.violated_adr_id else ""

    lines = [
        f"{emoji} *Architectural gap detected{adr_ref}*",
        "",
        result.alert_headline or "",
        "",
        result.alert_body or "",
        "",
        f"*PR:* <{pr_url}|View PR> | *Author:* @{engineer} | *Severity:* {result.severity}",
    ]

    if result.constraint_type:
        lines[-1] += f" | *Domain:* {result.constraint_type}"

    if result.rejected_alt_reintroduced:
        lines.append("\u26a0\ufe0f This approach was explicitly rejected in the ADR.")

    if result.corpus_conflict:
        lines.append(
            "\u26a0\ufe0f The corpus contains conflicting guidance. "
            "Review related items and resolve with update_item_status."
        )

    return "\n".join(line for line in lines if line is not None)


def _resolve_slack_channels(
    affected_services: list[str],
    settings: Settings,
) -> set[str]:
    """Resolve the set of Slack webhook URLs to notify.

    Looks up each affected service in SLACK_CHANNEL_MAP. Falls back to
    SLACK_WEBHOOK_URL if no service-specific channel is found.

    Returns an empty set if neither is configured (caller logs and skips).
    """
    channel_map = settings.slack_channel_map
    urls: set[str] = set()

    for svc in affected_services:
        if svc in channel_map:
            urls.add(channel_map[svc])

    if not urls and settings.SLACK_WEBHOOK_URL:
        urls.add(settings.SLACK_WEBHOOK_URL)

    return urls


async def _send_slack(
    message: str,
    affected_services: list[str],
    settings: Settings,
) -> None:
    """Send via Slack incoming webhooks with per-service channel routing."""
    urls = _resolve_slack_channels(affected_services, settings)

    if not urls:
        logger.warning(
            "No Slack webhook configured (set SLACK_WEBHOOK_URL or SLACK_CHANNEL_MAP) — logging only"
        )
        logger.info(message)
        return

    payload = {
        "text": message,  # notification fallback text
        "blocks": [
            {"type": "section", "text": {"type": "mrkdwn", "text": message}},
        ],
    }

    async with httpx.AsyncClient(timeout=30) as client:
        for url in urls:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"Slack webhook error {resp.status_code}: {resp.text}")

    logger.info("Slack alert sent to %d channel(s)", len(urls))