"""Alert formatter and channel dispatcher.

Formats a DetectionResult into a human-readable message and sends it
to the responsible engineer via the configured channel (Telegram, with
Slack planned). Kept short enough to read on mobile without scrolling.
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
    settings: Settings | None = None,
) -> None:
    """Format and send an alert for a detected architectural gap.

    Args:
        result: Detection pipeline output.
        engineer: GitHub username of the PR author.
        pr_url: Full PR URL.
        settings: Optional settings override (for tests).

    Raises:
        RuntimeError: If the configured channel fails to send.
    """
    s = settings or get_settings()
    message = _format_message(result=result, engineer=engineer, pr_url=pr_url)
    channel = s.ALERT_CHANNEL

    logger.info(
        "Dispatching %s alert to %s via %s",
        result.severity,
        engineer,
        channel,
    )
    logger.info("Headline: %s", result.alert_headline)

    if channel == "telegram":
        await _send_telegram(message, s)
    elif channel == "slack":
        await _send_slack(message, s)
    else:
        # Fallback: log to console if no channel configured
        logger.warning(
            "Unknown alert channel '%s' — printing alert:\n%s",
            channel,
            message,
        )


def _format_message(
    *,
    result: DetectionResult,
    engineer: str,
    pr_url: str,
) -> str:
    """Build the alert message text."""
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

    if result.rejected_alt_reintroduced:
        lines.append("\u26a0\ufe0f This approach was explicitly rejected in the ADR.")

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

    async with httpx.AsyncClient() as client:
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


async def _send_slack(message: str, settings: Settings) -> None:
    """Send via Slack webhook. Placeholder for future implementation."""
    # TODO: implement when Slack webhook URL is added to config
    logger.warning("Slack channel not yet implemented — logging only")
    logger.info(message)