"""Alert management with Slack webhook integration and rate limiting."""

from __future__ import annotations

import enum
import time
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)


class Severity(str, enum.Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# Emoji mapping for Slack message formatting.
_SEVERITY_EMOJI = {
    Severity.INFO: ":information_source:",
    Severity.WARNING: ":warning:",
    Severity.CRITICAL: ":rotating_light:",
}

# Minimum seconds between identical alerts (key = title).
_RATE_LIMIT_SECONDS = 300  # 5 minutes


class AlertManager:
    """Sends alerts via Slack webhook with rate limiting.

    Parameters:
        webhook_url: Slack incoming-webhook URL.  If ``None``, alerts are
            only logged (useful for development / testing).
    """

    def __init__(self, webhook_url: str | None = None) -> None:
        self._webhook_url = webhook_url
        self._last_sent: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send_alert(
        self,
        severity: Severity,
        title: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Send an alert, subject to rate limiting.

        The same *title* will not produce another alert within 5 minutes.
        All alerts are logged regardless of rate limiting.
        """
        log = logger.bind(severity=severity.value, title=title)

        # Always log
        log_method = {
            Severity.INFO: log.info,
            Severity.WARNING: log.warning,
            Severity.CRITICAL: log.error,
        }.get(severity, log.info)

        log_method(
            "alert.raised",
            message=message,
            details=details,
        )

        # Rate limiting
        now = time.monotonic()
        last = self._last_sent.get(title, 0.0)
        if now - last < _RATE_LIMIT_SECONDS:
            log.debug("alert.rate_limited", seconds_since_last=now - last)
            return

        self._last_sent[title] = now

        # Send to Slack if configured
        if self._webhook_url:
            await self._post_slack(severity, title, message, details)

    # ------------------------------------------------------------------
    # Slack integration
    # ------------------------------------------------------------------

    async def _post_slack(
        self,
        severity: Severity,
        title: str,
        message: str,
        details: dict[str, Any] | None,
    ) -> None:
        """POST a formatted message to the Slack webhook."""
        emoji = _SEVERITY_EMOJI.get(severity, "")
        text_parts = [f"{emoji} *[{severity.value.upper()}] {title}*", message]

        if details:
            detail_lines = [f"  - {k}: {v}" for k, v in details.items()]
            text_parts.append("\n".join(detail_lines))

        payload = {"text": "\n".join(text_parts)}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(self._webhook_url, json=payload)
                if resp.status_code != 200:
                    logger.warning(
                        "alert.slack.post_failed",
                        status=resp.status_code,
                        body=resp.text,
                    )
        except Exception:
            logger.exception("alert.slack.error")
