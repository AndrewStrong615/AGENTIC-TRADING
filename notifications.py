"""
notifications.py — Sends phone push notifications via ntfy.sh (free, no
signup). Works even when the dashboard browser tab is closed, because the
notification comes from ntfy's server, not your browser.

Setup:
    1. Install the ntfy app on your phone (App Store / Play Store), or just
       use ntfy.sh in a mobile browser.
    2. Pick a LONG, random, private topic name — anyone who knows it can
       see your notifications, since ntfy.sh is a public service.
       e.g. "trading-bot-a1str-9f3k2m8x"
    3. In the ntfy app, subscribe to that exact topic name.
    4. Set it as an environment variable before starting the dashboard:
           setx NTFY_TOPIC "trading-bot-a1str-9f3k2m8x"
       (open a NEW terminal after setx, same as DASHBOARD_PASSWORD)

If NTFY_TOPIC isn't set, every call here silently does nothing — the rest
of the dashboard works exactly as before.

Optional: Slack / Discord webhook alerts
    In addition to (or instead of) ntfy, you can have error/run/trade
    notifications also posted to a Slack or Discord channel via an
    incoming webhook. This is entirely opt-in and additive — if the
    corresponding env var isn't set, no webhook call is attempted.

    Slack:   set SLACK_WEBHOOK_URL to an "Incoming Webhook" URL.
    Discord: set DISCORD_WEBHOOK_URL to a channel webhook URL.
"""

import os
import json
import logging
import urllib.request
from typing import Optional

log = logging.getLogger("trader")

NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "")
NTFY_SERVER = os.environ.get("NTFY_SERVER", "https://ntfy.sh")

SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")


def send_notification(title: str, message: str, priority: str = "default", tags: str = ""):
    """priority: 'min'|'low'|'default'|'high'|'urgent'. tags: comma-separated
    ntfy emoji shortcodes, e.g. 'chart_with_upwards_trend,moneybag'."""
    if not NTFY_TOPIC:
        return
    url = f"{NTFY_SERVER}/{NTFY_TOPIC}"
    headers = {"Title": title, "Priority": priority}
    if tags:
        headers["Tags"] = tags
    try:
        req = urllib.request.Request(url, data=message.encode("utf-8"),
                                      headers=headers, method="POST")
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        log.error(f"notifications: failed to send ntfy notification: {e}")


def _post_webhook(url: str, payload: dict):
    """POST a small JSON payload to a Slack/Discord-style incoming webhook.
    Silently no-ops if url is falsy; any failure is logged, never raised."""
    if not url:
        return
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data,
                                      headers={"Content-Type": "application/json"},
                                      method="POST")
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        log.error(f"notifications: failed to send webhook notification: {e}")


def send_webhook_alert(title: str, message: str):
    """Fan out a notification to any configured Slack/Discord webhooks.
    No-op for any webhook whose env var isn't set."""
    text = f"*{title}*\n{message}" if title else message
    if SLACK_WEBHOOK_URL:
        _post_webhook(SLACK_WEBHOOK_URL, {"text": text})
    if DISCORD_WEBHOOK_URL:
        _post_webhook(DISCORD_WEBHOOK_URL, {"content": text})


def notify_run_finished(script_label: str, status: str, duration_seconds: Optional[float] = None):
    dur = f" ({int(duration_seconds)}s)" if duration_seconds else ""
    if status == "finished":
        title, msg = f"{script_label} finished", f"Completed successfully{dur}."
        send_notification(title, msg, tags="white_check_mark")
    elif status == "stopped":
        title, msg = f"{script_label} stopped", "Stopped by user."
        send_notification(title, msg, tags="octagonal_sign")
    else:
        title, msg = f"{script_label} ended", f"Status: {status}{dur}"
        send_notification(title, msg, priority="high", tags="warning")
    send_webhook_alert(title, msg)


def notify_new_trade(symbol: str, pnl_pct: float, exit_reason: str):
    emoji = "chart_with_upwards_trend" if pnl_pct >= 0 else "chart_with_downwards_trend"
    sign = "+" if pnl_pct >= 0 else ""
    title = f"{symbol} trade closed"
    msg = f"{sign}{pnl_pct:.2f}% — {exit_reason}"
    send_notification(title, msg, priority="default", tags=emoji)
    send_webhook_alert(title, msg)


def notify_order_placed(symbol: str, action: str, qty: int, price: float, reason: str = ""):
    """Immediate notification when an order is approved and sent.

    action: 'buy' or 'sell'
    """
    try:
        emoji = "chart_with_upwards_trend" if action.lower() == "buy" else "chart_with_downwards_trend"
        title = f"Order {action.upper()} {symbol}"
        msg = f"{action.upper()} {qty} @ ${price:.2f}"
        if reason:
            msg += f" — {reason}"
        send_notification(title, msg, priority="default", tags=emoji)
    except Exception:
        pass


def test_notification():
    send_notification("Test notification", "If you see this, notifications are working.",
                       tags="test_tube")

def notify_error(context: str, error: Exception):
    title = "Trading Dashboard ERROR"
    msg = f"{context}: {error}"
    send_notification(
        title,
        msg,
        priority="urgent",
        tags="rotating_light"
    )
    send_webhook_alert(title, msg)