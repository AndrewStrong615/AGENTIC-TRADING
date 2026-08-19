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
"""

import os
import logging
import urllib.request

log = logging.getLogger("trader")

NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "")
NTFY_SERVER = os.environ.get("NTFY_SERVER", "https://ntfy.sh")


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


def notify_run_finished(script_label: str, status: str, duration_seconds: float = None):
    dur = f" ({int(duration_seconds)}s)" if duration_seconds else ""
    if status == "finished":
        send_notification(f"{script_label} finished", f"Completed successfully{dur}.",
                           tags="white_check_mark")
    elif status == "stopped":
        send_notification(f"{script_label} stopped", "Stopped by user.", tags="octagonal_sign")
    else:
        send_notification(f"{script_label} ended", f"Status: {status}{dur}", priority="high",
                           tags="warning")


def notify_new_trade(symbol: str, pnl_pct: float, exit_reason: str):
    emoji = "chart_with_upwards_trend" if pnl_pct >= 0 else "chart_with_downwards_trend"
    sign = "+" if pnl_pct >= 0 else ""
    send_notification(f"{symbol} trade closed", f"{sign}{pnl_pct:.2f}% — {exit_reason}",
                       priority="default", tags=emoji)


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
    send_notification(
        "Trading Dashboard ERROR",
        f"{context}: {error}",
        priority="urgent",
        tags="rotating_light"
    )