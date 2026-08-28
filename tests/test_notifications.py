"""Tests for notifications.py.

urllib.request.urlopen is always monkeypatched to a fake -- these tests
never make a real HTTP request to ntfy.sh or any other host.
"""
import notifications as notif


def test_send_notification_is_a_noop_without_a_configured_topic(monkeypatch):
    monkeypatch.setattr(notif, "NTFY_TOPIC", "")
    called = []
    monkeypatch.setattr(notif.urllib.request, "urlopen", lambda *a, **k: called.append(True))
    notif.send_notification("Title", "Body")
    assert called == []  # never attempted a network call


def test_send_notification_builds_the_expected_request(monkeypatch):
    monkeypatch.setattr(notif, "NTFY_TOPIC", "fake-topic-for-tests")
    monkeypatch.setattr(notif, "NTFY_SERVER", "https://example.invalid")

    captured = {}

    def fake_urlopen(req, timeout=5):
        captured["url"] = req.full_url
        captured["headers"] = req.headers
        captured["data"] = req.data
        return None

    monkeypatch.setattr(notif.urllib.request, "urlopen", fake_urlopen)
    notif.send_notification("Hello", "World", priority="high", tags="warning")

    assert captured["url"] == "https://example.invalid/fake-topic-for-tests"
    assert captured["data"] == b"World"
    assert captured["headers"]["Title"] == "Hello"
    assert captured["headers"]["Priority"] == "high"
    assert captured["headers"]["Tags"] == "warning"


def test_send_notification_swallows_network_errors(monkeypatch):
    monkeypatch.setattr(notif, "NTFY_TOPIC", "fake-topic-for-tests")

    def boom(*a, **k):
        raise OSError("network is unreachable")

    monkeypatch.setattr(notif.urllib.request, "urlopen", boom)
    notif.send_notification("Title", "Body")  # must not raise


def test_notify_new_trade_uses_up_emoji_and_plus_sign_for_gains(monkeypatch):
    calls = []
    monkeypatch.setattr(notif, "send_notification", lambda *a, **k: calls.append((a, k)))
    notif.notify_new_trade("AAPL", 2.5, "take_profit")
    args, kwargs = calls[0]
    assert args[0] == "AAPL trade closed"
    assert "+2.50%" in args[1]
    assert kwargs["tags"] == "chart_with_upwards_trend"


def test_notify_new_trade_uses_down_emoji_for_losses(monkeypatch):
    calls = []
    monkeypatch.setattr(notif, "send_notification", lambda *a, **k: calls.append((a, k)))
    notif.notify_new_trade("AAPL", -1.2, "stop_loss")
    args, kwargs = calls[0]
    assert "-1.20%" in args[1]
    assert kwargs["tags"] == "chart_with_downwards_trend"


def test_notify_order_placed_formats_action_qty_and_price(monkeypatch):
    calls = []
    monkeypatch.setattr(notif, "send_notification", lambda *a, **k: calls.append((a, k)))
    notif.notify_order_placed("MSFT", "buy", 10, 123.456, reason="signal")
    args, kwargs = calls[0]
    assert args[0] == "Order BUY MSFT"
    assert "BUY 10 @ $123.46" in args[1]
    assert "signal" in args[1]


def test_notify_order_placed_never_raises_even_if_notification_fails(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("should be swallowed")

    monkeypatch.setattr(notif, "send_notification", boom)
    notif.notify_order_placed("MSFT", "buy", 10, 100.0)  # must not raise
