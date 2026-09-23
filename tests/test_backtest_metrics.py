"""Unit tests for the pure metrics helpers in backtest.py.

These functions are extracted directly from backtest.py's source (see
tests/_extract.py) rather than imported normally, because a plain
``import backtest`` pulls in Tradingcore.py and its heavy/optional
dependency chain (scikit-learn, hmmlearn, vaderSentiment,
alpaca_trade_api, ...) purely to build the live trading engine, which is
unnecessary just to exercise two pure math functions and is not guaranteed
to even succeed in every environment. This keeps the tests hermetic: no
network access, no credentials, no dependency on optional/live-trading
packages, and no execution of bot.py/Tradingcore.py/credentials_store.py.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from _extract import extract_functions

BACKTEST_PY = Path(__file__).resolve().parent.parent / "backtest.py"

STARTING_CASH_PER_SYMBOL = 10_000.0


@pytest.fixture(scope="module")
def metrics_ns():
    """Namespace containing compute_metrics/calculate_drawdown/
    _bar_interval_minutes, extracted straight from backtest.py's source."""
    return extract_functions(
        BACKTEST_PY,
        ["compute_metrics", "calculate_drawdown", "_bar_interval_minutes"],
        extra_globals={
            "np": np,
            "STARTING_CASH_PER_SYMBOL": STARTING_CASH_PER_SYMBOL,
            "BAR_TIMEFRAME": "1m",
        },
    )


@pytest.fixture(scope="module")
def calculate_drawdown(metrics_ns):
    return metrics_ns["calculate_drawdown"]


@pytest.fixture(scope="module")
def compute_metrics(metrics_ns):
    return metrics_ns["compute_metrics"]


# ---------------------------------------------------------------------------
# calculate_drawdown
# ---------------------------------------------------------------------------


def test_calculate_drawdown_empty_returns_empty_list(calculate_drawdown):
    assert calculate_drawdown([]) == []


def test_calculate_drawdown_monotonic_gain_is_all_zero(calculate_drawdown):
    result = calculate_drawdown([100, 110, 120, 130])
    assert result == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_calculate_drawdown_known_values(calculate_drawdown):
    # Peak at 100, drop to 80 (-20%), recover to 100 (0%), drop to 50 (-50%).
    equity_curve = [100, 80, 100, 50]
    result = calculate_drawdown(equity_curve)
    assert result == pytest.approx([0.0, -20.0, 0.0, -50.0])


def test_calculate_drawdown_never_recovers(calculate_drawdown):
    equity_curve = [100, 90, 80, 70]
    result = calculate_drawdown(equity_curve)
    assert result == pytest.approx([0.0, -10.0, -20.0, -30.0])


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


def test_compute_metrics_empty_equity_curve_returns_empty_dict(compute_metrics):
    assert compute_metrics([], []) == {}


def test_compute_metrics_no_trades_has_zeroed_trade_stats(compute_metrics):
    equity_curve = [10_000, 10_500, 11_000]
    result = compute_metrics(equity_curve, [])

    assert result["num_trades"] == 0
    assert result["win_rate"] == 0.0
    assert result["avg_win_pct"] == 0.0
    assert result["avg_loss_pct"] == 0.0
    assert result["profit_factor"] == 0.0
    assert result["expectancy_pct"] == 0.0
    assert result["total_return_pct"] == pytest.approx(10.0)
    assert result["max_drawdown_pct"] == pytest.approx(0.0)


def test_compute_metrics_known_total_return_and_drawdown(compute_metrics):
    # Starts at STARTING_CASH_PER_SYMBOL, dips, then ends up 20% higher.
    equity_curve = [10_000, 9_000, 12_000]
    result = compute_metrics(equity_curve, [])

    assert result["total_return_pct"] == pytest.approx(20.0)
    # Peak-to-trough from 10_000 -> 9_000 is a 10% drawdown.
    assert result["max_drawdown_pct"] == pytest.approx(-10.0)


def test_compute_metrics_win_rate_and_trade_stats(compute_metrics):
    equity_curve = [10_000, 10_100, 10_050, 10_200]
    trades = [
        {"pnl_pct": 2.0},
        {"pnl_pct": -1.0},
        {"pnl_pct": 3.0},
        {"pnl_pct": 0.0},  # zero counts as a loss per the source's <= 0 rule
    ]
    result = compute_metrics(equity_curve, trades)

    assert result["num_trades"] == 4
    assert result["win_rate"] == pytest.approx(50.0)  # 2 of 4 are > 0
    assert result["avg_win_pct"] == pytest.approx((2.0 + 3.0) / 2)
    assert result["avg_loss_pct"] == pytest.approx((-1.0 + 0.0) / 2)
    assert result["expectancy_pct"] == pytest.approx((2.0 - 1.0 + 3.0 + 0.0) / 4)


def test_compute_metrics_profit_factor_all_wins_is_capped(compute_metrics):
    equity_curve = [10_000, 10_500]
    trades = [{"pnl_pct": 5.0}, {"pnl_pct": 3.0}]
    result = compute_metrics(equity_curve, trades)

    # No losing trades -> gross_loss is 0, source caps this special case at 999.0
    assert result["profit_factor"] == pytest.approx(999.0)


def test_compute_metrics_profit_factor_no_wins_no_losses_is_zero(compute_metrics):
    equity_curve = [10_000, 10_000]
    trades = [{"pnl_pct": 0.0}]
    result = compute_metrics(equity_curve, trades)

    assert result["profit_factor"] == pytest.approx(0.0)


def test_compute_metrics_profit_factor_known_ratio(compute_metrics):
    equity_curve = [10_000, 10_100]
    trades = [{"pnl_pct": 4.0}, {"pnl_pct": -2.0}]
    result = compute_metrics(equity_curve, trades)

    # gross_profit = 4.0, gross_loss = 2.0 -> profit_factor = 2.0
    assert result["profit_factor"] == pytest.approx(2.0)


def test_compute_metrics_calmar_zero_when_no_drawdown(compute_metrics):
    equity_curve = [10_000, 11_000, 12_000]
    result = compute_metrics(equity_curve, [])
    # calculate_drawdown never goes negative on a monotonic rise, so calmar
    # falls back to the source's explicit 0.0 branch.
    assert result["calmar"] == 0.0


def test_compute_metrics_returns_all_expected_keys(compute_metrics):
    equity_curve = [10_000, 9_800, 10_200, 10_500]
    trades = [{"pnl_pct": 1.5}, {"pnl_pct": -0.5}]
    result = compute_metrics(equity_curve, trades)

    expected_keys = {
        "total_return_pct",
        "max_drawdown_pct",
        "num_trades",
        "win_rate",
        "avg_win_pct",
        "avg_loss_pct",
        "profit_factor",
        "expectancy_pct",
        "sharpe",
        "sortino",
        "calmar",
    }
    assert set(result.keys()) == expected_keys
    for key, value in result.items():
        assert isinstance(value, (int, float)), f"{key} was {type(value)}"
        assert not (isinstance(value, float) and math.isnan(value)), f"{key} is NaN"
