"""Unit tests for the pure helper functions in strategy_optimizer.py.

strategy_optimizer.py only imports the standard library at module scope
(argparse, csv, json, os, random, subprocess, sys, time, uuid, pathlib) and
performs no file or network I/O when merely imported — main() only runs
under ``if __name__ == "__main__":`` — so it's safe to import directly,
unlike backtest.py (see tests/test_backtest_metrics.py for why that one is
handled differently).
"""

import json

import pytest

import strategy_optimizer as optimizer


# ---------------------------------------------------------------------------
# config_id
# ---------------------------------------------------------------------------


def test_config_id_is_deterministic_json(sample_config):
    result = optimizer.config_id(sample_config)
    assert result == optimizer.config_id(sample_config)
    assert json.loads(result) == {k: sample_config.get(k) for k in optimizer.CONFIG_KEYS}


def test_config_id_ignores_key_order():
    cfg_a = {"BAR_TIMEFRAME": "5m", "MIN_SIGNAL_CONFIDENCE": 0.4}
    cfg_b = {"MIN_SIGNAL_CONFIDENCE": 0.4, "BAR_TIMEFRAME": "5m"}
    assert optimizer.config_id(cfg_a) == optimizer.config_id(cfg_b)


def test_config_id_ignores_keys_outside_config_keys():
    base = {"BAR_TIMEFRAME": "5m"}
    with_extra = {"BAR_TIMEFRAME": "5m", "some_unrelated_field": "ignored"}
    assert optimizer.config_id(base) == optimizer.config_id(with_extra)


def test_config_id_changes_when_a_config_value_changes():
    cfg_a = {"BAR_TIMEFRAME": "5m", "MIN_SIGNAL_CONFIDENCE": 0.4}
    cfg_b = {"BAR_TIMEFRAME": "5m", "MIN_SIGNAL_CONFIDENCE": 0.5}
    assert optimizer.config_id(cfg_a) != optimizer.config_id(cfg_b)


def test_config_id_missing_keys_become_null():
    result = optimizer.config_id({})
    payload = json.loads(result)
    assert set(payload.keys()) == set(optimizer.CONFIG_KEYS)
    assert all(v is None for v in payload.values())


# ---------------------------------------------------------------------------
# calculate_metrics
# ---------------------------------------------------------------------------


def test_calculate_metrics_no_windows_returns_empty_dict():
    assert optimizer.calculate_metrics({}) == {}
    assert optimizer.calculate_metrics({"windows": []}) == {}


def test_calculate_metrics_averages_single_window():
    summary = {
        "windows": [
            {
                "total_return_pct_mean": 10.0,
                "max_drawdown_pct_mean": -5.0,
                "sharpe_mean": 1.0,
                "sortino_mean": 1.5,
                "calmar_mean": 2.0,
                "profit_factor_mean": 1.8,
                "num_trades_mean": 30,
            }
        ]
    }
    result = optimizer.calculate_metrics(summary)

    assert result["return"] == pytest.approx(10.0)
    assert result["drawdown"] == pytest.approx(-5.0)
    assert result["sharpe"] == pytest.approx(1.0)
    assert result["sortino"] == pytest.approx(1.5)
    assert result["calmar"] == pytest.approx(2.0)
    assert result["profit_factor"] == pytest.approx(1.8)
    assert result["trades"] == pytest.approx(30)

    # score = 1.00*sharpe + 0.40*sortino + 0.20*calmar + 0.06*ret
    #        + 0.12*profit_factor - 0.10*abs(dd)
    expected_score = (
        1.00 * 1.0 + 0.40 * 1.5 + 0.20 * 2.0 + 0.06 * 10.0 + 0.12 * 1.8 - 0.10 * 5.0
    )
    assert result["score"] == pytest.approx(expected_score)


def test_calculate_metrics_averages_across_multiple_windows():
    summary = {
        "windows": [
            {"sharpe_mean": 1.0, "sortino_mean": 1.0, "calmar_mean": 1.0,
             "total_return_pct_mean": 0.0, "profit_factor_mean": 1.0,
             "max_drawdown_pct_mean": 0.0, "num_trades_mean": 10},
            {"sharpe_mean": 3.0, "sortino_mean": 3.0, "calmar_mean": 3.0,
             "total_return_pct_mean": 0.0, "profit_factor_mean": 3.0,
             "max_drawdown_pct_mean": 0.0, "num_trades_mean": 20},
        ]
    }
    result = optimizer.calculate_metrics(summary)

    assert result["sharpe"] == pytest.approx(2.0)  # (1.0 + 3.0) / 2
    assert result["sortino"] == pytest.approx(2.0)
    assert result["calmar"] == pytest.approx(2.0)
    assert result["profit_factor"] == pytest.approx(2.0)
    assert result["trades"] == pytest.approx(15.0)


def test_calculate_metrics_missing_fields_default_to_zero():
    # A row missing every metric key should contribute 0.0 via as_float's
    # default rather than raising.
    summary = {"windows": [{}]}
    result = optimizer.calculate_metrics(summary)
    assert result["sharpe"] == 0.0
    assert result["score"] == 0.0


def test_calculate_metrics_returns_all_expected_keys():
    summary = {
        "windows": [
            {"sharpe_mean": 0.5, "sortino_mean": 0.5, "calmar_mean": 0.5,
             "total_return_pct_mean": 1.0, "profit_factor_mean": 1.2,
             "max_drawdown_pct_mean": -2.0, "num_trades_mean": 5},
        ]
    }
    result = optimizer.calculate_metrics(summary)
    assert set(result.keys()) == {
        "return", "drawdown", "sharpe", "sortino", "calmar",
        "profit_factor", "trades", "score",
    }


# ---------------------------------------------------------------------------
# as_float (small helper used throughout calculate_metrics)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, default, expected",
    [
        ("1.5", 0.0, 1.5),
        (None, 2.5, 2.5),
        ("not-a-number", 9.0, 9.0),
        (3, 0.0, 3.0),
    ],
)
def test_as_float(value, default, expected):
    assert optimizer.as_float(value, default) == pytest.approx(expected)
