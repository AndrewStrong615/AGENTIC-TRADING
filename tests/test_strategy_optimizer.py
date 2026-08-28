"""Tests for the pure config/scoring/history logic in strategy_optimizer.py.

Nothing here runs a backtest, spawns a subprocess, or touches the real
trade_data/ directory -- history/config paths are always monkeypatched to
tmp_path.
"""
import json
import random

import pytest

import strategy_optimizer as opt


# ---------------------------------------------------------------------------
# as_float / config_id
# ---------------------------------------------------------------------------

def test_as_float_parses_valid_values():
    assert opt.as_float("1.5") == 1.5
    assert opt.as_float(3) == 3.0


def test_as_float_invalid_input_falls_back_to_default():
    assert opt.as_float(None, default=2.0) == 2.0
    assert opt.as_float("not-a-number", default=-1.0) == -1.0


def test_config_id_is_order_independent():
    cfg_a = {"BAR_TIMEFRAME": "5m", "MIN_SIGNAL_CONFIDENCE": 0.4}
    cfg_b = {"MIN_SIGNAL_CONFIDENCE": 0.4, "BAR_TIMEFRAME": "5m"}
    assert opt.config_id(cfg_a) == opt.config_id(cfg_b)


def test_config_id_differs_for_different_configs():
    assert opt.config_id({"BAR_TIMEFRAME": "5m"}) != opt.config_id({"BAR_TIMEFRAME": "10m"})


def test_config_id_ignores_unknown_keys():
    cfg_a = {"BAR_TIMEFRAME": "5m", "some_unrelated_field": "ignored"}
    cfg_b = {"BAR_TIMEFRAME": "5m"}
    assert opt.config_id(cfg_a) == opt.config_id(cfg_b)


# ---------------------------------------------------------------------------
# calculate_metrics
# ---------------------------------------------------------------------------

def _window(**overrides):
    row = {
        "total_return_pct_mean": 5.0,
        "max_drawdown_pct_mean": -3.0,
        "sharpe_mean": 1.0,
        "sortino_mean": 1.2,
        "calmar_mean": 0.8,
        "profit_factor_mean": 1.5,
        "num_trades_mean": 30,
    }
    row.update(overrides)
    return row


def test_calculate_metrics_empty_windows_returns_empty_dict():
    assert opt.calculate_metrics({"windows": []}) == {}
    assert opt.calculate_metrics({}) == {}


def test_calculate_metrics_averages_across_windows_and_computes_score():
    summary = {"windows": [_window(), _window(sharpe_mean=2.0)]}
    metrics = opt.calculate_metrics(summary)
    assert metrics["sharpe"] == pytest.approx(1.5)
    assert metrics["trades"] == pytest.approx(30.0)
    expected_score = (
        1.00 * metrics["sharpe"] + 0.40 * metrics["sortino"] + 0.20 * metrics["calmar"]
        + 0.06 * metrics["return"] + 0.12 * metrics["profit_factor"] - 0.10 * abs(metrics["drawdown"])
    )
    assert metrics["score"] == pytest.approx(expected_score)


# ---------------------------------------------------------------------------
# safe_holdout / evaluate_candidate_vs_current
# ---------------------------------------------------------------------------

def _good_result(score=1.0):
    return {"return": 1.0, "drawdown": -5.0, "sharpe": 0.5, "trades": 25, "score": score}


def test_safe_holdout_accepts_good_result():
    assert opt.safe_holdout(_good_result()) is True


def test_safe_holdout_empty_result_is_false():
    assert opt.safe_holdout({}) is False


@pytest.mark.parametrize("overrides", [
    {"return": -1.0},     # below MIN_RETURN
    {"drawdown": -20.0},  # worse than MAX_DRAWDOWN
    {"sharpe": 0.1},      # below MIN_SHARPE
    {"trades": 5},        # below MIN_TRADES
])
def test_safe_holdout_rejects_each_failing_threshold(overrides):
    result = _good_result()
    result.update(overrides)
    assert opt.safe_holdout(result) is False


def test_evaluate_candidate_vs_current_requires_matching_holdout_counts():
    result = opt.evaluate_candidate_vs_current([_good_result()], [])
    assert result["eligible"] is False


def test_evaluate_candidate_vs_current_promotes_clear_winner():
    candidate = [_good_result(1.0), _good_result(1.0)]
    current = [_good_result(0.5), _good_result(0.5)]
    result = opt.evaluate_candidate_vs_current(candidate, current)
    assert result["eligible"] is True
    assert result["beats"] == 2
    assert result["avg_gain"] == pytest.approx(0.5)


def test_evaluate_candidate_vs_current_rejects_gain_below_promotion_margin():
    candidate = [_good_result(0.51), _good_result(0.51)]
    current = [_good_result(0.50), _good_result(0.50)]
    result = opt.evaluate_candidate_vs_current(candidate, current)
    # avg_gain (0.01) < PROMOTION_MARGIN (0.03)
    assert result["eligible"] is False


def test_evaluate_candidate_vs_current_rejects_candidate_failing_safety_gates():
    unsafe = [dict(_good_result(5.0), **{"return": -5.0})] * 2
    current = [_good_result(0.5)] * 2
    result = opt.evaluate_candidate_vs_current(unsafe, current)
    assert result["eligible"] is False


# ---------------------------------------------------------------------------
# candidate generation
# ---------------------------------------------------------------------------

def test_make_candidate_from_current_stays_within_bounds():
    rng = random.Random(42)
    current = {"MIN_SIGNAL_CONFIDENCE": 0.40, "MIN_META_WIN_PROB": 0.62}
    for _ in range(50):
        cand = opt.make_candidate_from_current(current, rng)
        assert 0.30 <= cand["MIN_SIGNAL_CONFIDENCE"] <= 0.55
        assert 0.55 <= cand["MIN_META_WIN_PROB"] <= 0.72
        assert cand["BAR_TIMEFRAME"] in {"5m", "10m", "15m"}


def test_generate_candidates_produces_n_unique_configs_not_in_history():
    history = {opt.config_id({"BAR_TIMEFRAME": "5m"}): {}}
    candidates = opt.generate_candidates(10, current={}, history=history, seed=7)
    assert len(candidates) == 10
    ids = [opt.config_id(c) for c in candidates]
    assert len(set(ids)) == 10


def test_generate_candidates_is_deterministic_given_a_seed():
    a = opt.generate_candidates(5, current={}, history={}, seed=123)
    b = opt.generate_candidates(5, current={}, history={}, seed=123)
    assert a == b


# ---------------------------------------------------------------------------
# config/history persistence (all paths monkeypatched into tmp_path)
# ---------------------------------------------------------------------------

def test_load_current_config_missing_file_returns_empty_dict(tmp_path, monkeypatch):
    monkeypatch.setattr(opt, "BEST_PATH", tmp_path / "missing.json")
    assert opt.load_current_config() == {}


def test_load_current_config_reads_saved_config(tmp_path, monkeypatch):
    path = tmp_path / "best_strategy_config.json"
    path.write_text(json.dumps({"config": {"BAR_TIMEFRAME": "5m"}}))
    monkeypatch.setattr(opt, "BEST_PATH", path)
    assert opt.load_current_config() == {"BAR_TIMEFRAME": "5m"}


def test_load_current_config_handles_corrupt_json(tmp_path, monkeypatch):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json")
    monkeypatch.setattr(opt, "BEST_PATH", path)
    assert opt.load_current_config() == {}


def test_load_and_save_history_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(opt, "HISTORY_PATH", tmp_path / "optimizer_history.csv")
    monkeypatch.setattr(opt, "DATA_DIR", tmp_path)

    assert opt.load_history() == {}

    row = {
        "tested_at": "2026-08-28T00:00:00Z", "seed": 1, "config_id": "abc123",
        "run_id": "run-1", "returncode": 0, "return": 1.0, "drawdown": -2.0,
        "sharpe": 0.5, "sortino": 0.6, "calmar": 0.4, "profit_factor": 1.2,
        "trades": 30, "score": 1.5,
    }
    opt.save_history([row])
    history = opt.load_history()
    assert "abc123" in history
    assert history["abc123"]["run_id"] == "run-1"


def test_save_history_does_not_duplicate_existing_config_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(opt, "HISTORY_PATH", tmp_path / "optimizer_history.csv")
    monkeypatch.setattr(opt, "DATA_DIR", tmp_path)

    row = {"tested_at": "t", "seed": 1, "config_id": "dupe", "run_id": "run-1",
           "returncode": 0, "return": 1.0, "drawdown": -1.0, "sharpe": 0.5,
           "sortino": 0.5, "calmar": 0.5, "profit_factor": 1.0, "trades": 20, "score": 1.0}
    opt.save_history([row])
    opt.save_history([dict(row, run_id="run-2")])  # same config_id -> should be skipped
    history = opt.load_history()
    assert history["dupe"]["run_id"] == "run-1"
