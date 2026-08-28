"""Tests for strategy_config.py's config loading/env-application logic.

CONFIG_PATH is always monkeypatched to a throwaway file under tmp_path, and
any environment variables the tests set are restored automatically by
monkeypatch -- the real trade_data/best_strategy_config.json and the
process's real environment are never touched.
"""
import json
import os

import strategy_config as sc


def test_load_best_config_missing_file_returns_empty_dict(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "CONFIG_PATH", tmp_path / "missing.json")
    assert sc.load_best_config() == {}


def test_load_best_config_filters_to_known_keys_only(tmp_path, monkeypatch):
    path = tmp_path / "best_strategy_config.json"
    path.write_text(json.dumps({"config": {
        "BAR_TIMEFRAME": "5m",
        "MIN_SIGNAL_CONFIDENCE": 0.42,
        "totally_unknown_key": "ignored",
    }}))
    monkeypatch.setattr(sc, "CONFIG_PATH", path)
    assert sc.load_best_config() == {"BAR_TIMEFRAME": "5m", "MIN_SIGNAL_CONFIDENCE": 0.42}


def test_load_best_config_accepts_a_bare_config_dict(tmp_path, monkeypatch):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"BAR_TIMEFRAME": "10m"}))
    monkeypatch.setattr(sc, "CONFIG_PATH", path)
    assert sc.load_best_config() == {"BAR_TIMEFRAME": "10m"}


def test_load_best_config_corrupt_json_returns_empty_dict(tmp_path, monkeypatch):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    monkeypatch.setattr(sc, "CONFIG_PATH", path)
    assert sc.load_best_config() == {}


def test_apply_best_config_to_env_fills_only_unset_keys(tmp_path, monkeypatch):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"config": {"BAR_TIMEFRAME": "5m", "MIN_ATR_PCT": 0.05}}))
    monkeypatch.setattr(sc, "CONFIG_PATH", path)
    monkeypatch.delenv("BAR_TIMEFRAME", raising=False)
    monkeypatch.setenv("MIN_ATR_PCT", "0.99")  # explicit env should win over saved config

    applied = sc.apply_best_config_to_env()

    assert applied == {"BAR_TIMEFRAME": "5m"}
    assert os.environ["BAR_TIMEFRAME"] == "5m"
    assert os.environ["MIN_ATR_PCT"] == "0.99"
