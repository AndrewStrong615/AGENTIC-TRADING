"""Tests for analyze_trade_log.py's CSV parsing/aggregation logic.

Every test writes its own throwaway CSV under tmp_path -- the real
trade_data/trade_log.csv is never read.
"""
import csv

import pytest

import analyze_trade_log as atl

FIELDNAMES = ["symbol", "exit_timestamp", "pnl_pct", "entry_batch_weight", "entry_online_weight"]


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def test_load_rows_parses_numeric_fields(tmp_path):
    path = tmp_path / "trade_log.csv"
    _write_csv(path, [
        {"symbol": "AAPL", "exit_timestamp": "2026-01-01T00:00:00", "pnl_pct": "1.5",
         "entry_batch_weight": "0.6", "entry_online_weight": "0.4"},
    ])
    rows = atl.load_rows(path)
    assert len(rows) == 1
    assert rows[0]["pnl_pct"] == 1.5
    assert rows[0]["entry_batch_weight"] == 0.6


def test_load_rows_skips_rows_with_missing_weight_data(tmp_path):
    path = tmp_path / "trade_log.csv"
    _write_csv(path, [
        {"symbol": "AAPL", "exit_timestamp": "t1", "pnl_pct": "1.0",
         "entry_batch_weight": "", "entry_online_weight": "0.4"},
        {"symbol": "MSFT", "exit_timestamp": "t2", "pnl_pct": "2.0",
         "entry_batch_weight": "0.5", "entry_online_weight": "0.5"},
    ])
    rows = atl.load_rows(path)
    assert len(rows) == 1
    assert rows[0]["symbol"] == "MSFT"


def test_load_rows_skips_unparseable_pnl(tmp_path):
    path = tmp_path / "trade_log.csv"
    _write_csv(path, [
        {"symbol": "AAPL", "exit_timestamp": "t1", "pnl_pct": "not-a-number",
         "entry_batch_weight": "0.5", "entry_online_weight": "0.5"},
    ])
    assert atl.load_rows(path) == []


def test_load_rows_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        atl.load_rows(tmp_path / "does_not_exist.csv")


def test_summarize_with_no_rows_prints_a_clear_message(capsys):
    atl.summarize([])
    assert "No rows" in capsys.readouterr().out


def test_summarize_reports_per_symbol_stats_and_recent_trend(capsys):
    rows = [
        {"symbol": "AAPL", "exit_timestamp": "2026-01-01T00:00:00", "pnl_pct": 1.0,
         "entry_batch_weight": 0.7, "entry_online_weight": 0.3},
        {"symbol": "AAPL", "exit_timestamp": "2026-01-02T00:00:00", "pnl_pct": -1.0,
         "entry_batch_weight": 0.3, "entry_online_weight": 0.7},
    ]
    atl.summarize(rows)
    out = capsys.readouterr().out
    assert "Loaded 2 closed trades" in out
    assert "AAPL" in out
    assert "Recent trend" in out
