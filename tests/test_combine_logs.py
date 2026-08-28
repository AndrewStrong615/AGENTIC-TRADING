"""Tests for combine_logs.py.

combine_logs.py is a top-level script (no functions to import), so these
tests execute it with runpy inside a tmp_path used as the working directory
-- it only ever reads/writes a throwaway trade_data/ folder under tmp_path,
never the real one.
"""
import runpy
from pathlib import Path

import pandas as pd

COMBINE_LOGS_PATH = Path(__file__).resolve().parents[1] / "combine_logs.py"


def _run_combine_logs():
    runpy.run_path(str(COMBINE_LOGS_PATH), run_name="__combine_logs_test__")


def test_combine_logs_merges_multiple_csv_files(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "trade_data"
    data_dir.mkdir()
    pd.DataFrame({"symbol": ["AAPL"], "pnl_pct": [1.0]}).to_csv(
        data_dir / "backtest_trade_log_AAPL_0.csv", index=False)
    pd.DataFrame({"symbol": ["MSFT"], "pnl_pct": [2.0]}).to_csv(
        data_dir / "backtest_trade_log_MSFT_0.csv", index=False)

    _run_combine_logs()

    out_path = data_dir / "backtest_trade_log.csv"
    assert out_path.exists()
    combined = pd.read_csv(out_path)
    assert len(combined) == 2
    assert set(combined["symbol"]) == {"AAPL", "MSFT"}
    assert "Combined 2 files into 2 total trades" in capsys.readouterr().out


def test_combine_logs_reports_when_no_files_found(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _run_combine_logs()
    assert "No backtest_trade_log_*.csv files found" in capsys.readouterr().out
    # no output file should be created
    assert not (tmp_path / "trade_data" / "backtest_trade_log.csv").exists()
