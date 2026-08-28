"""Unit tests for the CSV-parsing logic in analyze_trade_log.py and
combine_logs.py.

All tests write small synthetic CSV fixtures to pytest's ``tmp_path`` —
nothing here reads from the real trade_data/ directory, so the suite is
hermetic and passes on a clean checkout with no trade history.

analyze_trade_log.py exposes a plain ``load_rows(path)`` function with no
import-time side effects, so it is imported directly.

combine_logs.py, by contrast, is a plain top-level script (no functions, no
``if __name__ == "__main__":`` guard) that globs ``trade_data/*.csv``
relative to the current working directory and writes a combined CSV as soon
as it is imported/run. Importing it directly would execute that logic
against whatever the current working directory happens to be, which is not
hermetic. Instead we run it as a subprocess with ``cwd`` pointed at a tmp_path
containing only synthetic fixture files, which exercises the exact same
CSV-parsing/combining code path without touching the real repo.
"""

import csv
import subprocess
import sys
from pathlib import Path

import pytest

import analyze_trade_log as atl

REPO_ROOT = Path(__file__).resolve().parent.parent
COMBINE_LOGS_PY = REPO_ROOT / "combine_logs.py"


# ---------------------------------------------------------------------------
# analyze_trade_log.load_rows
# ---------------------------------------------------------------------------


def _write_trade_log(path, rows, fieldnames=None):
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_load_rows_parses_valid_csv(tmp_path):
    csv_path = tmp_path / "trade_log.csv"
    _write_trade_log(
        csv_path,
        [
            {
                "symbol": "AAPL",
                "pnl_pct": "1.25",
                "entry_batch_weight": "0.6",
                "entry_online_weight": "0.4",
                "exit_timestamp": "2026-01-02T10:00:00",
            },
            {
                "symbol": "MSFT",
                "pnl_pct": "-0.5",
                "entry_batch_weight": "0.3",
                "entry_online_weight": "0.7",
                "exit_timestamp": "2026-01-02T11:00:00",
            },
        ],
    )

    rows = atl.load_rows(str(csv_path))

    assert len(rows) == 2
    assert rows[0]["symbol"] == "AAPL"
    assert rows[0]["pnl_pct"] == pytest.approx(1.25)
    assert rows[0]["entry_batch_weight"] == pytest.approx(0.6)
    assert rows[1]["entry_online_weight"] == pytest.approx(0.7)


def test_load_rows_skips_rows_missing_ensemble_weights(tmp_path):
    csv_path = tmp_path / "trade_log.csv"
    _write_trade_log(
        csv_path,
        [
            {
                "symbol": "AAPL",
                "pnl_pct": "1.0",
                "entry_batch_weight": "",
                "entry_online_weight": "0.4",
                "exit_timestamp": "2026-01-02T10:00:00",
            },
            {
                "symbol": "MSFT",
                "pnl_pct": "1.0",
                "entry_batch_weight": "None",
                "entry_online_weight": "0.4",
                "exit_timestamp": "2026-01-02T10:00:00",
            },
            {
                "symbol": "GOOGL",
                "pnl_pct": "1.0",
                "entry_batch_weight": "0.5",
                "entry_online_weight": "0.5",
                "exit_timestamp": "2026-01-02T10:00:00",
            },
        ],
    )

    rows = atl.load_rows(str(csv_path))

    assert len(rows) == 1
    assert rows[0]["symbol"] == "GOOGL"


def test_load_rows_skips_rows_with_unparseable_pnl(tmp_path):
    csv_path = tmp_path / "trade_log.csv"
    _write_trade_log(
        csv_path,
        [
            {
                "symbol": "AAPL",
                "pnl_pct": "not-a-number",
                "entry_batch_weight": "0.5",
                "entry_online_weight": "0.5",
                "exit_timestamp": "2026-01-02T10:00:00",
            },
            {
                "symbol": "MSFT",
                "pnl_pct": "2.0",
                "entry_batch_weight": "0.5",
                "entry_online_weight": "0.5",
                "exit_timestamp": "2026-01-02T10:00:00",
            },
        ],
    )

    rows = atl.load_rows(str(csv_path))

    assert len(rows) == 1
    assert rows[0]["symbol"] == "MSFT"


def test_load_rows_empty_file_returns_empty_list(tmp_path):
    csv_path = tmp_path / "trade_log.csv"
    _write_trade_log(
        csv_path,
        [],
        fieldnames=["symbol", "pnl_pct", "entry_batch_weight", "entry_online_weight", "exit_timestamp"],
    )

    rows = atl.load_rows(str(csv_path))
    assert rows == []


def test_load_rows_missing_file_raises_file_not_found(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        atl.load_rows(str(missing_path))


# ---------------------------------------------------------------------------
# combine_logs.py (run as a subprocess against a synthetic trade_data dir)
# ---------------------------------------------------------------------------


def test_combine_logs_combines_multiple_csvs(tmp_path):
    data_dir = tmp_path / "trade_data"
    data_dir.mkdir()

    _write_trade_log(
        data_dir / "backtest_trade_log_AAPL.csv",
        [
            {"symbol": "AAPL", "pnl_pct": "1.0"},
            {"symbol": "AAPL", "pnl_pct": "-0.5"},
        ],
    )
    _write_trade_log(
        data_dir / "backtest_trade_log_MSFT.csv",
        [
            {"symbol": "MSFT", "pnl_pct": "2.0"},
        ],
    )

    result = subprocess.run(
        [sys.executable, str(COMBINE_LOGS_PY)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr

    combined_path = data_dir / "backtest_trade_log.csv"
    assert combined_path.exists()

    with combined_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 3
    symbols = sorted(r["symbol"] for r in rows)
    assert symbols == ["AAPL", "AAPL", "MSFT"]
    assert "Combined 2 files into 3 total trades" in result.stdout


def test_combine_logs_no_matching_files_reports_none_found(tmp_path):
    data_dir = tmp_path / "trade_data"
    data_dir.mkdir()

    result = subprocess.run(
        [sys.executable, str(COMBINE_LOGS_PY)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "No backtest_trade_log_*.csv files found" in result.stdout
    assert not (data_dir / "backtest_trade_log.csv").exists()


def test_combine_logs_skips_unparseable_file_but_combines_the_rest(tmp_path):
    data_dir = tmp_path / "trade_data"
    data_dir.mkdir()

    _write_trade_log(
        data_dir / "backtest_trade_log_AAPL.csv",
        [{"symbol": "AAPL", "pnl_pct": "1.0"}],
    )
    # A completely empty file can't be parsed at all ("No columns to parse
    # from file") — combine_logs.py's per-file try/except should skip it and
    # still combine the remaining, valid file.
    (data_dir / "backtest_trade_log_BAD.csv").write_text("", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(COMBINE_LOGS_PY)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "SKIPPED" in result.stdout
    combined_path = data_dir / "backtest_trade_log.csv"
    assert combined_path.exists()
    with combined_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["symbol"] == "AAPL"
