"""
analyze_trade_log.py — Reads a trade log CSV and reports how the Layer 2
ensemble's blend weights evolved, and which sub-model tended to be right
when it carried more of the weight.


Defaults to trade_data/trade_log.csv if no path is given.
"""

import os
import sys
import csv
import math
from collections import defaultdict

import numpy as np

DEFAULT_PATH = os.path.join("trade_data", "trade_log.csv")

# Columns the rest of this script relies on being present in the header.
REQUIRED_COLUMNS = (
    "exit_timestamp",
    "symbol",
    "pnl_pct",
    "entry_batch_weight",
    "entry_online_weight",
)


def _parse_float(value):
    """Parse a CSV cell as a float, treating blank/None/"None" as missing.

    Returns None for missing values. Raises ValueError for anything else
    that isn't a finite number (including "nan"/"inf" strings, which
    `float()` happily accepts but which would silently poison every
    average downstream).
    """
    if value in (None, "", "None"):
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite value: {value!r}")
    return parsed


def load_rows(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            # Completely empty file (e.g. touched but never written to) —
            # treat the same as "no trades yet" rather than an error.
            return []

        missing_cols = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing_cols:
            raise ValueError(
                f"'{path}' is missing required column(s): {', '.join(missing_cols)} "
                f"— found columns: {', '.join(reader.fieldnames)}"
            )

        rows = []
        skipped_ragged = 0
        skipped_other = 0
        for r in reader:
            # csv.DictReader stuffs any columns beyond the header into a
            # list under the None key. That happens when a row was written
            # with more fields than the header declares (e.g. the log's
            # schema grew new columns mid-file without the header being
            # rewritten). In that case every field from the shifted point
            # onward is misaligned — e.g. entry_batch_weight/
            # entry_online_weight would actually hold some other column's
            # value, which can still parse as a valid float and silently
            # corrupt the stats below. Skip rows like this outright rather
            # than trusting the (wrong) positional mapping.
            if r.get(None) is not None:
                skipped_ragged += 1
                continue

            try:
                pnl_pct = _parse_float(r["pnl_pct"])
                batch_weight = _parse_float(r["entry_batch_weight"])
                online_weight = _parse_float(r["entry_online_weight"])
            except (TypeError, ValueError):
                skipped_other += 1
                continue

            if pnl_pct is None or batch_weight is None or online_weight is None:
                skipped_other += 1
                continue

            if not r.get("symbol") or not r.get("exit_timestamp"):
                skipped_other += 1
                continue

            r["pnl_pct"] = pnl_pct
            r["entry_batch_weight"] = batch_weight
            r["entry_online_weight"] = online_weight
            rows.append(r)

        if skipped_ragged:
            print(
                f"Warning: skipped {skipped_ragged} row(s) with more columns than the "
                f"CSV header declares (likely a schema change mid-file) — their fields "
                f"could not be trusted."
            )
        if skipped_other:
            print(
                f"Warning: skipped {skipped_other} row(s) with missing/invalid "
                f"pnl_pct, ensemble weights, symbol, or exit_timestamp."
            )

        return rows


def summarize(rows):
    if not rows:
        print("No rows with recorded ensemble weights found.")
        return

    by_symbol = defaultdict(list)
    for r in rows:
        by_symbol[r["symbol"]].append(r)

    print(f"Loaded {len(rows)} closed trades with ensemble weight data.\n")

    print(f"{'Symbol':<8}{'Trades':<8}{'Avg batch w':<14}{'Avg online w':<14}"
          f"{'Win% when batch-led':<22}{'Win% when online-led':<22}")
    for symbol, trades in sorted(by_symbol.items()):
        avg_batch = np.mean([t["entry_batch_weight"] for t in trades])
        avg_online = np.mean([t["entry_online_weight"] for t in trades])

        batch_led = [t for t in trades if t["entry_batch_weight"] > t["entry_online_weight"]]
        online_led = [t for t in trades if t["entry_online_weight"] >= t["entry_batch_weight"]]

        batch_win_pct = 100 * np.mean([t["pnl_pct"] > 0 for t in batch_led]) if batch_led else float("nan")
        online_win_pct = 100 * np.mean([t["pnl_pct"] > 0 for t in online_led]) if online_led else float("nan")

        print(f"{symbol:<8}{len(trades):<8}{avg_batch:<14.3f}{avg_online:<14.3f}"
              f"{batch_win_pct:<22.1f}{online_win_pct:<22.1f}")

    print("\nRecent trend (last 20 trades overall, oldest -> newest):")
    recent = sorted(rows, key=lambda r: r["exit_timestamp"])[-20:]
    for t in recent:
        tag = "BATCH-led" if t["entry_batch_weight"] > t["entry_online_weight"] else "ONLINE-led"
        outcome = "WIN " if t["pnl_pct"] > 0 else "loss"
        print(f"  {t['exit_timestamp'][:19]}  {t['symbol']:<6} {tag:<11} "
              f"batch={t['entry_batch_weight']:.2f} online={t['entry_online_weight']:.2f}  "
              f"{outcome} {t['pnl_pct']:+.3f}%")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    try:
        rows = load_rows(path)
    except FileNotFoundError:
        print(f"No trade log found at '{path}' yet — run the trader or backtest first.")
        sys.exit(1)
    except (OSError, ValueError) as e:
        print(f"Could not read trade log at '{path}': {e}")
        sys.exit(1)
    summarize(rows)
