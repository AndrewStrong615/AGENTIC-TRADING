import os
import sys
import csv
from collections import defaultdict

import numpy as np

DEFAULT_PATH = os.path.join("trade_data", "trade_log.csv")


def load_rows(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            try:
                r["pnl_pct"] = float(r["pnl_pct"])
                r["entry_batch_weight"] = float(r["entry_batch_weight"]) if r["entry_batch_weight"] not in (None, "", "None") else None
                r["entry_online_weight"] = float(r["entry_online_weight"]) if r["entry_online_weight"] not in (None, "", "None") else None
            except (TypeError, ValueError):
                continue
            if r["entry_batch_weight"] is None or r["entry_online_weight"] is None:
                continue
            rows.append(r)
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
    summarize(rows)