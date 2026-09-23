"""
analyze_trade_log.py — Reads a trade log CSV and reports how the Layer 2
ensemble's blend weights evolved, and which sub-model tended to be right
when it carried more of the weight.


Defaults to trade_data/trade_log.csv if no path is given.

Pass --output-dir <dir> to additionally save matplotlib charts (expert-weight
evolution over time, win rate by regime, PnL histogram) as PNGs into that
directory, alongside the usual console text summary. Without --output-dir,
behavior is unchanged from before (console-only, no matplotlib import).
"""

import os
import sys
import csv
import argparse
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


def save_charts(rows, output_dir):
    """Save expert-weight evolution, win-rate-by-regime, and PnL histogram
    charts as PNGs into output_dir, reusing the same fields already parsed
    by load_rows() (plus the raw regime strings DictReader gives us for
    free). No-op (with a message) if there's nothing to plot."""
    if not rows:
        print("No rows with recorded ensemble weights found — skipping charts.")
        return

    # Deferred import so a plain console run pays no matplotlib import cost.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # 1. Expert-weight evolution over time.
    ordered = sorted(rows, key=lambda r: r["exit_timestamp"])
    batch_weights = [r["entry_batch_weight"] for r in ordered]
    online_weights = [r["entry_online_weight"] for r in ordered]
    x = range(len(ordered))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, batch_weights, label="batch weight", linewidth=1)
    ax.plot(x, online_weights, label="online weight", linewidth=1)
    ax.set_xlabel("Trade # (chronological)")
    ax.set_ylabel("Blend weight")
    ax.set_title("Expert-weight evolution over time")
    ax.legend()
    fig.tight_layout()
    weight_path = os.path.join(output_dir, "weight_evolution.png")
    fig.savefig(weight_path)
    plt.close(fig)

    # 2. Win rate by regime (entry_hmm_regime, e.g. bull/bear/neutral).
    by_regime = defaultdict(list)
    for r in rows:
        regime = r.get("entry_hmm_regime") or "unknown"
        by_regime[regime].append(r)

    regimes = sorted(by_regime.keys())
    win_pcts = [
        100 * np.mean([t["pnl_pct"] > 0 for t in by_regime[regime]])
        for regime in regimes
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(regimes, win_pcts)
    ax.set_xlabel("Regime (entry_hmm_regime)")
    ax.set_ylabel("Win rate (%)")
    ax.set_title("Win rate by regime")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    regime_path = os.path.join(output_dir, "win_rate_by_regime.png")
    fig.savefig(regime_path)
    plt.close(fig)

    # 3. PnL histogram.
    pnl_values = [r["pnl_pct"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pnl_values, bins=40)
    ax.set_xlabel("PnL (%)")
    ax.set_ylabel("Trade count")
    ax.set_title("PnL distribution")
    fig.tight_layout()
    pnl_path = os.path.join(output_dir, "pnl_histogram.png")
    fig.savefig(pnl_path)
    plt.close(fig)

    print(f"\nSaved charts to '{output_dir}':")
    print(f"  {weight_path}")
    print(f"  {regime_path}")
    print(f"  {pnl_path}")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Analyze a trade log CSV: ensemble blend-weight trends "
        "and win rates, with optional chart PNGs."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_PATH,
        help=f"Path to trade log CSV (default: {DEFAULT_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="If given, also save weight-evolution, win-rate-by-regime, and "
        "PnL-histogram charts as PNGs into this directory.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    try:
        rows = load_rows(args.path)
    except FileNotFoundError:
        print(f"No trade log found at '{args.path}' yet — run the trader or backtest first.")
        sys.exit(1)
    summarize(rows)
    if args.output_dir:
        save_charts(rows, args.output_dir)