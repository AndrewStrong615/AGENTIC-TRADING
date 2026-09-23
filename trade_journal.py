"""
trade_journal.py — Turns a trade log CSV (trade_data/trade_log.csv, or a
backtest run's trade CSV) into a readable, per-trade Markdown "journal".

The raw trade log is wide (one row per event, lots of columns, mixed
open/closed rows) and hard to scan by eye. This script reads it, keeps only
*closed* trades (rows with both an exit_timestamp and exit_price), and
renders them grouped by symbol and then by calendar day, showing entry/exit
price, PnL, exit reason, and whatever signal-context fields (confidence,
regime, expert weights) actually happen to be present on each row.

This script is READ-ONLY with respect to the trade log: it never opens the
input CSV for writing and never modifies trade_data/trade_log.csv. It only
writes the journal file given by --output.

Usage:
    python3 trade_journal.py [input_csv] [--output OUTPUT_PATH]

Defaults:
    input_csv   trade_data/trade_log.csv
    --output    trade_data/trade_journal.md
"""

import argparse
import csv
import os
import sys
from collections import defaultdict, OrderedDict

DEFAULT_INPUT = os.path.join("trade_data", "trade_log.csv")
DEFAULT_OUTPUT = os.path.join("trade_data", "trade_journal.md")

# Known signal-context columns the bot may write on a trade row (see
# Tradingcore.py's record_trade_outcome / record_open_order / the
# reconciliation path in refresh_positions). Not every row — and not every
# version of trade_log.csv — has all of these; we only render the ones that
# are actually present with a non-empty value on a given row.
CONTEXT_FIELDS = [
    ("entry_confidence", "confidence"),
    ("entry_direction", "direction"),
    ("entry_trend_change_pct", "trend chg %"),
    ("entry_volatility_regime", "vol regime"),
    ("entry_trend_regime", "trend regime"),
    ("entry_hmm_regime", "hmm regime"),
    ("entry_hmm_regime_prob", "hmm prob"),
    ("entry_atr_pct", "atr %"),
    ("entry_news_sentiment", "news sentiment"),
    ("entry_batch_weight", "batch weight"),
    ("entry_online_weight", "online weight"),
    ("entry_markov_weight", "markov weight"),
    ("entry_meanrev_weight", "meanrev weight"),
    ("entry_meta_win_prob", "meta win prob"),
]


def _is_blank(value):
    return value is None or str(value).strip() in ("", "None")


def load_closed_trades(path):
    """Read the CSV and return a list of closed-trade dicts.

    Uses csv.DictReader so column order doesn't matter and extra/missing
    columns across different trade_log.csv versions are handled gracefully.
    Some historical rows in this repo's live log carry one more value than
    the header has columns for (an unlabeled trailing field); DictReader
    puts that under the None key, and since the code that writes trade rows
    appends entry_meta_win_prob last, we treat a lone unnamed trailing value
    as entry_meta_win_prob when the header itself doesn't already name that
    column.
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        has_named_meta_win_prob = "entry_meta_win_prob" in fieldnames

        rows = []
        for raw in reader:
            row = dict(raw)
            extra = row.pop(None, None)
            if extra and not has_named_meta_win_prob:
                # DictReader gives a list for the None key when there are
                # multiple overflow values; we only know how to interpret
                # a single trailing value (entry_meta_win_prob).
                value = extra[0] if isinstance(extra, list) else extra
                row["entry_meta_win_prob"] = value

            # A closed trade has both an exit timestamp and an exit price.
            # Open-order rows (record_open_order) leave both blank.
            if _is_blank(row.get("exit_timestamp")) or _is_blank(row.get("exit_price")):
                continue
            rows.append(row)
        return rows


def _fmt_price(value):
    try:
        return f"${float(value):,.4f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(value)


def _fmt_pct(value):
    try:
        v = float(value)
        return f"{v:+.2f}%"
    except (TypeError, ValueError):
        return str(value)


def _fmt_context_value(key, value):
    if key.endswith("_weight") or key in ("entry_confidence", "entry_hmm_regime_prob", "entry_meta_win_prob"):
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)
    if key in ("entry_trend_change_pct", "entry_atr_pct"):
        try:
            return f"{float(value):.2f}%"
        except (TypeError, ValueError):
            return str(value)
    if key == "entry_news_sentiment":
        try:
            return f"{float(value):+.2f}"
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def build_context_str(row):
    parts = []
    for key, label in CONTEXT_FIELDS:
        value = row.get(key)
        if _is_blank(value):
            continue
        parts.append(f"{label}={_fmt_context_value(key, value)}")
    return ", ".join(parts)


def group_by_symbol_and_day(trades):
    """Return OrderedDict[symbol] -> OrderedDict[day] -> [trades], both
    sorted (symbol alphabetically, days and trades chronologically)."""
    by_symbol = defaultdict(list)
    for t in trades:
        by_symbol[t["symbol"]].append(t)

    grouped = OrderedDict()
    for symbol in sorted(by_symbol.keys()):
        day_map = defaultdict(list)
        for t in by_symbol[symbol]:
            day = t["exit_timestamp"][:10]  # ISO8601 date prefix
            day_map[day].append(t)

        ordered_days = OrderedDict()
        for day in sorted(day_map.keys()):
            ordered_days[day] = sorted(day_map[day], key=lambda t: t["exit_timestamp"])
        grouped[symbol] = ordered_days

    return grouped


def render_markdown(grouped, source_path):
    total_trades = sum(len(day_trades) for days in grouped.values() for day_trades in days.values())
    lines = []
    lines.append("# Trade Journal")
    lines.append("")
    lines.append(f"Source: `{source_path}`  ")
    lines.append(f"Closed trades: {total_trades}  ")
    lines.append(f"Symbols: {len(grouped)}")
    lines.append("")

    for symbol, days in grouped.items():
        symbol_trades = [t for day_trades in days.values() for t in day_trades]
        symbol_pnls = [float(t["pnl_pct"]) for t in symbol_trades if not _is_blank(t.get("pnl_pct"))]
        wins = sum(1 for p in symbol_pnls if p > 0)
        total_pnl = sum(symbol_pnls)
        win_rate = (100.0 * wins / len(symbol_pnls)) if symbol_pnls else 0.0

        lines.append(f"## {symbol}")
        lines.append("")
        lines.append(
            f"{len(symbol_trades)} closed trades across {len(days)} day(s) — "
            f"win rate {win_rate:.0f}%, total PnL {total_pnl:+.2f}%"
        )
        lines.append("")

        for day, day_trades in days.items():
            day_pnls = [float(t["pnl_pct"]) for t in day_trades if not _is_blank(t.get("pnl_pct"))]
            day_total = sum(day_pnls)
            lines.append(f"### {day} ({len(day_trades)} trade(s), total PnL {day_total:+.2f}%)")
            lines.append("")
            lines.append("| Time (UTC) | Entry | Exit | PnL | Exit reason |")
            lines.append("|---|---|---|---|---|")
            for t in day_trades:
                time_part = t["exit_timestamp"][11:19] if len(t["exit_timestamp"]) >= 19 else t["exit_timestamp"]
                exit_reason = (t.get("exit_reason") or "").replace("|", "\\|")
                lines.append(
                    f"| {time_part} | {_fmt_price(t.get('entry_price'))} | {_fmt_price(t.get('exit_price'))} "
                    f"| {_fmt_pct(t.get('pnl_pct'))} | {exit_reason} |"
                )

                context_str = build_context_str(t)
                if context_str:
                    lines.append(f"| | | | | *{context_str}* |")

            lines.append("")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Turn a trade log CSV into a readable per-trade Markdown journal, "
                     "grouped by symbol and day."
    )
    parser.add_argument(
        "input", nargs="?", default=DEFAULT_INPUT,
        help=f"Path to the trade log CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Path to write the Markdown journal to (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"No trade log found at '{args.input}'.")
        sys.exit(1)

    trades = load_closed_trades(args.input)
    if not trades:
        print(f"'{args.input}' has no closed trades yet (no rows with both exit_timestamp and exit_price).")
        sys.exit(0)

    grouped = group_by_symbol_and_day(trades)
    markdown = render_markdown(grouped, args.input)

    out_parent = os.path.dirname(args.output)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(markdown)

    print(f"Wrote journal for {len(trades)} closed trades ({len(grouped)} symbols) -> {args.output}")


if __name__ == "__main__":
    main()
