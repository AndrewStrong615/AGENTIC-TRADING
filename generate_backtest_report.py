#!/usr/bin/env python3
"""
generate_backtest_report.py — Combine one backtest run's scattered output
files into a single self-contained HTML report.

This is a standalone, additive tool. It does NOT import or modify
backtest.py; it only reads the files backtest.py already writes into a run
directory under trade_data/backtests/<run_id>/:

    summary.json            - run metadata + per-symbol/per-window metrics
                               (the "windows" list)
    backtest_results.csv    - the same per-symbol/per-window metrics as CSV
                               (pd.DataFrame(all_window_results).to_csv(...))
    backtest_vs_spy.png     - equity / drawdown / SPY comparison chart
    backtest_trade_log_<SYMBOL>_<offset>.csv
                             - individual simulated trades for that symbol's
                               window (the real per-trade data)

Any of the above may be missing (e.g. a run that failed before completing,
or an older run with a different layout). This script degrades gracefully:
it reports what it found and what it could not find, rather than crashing.

Usage:
    python3 generate_backtest_report.py [run_dir_or_run_id]

If no argument is given, the most recently modified subdirectory under
trade_data/backtests/ is used. The report is written as report.html inside
the resolved run directory.
"""

import argparse
import base64
import csv
import glob
import html
import json
import os
import sys
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "trade_data")
BACKTEST_RUNS_DIR = os.path.join(DATA_DIR, "backtests")

# Per trade-log file, cap how many rows get embedded in the HTML so a run
# with an unusually large number of simulated trades doesn't produce an
# unwieldy multi-hundred-MB report.
MAX_TRADE_ROWS_PER_FILE = 2000


# ============================================================================
# RUN DIRECTORY RESOLUTION
# ============================================================================

def resolve_run_dir(arg):
    """Turn the (optional) CLI argument into a concrete run directory.

    Accepts, in order of preference:
      - an existing directory path (absolute or relative to cwd)
      - a bare run_id, resolved under trade_data/backtests/<run_id>
      - nothing, in which case the most recently modified subdirectory of
        trade_data/backtests/ is used.
    """

    if arg:
        candidate = os.path.abspath(arg)
        if os.path.isdir(candidate):
            return candidate

        by_id = os.path.join(BACKTEST_RUNS_DIR, arg)
        if os.path.isdir(by_id):
            return os.path.abspath(by_id)

        raise SystemExit(
            f"Could not find a backtest run at '{arg}' "
            f"(tried it as a directory and as a run_id under "
            f"{BACKTEST_RUNS_DIR!r})."
        )

    if not os.path.isdir(BACKTEST_RUNS_DIR):
        raise SystemExit(
            f"No backtests directory found at {BACKTEST_RUNS_DIR!r}. "
            f"Run a backtest first, or pass a run directory explicitly."
        )

    subdirs = [
        os.path.join(BACKTEST_RUNS_DIR, name)
        for name in os.listdir(BACKTEST_RUNS_DIR)
        if os.path.isdir(os.path.join(BACKTEST_RUNS_DIR, name))
    ]

    if not subdirs:
        raise SystemExit(
            f"No run directories found under {BACKTEST_RUNS_DIR!r}. "
            f"Run a backtest first, or pass a run directory explicitly."
        )

    subdirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return subdirs[0]


# ============================================================================
# LOADING RUN ARTIFACTS
# ============================================================================

def load_summary(run_dir):
    """Load summary.json if present. Returns (dict_or_None, note)."""

    path = os.path.join(run_dir, "summary.json")
    if not os.path.exists(path):
        return None, "summary.json not found (run may not have completed)."

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except (OSError, json.JSONDecodeError) as e:
        return None, f"summary.json could not be read: {e}"


def load_job(run_dir):
    """Load job.json if present, purely as a fallback status source."""

    path = os.path.join(run_dir, "job.json")
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def load_results_csv(run_dir):
    """Load backtest_results.csv if present. Returns (rows, fieldnames, note)."""

    path = os.path.join(run_dir, "backtest_results.csv")
    if not os.path.exists(path):
        return [], [], "backtest_results.csv not found."

    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames or []
            return rows, fieldnames, None
    except OSError as e:
        return [], [], f"backtest_results.csv could not be read: {e}"


def windows_from_summary(summary):
    """Pull the per-symbol/per-window metrics list out of summary.json.

    backtest.py writes this under summary["windows"] (a list of dicts
    produced by aggregate_metrics()). Older/alternate summaries might use a
    different key, so we fall back defensively.
    """

    if not summary:
        return []

    for key in ("windows", "results", "window_results"):
        val = summary.get(key)
        if isinstance(val, list):
            return val

    return []


def metrics_table(summary, csv_rows, csv_fields):
    """Prefer summary.json's windows list; fall back to the CSV rows.

    Both are, in practice, the same data (backtest.py writes
    backtest_results.csv as a straight dump of the same list that goes into
    summary.json), but a run might have one file without the other.
    Returns (rows, fieldnames, source_note).
    """

    windows = windows_from_summary(summary)
    if windows:
        fieldnames = []
        for row in windows:
            for k in row.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        return windows, fieldnames, "summary.json"

    if csv_rows:
        return csv_rows, csv_fields, "backtest_results.csv"

    return [], [], None


def find_chart(run_dir):
    path = os.path.join(run_dir, "backtest_vs_spy.png")
    return path if os.path.exists(path) else None


def find_trade_logs(run_dir):
    """Return sorted list of (label, path) for every per-symbol trade log.

    backtest.py names these backtest_trade_log_<SYMBOL>_<offset>.csv, one
    per symbol per walk-forward window, written directly into the run
    directory. This is where the actual individual simulated trades live —
    backtest_results.csv/summary.json only hold aggregated statistics.
    """

    pattern = os.path.join(run_dir, "backtest_trade_log_*.csv")
    paths = sorted(glob.glob(pattern))

    out = []
    for path in paths:
        base = os.path.basename(path)
        label = base[len("backtest_trade_log_"):-len(".csv")] if base.endswith(".csv") else base
        out.append((label, path))
    return out


def load_trade_log(path):
    """Read one trade-log CSV. Returns (rows, fieldnames, total_row_count)."""

    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            rows = []
            total = 0
            for row in reader:
                total += 1
                if len(rows) < MAX_TRADE_ROWS_PER_FILE:
                    rows.append(row)
            return rows, fieldnames, total
    except OSError:
        return [], [], 0


# ============================================================================
# HTML RENDERING
# ============================================================================

def esc(val):
    return html.escape("" if val is None else str(val))


def fmt_number(val):
    """Format numeric-looking values to a reasonable precision for display."""

    if val is None or val == "":
        return ""
    try:
        f = float(val)
    except (TypeError, ValueError):
        return esc(val)

    if f == int(f) and abs(f) < 1e15:
        return str(int(f))
    return f"{f:.4f}".rstrip("0").rstrip(".")


def render_table(rows, fieldnames, empty_message, numeric_hint=True, row_limit=None):
    if not rows:
        return f'<p class="empty-note">{esc(empty_message)}</p>'

    display_rows = rows if row_limit is None else rows[:row_limit]

    head = "".join(f"<th>{esc(f)}</th>" for f in fieldnames)
    body_lines = []
    for row in display_rows:
        cells = []
        for f in fieldnames:
            raw = row.get(f, "")
            cells.append(f"<td>{fmt_number(raw) if numeric_hint else esc(raw)}</td>")
        body_lines.append(f"<tr>{''.join(cells)}</tr>")

    truncation_note = ""
    if row_limit is not None and len(rows) > row_limit:
        truncation_note = (
            f'<p class="empty-note">Showing first {row_limit} of {len(rows)} rows.</p>'
        )

    return (
        '<div class="table-wrap"><table>'
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_lines)}</tbody>"
        "</table></div>"
        f"{truncation_note}"
    )


def render_chart_section(chart_path):
    if not chart_path:
        return '<p class="empty-note">backtest_vs_spy.png not found for this run.</p>'

    try:
        with open(chart_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return (
            f'<img class="chart" alt="Equity / drawdown / SPY comparison chart" '
            f'src="data:image/png;base64,{encoded}">'
        )
    except OSError as e:
        return f'<p class="empty-note">Could not embed backtest_vs_spy.png: {esc(e)}</p>'


def render_trade_logs_section(trade_logs):
    if not trade_logs:
        return '<p class="empty-note">No backtest_trade_log_*.csv files found for this run.</p>'

    sections = []
    for label, path in trade_logs:
        rows, fieldnames, total = load_trade_log(path)
        table_html = render_table(
            rows,
            fieldnames,
            empty_message=f"{os.path.basename(path)} is empty.",
            row_limit=MAX_TRADE_ROWS_PER_FILE,
        )
        sections.append(
            f'<details class="trade-log" open>'
            f"<summary>{esc(label)} &mdash; {total} trade(s)</summary>"
            f"{table_html}"
            f"</details>"
        )
    return "".join(sections)


def resolve_status(summary, job):
    if summary and summary.get("status"):
        return summary["status"]
    if job and job.get("status"):
        return job.get("status") + " (from job.json; run did not reach a completed summary.json)"
    return "unknown"


def resolve_completed_at(summary, job):
    if summary and summary.get("completed_at"):
        return summary["completed_at"]
    if job and job.get("updated_at"):
        return job.get("updated_at") + " (last update from job.json)"
    return "unknown"


PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Backtest Report — {run_id}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{
    --bg: #0f1115;
    --panel: #161a21;
    --border: #262c36;
    --text: #e6e9ef;
    --muted: #9aa4b2;
    --accent: #4f8cff;
    --good: #35c07a;
    --bad: #e15a5a;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 2rem;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.5;
  }}
  .wrap {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{ font-size: 1.6rem; margin-bottom: 0.25rem; }}
  h2 {{ font-size: 1.15rem; margin-top: 2.5rem; border-bottom: 1px solid var(--border); padding-bottom: 0.4rem; }}
  .meta {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 1.5rem; }}
  .meta span {{ margin-right: 1.5rem; display: inline-block; }}
  .meta b {{ color: var(--text); font-weight: 600; }}
  section {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-top: 1rem;
  }}
  .table-wrap {{ overflow-x: auto; max-width: 100%; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; white-space: nowrap; }}
  th, td {{ padding: 0.4rem 0.7rem; text-align: right; border-bottom: 1px solid var(--border); }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ color: var(--muted); font-weight: 600; position: sticky; top: 0; background: var(--panel); }}
  tbody tr:hover {{ background: rgba(255,255,255,0.03); }}
  .chart {{ max-width: 100%; height: auto; border-radius: 6px; display: block; margin: 0 auto; }}
  .empty-note {{ color: var(--muted); font-style: italic; }}
  details.trade-log {{ margin-bottom: 1.25rem; }}
  details.trade-log summary {{
    cursor: pointer;
    font-weight: 600;
    padding: 0.4rem 0;
    color: var(--accent);
  }}
  footer {{ color: var(--muted); font-size: 0.8rem; margin-top: 2.5rem; text-align: center; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Backtest Report</h1>
  <div class="meta">
    <span><b>Run ID:</b> {run_id}</span>
    <span><b>Status:</b> {status}</span>
    <span><b>Completed at:</b> {completed_at}</span>
    <span><b>Run directory:</b> {run_dir}</span>
  </div>

  {warnings_block}

  <h2>Metrics {metrics_source_note}</h2>
  <section>
    {metrics_table}
  </section>

  <h2>Equity / Drawdown / SPY Comparison</h2>
  <section>
    {chart_section}
  </section>

  <h2>Trade List</h2>
  <section>
    {trade_logs_section}
  </section>

  <footer>Generated by generate_backtest_report.py on {generated_at}</footer>
</div>
</body>
</html>
"""


def render_report(run_dir):
    run_id = os.path.basename(os.path.normpath(run_dir))

    summary, summary_note = load_summary(run_dir)
    job = load_job(run_dir)
    csv_rows, csv_fields, csv_note = load_results_csv(run_dir)
    chart_path = find_chart(run_dir)
    trade_logs = find_trade_logs(run_dir)

    metrics_rows, metrics_fields, metrics_source = metrics_table(summary, csv_rows, csv_fields)

    warnings = []
    if summary_note:
        warnings.append(summary_note)
    if csv_note and not metrics_rows:
        warnings.append(csv_note)
    if not chart_path:
        warnings.append("backtest_vs_spy.png not found.")
    if not trade_logs:
        warnings.append("No trade log CSVs found.")

    if not summary and not csv_rows and not chart_path and not trade_logs:
        raise SystemExit(
            f"No recognizable backtest output found in {run_dir!r}. "
            f"Expected at least one of: summary.json, backtest_results.csv, "
            f"backtest_vs_spy.png, or backtest_trade_log_*.csv. "
            f"This run may not have completed, or its output layout has "
            f"changed."
        )

    warnings_block = ""
    if warnings:
        items = "".join(f"<li>{esc(w)}</li>" for w in warnings)
        warnings_block = (
            '<section style="border-color:#5a4420;background:#241d0f;">'
            "<b>Note:</b> some expected files were missing for this run "
            f"and are shown as unavailable below.<ul>{items}</ul></section>"
        )

    metrics_source_note = f"<small style='color:#9aa4b2;'>(source: {esc(metrics_source)})</small>" if metrics_source else ""

    html_out = PAGE_TEMPLATE.format(
        run_id=esc(run_id),
        status=esc(resolve_status(summary, job)),
        completed_at=esc(resolve_completed_at(summary, job)),
        run_dir=esc(run_dir),
        warnings_block=warnings_block,
        metrics_source_note=metrics_source_note,
        metrics_table=render_table(
            metrics_rows,
            metrics_fields,
            empty_message="No metrics available for this run.",
        ),
        chart_section=render_chart_section(chart_path),
        trade_logs_section=render_trade_logs_section(trade_logs),
        generated_at=esc(datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )

    return html_out


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a single self-contained HTML report (metrics table, "
            "equity/drawdown/SPY chart, trade list) for one backtest run."
        )
    )
    parser.add_argument(
        "run",
        nargs="?",
        default=None,
        help=(
            "Run directory path or run_id under trade_data/backtests/. "
            "Defaults to the most recently modified run."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output HTML path. Defaults to report.html inside the run directory.",
    )
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run)
    html_out = render_report(run_dir)

    output_path = args.output or os.path.join(run_dir, "report.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_out)

    print(f"Wrote report: {output_path}")


if __name__ == "__main__":
    main()
