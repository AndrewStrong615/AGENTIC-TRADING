#!/usr/bin/env python3
"""
trade_log_to_jsonl.py — standalone CSV-to-JSONL trade log converter/sync.

WHAT THIS IS
------------
A standalone, read-only converter that mirrors trade_data/trade_log.csv into
trade_data/trade_log.jsonl, one JSON object per line (JSON Lines / JSONL),
using the same field names and values as the CSV — just easier to feed into
dashboards, alerting rules, or ad-hoc jq/pandas analysis than parsing CSV.

This script does NOT hook into the live trading bot's write path. It never
opens trade_log.csv for writing, and it is invoked separately (manually, or
via your own external cron job / hook) rather than being wired inline into
the CSV write logic. See the "Why standalone?" note below.

WHY STANDALONE?
---------------
The CSV row-write logic lives entirely inside Tradingcore.py (the header
write and all three row-append call sites). That file is off-limits to
automated changes in this project, so there is no safe way to emit JSON
inline at write-time. This script instead re-reads the CSV after the fact
and syncs any new rows to the JSONL file — fully additive, and it never
touches trade_log.csv itself.

OPT-IN / GATING
----------------
This script is a no-op unless explicitly enabled. Set:

    ENABLE_JSONL_TRADE_LOG=1

If that env var is not set to "1", the script prints a short message and
exits 0 without reading, writing, or creating anything.

INCREMENTAL / RE-RUNNABLE
--------------------------
The script tracks how many CSV data rows it has already converted in a small
state file (trade_data/.trade_log_jsonl_state.json, holding a row count).
Re-running the script only appends rows that are new since the last run — it
never rewrites or duplicates existing lines in the .jsonl file. If the CSV
ever has fewer rows than the state file expects (e.g. the CSV was replaced),
the script warns and reconverts from the beginning to stay safe.

USAGE
-----
    ENABLE_JSONL_TRADE_LOG=1 python3 trade_log_to_jsonl.py

Optional environment overrides (defaults shown):
    TRADE_LOG_CSV=trade_data/trade_log.csv
    TRADE_LOG_JSONL=trade_data/trade_log.jsonl
    TRADE_LOG_JSONL_STATE=trade_data/.trade_log_jsonl_state.json

Run it by hand, or point your own cron/hook at it. It exits 0 on success
(including "nothing to do") and non-zero only on unexpected errors.
"""

import csv
import json
import os
import re
import sys

DEFAULT_CSV_PATH = "trade_data/trade_log.csv"
DEFAULT_JSONL_PATH = "trade_data/trade_log.jsonl"
DEFAULT_STATE_PATH = "trade_data/.trade_log_jsonl_state.json"

# The trade_log.csv header line is only ever written once, the first time the
# file is created (see Tradingcore.py's _ensure_trade_log_header). Fields
# added to the row-writer later (e.g. entry_meta_win_prob) can therefore be
# present as trailing values in every data row while missing from an older
# header line already on disk. csv.DictReader collects any such unnamed
# trailing values under this restkey; we label them using the current
# canonical trailing field names so they don't silently get dropped.
KNOWN_TRAILING_FIELDS_BEYOND_HEADER = ["entry_meta_win_prob"]
_RESTKEY = "_extra_columns"

_INT_RE = re.compile(r"^[+-]?\d+$")


def coerce_value(value):
    """Turn obviously-numeric CSV strings into real JSON numbers.

    Empty string -> null. Integer-looking strings -> int. Other
    float-parseable strings -> float. Everything else is left as a string.
    """
    if value is None or value == "":
        return None
    if _INT_RE.match(value):
        try:
            return int(value)
        except ValueError:
            return value
    try:
        return float(value)
    except ValueError:
        return value


def row_to_json_object(row):
    extra = row.pop(_RESTKEY, None)
    if extra:
        for name, value in zip(KNOWN_TRAILING_FIELDS_BEYOND_HEADER, extra):
            row[name] = value
    return {key: coerce_value(value) for key, value in row.items()}


def load_state(state_path):
    if not os.path.exists(state_path):
        return {"rows_processed": 0}
    try:
        with open(state_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("rows_processed"), int):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {"rows_processed": 0}


def save_state(state_path, rows_processed):
    tmp_path = state_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump({"rows_processed": rows_processed}, f)
    os.replace(tmp_path, state_path)


def main():
    if os.environ.get("ENABLE_JSONL_TRADE_LOG") != "1":
        print(
            "ENABLE_JSONL_TRADE_LOG is not set to '1' — the JSONL trade log "
            "sync is opt-in and disabled. Exiting without reading or "
            "writing anything. Set ENABLE_JSONL_TRADE_LOG=1 to enable it."
        )
        return 0

    csv_path = os.environ.get("TRADE_LOG_CSV", DEFAULT_CSV_PATH)
    jsonl_path = os.environ.get("TRADE_LOG_JSONL", DEFAULT_JSONL_PATH)
    state_path = os.environ.get("TRADE_LOG_JSONL_STATE", DEFAULT_STATE_PATH)

    if not os.path.exists(csv_path):
        print(f"No CSV trade log found at {csv_path} — nothing to convert.")
        return 0

    # Read-only against the CSV: open in read mode only, never append/write.
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f, restkey=_RESTKEY)
        all_rows = list(reader)

    state = load_state(state_path)
    rows_processed = state.get("rows_processed", 0)

    if rows_processed > len(all_rows):
        print(
            f"Warning: state file expected {rows_processed} row(s) already "
            f"processed, but {csv_path} only has {len(all_rows)} data "
            "row(s) now. The CSV may have been replaced/truncated — "
            "reconverting from the start to stay safe.",
            file=sys.stderr,
        )
        rows_processed = 0

    new_rows = all_rows[rows_processed:]

    if not new_rows:
        print("No new rows since last run — trade_log.jsonl is already up to date.")
        return 0

    parent = os.path.dirname(jsonl_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(jsonl_path, "a") as f:
        for row in new_rows:
            f.write(json.dumps(row_to_json_object(row)) + "\n")

    rows_processed = len(all_rows)
    save_state(state_path, rows_processed)

    print(
        f"Appended {len(new_rows)} new row(s) to {jsonl_path} "
        f"(total rows processed so far: {rows_processed})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
