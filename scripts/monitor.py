"""
Monitor agent: run backtest and unit tests, compare to baseline, and trigger reproduce/triage if thresholds exceeded.
Writes structured JSON summary and artifacts.
"""
from __future__ import annotations

import os
import sys
import subprocess
import json
import shutil
import glob
from datetime import datetime
import csv

REPO_ROOT = os.getcwd()
ARTIFACTS_ROOT = os.path.join(REPO_ROOT, "artifacts")
os.makedirs(ARTIFACTS_ROOT, exist_ok=True)
CONFIG_DIR = os.path.join(REPO_ROOT, "config")
BASELINE_PATH = os.path.join(CONFIG_DIR, "baseline_metrics.json")
MONITOR_CONFIG = os.path.join(CONFIG_DIR, "monitor.yaml")

DEFAULT_THRESHOLDS = {
    "total_return_drop_pct": 20,
    "drawdown_increase_pct": 15,
    "win_rate_drop_pct": 10,
}


def _read_simple_yaml(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip()
            try:
                if "." in v or "e" in v.lower():
                    out[k] = float(v)
                else:
                    out[k] = int(v)
            except Exception:
                out[k] = v
    return out


def _find_latest_run_artifact() -> str | None:
    # Look for the most-recent timestamped folder in artifacts
    subs = [os.path.join(ARTIFACTS_ROOT, p) for p in os.listdir(ARTIFACTS_ROOT) if os.path.isdir(os.path.join(ARTIFACTS_ROOT, p))]
    if not subs:
        return None
    subs_sorted = sorted(subs, key=lambda p: os.path.getmtime(p), reverse=True)
    return subs_sorted[0]


def _load_backtest_results(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _compute_summary_metrics(rows: list[dict]) -> dict:
    keys = ["total_return_pct_mean", "max_drawdown_pct_mean", "win_rate_mean"]
    out = {}
    if not rows:
        return {k: None for k in keys}
    for k in keys:
        vals = []
        for r in rows:
            v = r.get(k)
            if v is None or v == "":
                continue
            try:
                vals.append(float(v))
            except Exception:
                pass
        out[k] = float(sum(vals) / len(vals)) if vals else None
    return out


def _percent_change(old, new):
    try:
        if old is None or new is None:
            return None
        if old == 0:
            return None
        return (old - new) / abs(old) * 100.0
    except Exception:
        return None


def main():
    monitor_cfg = DEFAULT_THRESHOLDS.copy()
    monitor_cfg.update(_read_simple_yaml(MONITOR_CONFIG))

    # Run backtest and tests sequentially
    print("[monitor] Running backtest...")
    try:
        proc = subprocess.run([sys.executable, os.path.join(REPO_ROOT, "scripts", "run_backtest.py")], cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        # Attempt parse JSON output
        summary = None
        try:
            # The runner prints a JSON summary as last line
            j = proc.stdout.strip().splitlines()[-1]
            summary = json.loads(j)
        except Exception:
            summary = None
    except Exception as e:
        print(f"[monitor] Failed to launch backtest: {e}")
        summary = None

    # Find the run artifacts
    run_dir = _find_latest_run_artifact()
    if run_dir is None:
        print("[monitor] No artifacts found after backtest.")
        sys.exit(1)

    print(f"[monitor] Using run artifacts: {run_dir}")

    results_csv = os.path.join(run_dir, "backtest_results.csv")
    # Also accept trade_data/backtest_results.csv copied inside run_dir
    if not os.path.exists(results_csv):
        # check for any backtest_results in run_dir
        for p in os.listdir(run_dir):
            if p.endswith("backtest_results.csv"):
                results_csv = os.path.join(run_dir, p)
                break

    rows = _load_backtest_results(results_csv)
    current_metrics = _compute_summary_metrics(rows)

    # Baseline handling
    if not os.path.exists(BASELINE_PATH):
        # Save baseline and exit
        try:
            os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
            with open(BASELINE_PATH, "w", encoding="utf-8") as f:
                json.dump({"created": datetime.utcnow().isoformat() + "Z", "metrics": current_metrics}, f, indent=2)
            print(f"[monitor] Baseline not found; saved current run as baseline to {BASELINE_PATH}")
        except Exception as e:
            print(f"[monitor] Failed to save baseline: {e}")
        sys.exit(0)

    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        baseline = json.load(f).get("metrics", {})

    # Compare key metrics
    incidents = []

    # total return drop
    baseline_return = baseline.get("total_return_pct_mean")
    current_return = current_metrics.get("total_return_pct_mean")
    if baseline_return is not None and current_return is not None:
        pct_drop = _percent_change(baseline_return, current_return)
        thr = float(monitor_cfg.get("total_return_drop_pct", DEFAULT_THRESHOLDS["total_return_drop_pct"]))
        if pct_drop is not None and pct_drop >= thr:
            incidents.append({"metric": "total_return_pct_mean", "percent_drop": pct_drop, "threshold": thr})

    # drawdown increase by magnitude
    baseline_dd = baseline.get("max_drawdown_pct_mean")
    current_dd = current_metrics.get("max_drawdown_pct_mean")
    if baseline_dd is not None and current_dd is not None:
        # compare absolute drawdown magnitudes
        if abs(current_dd) > abs(baseline_dd) * (1.0 + float(monitor_cfg.get("drawdown_increase_pct", DEFAULT_THRESHOLDS["drawdown_increase_pct"])) / 100.0):
            incidents.append({"metric": "max_drawdown_pct_mean", "baseline": baseline_dd, "current": current_dd,
                              "threshold_pct": float(monitor_cfg.get("drawdown_increase_pct", DEFAULT_THRESHOLDS["drawdown_increase_pct"]))})

    # win rate drop
    baseline_win = baseline.get("win_rate_mean")
    current_win = current_metrics.get("win_rate_mean")
    if baseline_win is not None and current_win is not None and baseline_win != 0:
        win_drop_pct = (baseline_win - current_win) / abs(baseline_win) * 100.0
        thr = float(monitor_cfg.get("win_rate_drop_pct", DEFAULT_THRESHOLDS["win_rate_drop_pct"]))
        if win_drop_pct >= thr:
            incidents.append({"metric": "win_rate_mean", "percent_drop": win_drop_pct, "threshold": thr})

    summary_out = {
        "run_dir": run_dir,
        "baseline_path": BASELINE_PATH,
        "baseline_metrics": baseline,
        "current_metrics": current_metrics,
        "incidents": incidents,
        "checked_at": datetime.utcnow().isoformat() + "Z",
    }

    summary_path = os.path.join(run_dir, "monitor_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)

    if not incidents:
        print("[monitor] No incidents detected. Monitor complete.")
        sys.exit(0)

    # On incident: create incident folder and copy artifacts
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    incident_dir = os.path.join(ARTIFACTS_ROOT, f"incident-{ts}")
    os.makedirs(incident_dir, exist_ok=True)

    try:
        # copy run_dir contents
        for item in os.listdir(run_dir):
            s = os.path.join(run_dir, item)
            d = os.path.join(incident_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        # copy monitor_summary
        shutil.copy2(summary_path, os.path.join(incident_dir, "monitor_summary.json"))
    except Exception as e:
        print(f"[monitor] Failed to copy artifacts to incident dir: {e}")

    print(f"[monitor] Incident detected; artifacts copied to {incident_dir}")

    # Run reproduce and triage
    try:
        print("[monitor] Running reproduce_and_triage.py...")
        subprocess.run([sys.executable, os.path.join(REPO_ROOT, "scripts", "reproduce_and_triage.py"), incident_dir], cwd=REPO_ROOT, check=False)
    except Exception as e:
        print(f"[monitor] reproduce_and_triage failed: {e}")

    # Run propose_fix (best-effort)
    try:
        print("[monitor] Running propose_fix.py (best-effort)...")
        subprocess.run([sys.executable, os.path.join(REPO_ROOT, "scripts", "propose_fix.py"), os.path.join(incident_dir, "triage_report.json")], cwd=REPO_ROOT, check=False)
    except Exception as e:
        print(f"[monitor] propose_fix failed: {e}")

    print("[monitor] Monitor run finished with incidents.")
    sys.exit(0)


if __name__ == "__main__":
    main()
