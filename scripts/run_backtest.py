"""
Wrapper to run backtest.py deterministically and collect artifacts.
- Creates artifacts/<timestamp>/ and copies trade_data/backtest_results.csv and any backtest_trade_log_*.csv into it.
- Exits non-zero if backtest fails.
- Appends a human-readable and machine-readable summary JSON.
"""
from __future__ import annotations

import os
import sys
import subprocess
import shutil
import glob
import json
from datetime import datetime
import traceback

REPO_ROOT = os.getcwd()
ARTIFACTS_ROOT = os.path.join(REPO_ROOT, "artifacts")
os.makedirs(ARTIFACTS_ROOT, exist_ok=True)


def _git_info():
    try:
        import subprocess
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        return commit, branch
    except Exception:
        return None, None


def main():
    start = datetime.utcnow()
    ts = start.strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(ARTIFACTS_ROOT, ts)
    os.makedirs(run_dir, exist_ok=True)

    summary = {
        "start_time": start.isoformat() + "Z",
        "artifact_dir": run_dir,
    }

    env = os.environ.copy()
    # Deterministic seeds for reproducibility
    env["PYTHONHASHSEED"] = env.get("PYTHONHASHSEED", "0")
    env.setdefault("PYTHONWARNINGS", "ignore")

    backtest_cmd = [sys.executable, os.path.join(REPO_ROOT, "backtest.py")]

    try:
        proc = subprocess.run(backtest_cmd, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out = proc.stdout or ""
    except Exception as e:
        out = f"Exception while launching backtest: {e}\n" + traceback.format_exc()
        proc = type("P", (), {"returncode": 2})()

    # Write captured output
    with open(os.path.join(run_dir, "backtest_output.txt"), "w", encoding="utf-8") as f:
        f.write(out)

    # Copy expected artifacts if they exist
    try:
        trade_results = os.path.join(REPO_ROOT, "trade_data", "backtest_results.csv")
        if os.path.exists(trade_results):
            shutil.copy2(trade_results, os.path.join(run_dir, "backtest_results.csv"))

        for path in glob.glob(os.path.join(REPO_ROOT, "trade_data", "backtest_trade_log_*.csv")):
            try:
                shutil.copy2(path, run_dir)
            except Exception:
                pass
    except Exception:
        # don't fail on copy errors
        pass

    # Fill summary
    summary["end_time"] = datetime.utcnow().isoformat() + "Z"
    summary["returncode"] = int(getattr(proc, "returncode", 1))
    commit, branch = _git_info()
    if commit:
        summary["git_commit"] = commit
    if branch:
        summary["git_branch"] = branch

    # Write machine-readable summary
    try:
        with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    except Exception:
        pass

    # Append to global agent log
    agent_log = os.path.join(ARTIFACTS_ROOT, "agent_log.txt")
    try:
        with open(agent_log, "a", encoding="utf-8") as f:
            f.write(f"{datetime.utcnow().isoformat()}Z RUN_BACKTEST commit={commit} branch={branch} ret={summary['returncode']} dir={run_dir}\n")
    except Exception:
        pass

    # Print summary to stdout for callers
    print(json.dumps(summary))

    if summary["returncode"] != 0:
        # Exit non-zero to signal failure
        sys.exit(summary["returncode"]) 


if __name__ == "__main__":
    main()
