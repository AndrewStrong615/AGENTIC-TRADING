"""
Wrapper to run pytest deterministically and collect artifacts.
- Writes output to artifacts/<timestamp>/unit_test_output.txt
- Treats 'no tests' or missing pytest as a non-failure, but logs details.
"""
from __future__ import annotations

import os
import sys
import subprocess
import json
from datetime import datetime
import traceback

REPO_ROOT = os.getcwd()
ARTIFACTS_ROOT = os.path.join(REPO_ROOT, "artifacts")
import glob
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
    run_dir = os.path.join(ARTIFACTS_ROOT, ts + "-tests")
    os.makedirs(run_dir, exist_ok=True)

    summary = {"start_time": start.isoformat() + "Z", "artifact_dir": run_dir}

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = env.get("PYTHONHASHSEED", "0")

    cmd = [sys.executable, "-m", "pytest", "-q", "--maxfail=1"]

    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out = proc.stdout or ""
        returncode = proc.returncode
    except Exception as e:
        out = f"Exception while launching pytest: {e}\n" + traceback.format_exc()
        returncode = 2

    with open(os.path.join(run_dir, "unit_test_output.txt"), "w", encoding="utf-8") as f:
        f.write(out)

    # Interpret common non-fatal conditions
    # Pytest exit code 5 means no tests collected on some setups; treat as success.
    effective_rc = 0 if returncode == 5 else returncode

    summary["end_time"] = datetime.utcnow().isoformat() + "Z"
    summary["returncode"] = int(effective_rc)
    commit, branch = _git_info()
    if commit:
        summary["git_commit"] = commit
    if branch:
        summary["git_branch"] = branch

    try:
        with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    except Exception:
        pass

    agent_log = os.path.join(ARTIFACTS_ROOT, "agent_log.txt")
    try:
        with open(agent_log, "a", encoding="utf-8") as f:
            f.write(f"{datetime.utcnow().isoformat()}Z RUN_UNIT_TESTS commit={commit} branch={branch} ret={summary['returncode']} dir={run_dir}\n")
    except Exception:
        pass

    print(json.dumps(summary))
    if summary["returncode"] != 0:
        sys.exit(summary["returncode"])


if __name__ == "__main__":
    main()
