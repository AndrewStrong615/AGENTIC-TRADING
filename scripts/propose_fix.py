"""
Propose a minimal fix given a triage report.
- Creates a branch autofix/<issue-id>-<short> and commits a single-file patch (PATCH.md) describing the change.
- If GH_TOKEN is present in env, will attempt to push and create a draft PR using `gh` if available.
- Runs unit tests and a short backtest to sanity-check the change.

This script is conservative: it does NOT modify production code automatically unless a clear, safe rule is matched.
"""
from __future__ import annotations

import os
import sys
import json
import subprocess
from datetime import datetime
import uuid

REPO_ROOT = os.getcwd()
ARTIFACTS_ROOT = os.path.join(REPO_ROOT, "artifacts")

SAFE_RULES = ["data-fetch", "environment", "bug/exception"]


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _run(cmd, **kwargs):
    print("+ ", " ".join(cmd))
    return subprocess.run(cmd, cwd=REPO_ROOT, **kwargs)


def main():
    if len(sys.argv) < 2:
        print("Usage: propose_fix.py <triage_report.json>")
        sys.exit(2)
    triage_path = sys.argv[1]
    triage = _read_json(triage_path) or {}
    causes = triage.get("likely_causes", [])
    issue_id = os.path.basename(triage.get("incident_dir", "incident")) if triage.get("incident_dir") else datetime.utcnow().strftime("incident-%Y%m%dT%H%M%SZ")

    # Pick a primary cause we know how to handle
    primary = None
    for c in causes:
        if c in SAFE_RULES:
            primary = c
            break
    if primary is None:
        primary = causes[0] if causes else "unknown"

    short = primary.replace("/", "-")
    branch = f"autofix/{issue_id}-{short}-{str(uuid.uuid4())[:6]}"

    # Create branch (if git CLI is available). Fall back to writing patch files locally if git isn't present.
    git_available = True
    try:
        res = _run(["git", "checkout", "-b", branch], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if res.returncode != 0:
            print("Failed to create branch via git; continuing without branch (patch will be left in autofix/)")
            git_available = False
    except FileNotFoundError:
        print("git CLI not found in PATH; leaving patch files on disk for manual review/push.")
        git_available = False

    # Prepare a conservative single-file patch: a markdown explaining the small change and a suggested diff
    patch_dir = os.path.join(REPO_ROOT, "autofix", branch)
    os.makedirs(patch_dir, exist_ok=True)
    patch_path = os.path.join(patch_dir, "PATCH.md")

    suggested_patch = """
The proposed conservative change addresses missing Alpaca credentials and protects backtest from raising a KeyError on import.
Suggested minimal code patch (apply to backtest.py):

--- a/backtest.py
+++ b/backtest.py
@@
-API_KEY = os.environ["ALPACA_API_KEY"]
-SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
-BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
-
-api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")
+API_KEY = os.environ.get("ALPACA_API_KEY")
+SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
+BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
+
+api = None
+try:
+    if API_KEY and SECRET_KEY:
+        api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")
+    else:
+        print("[autofix] ALPACA credentials not found; running in offline/mock mode")
+except Exception as e:
+    print(f"[autofix] Failed to initialize Alpaca API: {e}")
+    api = None

And add guards in fetch_bars() to fallback to deterministic synthetic data if `api` is None.
"""

    with open(patch_path, "w", encoding="utf-8") as f:
        f.write(f"# Proposed autofix for {issue_id}\n\n")
        f.write(f"Primary cause: {primary}\n\n")
        f.write(suggested_patch)

    # Add and commit the patch file if git is available; otherwise leave files in autofix/ for manual review
    if git_available:
        _run(["git", "add", patch_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        commit_msg = f"autofix: propose conservative patch for {issue_id} ({primary})\n\nCo-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
        res2 = _run(["git", "commit", "-m", commit_msg], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if res2.returncode != 0:
            print("Commit failed (possibly empty). Leaving branch locally with patch file.")
        else:
            print(f"Committed patch on branch {branch}")
    else:
        print(f"git not available — wrote patch to {patch_path}. To publish, create a branch and commit the file manually.")

    # Run unit tests and short backtest as a sanity check
    print("Running unit tests (best-effort)...")
    _run([sys.executable, os.path.join(REPO_ROOT, "scripts", "run_unit_tests.py")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print("Running a short backtest (best-effort)...")
    _run([sys.executable, os.path.join(REPO_ROOT, "scripts", "run_backtest.py")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Try to push and create draft PR if GH_TOKEN is present
    gh_token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if gh_token and git_available:
        print("GH_TOKEN found and git available; attempting to push branch and create draft PR (requires 'gh' CLI).")
        push = _run(["git", "push", "-u", "origin", branch], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if push.returncode != 0:
            print("Failed to push branch; please push manually.")
        else:
            # Create draft PR using gh if available
            gh_check = subprocess.run(["gh", "--version"], capture_output=True, text=True)
            if gh_check.returncode == 0:
                # Authenticate gh using token if provided in env
                token = os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
                if token:
                    try:
                        auth = subprocess.run(["gh", "auth", "login", "--with-token"], input=token, text=True, capture_output=True)
                        if auth.returncode != 0:
                            print('gh auth login failed:', auth.stderr)
                    except Exception as e:
                        print('gh auth login error:', e)

                title = f"Autofix: {primary} ({issue_id})"
                body = open(patch_path, "r", encoding="utf-8").read()
                pr_create = subprocess.run(["gh", "pr", "create", "--draft", "--title", title, "--body", body], text=True)
                if pr_create.returncode == 0:
                    print("Draft PR created via gh CLI.")
                else:
                    print("Failed to create PR via gh CLI; please create it manually.")
            else:
                print("gh CLI not found; please create a draft PR manually.")
    else:
        if not git_available:
            print("Git CLI not available — patch files are left in autofix/ for manual publishing.")
        else:
            print("No GH_TOKEN found — left branch locally. To publish and open a draft PR run:")
            print(f"  git push -u origin {branch}")
            print(f"  gh pr create --draft --title 'Autofix: {primary} ({issue_id})' --body '{patch_path}'")

    # Log our action
    agent_log = os.path.join(ARTIFACTS_ROOT, "agent_log.txt")
    try:
        with open(agent_log, "a", encoding="utf-8") as f:
            f.write(f"{datetime.utcnow().isoformat()}Z PROPOSE_FIX branch={branch} incident={issue_id} primary={primary}\n")
    except Exception:
        pass

    print(f"Propose-fix finished; branch: {branch}")


if __name__ == "__main__":
    main()
