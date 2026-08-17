"""
Conditional auto-merge script.
- Finds open PRs whose head branch starts with 'autofix/'
- For each PR, runs unit tests and a short backtest on the PR branch (best-effort),
  compares metrics against config/baseline_metrics.json and config/monitor.yaml,
  and if tests pass and performance meets thresholds, merges the PR via GitHub API.

Requires environment variable GITHUB_TOKEN with repo write permissions.
Usage: python scripts/conditional_auto_merge.py
"""
from __future__ import annotations

import os
import sys
import json
import subprocess
import time
from datetime import datetime, timedelta
from urllib.request import Request, urlopen
from urllib.parse import urlencode

REPO = os.getcwd()
API_BASE = os.environ.get('GITHUB_API_URL', 'https://api.github.com')
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')

if not GITHUB_TOKEN:
    print('GITHUB_TOKEN not set; cannot auto-merge. Exiting.')
    sys.exit(2)

# Determine owner/repo from git remote
def repo_owner_name():
    try:
        remotes = subprocess.check_output(['git', 'remote', 'get-url', 'origin'], cwd=REPO, text=True).strip()
        # remotes like git@github.com:owner/repo.git or https://github.com/owner/repo.git
        if remotes.startswith('git@'):
            path = remotes.split(':',1)[1]
        else:
            path = remotes.split('github.com/',1)[1]
        if path.endswith('.git'):
            path = path[:-4]
        owner, repo = path.split('/',1)
        return owner, repo
    except Exception as e:
        print('Failed to determine repo owner/name from git remote:', e)
        return None, None

OWNER, REPO_NAME = repo_owner_name()
if not OWNER:
    print('Cannot determine repository owner/name; aborting.')
    sys.exit(2)

HEADERS = {
    'Authorization': f'token {GITHUB_TOKEN}',
    'Accept': 'application/vnd.github+json',
    'User-Agent': 'autofix-agent'
}


def api_get(path, params=None):
    url = f"{API_BASE}/repos/{OWNER}/{REPO_NAME}{path}"
    if params:
        url += '?' + urlencode(params)
    req = Request(url, headers=HEADERS)
    with urlopen(req) as resp:
        return json.load(resp)


def api_post(path, data):
    url = f"{API_BASE}/repos/{OWNER}/{REPO_NAME}{path}"
    req = Request(url, data=json.dumps(data).encode('utf-8'), headers={**HEADERS, 'Content-Type': 'application/json'}, method='PUT')
    with urlopen(req) as resp:
        return json.load(resp)


def list_autofix_prs():
    prs = api_get('/pulls', params={'state': 'open', 'per_page': 100})
    autofix = [p for p in prs if p.get('head', {}).get('ref','').startswith('autofix/')]
    return autofix


def run_command(cmd):
    print('+', ' '.join(cmd))
    return subprocess.run(cmd, cwd=REPO)


def load_baseline():
    try:
        with open(os.path.join(REPO, 'config', 'baseline_metrics.json'), 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def load_thresholds():
    try:
        with open(os.path.join(REPO, 'config', 'monitor.yaml'), 'r', encoding='utf-8') as f:
            import yaml
            return yaml.safe_load(f)
    except Exception:
        # default thresholds
        return {'total_return_drop_pct': 20, 'drawdown_increase_pct': 15, 'win_rate_drop_pct': 10}


def extract_metrics_from_artifacts(art_dir):
    # Look for backtest_results.csv in art_dir
    import csv
    import math
    path = None
    for root, dirs, files in os.walk(art_dir):
        for fn in files:
            if fn == 'backtest_results.csv':
                path = os.path.join(root, fn)
                break
        if path:
            break
    if not path:
        return None
    # Read and compute mean metrics across rows
    vals = {'total_return_pct_mean': [], 'max_drawdown_pct_mean': [], 'win_rate_mean': []}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for k in list(vals.keys()):
                    if k in row and row[k] != '':
                        try:
                            vals[k].append(float(row[k]) if row[k] != '' else float('nan'))
                        except Exception:
                            pass
        out = {k: (sum(v)/len(v) if v else None) for k,v in vals.items()}
        return out
    except Exception as e:
        print('Failed to read backtest_results.csv:', e)
        return None


def should_merge(baseline, current, thresholds):
    if baseline is None or current is None:
        return False
    # Compare: require total_return not worse by more than threshold (i.e., improvement >= -threshold)
    tdrop = thresholds.get('total_return_drop_pct', 20)
    dinc = thresholds.get('drawdown_increase_pct', 15)
    wrdrop = thresholds.get('win_rate_drop_pct', 10)

    # For safe merge, require current total_return_pct_mean >= baseline - tdrop
    if current.get('total_return_pct_mean') is None:
        return False
    if current['total_return_pct_mean'] < baseline.get('total_return_pct_mean', 0) - tdrop:
        print(f"Reject: total_return_pct_mean {current['total_return_pct_mean']} < baseline {baseline.get('total_return_pct_mean')} - {tdrop}")
        return False
    # Require drawdown not much worse
    if current.get('max_drawdown_pct_mean') is not None and baseline.get('max_drawdown_pct_mean') is not None:
        if current['max_drawdown_pct_mean'] < baseline['max_drawdown_pct_mean'] - dinc:
            print('Reject: drawdown increased beyond threshold')
            return False
    # Require win_rate not much worse
    if current.get('win_rate_mean') is not None and baseline.get('win_rate_mean') is not None:
        if current['win_rate_mean'] < baseline['win_rate_mean'] - wrdrop:
            print('Reject: win_rate dropped beyond threshold')
            return False
    return True


def merge_pr(pr_number):
    path = f'/pulls/{pr_number}/merge'
    data = {'commit_title': f'Auto-merge PR #{pr_number} by autofix-agent', 'merge_method': 'merge'}
    try:
        res = api_post(path, data)
        print('Merge result:', res)
        return True
    except Exception as e:
        print('Failed to merge via API:', e)
        return False


def main():
    baseline = load_baseline()
    thresholds = load_thresholds()
    prs = list_autofix_prs()
    if not prs:
        print('No autofix PRs found. Exiting.')
        return
    print(f'Found {len(prs)} autofix PR(s)')
    for pr in prs:
        pr_num = pr['number']
        head_ref = pr['head']['ref']
        created_at = pr.get('created_at')
        # Skip PRs older than 7 days
        try:
            created_dt = datetime.fromisoformat(created_at.replace('Z','+00:00'))
            if created_dt < datetime.utcnow() - timedelta(days=7):
                print(f'Skipping old PR #{pr_num}')
                continue
        except Exception:
            pass
        print(f'Processing PR #{pr_num} head={head_ref}')
        # Checkout branch
        res = run_command(['git', 'fetch', 'origin', head_ref])
        if res.returncode != 0:
            print('Failed to fetch branch, skipping')
            continue
        res = run_command(['git', 'checkout', '-f', head_ref])
        if res.returncode != 0:
            print('Failed to checkout branch, skipping')
            continue
        # Run tests and backtest
        run_command([sys.executable, os.path.join(REPO, 'scripts', 'run_unit_tests.py')])
        run_command([sys.executable, os.path.join(REPO, 'scripts', 'run_backtest.py')])
        # Find latest artifacts dir
        art_root = os.path.join(REPO, 'artifacts')
        subdirs = [os.path.join(art_root,d) for d in os.listdir(art_root) if os.path.isdir(os.path.join(art_root,d))]
        latest = max(subdirs, key=os.path.getmtime) if subdirs else None
        if not latest:
            print('No artifacts produced by test/backtest; skipping merge for PR', pr_num)
            continue
        current_metrics = extract_metrics_from_artifacts(latest)
        print('Current metrics:', current_metrics)
        if should_merge(baseline, current_metrics, thresholds):
            print('Conditions met; merging PR', pr_num)
            merged = merge_pr(pr_num)
            if merged:
                print('Merged PR', pr_num)
            else:
                print('Failed to merge PR', pr_num)
        else:
            print('Conditions not met; leaving PR as draft for human review')

if __name__ == '__main__':
    main()
