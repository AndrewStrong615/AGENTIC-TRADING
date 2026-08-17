"""
Reproducer and triage:
- Given an incident artifact folder, attempt to reproduce the failure by re-running the deterministic backtest a few times
- Classify likely cause into categories: data-fetch, environment, model-drift, bug/exception, stochastic noise
- Produce triage_report.json in the incident folder with findings
"""
from __future__ import annotations

import os
import sys
import subprocess
import json
import time
import re
from datetime import datetime


def _read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _classify_from_log(log_text: str) -> list[str]:
    reasons = set()
    text = log_text.lower()
    if not text:
        return ["unknown"]
    # data-fetch related
    if any(k in text for k in ["get_bars", "news fetch", "api", "timeout", "connectionerror", "httperror"]):
        reasons.add("data-fetch")
    # environment / credentials
    if "alpaca_api_key" in text or "alpaca_secret_key" in text or "keyerror" in text or "environment" in text:
        reasons.add("environment")
    # code exceptions
    if "traceback" in text or "exception" in text or "attributeerror" in text or "typeerror" in text or "keyerror" in text:
        reasons.add("bug/exception")
    # model drift (degraded metrics without exceptions)
    if "no exception" in text or ("total return" in text and "-" in text and "exception" not in text):
        reasons.add("model-drift")
    # fallback stochastic
    # detect noisy results across repeated runs will be determined outside this helper
    if not reasons:
        reasons.add("unknown")
    return list(reasons)


def _run_backtest_once(timeout=900):
    cmd = [sys.executable, os.path.join(os.getcwd(), "scripts", "run_backtest.py")]
    try:
        proc = subprocess.run(cmd, cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        out = proc.stdout or ""
        rc = proc.returncode
    except subprocess.TimeoutExpired as e:
        out = f"TIMEOUT after {timeout}s\n"
        rc = 124
    except Exception as e:
        out = f"EXC: {e}\n"
        rc = 2
    return rc, out


def main():
    if len(sys.argv) < 2:
        print("Usage: reproduce_and_triage.py <incident_dir>")
        sys.exit(2)
    incident_dir = sys.argv[1]
    if not os.path.isdir(incident_dir):
        print(f"Incident dir not found: {incident_dir}")
        sys.exit(2)

    # Read monitor summary if present
    monitor_summary = {}
    try:
        with open(os.path.join(incident_dir, "monitor_summary.json"), "r", encoding="utf-8") as f:
            monitor_summary = json.load(f)
    except Exception:
        pass

    attempts = []
    logs = []
    # Re-run a few times to detect reproducibility
    for i in range(3):
        rc, out = _run_backtest_once(timeout=300)
        attempts.append({"attempt": i + 1, "returncode": rc})
        logs.append(out)
        # small delay between runs
        time.sleep(2)

    # Analyze logs for exceptions and patterns
    combined = "\n\n---\n\n".join(logs)
    probable_reasons = _classify_from_log(combined)

    # Heuristic: if runs sometimes fail and sometimes pass -> stochastic noise or environment
    rc_set = set([a["returncode"] for a in attempts])
    if len(rc_set) > 1:
        if 0 in rc_set:
            probable_reasons.append("stochastic-noise")
        else:
            probable_reasons.append("environment")

    # If there are obvious tracebacks, prefer bug/exception
    if "traceback" in combined.lower() or re.search(r"\b(error|exception)\b", combined, re.IGNORECASE):
        probable_reasons.append("bug/exception")

    # Construct triage report
    triage = {
        "incident_dir": incident_dir,
        "checked_at": datetime.utcnow().isoformat() + "Z",
        "monitor_summary": monitor_summary,
        "reproduce_attempts": attempts,
        "likely_causes": sorted(set(probable_reasons)),
        "log_excerpt": combined[:2000],
    }

    # Write report to incident dir
    try:
        with open(os.path.join(incident_dir, "triage_report.json"), "w", encoding="utf-8") as f:
            json.dump(triage, f, indent=2)
        print(f"Wrote triage report to {os.path.join(incident_dir, 'triage_report.json')}")
    except Exception as e:
        print(f"Failed to write triage report: {e}")

    # Append to agent log
    agent_log = os.path.join(os.getcwd(), "artifacts", "agent_log.txt")
    try:
        with open(agent_log, "a", encoding="utf-8") as f:
            f.write(f"{datetime.utcnow().isoformat()}Z TRIAGE incident={incident_dir} causes={triage['likely_causes']}\n")
    except Exception:
        pass

    print(json.dumps(triage, indent=2))


if __name__ == "__main__":
    main()
