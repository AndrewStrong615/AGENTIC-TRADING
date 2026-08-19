"""
dashboard_server.py
Mobile-friendly trading bot dashboard.

Features
--------
- Nice password login screen
- Run / stop trading scripts
- Backtest ETA and elapsed time
- Live backtest charts while results are being written
- Backtest summary charts
- Live trading dashboard
- Hypothesis experiment history
- Mobile-first UI
- Works behind Tailscale / Funnel / Cloudflare Tunnel

Run:
    python dashboard_server.py

Password:
    setx DASHBOARD_PASSWORD "your-password"

Then open a NEW terminal and run the dashboard.

The server listens on:
    http://0.0.0.0:5000
"""

import os
import sys
import time
import signal
import json
import threading
import subprocess

from collections import deque
from datetime import datetime

import pandas as pd
from flask import (
    Flask,
    jsonify,
    request,
    send_file,
    Response,
    session,
    redirect,
)


# ============================================================================
# CONFIG
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "trade_data")
from notifications import send_notification, notify_run_finished, notify_new_trade, test_notification
os.makedirs(DATA_DIR, exist_ok=True)

PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "")

# Used to remember previous backtest durations.
TIMINGS_FILE = os.path.join(BASE_DIR, ".dashboard_timings.json")

app = Flask(__name__)

# Randomize this in a real deployment if desired.
app.secret_key = os.environ.get(
    "DASHBOARD_SECRET",
    "change-this-dashboard-secret"
)

# Prevent browsers from caching API responses.
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


# ============================================================================
# SCRIPTS
# ============================================================================

SCRIPTS = {
    "pretest_check": {
        "file": "pre_test.py",
        "label": "pre_test Check",
        "long_running": False,
    },

    "backtest": {
        "file": "backtest.py",
        "label": "Run Backtest",
        "long_running": False,
    },

    "combine_logs": {
        "file": "combine_logs.py",
        "label": "Combine Trade Logs",
        "long_running": False,
    },

    "train_meta_model": {
        "file": "train_meta_model.py",
        "label": "Train Meta-Model",
        "long_running": False,
    },

    "bot": {
        "file": "bot.py",
        "label": "Trading Bot",
        "long_running": True,
    },
}


# ============================================================================
# RUN STATE
# ============================================================================

RUNS = {}
RUNS_LOCK = threading.Lock()


# ============================================================================
# TIMING STORAGE
# ============================================================================

def load_timings():
    try:
        if not os.path.exists(TIMINGS_FILE):
            return {}

        with open(TIMINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data if isinstance(data, dict) else {}

    except Exception:
        return {}


def save_timings(data):
    try:
        with open(TIMINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


TIMINGS = load_timings()


def record_run_duration(script_key, seconds):
    if seconds is None or seconds <= 0:
        return

    history = TIMINGS.setdefault(script_key, [])

    history.append(round(seconds, 2))

    # Keep the last 10 runs.
    history[:] = history[-10:]

    save_timings(TIMINGS)


def estimated_duration(script_key):
    """
    Return a reasonable ETA estimate.

    For a backtest:
      - use average of previous runs when available
      - otherwise use 5 minutes as a starting estimate

    This is intentionally only an estimate.
    """

    history = TIMINGS.get(script_key, [])

    if history:
        return sum(history) / len(history)

    if script_key == "backtest":
        return 300.0

    if script_key == "train_meta_model":
        return 180.0

    if script_key == "combine_logs":
        return 30.0

    if script_key == "pre_test":
        return 20.0

    return None


# ============================================================================
# AUTHENTICATION
# ============================================================================

def logged_in():
    if not PASSWORD:
        return True

    return session.get("dashboard_authenticated") is True


def _check_password(value):
    if not PASSWORD:
        return True

    return value == PASSWORD


@app.before_request
def auth_gate():

    # Static-ish routes / login itself don't require auth.
    allowed = {
        "/login",
        "/logout",
        "/favicon.ico",
    }

    if request.path in allowed:
        return None

    if not PASSWORD:
        return None

    if logged_in():
        return None

    # API calls should get JSON rather than an HTML redirect.
    if request.path.startswith("/api/"):
        return jsonify({
            "error": "unauthorized",
            "login_required": True,
        }), 401

    return redirect("/login")


# ============================================================================
# LOGIN
# ============================================================================

LOGIN_HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">

<meta
    name="viewport"
    content="width=device-width,
             initial-scale=1,
             maximum-scale=1,
             user-scalable=no"
>

<meta name="theme-color" content="#090b10">

<title>Trading Bot — Login</title>

<style>

:root {
    --bg: #090b10;
    --card: #11151d;
    --card2: #171c25;
    --border: #252c38;
    --text: #f4f6fa;
    --muted: #8993a4;
    --blue: #557cff;
    --blue2: #765cff;
    --red: #ff6b73;
}

* {
    box-sizing: border-box;
}

html,
body {
    margin: 0;
    width: 100%;
    min-height: 100%;
}

body {
    background:
        radial-gradient(
            circle at 50% -10%,
            rgba(85,124,255,.22),
            transparent 45%
        ),
        var(--bg);

    color: var(--text);

    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "SF Pro Display",
        "Segoe UI",
        sans-serif;

    display: flex;
    justify-content: center;
    align-items: center;

    padding: 24px;
}

.login {
    width: 100%;
    max-width: 390px;
}

.logo {
    width: 58px;
    height: 58px;
    margin: 0 auto 18px;

    border-radius: 17px;

    background:
        linear-gradient(
            145deg,
            var(--blue),
            var(--blue2)
        );

    display: flex;
    align-items: center;
    justify-content: center;

    box-shadow:
        0 15px 45px rgba(85,124,255,.25);
}

.logo svg {
    width: 31px;
    height: 31px;
}

h1 {
    text-align: center;
    margin: 0;

    font-size: 25px;
    letter-spacing: -.03em;
}

.subtitle {
    text-align: center;
    color: var(--muted);
    font-size: 13px;
    margin-top: 7px;
    margin-bottom: 26px;
}

.card {
    background: rgba(17,21,29,.94);

    border:
        1px solid
        rgba(255,255,255,.07);

    border-radius: 22px;

    padding: 22px;

    box-shadow:
        0 25px 70px rgba(0,0,0,.45);
}

label {
    display: block;

    color: var(--muted);

    font-size: 11px;
    font-weight: 700;

    text-transform: uppercase;
    letter-spacing: .08em;

    margin-bottom: 8px;
}

input {
    width: 100%;

    background: var(--card2);

    color: var(--text);

    border:
        1px solid
        var(--border);

    border-radius: 12px;

    padding:
        14px
        14px;

    font-size: 16px;

    outline: none;

    transition:
        border .15s,
        box-shadow .15s;
}

input:focus {
    border-color: var(--blue);

    box-shadow:
        0 0 0 3px
        rgba(85,124,255,.14);
}

button {
    width: 100%;

    margin-top: 12px;

    border: 0;

    border-radius: 12px;

    padding: 14px;

    background:
        linear-gradient(
            135deg,
            var(--blue),
            var(--blue2)
        );

    color: white;

    font-size: 14px;
    font-weight: 700;

    cursor: pointer;
}

.error {
    background: rgba(255,107,115,.10);

    color: var(--red);

    border:
        1px solid
        rgba(255,107,115,.20);

    border-radius: 10px;

    padding: 10px 12px;

    font-size: 12px;

    margin-bottom: 12px;
}

.footer {
    text-align: center;

    color: #555e6d;

    font-size: 10px;

    margin-top: 18px;
}

</style>
</head>

<body>

<div class="login">

    <div class="logo">
        <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="white"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
        >
            <path d="M3 17l6-6 4 4 8-8"/>
            <path d="M15 7h6v6"/>
        </svg>
    </div>

    <h1>Trading Bot</h1>

    <div class="subtitle">
        Secure dashboard access
    </div>

    <div class="card">

        {% if error %}
        <div class="error">
            Incorrect password.
        </div>
        {% endif %}

        <form method="POST">

            <label for="password">
                Dashboard password
            </label>

            <input
                id="password"
                name="password"
                type="password"
                autocomplete="current-password"
                autofocus
                placeholder="Enter password"
            >

            <button type="submit">
                Unlock Dashboard
            </button>

        </form>

    </div>

    <div class="footer">
        Private trading control panel
    </div>

</div>

</body>
</html>
"""


@app.route("/login", methods=["GET", "POST"])
def login():

    if not PASSWORD:
        return redirect("/")

    if request.method == "POST":

        password = request.form.get("password", "")

        if _check_password(password):

            session["dashboard_authenticated"] = True

            return redirect("/")

        return Response(
            LOGIN_HTML.replace(
                "{% if error %}",
                ""
            ),
            mimetype="text/html"
        )

    return Response(
        LOGIN_HTML.replace(
            "{% if error %}",
            ""
        ),
        mimetype="text/html"
    )


@app.route("/logout")
def logout():

    session.clear()

    return redirect("/login")


# ============================================================================
# PROCESS READER
# ============================================================================

def _reader_thread(run_id, proc):

    entry = RUNS[run_id]

    try:

        for line in proc.stdout:

            entry["output"].append(
                line.rstrip("\n")
            )

    except Exception as e:

        entry["output"].append(
            f"[dashboard] reader error: {e}"
        )

    finally:

        proc.wait()

        entry["status"] = (
            "stopped"
            if entry["status"] == "stopping"
            else "finished"
        )

        entry["returncode"] = proc.returncode

        entry["ended_at"] = datetime.now().isoformat()
        script_label = SCRIPTS[entry["script"]]["label"]
        notify_run_finished(script_label, entry["status"], entry.get("duration_seconds"))

        started = entry.get("started_timestamp")
        ended = time.time()

        if started:

            duration = ended - started

            entry["duration_seconds"] = duration

            record_run_duration(
                entry["script"],
                duration
            )


# ============================================================================
# SCRIPT API
# ============================================================================
@app.route("/api/notifications/test", methods=["POST"])
def api_test_notification():
    test_notification()
    return jsonify({"sent": True})
@app.route("/api/scripts")
def api_scripts():

    out = []

    for key, cfg in SCRIPTS.items():

        running_id = None
        running_started = None

        for r in RUNS.values():

            if (
                r["script"] == key
                and r["status"] in ("running", "stopping")
            ):

                running_id = r["run_id"]
                running_started = r.get(
                    "started_timestamp"
                )

        estimate = estimated_duration(key)

        elapsed = None
        eta = None

        if running_started:

            elapsed = time.time() - running_started

            if estimate:
                eta = max(
                    estimate - elapsed,
                    0
                )

        out.append({
            "key": key,
            "label": cfg["label"],
            "long_running": cfg["long_running"],
            "running_id": running_id,
            "estimated_seconds": estimate,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
        })

    return jsonify(out)


@app.route("/api/run/<script_key>", methods=["POST"])
def api_run(script_key):

    if script_key not in SCRIPTS:
        return jsonify({
            "error": "unknown script"
        }), 404

    with RUNS_LOCK:

        for r in RUNS.values():

            if (
                r["script"] == script_key
                and r["status"] in ("running", "stopping")
            ):

                return jsonify({
                    "error":
                        f"{script_key} is already running",
                    "run_id":
                        r["run_id"],
                }), 409

        file_path = os.path.join(
            BASE_DIR,
            SCRIPTS[script_key]["file"]
        )

        if not os.path.exists(file_path):

            return jsonify({
                "error":
                    f"{SCRIPTS[script_key]['file']} "
                    "not found next to dashboard_server.py"
            }), 404

        run_id = (
            f"{script_key}-"
            f"{int(time.time() * 1000)}"
        )

        kwargs = dict(
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if os.name == "nt":

            kwargs["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP
            )

        proc = subprocess.Popen(
            [sys.executable, file_path],
            **kwargs
        )

        RUNS[run_id] = {

            "run_id": run_id,

            "process": proc,

            "output":
                deque(maxlen=3000),

            "status":
                "running",

            "script":
                script_key,

            "started_at":
                datetime.now().isoformat(),

            "started_timestamp":
                time.time(),

            "returncode":
                None,

            "ended_at":
                None,

            "duration_seconds":
                None,
        }

        threading.Thread(
            target=_reader_thread,
            args=(run_id, proc),
            daemon=True,
        ).start()

    return jsonify({
        "run_id":
            run_id,

        "estimated_seconds":
            estimated_duration(script_key),
    })


@app.route("/api/run/<run_id>/stop", methods=["POST"])
def api_stop(run_id):

    entry = RUNS.get(run_id)

    if not entry:

        return jsonify({
            "error": "unknown run_id"
        }), 404

    if entry["status"] not in (
        "running",
        "stopping",
    ):

        return jsonify({
            "status":
                entry["status"]
        })

    entry["status"] = "stopping"

    proc = entry["process"]

    try:

        if os.name == "nt":

            proc.send_signal(
                signal.CTRL_BREAK_EVENT
            )

        else:

            proc.send_signal(
                signal.SIGINT
            )

    except Exception as e:

        entry["output"].append(
            "[dashboard] stop signal failed "
            f"({e}), forcing terminate."
        )

        proc.terminate()

    return jsonify({
        "status": "stopping"
    })


@app.route("/api/run/<run_id>/status")
def api_run_status(run_id):

    entry = RUNS.get(run_id)

    if not entry:

        return jsonify({
            "error": "unknown run_id"
        }), 404

    since = int(
        request.args.get(
            "since",
            0
        )
    )

    lines = list(
        entry["output"]
    )

    elapsed = None
    eta = None
    estimate = estimated_duration(
        entry["script"]
    )

    if entry.get("started_timestamp"):

        elapsed = (
            time.time()
            - entry["started_timestamp"]
        )

        if estimate:

            eta = max(
                estimate - elapsed,
                0
            )

    return jsonify({

        "status":
            entry["status"],

        "returncode":
            entry["returncode"],

        "started_at":
            entry["started_at"],

        "ended_at":
            entry["ended_at"],

        "duration_seconds":
            entry.get("duration_seconds"),

        "estimated_seconds":
            estimate,

        "elapsed_seconds":
            elapsed,

        "eta_seconds":
            eta,

        "output":
            lines[since:],

        "total_lines":
            len(lines),
    })


# ============================================================================
# DATA HELPERS
# ============================================================================

def _read_csv_safe(path, **kwargs):

    if not os.path.exists(path):
        return None

    try:

        df = pd.read_csv(
            path,
            **kwargs
        )

        return (
            df
            if not df.empty
            else None
        )

    except Exception:

        return None


# ============================================================================
# BACKTEST DATA
# ============================================================================

@app.route("/api/data/backtest_summary")
def api_backtest_summary():

    path = os.path.join(
        DATA_DIR,
        "backtest_results.csv"
    )

    df = _read_csv_safe(path)

    if df is None:

        return jsonify({
            "available": False
        })

    result = {
        "available": True,
        "rows": len(df),
        "symbols": [],
        "returns_pct": [],
        "win_rates": [],
        "num_trades": [],
        "windows": [],
        "win_loss_pie": {
            "wins": 0,
            "losses": 0,
        },
        "has_chart_image":
            os.path.exists(
                os.path.join(
                    DATA_DIR,
                    "backtest_vs_spy.png"
                )
            ),
    }

    if "window" in df.columns:

        result["windows"] = (
            df["window"]
            .dropna()
            .astype(str)
            .tolist()
        )

        latest_window = (
            df["window"].iloc[-1]
        )

        latest = df[
            df["window"] ==
            latest_window
        ].copy()

        result["window"] = str(
            latest_window
        )

    else:

        latest = df.copy()
        result["window"] = "Latest"

    if "symbol" in latest.columns:

        result["symbols"] = (
            latest["symbol"]
            .astype(str)
            .tolist()
        )

    if "total_return_pct_mean" in latest.columns:

        result["returns_pct"] = [
            round(float(x), 2)
            for x in pd.to_numeric(
                latest[
                    "total_return_pct_mean"
                ],
                errors="coerce"
            ).fillna(0)
        ]

    if "win_rate_mean" in latest.columns:

        result["win_rates"] = [
            round(float(x), 1)
            for x in pd.to_numeric(
                latest[
                    "win_rate_mean"
                ],
                errors="coerce"
            ).fillna(0)
        ]

    if "num_trades_mean" in latest.columns:

        result["num_trades"] = [
            round(float(x), 0)
            for x in pd.to_numeric(
                latest[
                    "num_trades_mean"
                ],
                errors="coerce"
            ).fillna(0)
        ]

    if (
        result["num_trades"]
        and result["win_rates"]
    ):

        total_trades = sum(
            result["num_trades"]
        )

        est_wins = sum(
            n * (w / 100)
            for n, w in zip(
                result["num_trades"],
                result["win_rates"]
            )
        )

        est_losses = max(
            total_trades - est_wins,
            0
        )

        result["win_loss_pie"] = {
            "wins":
                round(est_wins, 1),

            "losses":
                round(est_losses, 1),
        }

    return jsonify(result)


# ============================================================================
# LIVE BACKTEST DATA
# ============================================================================

@app.route("/api/data/backtest_live")
def api_backtest_live():

    """
    This endpoint is intentionally tolerant of partial CSV files.

    If backtest.py writes rows as it goes, the charts update automatically.

    If backtest.py only writes backtest_results.csv after the entire run
    finishes, the charts will remain empty until the file is written.
    """

    path = os.path.join(
        DATA_DIR,
        "backtest_results.csv"
    )

    df = _read_csv_safe(path)

    if df is None:

        return jsonify({
            "available": False,
            "rows": 0,
            "symbols": [],
            "returns": [],
            "win_rates": [],
            "trades": [],
            "windows": [],
        })

    result = {
        "available": True,
        "rows": len(df),
        "symbols": [],
        "returns": [],
        "win_rates": [],
        "trades": [],
        "windows": [],
    }

    if "symbol" in df.columns:

        result["symbols"] = (
            df["symbol"]
            .astype(str)
            .tolist()
        )

    if "total_return_pct_mean" in df.columns:

        result["returns"] = [
            round(float(x), 3)
            for x in pd.to_numeric(
                df[
                    "total_return_pct_mean"
                ],
                errors="coerce"
            ).fillna(0)
        ]

    if "win_rate_mean" in df.columns:

        result["win_rates"] = [
            round(float(x), 2)
            for x in pd.to_numeric(
                df[
                    "win_rate_mean"
                ],
                errors="coerce"
            ).fillna(0)
        ]

    if "num_trades_mean" in df.columns:

        result["trades"] = [
            round(float(x), 0)
            for x in pd.to_numeric(
                df[
                    "num_trades_mean"
                ],
                errors="coerce"
            ).fillna(0)
        ]

    if "window" in df.columns:

        result["windows"] = (
            df["window"]
            .astype(str)
            .tolist()
        )

    return jsonify(result)


# ============================================================================
# BACKTEST IMAGE
# ============================================================================

@app.route("/api/chart-image/backtest_vs_spy")
def api_chart_image():

    path = os.path.join(
        DATA_DIR,
        "backtest_vs_spy.png"
    )

    if not os.path.exists(path):

        return jsonify({
            "error":
                "not generated yet — "
                "run a backtest first"
        }), 404

    return send_file(
        path,
        mimetype="image/png"
    )


# ============================================================================
# BACKTEST RUN ARCHIVE
#
# Expose past backtest runs saved under trade_data/backtests/ so the dashboard
# can list and inspect historical backtest results and charts.
# ============================================================================

@app.route("/api/backtests")
def api_backtest_runs():
    runs_dir = os.path.join(DATA_DIR, "backtests")
    out = []
    if not os.path.exists(runs_dir):
        return jsonify(out)

    for name in sorted(os.listdir(runs_dir), reverse=True):
        run_path = os.path.join(runs_dir, name)
        if not os.path.isdir(run_path):
            continue
        summary_path = os.path.join(run_path, "summary.json")
        results_path = os.path.join(run_path, "backtest_results.csv")
        chart_path = os.path.join(run_path, "backtest_vs_spy.png")
        job_path = os.path.join(run_path, "job.json")
        summary = {}
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
            except Exception:
                summary = {}
        # include list of files and whether there is a job.json for older runs
        files = []
        try:
            files = sorted(os.listdir(run_path))
        except Exception:
            files = []

        out.append({
            "run_id": name,
            "summary": summary,
            "results_available": os.path.exists(results_path),
            "chart_available": os.path.exists(chart_path),
            "job_json_available": os.path.exists(job_path),
            "files": files,
        })

    return jsonify(out)


@app.route("/api/backtests/<run_id>/results")
def api_backtest_run_results(run_id):
    path = os.path.join(DATA_DIR, "backtests", run_id, "backtest_results.csv")
    if not os.path.exists(path):
        return jsonify({"error": "not found"}), 404
    return send_file(path, mimetype="text/csv")


@app.route("/api/backtests/<run_id>/chart")
def api_backtest_run_chart(run_id):
    path = os.path.join(DATA_DIR, "backtests", run_id, "backtest_vs_spy.png")
    if not os.path.exists(path):
        return jsonify({"error": "not found"}), 404
    return send_file(path, mimetype="image/png")


@app.route("/api/backtests/<run_id>/files")
def api_backtest_run_files(run_id):
    run_path = os.path.join(DATA_DIR, "backtests", run_id)
    if not os.path.exists(run_path) or not os.path.isdir(run_path):
        return jsonify({"error": "not found"}), 404
    try:
        files = sorted(os.listdir(run_path))
        return jsonify({"files": files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtests/<run_id>/artifact/<path:filename>")
def api_backtest_run_artifact(run_id, filename):
    # Serve arbitrary artifacts from the run directory but ensure path safety
    run_path = os.path.join(DATA_DIR, "backtests", run_id)
    if not os.path.exists(run_path) or not os.path.isdir(run_path):
        return jsonify({"error": "not found"}), 404
    # Normalize and check path is inside run_path
    abs_requested = os.path.realpath(os.path.join(run_path, filename))
    abs_run = os.path.realpath(run_path)
    if not abs_requested.startswith(abs_run):
        return jsonify({"error": "forbidden"}), 403
    if not os.path.exists(abs_requested):
        return jsonify({"error": "not found"}), 404
    # Guess mimetype via extension for common types; leave to send_file otherwise
    ext = os.path.splitext(abs_requested)[1].lower()
    if ext in (".png", ".jpg", ".jpeg", ".gif"):
        return send_file(abs_requested, mimetype=f"image/{ext.lstrip('.')}")
    if ext in (".csv", ".txt", ".json"):
        return send_file(abs_requested, mimetype="text/plain")
    return send_file(abs_requested)


@app.route("/api/backtests/<run_id>/summary")
def api_backtest_run_summary(run_id):
    """
    Build a lightweight summary from per-symbol backtest trade logs when
    backtest_results.csv/summary.json are not available. Returns JSON with
    per-symbol metrics: total_return_pct, num_trades, win_rate_pct, avg_pnl_pct.
    """
    run_path = os.path.join(DATA_DIR, "backtests", run_id)
    if not os.path.exists(run_path) or not os.path.isdir(run_path):
        return jsonify({"error": "not found"}), 404
    try:
        summaries = []
        for fname in sorted(os.listdir(run_path)):
            if not fname.startswith("backtest_trade_log_") or not fname.endswith(".csv"):
                continue
            fpath = os.path.join(run_path, fname)
            try:
                df = pd.read_csv(fpath)
            except Exception:
                continue
            if "pnl_pct" not in df.columns:
                continue
            df["pnl_pct"] = pd.to_numeric(df["pnl_pct"], errors="coerce")
            df = df.dropna(subset=["pnl_pct"])
            if df.empty:
                continue
            symbol = df.get("symbol", pd.Series([None])).iloc[0]
            num_trades = int(len(df))
            total_return = float(df["pnl_pct"].sum())
            wins = int((df["pnl_pct"] > 0).sum())
            win_rate = float(wins / num_trades * 100) if num_trades else 0.0
            avg_pnl = float(df["pnl_pct"].mean())
            summaries.append({
                "symbol": symbol,
                "num_trades": num_trades,
                "total_return_pct": round(total_return, 4),
                "win_rate_pct": round(win_rate, 2),
                "avg_pnl_pct": round(avg_pnl, 4),
                "file": fname,
            })
        # Sort by total_return_pct desc
        summaries = sorted(summaries, key=lambda x: x["total_return_pct"], reverse=True)
        return jsonify({"available": True, "summaries": summaries})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# LIVE TRADING
# ============================================================================

@app.route("/api/data/live_summary")
def api_live_summary():

    path = os.path.join(
        DATA_DIR,
        "trade_log.csv"
    )

    df = _read_csv_safe(path)

    if df is None:

        return jsonify({
            "available": False
        })

    if "exit_timestamp" in df.columns:

        df = df.sort_values(
            "exit_timestamp"
        )

    if "pnl_pct" not in df.columns:

        return jsonify({
            "available": False
        })

    df["pnl_pct"] = pd.to_numeric(
        df["pnl_pct"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["pnl_pct"]
    )

    if df.empty:

        return jsonify({
            "available": False
        })

    wins = int(
        (df["pnl_pct"] > 0).sum()
    )

    losses = int(
        (df["pnl_pct"] <= 0).sum()
    )

    cum = [
        round(float(x), 3)
        for x in df[
            "pnl_pct"
        ].cumsum()
    ]

    timestamps = [
        str(x)[:19]
        for x in df[
            "exit_timestamp"
        ]
    ]

    recent = df.tail(15).copy()

    cols = [
        c for c in [
            "exit_timestamp",
            "symbol",
            "pnl_pct",
            "exit_reason",
        ]
        if c in recent.columns
    ]

    recent_trades = (
        recent[cols]
        .fillna("")
        .to_dict(
            orient="records"
        )
    )

    trader_log_tail = []

    log_file = os.path.join(
        DATA_DIR,
        "trader.log"
    )

    if os.path.exists(log_file):

        try:

            with open(
                log_file,
                "r",
                errors="ignore"
            ) as f:

                trader_log_tail = [
                    line.rstrip("\n")
                    for line in f.readlines()[-30:]
                ]

        except Exception:
            pass

    return jsonify({

        "available":
            True,

        "win_loss_pie": {
            "wins":
                wins,

            "losses":
                losses,
        },

        "cumulative_pnl_pct":
            cum,

        "timestamps":
            timestamps,

        "recent_trades":
            recent_trades,

        "trader_log_tail":
            trader_log_tail,
    })


# ============================================================================
# HYPOTHESIS
# ============================================================================

@app.route("/api/data/hypothesis")
def api_hypothesis():

    path = os.path.join(
        DATA_DIR,
        "hypothesis_log.csv"
    )

    df = _read_csv_safe(path)

    if df is None:

        return jsonify({
            "available": False
        })

    df = df.fillna("")

    return jsonify({
        "available": True,
        "rows":
            df.tail(50).to_dict(
                orient="records"
            ),
    })


# ============================================================================
# FRONTEND
# ============================================================================

INDEX_HTML = r"""
<!doctype html>

<html lang="en">

<head>

<meta charset="utf-8">

<meta
    name="viewport"
    content="width=device-width,
             initial-scale=1,
             maximum-scale=1,
             user-scalable=no"
>

<meta
    name="theme-color"
    content="#090b10"
>

<meta
    name="apple-mobile-web-app-capable"
    content="yes"
>

<title>Trading Bot Dashboard</title>

<script
src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js">
</script>

<style>

/* ========================================================================
   VARIABLES
   ======================================================================== */

:root {

    --bg:
        #090b10;

    --surface:
        #11151c;

    --surface2:
        #171c24;

    --surface3:
        #202631;

    --border:
        #252c37;

    --border-soft:
        #1b2029;

    --text:
        #f1f4f8;

    --muted:
        #8e98a8;

    --faint:
        #596271;

    --blue:
        #557cff;

    --blue-soft:
        rgba(85,124,255,.12);

    --green:
        #31d58a;

    --green-soft:
        rgba(49,213,138,.12);

    --red:
        #ff6b73;

    --red-soft:
        rgba(255,107,115,.12);

    --amber:
        #f2b84b;

    --radius:
        15px;

    --radius-sm:
        10px;

    --mono:
        "SF Mono",
        "Roboto Mono",
        Menlo,
        Consolas,
        monospace;
}


/* ========================================================================
   BASE
   ======================================================================== */

* {
    box-sizing:
        border-box;

    -webkit-tap-highlight-color:
        transparent;
}

html,
body {

    margin:
        0;

    padding:
        0;

    background:
        var(--bg);

    color:
        var(--text);

    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "SF Pro Text",
        "Segoe UI",
        Inter,
        sans-serif;

    -webkit-font-smoothing:
        antialiased;
}

body {
    min-height:
        100vh;
}

button {
    font-family:
        inherit;
}


/* ========================================================================
   APP
   ======================================================================== */

#app {

    max-width:
        700px;

    margin:
        0 auto;

    padding:
        18px 15px 110px;
}


/* ========================================================================
   HEADER
   ======================================================================== */

.top {

    display:
        flex;

    align-items:
        center;

    justify-content:
        space-between;

    padding-bottom:
        17px;

    margin-bottom:
        20px;

    border-bottom:
        1px solid
        var(--border-soft);
}

.brand {

    display:
        flex;

    align-items:
        center;

    gap:
        10px;
}

.mark {

    width:
        34px;

    height:
        34px;

    border-radius:
        10px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    background:
        linear-gradient(
            145deg,
            #557cff,
            #765cff
        );
}

.mark svg {

    width:
        18px;

    height:
        18px;
}

h1 {

    margin:
        0;

    font-size:
        16px;

    font-weight:
        700;
}

.subtitle {

    color:
        var(--faint);

    font-size:
        10px;

    margin-top:
        2px;
}

.connection {

    display:
        flex;

    align-items:
        center;

    gap:
        6px;

    color:
        var(--faint);

    font-size:
        10px;

    font-weight:
        700;
}

.connection-dot {

    width:
        7px;

    height:
        7px;

    border-radius:
        50%;

    background:
        var(--faint);
}

.connection-dot.ok {

    background:
        var(--green);

    box-shadow:
        0 0 0 3px
        var(--green-soft);
}

.connection-dot.bad {

    background:
        var(--red);

    box-shadow:
        0 0 0 3px
        var(--red-soft);
}


/* ========================================================================
   SECTIONS
   ======================================================================== */

.section-title {

    display:
        flex;

    align-items:
        center;

    justify-content:
        space-between;

    margin:
        22px 2px 9px;
}

.section-title h2 {

    margin:
        0;

    color:
        var(--faint);

    font-size:
        10px;

    text-transform:
        uppercase;

    letter-spacing:
        .1em;

    font-weight:
        800;
}


/* ========================================================================
   CARDS
   ======================================================================== */

.card {

    background:
        var(--surface);

    border:
        1px solid
        var(--border-soft);

    border-radius:
        var(--radius);

    padding:
        15px;

    margin-bottom:
        9px;
}

.card-header {

    display:
        flex;

    justify-content:
        space-between;

    align-items:
        center;

    gap:
        10px;
}

.card-title {

    font-size:
        14px;

    font-weight:
        650;
}

.muted {

    color:
        var(--muted);

    font-size:
        11px;
}

.faint {

    color:
        var(--faint);
}


/* ========================================================================
   BUTTONS
   ======================================================================== */

.btn-row {

    display:
        flex;

    gap:
        8px;

    margin-top:
        12px;
}

.btn {

    flex:
        1;

    border:
        0;

    border-radius:
        var(--radius-sm);

    padding:
        11px 13px;

    font-size:
        12px;

    font-weight:
        700;

    cursor:
        pointer;
}

.btn.primary {

    color:
        white;

    background:
        var(--blue);
}

.btn.primary:disabled {

    background:
        var(--surface3);

    color:
        var(--faint);
}

.btn.stop {

    background:
        var(--surface2);

    color:
        var(--red);

    border:
        1px solid
        var(--border);
}

.btn.stop:disabled {

    color:
        var(--faint);
}


/* ========================================================================
   BADGES
   ======================================================================== */

.badge {

    border-radius:
        999px;

    padding:
        4px 9px;

    font-size:
        9px;

    font-weight:
        800;

    text-transform:
        uppercase;

    background:
        var(--surface2);

    color:
        var(--muted);
}

.badge.running {

    background:
        var(--green-soft);

    color:
        var(--green);
}


/* ========================================================================
   ETA
   ======================================================================== */

.eta {

    margin-top:
        10px;

    padding:
        10px 11px;

    border-radius:
        10px;

    background:
        var(--blue-soft);

    border:
        1px solid
        rgba(85,124,255,.18);

    color:
        #aebeff;

    font-size:
        11px;

    line-height:
        1.45;
}

.eta strong {

    color:
        white;
}

.progress {

    margin-top:
        8px;

    height:
        5px;

    overflow:
        hidden;

    background:
        var(--surface3);

    border-radius:
        999px;
}

.progress-bar {

    height:
        100%;

    width:
        0%;

    background:
        linear-gradient(
            90deg,
            var(--blue),
            #8b6cff
        );

    transition:
        width .4s ease;
}


/* ========================================================================
   OUTPUT
   ======================================================================== */

pre.output {

    display:
        none;

    margin:
        12px 0 0;

    padding:
        11px;

    max-height:
        220px;

    overflow:
        auto;

    white-space:
        pre-wrap;

    word-break:
        break-word;

    background:
        #080a0e;

    border:
        1px solid
        var(--border-soft);

    border-radius:
        var(--radius-sm);

    color:
        #aeb6c4;

    font-family:
        var(--mono);

    font-size:
        10px;

    line-height:
        1.55;
}


/* ========================================================================
   TABS
   ======================================================================== */

.tab-page {
    display:
        none;
}

.tab-page.active {
    display:
        block;
    animation:
        fade .15s ease;
}

@keyframes fade {

    from {
        opacity:
            0;
        transform:
            translateY(3px);
    }

    to {
        opacity:
            1;
        transform:
            none;
    }
}


/* ========================================================================
   BOTTOM NAV
   ======================================================================== */

#tabbar {

    position:
        fixed;

    bottom:
        12px;

    left:
        0;

    right:
        0;

    display:
        flex;

    justify-content:
        center;

    z-index:
        100;
}

.tab-inner {

    width:
        calc(100% - 26px);

    max-width:
        674px;

    padding:
        5px;

    display:
        flex;

    background:
        rgba(17,21,28,.94);

    border:
        1px solid
        var(--border);

    border-radius:
        18px;

    backdrop-filter:
        blur(20px);

    box-shadow:
        0 15px 45px
        rgba(0,0,0,.45);
}

.tab-button {

    flex:
        1;

    border:
        0;

    background:
        none;

    color:
        var(--faint);

    border-radius:
        13px;

    padding:
        8px 2px;

    font-size:
        9px;

    font-weight:
        700;
}

.tab-button.active {

    background:
        var(--surface3);

    color:
        var(--text);
}

.tab-button svg {

    display:
        block;

    width:
        18px;

    height:
        18px;

    margin:
        0 auto 3px;
}


/* ========================================================================
   STATS
   ======================================================================== */

.stat-grid {

    display:
        grid;

    grid-template-columns:
        repeat(2, 1fr);

    gap:
        9px;
}

.stat {

    background:
        var(--surface);

    border:
        1px solid
        var(--border-soft);

    border-radius:
        var(--radius);

    padding:
        13px;
}

.stat-label {

    color:
        var(--faint);

    font-size:
        9px;

    text-transform:
        uppercase;

    letter-spacing:
        .06em;

    font-weight:
        800;
}

.stat-value {

    margin-top:
        5px;

    font-family:
        var(--mono);

    font-size:
        18px;

    font-weight:
        700;
}

.green {
    color:
        var(--green);
}

.red {
    color:
        var(--red);
}

.amber {
    color:
        var(--amber);
}


/* ========================================================================
   CHARTS
   ======================================================================== */

.chart-card {

    padding:
        15px;
}

.chart-title {

    color:
        var(--muted);

    font-size:
        11px;

    font-weight:
        650;

    margin-bottom:
        12px;
}

canvas {

    width:
        100% !important;

    max-height:
        230px;
}

.chart-placeholder {

    height:
        150px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    color:
        var(--faint);

    font-size:
        11px;

    text-align:
        center;
}


/* ========================================================================
   TABLE
   ======================================================================== */

.table-wrap {

    overflow-x:
        auto;
}

table {

    width:
        100%;

    border-collapse:
        collapse;

    font-size:
        11px;
}

th,
td {

    padding:
        8px 5px;

    border-bottom:
        1px solid
        var(--border-soft);

    text-align:
        left;

    white-space:
        nowrap;
}

th {

    color:
        var(--faint);

    font-size:
        9px;

    text-transform:
        uppercase;
}

td.num,
th.num {

    text-align:
        right;

    font-family:
        var(--mono);
}

tr:last-child td {

    border-bottom:
        0;
}


/* ========================================================================
   EMPTY
   ======================================================================== */

.empty {

    text-align:
        center;

    padding:
        35px 12px;

    color:
        var(--faint);

    font-size:
        12px;
}


/* ========================================================================
   LOGOUT
   ======================================================================== */

.logout {

    color:
        var(--faint);

    text-decoration:
        none;

    font-size:
        10px;

    font-weight:
        700;
}

</style>
</head>


<body>

<div id="app">

<header class="top">

    <div class="brand">

        <div class="mark">

            <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="white"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
            >
                <path d="M3 17l6-6 4 4 8-8"/>
                <path d="M15 7h6v6"/>
            </svg>

        </div>

        <div>

            <h1>Trading Bot</h1>

            <div class="subtitle">
                Control & monitoring
            </div>

        </div>

    </div>

    <div class="connection">

        <span
            id="conn-dot"
            class="connection-dot"
        ></span>

        <span id="conn-label">
            Connecting
        </span>

    </div>

</header>


<!-- =====================================================================
     RUN
     ===================================================================== -->

<div
    class="tab-page active"
    id="page-run"
>

    <div class="section-title">

        <h2>Scripts</h2>

        <a
            href="/logout"
            class="logout"
        >
            Logout
        </a>

    </div>

    <div id="scripts-list">
        <div class="card">
        <div class="card-title">Notifications</div>
        <div class="muted" style="margin-top:4px;">Set NTFY_TOPIC before starting the dashboard to enable.</div>
            <div class="btn-row">
        <button class="btn primary" onclick="apiFetch('/api/notifications/test',{method:'POST'})">Send Test</button>
    </div>
</div>

        <div class="card">
            Loading scripts...
        </div>

    </div>

</div>


<!-- =====================================================================
     BACKTEST
     ===================================================================== -->

<div
    class="tab-page"
    id="page-backtest"
>

    <div class="section-title">

        <h2>Backtesting</h2>

        <span
            id="backtest-live-label"
            class="badge"
        >
            Waiting
        </span>

    </div>


    <div id="backtest-status"></div>


    <div id="backtest-content">

        <div class="empty">
            Loading backtest data...
        </div>

    </div>

</div>


<!-- =====================================================================
     LIVE
     ===================================================================== -->

<div
    class="tab-page"
    id="page-live"
>

    <div class="section-title">

        <h2>Live Trading</h2>

        <span class="badge">
            Auto
        </span>

    </div>

    <div id="live-content">

        <div class="empty">
            Loading live data...
        </div>

    </div>

</div>


<!-- =====================================================================
     HYPOTHESIS
     ===================================================================== -->

<div
    class="tab-page"
    id="page-hyp"
>

    <div class="section-title">

        <h2>Experiments</h2>

    </div>

    <div id="hyp-content">

        <div class="empty">
            Loading experiments...
        </div>

    </div>

</div>


</div>


<!-- =====================================================================
     NAV
     ===================================================================== -->

<div id="tabbar">

<div class="tab-inner">

<button
    class="tab-button active"
    data-tab="run"
>

    <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
    >
        <path d="M6 4l14 8-14 8V4z"/>
    </svg>

    Run

</button>


<button
    class="tab-button"
    data-tab="backtest"
>

    <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
    >
        <path d="M4 20V10"/>
        <path d="M10 20V4"/>
        <path d="M16 20v-7"/>
        <path d="M22 20H2"/>
    </svg>

    Backtest

</button>


<button
    class="tab-button"
    data-tab="live"
>

    <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
    >
        <path d="M2 12h4l3-8 6 16 3-8h4"/>
    </svg>

    Live

</button>


<button
    class="tab-button"
    data-tab="hyp"
>

    <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
    >
        <path d="M9 2h6"/>
        <path d="M10 2v6l-6 11a2 2 0 002 3h12a2 2 0 002-3l-6-11V2"/>
    </svg>

    Tests

</button>

</div>

</div>


<script>

/* ========================================================================
   GLOBALS
   ======================================================================== */

const scriptEls = {};
const runningPolls = {};

let backtestCharts = {};
let liveCharts = {};

let backtestRefreshTimer = null;
let liveRefreshTimer = null;


/* ========================================================================
   CONNECTION
   ======================================================================== */

function setConnection(ok) {

    const dot =
        document.getElementById(
            "conn-dot"
        );

    const label =
        document.getElementById(
            "conn-label"
        );

    dot.className =
        "connection-dot " +
        (ok ? "ok" : "bad");

    label.textContent =
        ok
            ? "Connected"
            : "Offline";
}


/* ========================================================================
   API
   ======================================================================== */

async function apiFetch(
    path,
    options = {}
) {

    try {

        const response =
            await fetch(
                path,
                {
                    ...options,
                    cache:
                        "no-store",
                    credentials:
                        "same-origin",
                }
            );

        if (
            response.status === 401
        ) {

            window.location.href =
                "/login";

            throw new Error(
                "Login required"
            );
        }

        setConnection(
            response.ok
        );

        return response;

    } catch (error) {

        setConnection(false);

        throw error;
    }
}


/* ========================================================================
   FORMATTING
   ======================================================================== */

function fmtPct(value) {

    const n =
        Number(value);

    if (
        Number.isNaN(n)
    ) {
        return "—";
    }

    return (
        (n > 0 ? "+" : "") +
        n.toFixed(2) +
        "%"
    );
}


function fmtSeconds(seconds) {

    if (
        seconds === null ||
        seconds === undefined ||
        !Number.isFinite(
            Number(seconds)
        )
    ) {

        return "calculating…";
    }

    seconds =
        Math.max(
            0,
            Number(seconds)
        );

    const h =
        Math.floor(
            seconds / 3600
        );

    const m =
        Math.floor(
            (seconds % 3600) / 60
        );

    const s =
        Math.floor(
            seconds % 60
        );

    if (h > 0) {

        return (
            h +
            "h " +
            m +
            "m"
        );
    }

    if (m > 0) {

        return (
            m +
            "m " +
            s +
            "s"
        );
    }

    return (
        s +
        "s"
    );
}


/* ========================================================================
   TABS
   ======================================================================== */

document
.querySelectorAll(
    ".tab-button"
)
.forEach(button => {

    button.addEventListener(
        "click",
        () => {

            document
            .querySelectorAll(
                ".tab-button"
            )
            .forEach(
                b =>
                    b.classList.remove(
                        "active"
                    )
            );

            document
            .querySelectorAll(
                ".tab-page"
            )
            .forEach(
                page =>
                    page.classList.remove(
                        "active"
                    )
            );

            button.classList.add(
                "active"
            );

            const tab =
                button.dataset.tab;

            document
                .getElementById(
                    "page-" + tab
                )
                .classList.add(
                    "active"
                );

            if (
                tab === "backtest"
            ) {

                loadBacktest();

            } else if (
                tab === "live"
            ) {

                loadLive();
            }else if(
                tab=="hyp")
                {
                    loadHypothesis();

                }

        }
    );

});


/* ========================================================================
   SCRIPT CARDS
   ======================================================================== */

function buildScriptCard(
    script
) {

    const card =
        document.createElement(
            "div"
        );

    card.className =
        "card";

    card.innerHTML = `

        <div class="card-header">

            <span class="card-title">
                ${script.label}
            </span>

            <span
                class="badge
                ${script.running_id
                    ? "running"
                    : ""}"
            >
                ${script.running_id
                    ? "Running"
                    : "Idle"}
            </span>

        </div>

        <div
            class="eta"
            style="display:none"
        ></div>

        <div
            class="progress"
            style="display:none"
        >
            <div
                class="progress-bar"
            ></div>
        </div>

        <div class="btn-row"></div>

        <pre class="output"></pre>
    `;


    const row =
        card.querySelector(
            ".btn-row"
        );

    const badge =
        card.querySelector(
            ".badge"
        );

    const eta =
        card.querySelector(
            ".eta"
        );

    const progress =
        card.querySelector(
            ".progress"
        );

    const progressBar =
        card.querySelector(
            ".progress-bar"
        );

    const output =
        card.querySelector(
            ".output"
        );


    const runButton =
        document.createElement(
            "button"
        );

    runButton.className =
        "btn primary";

    runButton.textContent =
        script.running_id
            ? "Running…"
            : "Run";

    runButton.disabled =
        !!script.running_id;


    const stopButton =
        document.createElement(
            "button"
        );

    stopButton.className =
        "btn stop";

    stopButton.textContent =
        "Stop";

    stopButton.disabled =
        !script.running_id;


    const refs = {

        card,

        badge,

        eta,

        progress,

        progressBar,

        output,

        runButton,

        stopButton,

        currentRunId:
            script.running_id ||
            null,
    };


    runButton.onclick =
        () =>
            runScript(
                script.key
            );


    stopButton.onclick =
        () =>
            stopScript(
                refs.currentRunId
            );


    row.appendChild(
        runButton
    );

    row.appendChild(
        stopButton
    );


    if (
        script.running_id
    ) {

        updateEtaUI(
            refs,
            script
        );

        output.style.display =
            "block";

        pollRun(
            script.running_id,
            script.key
        );
    }


    return refs;
}


function updateEtaUI(
    refs,
    data
) {

    const running =
        !!data.running_id;

    if (!running) {

        refs.eta.style.display =
            "none";

        refs.progress.style.display =
            "none";

        return;
    }


    refs.eta.style.display =
        "block";

    refs.progress.style.display =
        "block";


    const elapsed =
        Number(
            data.elapsed_seconds
        ) || 0;

    const estimate =
        Number(
            data.estimated_seconds
        ) || 0;

    const eta =
        Number(
            data.eta_seconds
        );


    if (
        estimate > 0
    ) {

        let percent =
            (
                elapsed /
                estimate
            ) * 100;

        percent =
            Math.min(
                Math.max(
                    percent,
                    2
                ),
                95
            );

        refs.progressBar.style.width =
            percent + "%";
    }


    if (
        Number.isFinite(
            eta
        )
    ) {

        refs.eta.innerHTML = `

            <strong>
                Estimated remaining:
                ${fmtSeconds(eta)}
            </strong>

            <br>

            Elapsed:
            ${fmtSeconds(elapsed)}

            · Typical run:
            ${fmtSeconds(estimate)}

        `;

    } else {

        refs.eta.innerHTML = `

            <strong>
                Estimated time:
                ${fmtSeconds(estimate)}
            </strong>

            <br>

            Elapsed:
            ${fmtSeconds(elapsed)}

        `;
    }
}


function patchScriptCard(
    refs,
    script
) {

    const running =
        !!script.running_id;

    refs.currentRunId =
        script.running_id ||
        null;

    refs.badge.textContent =
        running
            ? "Running"
            : "Idle";

    refs.badge.className =
        "badge" +
        (
            running
                ? " running"
                : ""
        );

    refs.runButton.disabled =
        running;

    refs.runButton.textContent =
        running
            ? "Running…"
            : "Run";

    refs.stopButton.disabled =
        !running;

    updateEtaUI(
        refs,
        script
    );
}


/* ========================================================================
   LOAD SCRIPTS
   ======================================================================== */

async function loadScripts() {

    try {

        const response =
            await apiFetch(
                "/api/scripts"
            );

        if (!response.ok)
            return;

        const scripts =
            await response.json();

        const container =
            document.getElementById(
                "scripts-list"
            );

        container.innerHTML = "";

        scripts.forEach(
            script => {

                const refs =
                    buildScriptCard(
                        script
                    );

                scriptEls[
                    script.key
                ] = refs;

                container.appendChild(
                    refs.card
                );

            }
        );

    } catch (error) {

        console.error(error);
    }
}


async function refreshScriptStatuses() {

    try {

        const response =
            await apiFetch(
                "/api/scripts"
            );

        if (!response.ok)
            return;

        const scripts =
            await response.json();

        for (
            const script of scripts
        ) {

            if (
                !scriptEls[
                    script.key
                ]
            ) {

                await loadScripts();
                return;
            }

            patchScriptCard(
                scriptEls[
                    script.key
                ],
                script
            );

            if (
                script.running_id
            ) {

                pollRun(
                    script.running_id,
                    script.key
                );
            }
        }

    } catch (error) {

        console.error(error);
    }
}


/* ========================================================================
   RUN SCRIPT
   ======================================================================== */

async function runScript(
    key
) {

    try {

        const response =
            await apiFetch(
                "/api/run/" + key,
                {
                    method:
                        "POST",
                }
            );

        const data =
            await response.json();

        if (!response.ok) {

            alert(
                data.error ||
                "Failed to start"
            );

            return;
        }

        const refs =
            scriptEls[key];

        if (refs) {

            patchScriptCard(
                refs,
                {
                    running_id:
                        data.run_id,

                    estimated_seconds:
                        data.estimated_seconds,

                    elapsed_seconds:
                        0,

                    eta_seconds:
                        data.estimated_seconds,
                }
            );
        }

        pollRun(
            data.run_id,
            key
        );

        /*
         * If this is a backtest, immediately switch to the backtest
         * page so the user can watch it.
         */
        if (
            key === "backtest"
        ) {

            document
                .querySelector(
                    '[data-tab="backtest"]'
                )
                .click();
        }

    } catch (error) {

        console.error(error);

        alert(
            "Could not contact dashboard."
        );
    }
}


/* ========================================================================
   STOP
   ======================================================================== */

async function stopScript(
    runId
) {

    if (!runId)
        return;

    await apiFetch(
        "/api/run/" +
        runId +
        "/stop",
        {
            method:
                "POST",
        }
    );
}


/* ========================================================================
   POLL PROCESS
   ======================================================================== */

function pollRun(
    runId,
    key
) {

    if (
        runningPolls[runId]
    )
        return;

    const refs =
        scriptEls[key];

    if (!refs)
        return;

    refs.output.style.display =
        "block";


    let since = 0;


    runningPolls[runId] =
        setInterval(
            async () => {

                try {

                    const response =
                        await apiFetch(
                            "/api/run/" +
                            runId +
                            "/status?since=" +
                            since
                        );

                    if (
                        !response.ok
                    )
                        return;

                    const data =
                        await response.json();


                    if (
                        data.output &&
                        data.output.length
                    ) {

                        data.output.forEach(
                            line => {

                                refs.output.textContent +=
                                    line + "\n";

                            }
                        );

                        refs.output.scrollTop =
                            refs.output.scrollHeight;

                        since =
                            data.total_lines;
                    }


                    updateEtaUI(
                        refs,
                        {
                            running_id:
                                (
                                    data.status ===
                                    "running" ||
                                    data.status ===
                                    "stopping"
                                )
                                    ? runId
                                    : null,

                            elapsed_seconds:
                                data.elapsed_seconds,

                            estimated_seconds:
                                data.estimated_seconds,

                            eta_seconds:
                                data.eta_seconds,
                        }
                    );


                    /*
                     * BACKTEST LIVE CHARTS
                     *
                     * Refresh whenever the backtest process is alive.
                     */

                    if (
                        key === "backtest"
                    ) {

                        loadBacktest(
                            true
                        );
                    }


                    if (
                        data.status !==
                            "running" &&
                        data.status !==
                            "stopping"
                    ) {

                        clearInterval(
                            runningPolls[
                                runId
                            ]
                        );

                        delete runningPolls[
                            runId
                        ];


                        patchScriptCard(
                            refs,
                            {
                                running_id:
                                    null
                            }
                        );


                        if (
                            key ===
                            "backtest"
                        ) {

                            await loadBacktest(
                                true
                            );
                        }
                    }

                } catch (error) {

                    console.error(
                        error
                    );
                }

            },
            1200
        );
}


/* ========================================================================
   DESTROY CHARTS
   ======================================================================== */

function destroyBacktestCharts() {

    Object.values(
        backtestCharts
    ).forEach(
        chart => {

            try {
                chart.destroy();
            } catch (_) {}

        }
    );

    backtestCharts = {};
}


/* ========================================================================
   BACKTEST
   ======================================================================== */

async function loadBacktest(
    silent = false
) {

    try {

        const response =
            await apiFetch(
                "/api/data/backtest_summary"
            );

        const data =
            await response.json();

        const content =
            document.getElementById(
                "backtest-content"
            );


        /*
         * If no results exist yet, still show the live area.
         */

        if (
            !data.available
        ) {

            content.innerHTML = `

                <div class="card">

                    <div class="empty">
                        No completed backtest results yet.
                        <br><br>
                        Start a backtest from the
                        <b>Run</b> tab.
                    </div>

                </div>

                <div
                    class="card chart-card"
                >

                    <div class="chart-title">
                        Live backtest results
                    </div>

                    <div
                        class="chart-placeholder"
                    >
                        Waiting for
                        backtest_results.csv…
                    </div>

                </div>
            `;

            destroyBacktestCharts();

            return;
        }


        const returns =
            data.returns_pct || [];

        const winRates =
            data.win_rates || [];

        const trades =
            data.num_trades || [];

        const symbols =
            data.symbols || [];


        const best =
            returns.length
                ? Math.max(
                    ...returns
                )
                : 0;


        const avgWin =
            winRates.length
                ? winRates.reduce(
                    (a, b) =>
                        a + b,
                    0
                ) /
                  winRates.length
                : 0;


        /*
         * Don't rebuild the whole page on every poll if it already exists.
         */

        if (
            !document.getElementById(
                "backtestPie"
            )
        ) {

            content.innerHTML = `

                <div class="stat-grid">

                    <div class="stat">

                        <div class="stat-label">
                            Best Return
                        </div>

                        <div
                            id="bt-best"
                            class="stat-value"
                        >
                            —
                        </div>

                    </div>


                    <div class="stat">

                        <div class="stat-label">
                            Avg Win Rate
                        </div>

                        <div
                            id="bt-winrate"
                            class="stat-value"
                        >
                            —
                        </div>

                    </div>

                </div>


                <div class="card">

                    <div class="muted">
                        Backtest window
                    </div>

                    <div
                        id="bt-window"
                        style="
                            margin-top:5px;
                            font-size:13px;
                            font-weight:650;
                        "
                    >
                        —
                    </div>

                </div>


                <div class="card chart-card">

                    <div class="chart-title">
                        Win / Loss
                    </div>

                    <canvas
                        id="backtestPie"
                    ></canvas>

                </div>


                <div class="card chart-card">

                    <div class="chart-title">
                        Return % by Symbol
                    </div>

                    <canvas
                        id="backtestReturns"
                    ></canvas>

                </div>


                <div class="card chart-card">

                    <div class="chart-title">
                        Win Rate by Symbol
                    </div>

                    <canvas
                        id="backtestWinRate"
                    ></canvas>

                </div>


                <div class="card chart-card">

                    <div class="chart-title">
                        Number of Trades
                    </div>

                    <canvas
                        id="backtestTrades"
                    ></canvas>

                </div>


                <div
                    id="spy-chart-container"
                ></div>

            `;
        }


        document.getElementById(
            "bt-best"
        ).textContent =
            fmtPct(best);


        document.getElementById(
            "bt-best"
        ).className =
            "stat-value " +
            (
                best >= 0
                    ? "green"
                    : "red"
            );


        document.getElementById(
            "bt-winrate"
        ).textContent =
            avgWin.toFixed(1) +
            "%";


        document.getElementById(
            "bt-window"
        ).textContent =
            data.window ||
            "Latest";


        /*
         * WIN/LOSS
         */

        if (
            backtestCharts.pie
        ) {

            backtestCharts.pie.data.datasets[0].data =
                [
                    data.win_loss_pie.wins,
                    data.win_loss_pie.losses,
                ];

            backtestCharts.pie.update(
                "none"
            );

        } else {

            backtestCharts.pie =
                new Chart(
                    document
                        .getElementById(
                            "backtestPie"
                        )
                        .getContext("2d"),
                    {
                        type:
                            "doughnut",

                        data: {

                            labels:
                                [
                                    "Wins",
                                    "Losses"
                                ],

                            datasets: [{
                                data:
                                    [
                                        data.win_loss_pie.wins,
                                        data.win_loss_pie.losses
                                    ],

                                backgroundColor:
                                    [
                                        "#31d58a",
                                        "#ff6b73"
                                    ],

                                borderWidth:
                                    0
                            }]
                        },

                        options: {

                            cutout:
                                "70%",

                            plugins: {

                                legend: {

                                    labels: {

                                        color:
                                            "#8e98a8",

                                        boxWidth:
                                            11,

                                        font: {
                                            size:
                                                10
                                        }
                                    }
                                }
                            }
                        }
                    }
                );
        }


        /*
         * BAR CHART HELPER
         */

        function makeBar(
            key,
            canvasId,
            values,
            label,
            colors
        ) {

            if (
                backtestCharts[key]
            ) {

                backtestCharts[key]
                    .data
                    .labels =
                    symbols;

                backtestCharts[key]
                    .data
                    .datasets[0]
                    .data =
                    values;

                backtestCharts[key]
                    .update("none");

                return;
            }


            backtestCharts[key] =
                new Chart(
                    document
                        .getElementById(
                            canvasId
                        )
                        .getContext("2d"),
                    {

                        type:
                            "bar",

                        data: {

                            labels:
                                symbols,

                            datasets: [{

                                label,

                                data:
                                    values,

                                backgroundColor:
                                    colors,

                                borderRadius:
                                    5,

                                maxBarThickness:
                                    34
                            }]
                        },

                        options: {

                            responsive:
                                true,

                            scales: {

                                x: {

                                    ticks: {
                                        color:
                                            "#8e98a8",
                                        font: {
                                            size:
                                                9
                                        }
                                    },

                                    grid: {
                                        display:
                                            false
                                    }
                                },

                                y: {

                                    ticks: {
                                        color:
                                            "#8e98a8",
                                        font: {
                                            size:
                                                9
                                        }
                                    },

                                    grid: {
                                        color:
                                            "#1b2029"
                                    }
                                }
                            },

                            plugins: {

                                legend: {
                                    display:
                                        false
                                }
                            }
                        }
                    }
                );
        }


        makeBar(
            "returns",
            "backtestReturns",
            returns,
            "Return %",
            returns.map(
                x =>
                    x >= 0
                        ? "#31d58a"
                        : "#ff6b73"
            )
        );


        makeBar(
            "winrate",
            "backtestWinRate",
            winRates,
            "Win Rate",
            "#557cff"
        );


        makeBar(
            "trades",
            "backtestTrades",
            trades,
            "Trades",
            "#765cff"
        );


        /*
         * SPY IMAGE
         */

        const spyContainer =
            document.getElementById(
                "spy-chart-container"
            );


        if (
            data.has_chart_image
        ) {

            spyContainer.innerHTML = `

                <div class="card">

                    <div class="chart-title">
                        Strategy vs SPY
                    </div>

                    <img
                        src="/api/chart-image/backtest_vs_spy"
                        style="
                            width:100%;
                            display:block;
                            border-radius:10px;
                        "
                    >

                </div>

            `;
        }


        /*
         * Check whether a backtest is currently running.
         */
 
        updateBacktestLiveStatus();
        loadBacktestRuns();

    } catch (error) {

        if (!silent)
            console.error(error);
    }
}


/* ========================================================================
   BACKTEST LIVE STATUS
   ======================================================================== */

async function updateBacktestLiveStatus() {

    try {

        const response =
            await apiFetch(
                "/api/scripts"
            );

        const scripts =
            await response.json();

        const backtest =
            scripts.find(
                s =>
                    s.key ===
                    "backtest"
            );

        const label =
            document.getElementById(
                "backtest-live-label"
            );

        const status =
            document.getElementById(
                "backtest-status"
            );


        if (
            backtest &&
            backtest.running_id
        ) {

            label.textContent =
                "Running";

            label.className =
                "badge running";


            status.innerHTML = `

                <div class="card">

                    <div class="card-header">

                        <span
                            class="card-title"
                        >
                            Backtest in progress
                        </span>

                        <span
                            class="muted"
                        >
                            Live
                        </span>

                    </div>

                    <div class="eta">

                        <strong>
                            Estimated remaining:
                            ${
                                fmtSeconds(
                                    backtest.eta_seconds
                                )
                            }
                        </strong>

                        <br>

                        Elapsed:
                        ${
                            fmtSeconds(
                                backtest.elapsed_seconds
                            )
                        }

                        <br>

                        Typical run:
                        ${
                            fmtSeconds(
                                backtest.estimated_seconds
                            )
                        }

                        <div class="progress">

                            <div
                                class="progress-bar"
                                style="
                                    width:
                                    ${
                                        Math.min(
                                            Math.max(
                                                (
                                                    (
                                                        backtest.elapsed_seconds ||
                                                        0
                                                    ) /
                                                    (
                                                        backtest.estimated_seconds ||
                                                        1
                                                    )
                                                ) * 100,
                                                2
                                            ),
                                            95
                                        )
                                    }%;
                                "
                            ></div>

                        </div>

                    </div>

                </div>

            `;

        } else {

            label.textContent =
                "Complete";

            label.className =
                "badge";

            status.innerHTML = "";
        }

    } catch (error) {

        console.error(error);
    }
}


/* ========================================================================
   BACKTEST HISTORY (client)
   ======================================================================== */

function escapeHtml(s) {
    return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

async function loadBacktestRuns() {
    try {
        const resp = await apiFetch('/api/backtests');
        const runs = await resp.json();

        // Ensure container exists
        let container = document.getElementById('backtest-runs');
        if (!container) {
            const content = document.getElementById('backtest-content');
            const el = document.createElement('div');
            el.id = 'backtest-runs';
            content.appendChild(el);
            container = el;
        }

        const listHtml = runs.map(r => {
            const completed = r.summary && r.summary.completed_at ? r.summary.completed_at : '';
            return `<div class="backtest-run-item" style="padding:8px;border-bottom:1px solid #222;display:flex;justify-content:space-between;align-items:center;">
                <div style="font-size:13px">
                    <strong>${r.run_id}</strong><br>
                    <span class="muted">${completed}</span>
                </div>
                <div>
                    ${r.chart_available ? `<button class="btn" onclick="viewBacktestRunChart('${r.run_id}')">View</button>` : ''}
                    ${r.results_available ? `<a class="link" href="/api/backtests/${r.run_id}/results" target="_blank">CSV</a>` : ''}
                    <button class="btn" onclick="viewBacktestRunSummary('${r.run_id}')">Summary</button>
                </div>
            </div>`;
        }).join('');

        container.innerHTML = `<div class="card"><div class="card-header"><span class="card-title">Backtest History</span><span class="muted">Past runs</span></div><div style="max-height:260px;overflow:auto;">${listHtml || '<div class="empty">No past runs found.</div>'}</div></div>`;

    } catch (e) {
        console.error('Failed to load backtest runs', e);
    }
}

window.viewBacktestRunChart = async function(run_id) {
    try {
        const spyContainer = document.getElementById('spy-chart-container');
        if (!spyContainer) return;

        spyContainer.innerHTML = `\n            <div class="card">\n                <div class="chart-title">Strategy vs SPY (run ${run_id})</div>\n                <img src="/api/backtests/${run_id}/chart" style="width:100%;display:block;border-radius:10px;">\n            </div>\n        `;

        // remove any existing preview
        const existing = document.getElementById('backtest-csv-preview');
        if (existing) existing.remove();

        // fetch a small preview of the CSV
        const r = await apiFetch(`/api/backtests/${run_id}/results`);
        if (r.ok) {
            const text = await r.text();
            const preview = text.split('\n').slice(0, 6).join('\n');
            const previewHtml = `\n                <div class="card" id="backtest-csv-preview">\n                    <div class="chart-title">Results CSV (preview)</div>\n                    <pre style="max-height:200px;overflow:auto;background:#0f141a;padding:10px;border-radius:6px;color:#9aa4b2;">${escapeHtml(preview)}</pre>\n                </div>\n            `;
            spyContainer.insertAdjacentHTML('afterend', previewHtml);
        }
    } catch (e) {
        console.error(e);
    }
}


window.viewBacktestRunSummary = async function(run_id) {
    try {
        const r = await apiFetch(`/api/backtests/${run_id}/summary`);
        const data = await r.json();
        const spyContainer = document.getElementById('spy-chart-container');
        if (!spyContainer) return;

        // remove an existing summary preview if present
        const existing = document.getElementById('backtest-summary');
        if (existing) existing.remove();

        let html = '';
        if (data && data.available && Array.isArray(data.summaries) && data.summaries.length) {
            html += `<div class="card" id="backtest-summary"><div class="card-header"><span class="card-title">Summary — ${escapeHtml(run_id)}</span></div><div class="table-wrap"><table><thead><tr><th>Symbol</th><th>Trades</th><th>Total Return %</th><th>Win Rate %</th><th>Avg PnL %</th><th>CSV</th></tr></thead><tbody>`;
            data.summaries.forEach(s => {
                html += `<tr>` +
                    `<td>${escapeHtml(s.symbol || '')}</td>` +
                    `<td class="num">${s.num_trades || 0}</td>` +
                    `<td class="${(s.total_return_pct || 0) >= 0 ? 'green' : 'red'}">${fmtPct(s.total_return_pct || 0)}</td>` +
                    `<td>${(s.win_rate_pct || 0).toFixed ? (s.win_rate_pct.toFixed(2) + '%') : (s.win_rate_pct + '%')}</td>` +
                    `<td>${(s.avg_pnl_pct || 0)}</td>` +
                    `<td>${s.file ? `<a class="link" href="/api/backtests/${run_id}/artifact/${encodeURIComponent(s.file)}" target="_blank">CSV</a>` : ''}</td>` +
                    `</tr>`;
            });
            html += `</tbody></table></div></div>`;
        } else if (data && data.error) {
            html = `<div class="card" id="backtest-summary"><div class="empty">Error: ${escapeHtml(data.error)}</div></div>`;
        } else {
            html = `<div class="card" id="backtest-summary"><div class="empty">No summary available for this run.</div></div>`;
        }

        spyContainer.insertAdjacentHTML('afterend', html);
    } catch (e) {
        console.error(e);
    }
}


/* ========================================================================
   LIVE TRADING
   ======================================================================== */

async function loadLive() {

    try {

        const response =
            await apiFetch(
                "/api/data/live_summary"
            );

        const data =
            await response.json();

        const el =
            document.getElementById(
                "live-content"
            );


        if (
            !data.available
        ) {

            el.innerHTML = `

                <div class="card">

                    <div class="empty">
                        No live trades logged yet.
                    </div>

                </div>

            `;

            return;
        }


        const net =
            data.cumulative_pnl_pct.length
                ? data
                    .cumulative_pnl_pct[
                        data.cumulative_pnl_pct.length - 1
                    ]
                : 0;


        const total =
            data.win_loss_pie.wins +
            data.win_loss_pie.losses;


        const winRate =
            total
                ? (
                    data.win_loss_pie.wins /
                    total
                ) * 100
                : 0;


        el.innerHTML = `

            <div class="stat-grid">

                <div class="stat">

                    <div class="stat-label">
                        Net P&L
                    </div>

                    <div
                        class="stat-value
                        ${net >= 0
                            ? "green"
                            : "red"}"
                    >
                        ${fmtPct(net)}
                    </div>

                </div>


                <div class="stat">

                    <div class="stat-label">
                        Win Rate
                    </div>

                    <div class="stat-value">
                        ${winRate.toFixed(1)}%
                    </div>

                </div>

            </div>


            <div class="card chart-card">

                <div class="chart-title">
                    Win / Loss
                </div>

                <canvas
                    id="livePie"
                ></canvas>

            </div>


            <div class="card chart-card">

                <div class="chart-title">
                    Cumulative P&L
                </div>

                <canvas
                    id="liveLine"
                ></canvas>

            </div>


            <div class="card">

                <div class="chart-title">
                    Recent Trades
                </div>

                <div class="table-wrap">

                    <table>

                        <thead>

                            <tr>

                                <th>
                                    Time
                                </th>

                                <th>
                                    Symbol
                                </th>

                                <th class="num">
                                    PnL
                                </th>

                                <th>
                                    Reason
                                </th>

                            </tr>

                        </thead>

                        <tbody>

                            ${
                                data.recent_trades
                                    .slice()
                                    .reverse()
                                    .map(
                                        trade => `

                                        <tr>

                                            <td>
                                                ${
                                                    String(
                                                        trade.exit_timestamp
                                                    )
                                                    .slice(
                                                        5,
                                                        16
                                                    )
                                                }
                                            </td>

                                            <td>
                                                ${
                                                    trade.symbol ||
                                                    ""
                                                }
                                            </td>

                                            <td
                                                class="num
                                                ${
                                                    Number(
                                                        trade.pnl_pct
                                                    ) >= 0
                                                        ? "green"
                                                        : "red"
                                                }"
                                            >
                                                ${
                                                    fmtPct(
                                                        trade.pnl_pct
                                                    )
                                                }
                                            </td>

                                            <td class="muted">
                                                ${
                                                    (
                                                        trade.exit_reason ||
                                                        ""
                                                    ).slice(
                                                        0,
                                                        24
                                                    )
                                                }
                                            </td>

                                        </tr>

                                    `
                                    )
                                    .join("")
                            }

                        </tbody>

                    </table>

                </div>

            </div>


            <div class="card">

                <div class="chart-title">
                    trader.log
                </div>

                <pre
                    class="output"
                    style="
                        display:block;
                        margin:0;
                    "
                >${
                    (
                        data.trader_log_tail ||
                        []
                    ).join("\n")
                }</pre>

            </div>
        `;


        if (
            liveCharts.pie
        ) {

            liveCharts.pie.destroy();
        }


        liveCharts.pie =
            new Chart(
                document
                    .getElementById(
                        "livePie"
                    )
                    .getContext("2d"),
                {

                    type:
                        "doughnut",

                    data: {

                        labels:
                            [
                                "Wins",
                                "Losses"
                            ],

                        datasets: [{

                            data:
                                [
                                    data.win_loss_pie.wins,
                                    data.win_loss_pie.losses
                                ],

                            backgroundColor:
                                [
                                    "#31d58a",
                                    "#ff6b73"
                                ],

                            borderWidth:
                                0
                        }]
                    },

                    options: {

                        cutout:
                            "70%",

                        plugins: {

                            legend: {

                                labels: {

                                    color:
                                        "#8e98a8",

                                    boxWidth:
                                        11,

                                    font: {
                                        size:
                                            10
                                    }
                                }
                            }
                        }
                    }
                }
            );


        if (
            liveCharts.line
        ) {

            liveCharts.line.destroy();
        }


        liveCharts.line =
            new Chart(
                document
                    .getElementById(
                        "liveLine"
                    )
                    .getContext("2d"),
                {

                    type:
                        "line",

                    data: {

                        labels:
                            data.timestamps.map(
                                x =>
                                    String(
                                        x
                                    ).slice(
                                        5,
                                        16
                                    )
                            ),

                        datasets: [{

                            data:
                                data.cumulative_pnl_pct,

                            borderColor:
                                "#557cff",

                            backgroundColor:
                                "rgba(85,124,255,.08)",

                            fill:
                                true,

                            tension:
                                .25,

                            pointRadius:
                                0,

                            borderWidth:
                                2
                        }]
                    },

                    options: {

                        scales: {

                            x: {

                                ticks: {
                                    color:
                                        "#8e98a8",
                                    maxTicksLimit:
                                        6,
                                    font: {
                                        size:
                                            9
                                    }
                                },

                                grid: {
                                    display:
                                        false
                                }
                            },

                            y: {

                                ticks: {
                                    color:
                                        "#8e98a8",
                                    font: {
                                        size:
                                            9
                                    }
                                },

                                grid: {
                                    color:
                                        "#1b2029"
                                }
                            }
                        },

                        plugins: {

                            legend: {
                                display:
                                    false
                            }
                        }
                    }
                }
            );


        clearTimeout(
            liveRefreshTimer
        );

        liveRefreshTimer =
            setTimeout(
                loadLive,
                5000
            );

    } catch (error) {

        console.error(
            error
        );
    }
}


/* ========================================================================
   HYPOTHESIS
   ======================================================================== */

async function loadHypothesis() {

    try {

        const response =
            await apiFetch(
                "/api/data/hypothesis"
            );

        const data =
            await response.json();

        const el =
            document.getElementById(
                "hyp-content"
            );


        if (
            !data.available ||
            !data.rows.length
        ) {

            el.innerHTML = `

                <div class="card">

                    <div class="empty">
                        No hypothesis tests logged yet.
                    </div>

                </div>

            `;

            return;
        }


        el.innerHTML = `

            <div
                class="card"
                style="
                    overflow-x:auto;
                    padding:7px;
                "
            >

                <table>

                    <thead>

                        <tr>

                            <th>
                                When
                            </th>

                            <th>
                                Param
                            </th>

                            <th>
                                Old → New
                            </th>

                            <th class="num">
                                Base
                            </th>

                            <th class="num">
                                Trial
                            </th>

                            <th>
                                Result
                            </th>

                        </tr>

                    </thead>

                    <tbody>

                        ${
                            data.rows
                                .slice()
                                .reverse()
                                .map(
                                    row => `

                                    <tr>

                                        <td class="muted">
                                            ${
                                                String(
                                                    row.timestamp ||
                                                    ""
                                                ).slice(
                                                    5,
                                                    16
                                                )
                                            }
                                        </td>

                                        <td>
                                            ${
                                                row.parameter ||
                                                ""
                                            }
                                        </td>

                                        <td class="muted">
                                            ${
                                                row.old_value ||
                                                ""
                                            }
                                            →
                                            ${
                                                row.new_value ||
                                                ""
                                            }
                                        </td>

                                        <td class="num">
                                            ${
                                                row.baseline_arm_return_pct === ""
                                                    ? "—"
                                                    : fmtPct(
                                                        row.baseline_arm_return_pct
                                                    )
                                            }
                                        </td>

                                        <td class="num">
                                            ${
                                                row.trial_arm_return_pct === ""
                                                    ? "—"
                                                    : fmtPct(
                                                        row.trial_arm_return_pct
                                                    )
                                            }
                                        </td>

                                        <td>
                                            ${
                                                row.result ||
                                                ""
                                            }
                                        </td>

                                    </tr>

                                `
                                )
                                .join("")
                        }

                    </tbody>

                </table>

            </div>

        `;

    } catch (error) {

        console.error(
            error
        );
    }
}


/* ========================================================================
   INITIALIZATION
   ======================================================================== */

loadScripts();

setInterval(
    () => {

        const runPage =
            document.getElementById(
                "page-run"
            );

        if (
            runPage.classList.contains(
                "active"
            )
        ) {

            refreshScriptStatuses();
        }

    },
    4000
);


/*
 * Backtest page refresh.
 *
 * This means that if backtest.py is writing backtest_results.csv while
 * running, the charts will update automatically.
 */

setInterval(
    () => {

        const page =
            document.getElementById(
                "page-backtest"
            );

        if (
            page.classList.contains(
                "active"
            )
        ) {

            loadBacktest(true);
            updateBacktestLiveStatus();
        }

    },
    2500
);


// When the page becomes visible again (e.g., phone unlocked or tab foregrounded),
// immediately refresh relevant data and restart any stalled polls. Mobile browsers
// may throttle timers while in the background; this helps the UI catch up when
// visibility is restored.
document.addEventListener('visibilitychange', () => {
    try {
        if (document.visibilityState === 'visible') {

            // Refresh script statuses and any active page content
            refreshScriptStatuses();

            const backtestPage = document.getElementById('page-backtest');
            if (backtestPage && backtestPage.classList.contains('active')) {
                loadBacktest(true);
                updateBacktestLiveStatus();
            }

            const livePage = document.getElementById('page-live');
            if (livePage && livePage.classList.contains('active')) {
                loadLive();
            }

            // Restart polling for any running scripts in case timers were throttled
            for (const key in scriptEls) {
                if (!Object.prototype.hasOwnProperty.call(scriptEls, key)) continue;
                const refs = scriptEls[key];
                if (refs && refs.currentRunId) {
                    try {
                        if (runningPolls[refs.currentRunId]) {
                            clearInterval(runningPolls[refs.currentRunId]);
                            delete runningPolls[refs.currentRunId];
                        }
                    } catch (e) {
                        // ignore
                    }
                    pollRun(refs.currentRunId, key);
                }
            }
        }
    } catch (err) {
        console.error('visibilitychange handler error:', err);
    }
});


</script>

</body>
</html>
"""

# ============================================================================
# HOME
# ============================================================================

@app.route("/")
def index():

    return Response(
        INDEX_HTML,
        mimetype="text/html"
    )


# ============================================================================
# START
# ============================================================================
def _trade_notification_watcher():
    last_row_count = 0
    path = os.path.join(DATA_DIR, "trade_log.csv")
    while True:
        try:
            df = _read_csv_safe(path)
            if df is not None:
                if last_row_count == 0:
                    last_row_count = len(df)  # don't spam on first startup
                elif len(df) > last_row_count:
                    for _, row in df.iloc[last_row_count:].iterrows():
                        notify_new_trade(row.get("symbol", "?"),
                                          float(row.get("pnl_pct", 0)),
                                          str(row.get("exit_reason", "")))
                    last_row_count = len(df)
        except Exception as e:
            print(f"[dashboard] trade watcher error: {e}")
        time.sleep(15)

threading.Thread(target=_trade_notification_watcher, daemon=True).start()
if __name__ == "__main__":

    print()
    print("=" * 60)
    print("TRADING BOT DASHBOARD")
    print("=" * 60)

    if not PASSWORD:

        print()
        print(
            "[WARNING] DASHBOARD_PASSWORD is NOT set."
        )

        print(
            "Anyone who can reach this dashboard "
            "may be able to run your scripts."
        )

        print()
        print(
            'Set it with:'
        )

        print(
            'setx DASHBOARD_PASSWORD "your-password"'
        )

        print(
            "Then open a NEW terminal."
        )

    else:

        print(
            "[dashboard] Password protection: ENABLED"
        )

    print()
    print(
        "[dashboard] Local:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print()
    print(
        "[dashboard] Tailscale:"
    )

    print(
        "http://<your-tailscale-ip>:5000"
    )

    print()
    print(
        "[dashboard] Listening on 0.0.0.0:5000"
    )

    print("=" * 60)
    print()


    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True,
    )