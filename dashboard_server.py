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
from typing import Any

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
from credentials_store import load_credentials, save_credentials, status as credentials_status, build_subprocess_env
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
        "label": "Pre-Test Check",
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
    "optimizer": {
        "file": "strategy_optimizer.py",
        "label": "Strategy Optimizer",
        "long_running": False,
    },
    "bot": {
        "file": "trader_v3.py" if os.path.exists(os.path.join(BASE_DIR, "trader_v3.py")) else "bot.py",
        "label": "Live Trading Bot",
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
# ALPACA CREDENTIALS
# ============================================================================
@app.route("/api/credentials", methods=["GET"])
def api_credentials():
    # Never return the raw API secret/key to the browser.
    return jsonify(credentials_status())


@app.route("/api/credentials", methods=["POST"])
def api_save_credentials():
    data = request.get_json(silent=True) or {}

    # Only update fields explicitly supplied. This lets the UI preserve a
    # previously saved secret when the user leaves that field blank.
    api_key = data.get("alpaca_api_key")
    secret_key = data.get("alpaca_secret_key")
    base_url = data.get("alpaca_base_url")

    if api_key is not None and not isinstance(api_key, str):
        return jsonify({"error": "alpaca_api_key must be a string"}), 400
    if secret_key is not None and not isinstance(secret_key, str):
        return jsonify({"error": "alpaca_secret_key must be a string"}), 400
    if base_url is not None and not isinstance(base_url, str):
        return jsonify({"error": "alpaca_base_url must be a string"}), 400

    # Prevent accidentally saving a non-Alpaca URL.
    if base_url is not None:
        base_url = base_url.strip().rstrip("/")
        allowed_hosts = (
            "https://paper-api.alpaca.markets",
            "https://api.alpaca.markets",
        )
        if base_url not in allowed_hosts:
            return jsonify({
                "error": "base_url must be an Alpaca paper or live endpoint"
            }), 400

    try:
        save_credentials(
            api_key=api_key,
            secret_key=secret_key,
            base_url=base_url,
        )
        return jsonify({
            "saved": True,
            "credentials": credentials_status(),
        })
    except Exception as e:
        return jsonify({"error": f"could not save credentials: {e}"}), 500


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

        kwargs: dict[str, Any] = dict(
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

        # Read credentials from credentials.json immediately before launch.
        # This means newly saved keys are used on the very next run without
        # restarting the dashboard or changing terminal environment variables.
        kwargs["env"] = build_subprocess_env()

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
# ANALYTICS / MODEL / OPTIMIZER
# ============================================================================

def _read_json_safe(path):
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _coerce_num(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


@app.route("/api/analytics/optimizer")
def api_optimizer_analytics():
    history_path = os.path.join(DATA_DIR, "optimizer_history.csv")
    validation_path = os.path.join(DATA_DIR, "optimizer_validation.csv")
    results_path = os.path.join(DATA_DIR, "optimizer_results.csv")
    best_path = os.path.join(DATA_DIR, "best_strategy_config.json")

    history = _read_csv_safe(history_path)
    validation = _read_csv_safe(validation_path)
    results = _read_csv_safe(results_path)
    best = _read_json_safe(best_path) or {}

    out = {
        "available": bool(history is not None or validation is not None or best),
        "history_count": int(len(history)) if history is not None else 0,
        "recent_trials": [],
        "validation": [],
        "best_config": best.get("config", {}) if isinstance(best, dict) else {},
        "best_meta": best,
    }

    if history is not None:
        h = history.copy().fillna("")
        if "score" in h.columns:
            h["score_num"] = pd.to_numeric(h["score"], errors="coerce")
            h = h.sort_values("score_num", ascending=False, na_position="last")
        cols = [c for c in [
            "tested_at", "config_id", "BAR_TIMEFRAME",
            "MIN_SIGNAL_CONFIDENCE", "MIN_META_WIN_PROB",
            "RISK_PER_TRADE_PCT", "STOP_LOSS_PCT",
            "TAKE_PROFIT_PCT", "ATR_STOP_MULTIPLIER",
            "return", "sharpe", "drawdown", "trades", "score"
        ] if c in h.columns]
        out["recent_trials"] = h[cols].head(60).to_dict(orient="records")

    if validation is not None:
        v = validation.copy().fillna("")
        cols = [c for c in [
            "config_id", "validation_offset", "return", "sharpe",
            "drawdown", "profit_factor", "trades", "score"
        ] if c in v.columns]
        out["validation"] = v[cols].tail(100).to_dict(orient="records")

    return jsonify(out)


@app.route("/api/analytics/model")
def api_model_analytics():
    info = _read_json_safe(os.path.join(DATA_DIR, "meta_model_info.json")) or {}
    model_path = os.path.join(DATA_DIR, "meta_model.joblib")
    training_path = os.path.join(DATA_DIR, "meta_training_data.csv")
    training = _read_csv_safe(training_path)

    out = {
        "available": bool(info or os.path.exists(model_path) or training is not None),
        "model_exists": os.path.exists(model_path),
        "info": info,
        "training_rows": int(len(training)) if training is not None else 0,
        "sources": {},
        "feature_importance": [],
    }

    if training is not None and "_data_source" in training.columns:
        counts = training["_data_source"].astype(str).value_counts().to_dict()
        out["sources"] = {str(k): int(v) for k, v in counts.items()}

    try:
        import joblib
        if os.path.exists(model_path):
            saved = joblib.load(model_path)
            model = saved.get("model")
            feature_cols = saved.get("feature_cols") or []
            if model is not None and hasattr(model, "feature_importances_"):
                items = [
                    {"feature": str(f), "importance": float(i)}
                    for f, i in zip(feature_cols, model.feature_importances_)
                ]
                items.sort(key=lambda x: x["importance"], reverse=True)
                out["feature_importance"] = items[:20]
    except Exception as exc:
        out["model_load_error"] = str(exc)

    return jsonify(out)


@app.route("/api/analytics/portfolio")
def api_portfolio_analytics():
    path = os.path.join(DATA_DIR, "trade_log.csv")
    df = _read_csv_safe(path)
    if df is None or "pnl_pct" not in df.columns:
        return jsonify({"available": False, "trades": []})

    df = df.copy()
    df["pnl_pct"] = pd.to_numeric(df["pnl_pct"], errors="coerce")
    df = df.dropna(subset=["pnl_pct"])
    if df.empty:
        return jsonify({"available": False, "trades": []})

    if "exit_timestamp" in df.columns:
        df["exit_timestamp"] = df["exit_timestamp"].astype(str)
        df = df.sort_values("exit_timestamp")

    df["cum_pnl_pct"] = df["pnl_pct"].cumsum()
    df["running_peak"] = df["cum_pnl_pct"].cummax()
    df["drawdown"] = df["cum_pnl_pct"] - df["running_peak"]

    trades = df.tail(100).fillna("")
    cols = [c for c in [
        "exit_timestamp", "symbol", "entry_price", "exit_price",
        "pnl_pct", "exit_reason", "entry_confidence",
        "entry_meta_win_prob", "entry_trend_regime",
        "entry_volatility_regime", "entry_hmm_regime"
    ] if c in trades.columns]

    return jsonify({
        "available": True,
        "total_trades": int(len(df)),
        "win_rate": float((df["pnl_pct"] > 0).mean() * 100),
        "total_pnl_pct": float(df["pnl_pct"].sum()),
        "avg_trade_pct": float(df["pnl_pct"].mean()),
        "max_drawdown_pct": float(df["drawdown"].min()),
        "timeline": {
            "timestamps": df["exit_timestamp"].tail(250).tolist() if "exit_timestamp" in df.columns else [],
            "cum_pnl": [float(x) for x in df["cum_pnl_pct"].tail(250)],
            "drawdown": [float(x) for x in df["drawdown"].tail(250)],
        },
        "trades": trades[cols].to_dict(orient="records"),
    })


# ============================================================================
# FRONTEND
# ============================================================================

INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#0a0f18">
<title>Trading Bot Control Center</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
:root{
  --bg:#080c13;--panel:#0f1622;--panel2:#121b29;--panel3:#172233;--line:#233044;
  --text:#edf2f8;--muted:#8794a7;--faint:#5c6a7e;--blue:#6c8cff;--cyan:#50c9ff;
  --green:#2dd78b;--red:#ff6673;--amber:#f2b84b;--purple:#9a7cff;
  --radius:16px;--shadow:0 16px 45px rgba(0,0,0,.25);--mono:ui-monospace,SFMono-Regular,Consolas,monospace;
}
*{box-sizing:border-box}html,body{margin:0;min-height:100%;background:var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
body{overflow-x:hidden}.app{display:grid;grid-template-columns:250px 1fr;min-height:100vh}.sidebar{position:sticky;top:0;height:100vh;border-right:1px solid var(--line);background:linear-gradient(180deg,#0b111a,#0a0f17);padding:22px 16px;display:flex;flex-direction:column;gap:18px}.brand{display:flex;align-items:center;gap:11px;padding:4px 8px 18px}.logo{width:38px;height:38px;border-radius:12px;background:linear-gradient(135deg,var(--blue),var(--purple));display:grid;place-items:center;font-weight:800;box-shadow:0 12px 30px rgba(108,140,255,.24)}.brand h1{font-size:15px;margin:0}.brand p{font-size:11px;color:var(--faint);margin:3px 0 0}.nav{display:grid;gap:6px}.nav button{border:1px solid transparent;background:transparent;color:var(--muted);text-align:left;padding:12px 13px;border-radius:11px;font-weight:650;cursor:pointer}.nav button:hover{background:var(--panel2);color:var(--text)}.nav button.active{background:#17233b;color:#fff;border-color:#263b62}.side-bottom{margin-top:auto;display:grid;gap:10px}.system-badge{display:flex;align-items:center;gap:8px;padding:10px 12px;border:1px solid var(--line);border-radius:12px;background:var(--panel)}.dot{width:8px;height:8px;border-radius:50%;background:var(--faint)}.dot.ok{background:var(--green);box-shadow:0 0 0 4px rgba(45,215,139,.10)}.dot.bad{background:var(--red);box-shadow:0 0 0 4px rgba(255,102,115,.10)}.main{padding:24px 28px 42px;max-width:1700px;width:100%;margin:0 auto}.topbar{display:flex;align-items:center;justify-content:space-between;gap:20px;margin-bottom:22px}.topbar h2{margin:0;font-size:25px;letter-spacing:-.03em}.topbar p{margin:6px 0 0;color:var(--muted);font-size:13px}.top-actions{display:flex;gap:10px;align-items:center}.btn{border:1px solid var(--line);background:var(--panel);color:var(--text);border-radius:10px;padding:10px 13px;cursor:pointer;font-weight:650}.btn:hover{background:var(--panel3)}.btn.primary{background:linear-gradient(135deg,#5f82ff,#8366e8);border-color:#6d85e8}.btn.danger{color:#ff9aa3;border-color:#5b2830}.status-pill{display:inline-flex;align-items:center;gap:8px;padding:8px 11px;border:1px solid var(--line);border-radius:999px;background:var(--panel);font-size:12px;color:var(--muted)}.grid{display:grid;gap:16px}.kpis{grid-template-columns:repeat(6,minmax(0,1fr))}.kpi,.card{background:linear-gradient(180deg,rgba(16,24,36,.98),rgba(12,19,29,.98));border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow)}.kpi{padding:16px}.kpi .label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;font-weight:800}.kpi .value{font-size:26px;font-weight:750;margin-top:8px}.kpi .sub{font-size:11px;color:var(--faint);margin-top:4px}.positive{color:var(--green)}.negative{color:var(--red)}.amber{color:var(--amber)}.blue{color:var(--cyan)}.content-grid{grid-template-columns:1.6fr 1fr}.card{padding:17px}.card-head{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:14px}.card-title{font-weight:750}.card-sub{color:var(--faint);font-size:11px}.chart-wrap{position:relative;height:300px}.chart-wrap.small{height:230px}.two{grid-template-columns:1fr 1fr}.three{grid-template-columns:repeat(3,1fr)}.section{margin-top:18px}.section-label{font-size:11px;color:var(--faint);text-transform:uppercase;letter-spacing:.12em;font-weight:800;margin:26px 0 10px}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:12px}.table{width:100%;border-collapse:collapse;min-width:720px}.table th,.table td{padding:10px 11px;border-bottom:1px solid #1b2636;font-size:12px;text-align:left;white-space:nowrap}.table th{color:#95a4b7;background:#101826;position:sticky;top:0}.table tr:last-child td{border-bottom:0}.tag{display:inline-flex;padding:4px 8px;border-radius:999px;font-size:10px;font-weight:800;background:#1a2638;color:#b9c7da}.tag.green{background:rgba(45,215,139,.10);color:#7ceab2}.tag.red{background:rgba(255,102,115,.10);color:#ff9aa3}.tag.blue{background:rgba(108,140,255,.12);color:#9cafFF}.run-grid{grid-template-columns:repeat(3,1fr)}.run-card{padding:16px}.run-card h3{margin:0 0 4px;font-size:15px}.run-card p{margin:0 0 13px;color:var(--muted);font-size:12px}.run-actions{display:flex;gap:8px;flex-wrap:wrap}.progress{height:7px;background:#1a2433;border-radius:999px;overflow:hidden}.progress i{display:block;height:100%;background:linear-gradient(90deg,var(--blue),var(--cyan));width:0}.console{background:#070a0f;border:1px solid #1a2534;border-radius:12px;padding:12px;max-height:300px;overflow:auto;color:#9fb0c3;font:11px/1.5 var(--mono);white-space:pre-wrap}.empty{padding:40px;text-align:center;color:var(--muted);border:1px dashed var(--line);border-radius:12px}.config-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.config-item{padding:10px 12px;border:1px solid var(--line);border-radius:10px;background:#0d1520}.config-item .k{font-size:10px;color:var(--faint);margin-bottom:5px}.config-item .v{font-family:var(--mono);font-size:12px}.tabs-inline{display:flex;gap:6px;flex-wrap:wrap}.tab-chip{padding:7px 10px;border:1px solid var(--line);border-radius:999px;background:transparent;color:var(--muted);cursor:pointer}.tab-chip.active{background:#18243a;color:#fff;border-color:#2c4470}.page{display:none}.page.active{display:block}.toast{position:fixed;right:20px;bottom:20px;z-index:30;padding:12px 14px;border:1px solid var(--line);background:#111b29;border-radius:12px;box-shadow:var(--shadow);font-size:12px;display:none}.modal{position:fixed;inset:0;background:rgba(4,8,13,.74);display:none;align-items:center;justify-content:center;z-index:40;padding:20px}.modal.open{display:flex}.modal-card{width:min(640px,100%);max-height:90vh;overflow:auto;background:#0e1622;border:1px solid var(--line);border-radius:18px;padding:20px}.field{display:grid;gap:6px;margin-bottom:13px}.field label{font-size:11px;color:var(--muted);font-weight:800;text-transform:uppercase;letter-spacing:.07em}.field input{width:100%;padding:11px 12px;background:#09111b;color:#fff;border:1px solid var(--line);border-radius:10px;outline:none}.field input:focus{border-color:#557cff}.muted{color:var(--muted)}.mono{font-family:var(--mono)}.hide{display:none!important}
@media(max-width:1100px){.kpis{grid-template-columns:repeat(3,1fr)}.content-grid{grid-template-columns:1fr}.run-grid{grid-template-columns:1fr 1fr}.three{grid-template-columns:1fr 1fr}.config-grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:760px){.app{grid-template-columns:1fr}.sidebar{position:sticky;z-index:20;height:auto;padding:10px;border-right:0;border-bottom:1px solid var(--line);flex-direction:row;align-items:center}.brand{padding:0}.brand h1{font-size:13px}.nav{display:flex;overflow:auto;flex:1}.nav button{white-space:nowrap;padding:9px 10px}.side-bottom{display:none}.main{padding:18px 13px 34px}.kpis{grid-template-columns:1fr 1fr}.run-grid,.two,.three{grid-template-columns:1fr}.topbar{align-items:flex-start}.top-actions .status-pill{display:none}.config-grid{grid-template-columns:1fr}.chart-wrap{height:250px}}
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="brand"><div class="logo">↗</div><div><h1>Trading Control Center</h1><p>Research • backtest • live</p></div></div>
    <div class="nav">
      <button class="active" data-page="overview">Overview</button>
      <button data-page="live">Live Trading</button>
      <button data-page="backtest">Backtest Lab</button>
      <button data-page="optimizer">Strategy Optimizer</button>
      <button data-page="model">Meta Model</button>
      <button data-page="experiments">Experiments</button>
      <button data-page="runs">Runs & Logs</button>
    </div>
    <div class="side-bottom">
      <div class="system-badge"><span id="systemDot" class="dot"></span><div><div style="font-size:12px;font-weight:700">Dashboard</div><div class="muted" style="font-size:10px">Auto-refreshing</div></div></div>
      <button class="btn" onclick="openSettings()">Settings</button>
      <a class="btn" href="/logout" style="text-decoration:none;text-align:center">Logout</a>
    </div>
  </aside>

  <main class="main">
    <div class="topbar">
      <div><h2 id="pageTitle">System Overview</h2><p id="pageSubtitle">A live view of trading, research, and model health.</p></div>
      <div class="top-actions"><span class="status-pill"><span id="clockDot" class="dot ok"></span><span id="lastRefresh">Updating…</span></span><button class="btn primary" onclick="openSettings()">Settings</button></div>
    </div>

    <section id="page-overview" class="page active">
      <div class="grid kpis">
        <div class="kpi"><div class="label">Live P&amp;L</div><div id="kLivePnl" class="value">—</div><div id="kLivePnlSub" class="sub">waiting for trades</div></div>
        <div class="kpi"><div class="label">Live Win Rate</div><div id="kWinRate" class="value">—</div><div id="kTrades" class="sub">0 completed trades</div></div>
        <div class="kpi"><div class="label">Backtest Return</div><div id="kBacktest" class="value">—</div><div id="kBacktestSub" class="sub">latest saved window</div></div>
        <div class="kpi"><div class="label">Meta AUC</div><div id="kAuc" class="value">—</div><div id="kAucSub" class="sub">validation</div></div>
        <div class="kpi"><div class="label">Training Rows</div><div id="kTrainRows" class="value">—</div><div class="sub">combined dataset</div></div>
        <div class="kpi"><div class="label">Best Strategy</div><div id="kStrategy" class="value" style="font-size:18px">—</div><div id="kStrategySub" class="sub">current configuration</div></div>
      </div>
      <div class="section grid content-grid">
        <div class="card"><div class="card-head"><div><div class="card-title">Live equity / cumulative P&amp;L</div><div class="card-sub">Completed live trade outcomes</div></div><span class="tag blue">LIVE</span></div><div class="chart-wrap"><canvas id="overviewLiveChart"></canvas></div></div>
        <div class="card"><div class="card-head"><div><div class="card-title">Current activity</div><div class="card-sub">Processes running now</div></div></div><div id="overviewRuns" class="run-grid" style="grid-template-columns:1fr"></div></div>
      </div>
      <div class="section grid two">
        <div class="card"><div class="card-head"><div><div class="card-title">Backtest performance by symbol</div><div class="card-sub">Latest completed window</div></div></div><div class="chart-wrap small"><canvas id="overviewBacktestChart"></canvas></div></div>
        <div class="card"><div class="card-head"><div><div class="card-title">Optimizer frontier</div><div class="card-sub">Recent tested configurations</div></div></div><div class="chart-wrap small"><canvas id="overviewOptimizerChart"></canvas></div></div>
      </div>
    </section>

    <section id="page-live" class="page">
      <div class="grid content-grid">
        <div class="card"><div class="card-head"><div><div class="card-title">Live cumulative P&amp;L</div><div class="card-sub">Updates automatically from trade_log.csv</div></div><span class="tag green">LIVE</span></div><div class="chart-wrap"><canvas id="livePnlChart"></canvas></div></div>
        <div class="card"><div class="card-head"><div><div class="card-title">Win / loss mix</div><div class="card-sub">Completed trades</div></div></div><div class="chart-wrap"><canvas id="livePieChart"></canvas></div></div>
      </div>
      <div class="section grid three"><div class="kpi"><div class="label">Total P&amp;L</div><div id="liveTotal" class="value">—</div></div><div class="kpi"><div class="label">Avg Trade</div><div id="liveAvg" class="value">—</div></div><div class="kpi"><div class="label">Max Drawdown</div><div id="liveDD" class="value">—</div></div></div>
      <div class="section card"><div class="card-head"><div><div class="card-title">Recent live trades</div><div class="card-sub">Newest completed outcomes</div></div></div><div id="liveTradesTable"></div></div>
      <div class="section card"><div class="card-head"><div><div class="card-title">Trader log</div><div class="card-sub">Latest process messages</div></div></div><div id="liveLog" class="console">No log data.</div></div>
    </section>

    <section id="page-backtest" class="page">
      <div class="card"><div class="card-head"><div><div class="card-title">Backtest control</div><div class="card-sub">Launch the same backtest process as the original dashboard.</div></div><div id="backtestBadge" class="tag">IDLE</div></div><div class="run-actions"><button class="btn primary" onclick="runScript('backtest')">Run Backtest</button><button class="btn danger" onclick="stopScriptByKey('backtest')">Stop</button></div><div style="margin-top:14px"><div class="progress"><i id="backtestProgress"></i></div><div id="backtestTiming" class="card-sub" style="margin-top:8px">No active run.</div></div></div>
      <div class="section grid content-grid">
        <div class="card"><div class="card-head"><div class="card-title">Latest backtest returns</div></div><div class="chart-wrap"><canvas id="backtestReturnChart"></canvas></div></div>
        <div class="card"><div class="card-head"><div class="card-title">Win rate by symbol</div></div><div class="chart-wrap"><canvas id="backtestWinChart"></canvas></div></div>
      </div>
      <div class="section card"><div class="card-head"><div><div class="card-title">Live backtest equity</div><div class="card-sub">Updates while backtest_live.json is being written.</div></div><span id="backtestLiveTag" class="tag">WAITING</span></div><div class="chart-wrap"><canvas id="backtestLiveChart"></canvas></div></div>
      <div class="section card"><div class="card-head"><div><div class="card-title">Latest backtest table</div><div class="card-sub">Symbols, return, win rate, and trade count.</div></div></div><div id="backtestTable"></div></div>
      <div class="section card"><div class="card-head"><div class="card-title">Backtest history</div></div><div id="backtestHistory"></div></div>
    </section>

    <section id="page-optimizer" class="page">
      <div class="card"><div class="card-head"><div><div class="card-title">Strategy optimizer</div><div class="card-sub">Searches new configurations without repeating the persistent optimizer history.</div></div><div id="optimizerStatusTag" class="tag">IDLE</div></div><div class="run-actions"><button class="btn primary" onclick="runScript('optimizer')">Run Optimizer</button><button class="btn danger" onclick="stopScriptByKey('optimizer')">Stop</button><button class="btn" onclick="refreshOptimizer()">Refresh Results</button></div></div>
      <div class="section grid two"><div class="card"><div class="card-head"><div class="card-title">Trial score distribution</div></div><div class="chart-wrap"><canvas id="optimizerScoreChart"></canvas></div></div><div class="card"><div class="card-head"><div class="card-title">Current strategy</div></div><div id="currentConfig" class="config-grid"></div></div></div>
      <div class="section card"><div class="card-head"><div><div class="card-title">Optimizer trial history</div><div class="card-sub">Best candidates first.</div></div></div><div id="optimizerTable"></div></div>
      <div class="section card"><div class="card-head"><div class="card-title">Validation history</div></div><div id="validationTable"></div></div>
    </section>

    <section id="page-model" class="page">
      <div class="grid three"><div class="kpi"><div class="label">Model Status</div><div id="modelStatus" class="value" style="font-size:18px">—</div></div><div class="kpi"><div class="label">Validation AUC</div><div id="modelAuc" class="value">—</div></div><div class="kpi"><div class="label">Combined Training Rows</div><div id="modelRows" class="value">—</div></div></div>
      <div class="section grid content-grid"><div class="card"><div class="card-head"><div class="card-title">Feature importance</div><div class="card-sub">Top learned predictors</div></div><div class="chart-wrap"><canvas id="featureChart"></canvas></div></div><div class="card"><div class="card-head"><div class="card-title">Training data sources</div></div><div class="chart-wrap"><canvas id="sourceChart"></canvas></div></div></div>
      <div class="section card"><div class="card-head"><div class="card-title">Model details</div></div><div id="modelDetails" class="config-grid"></div></div>
      <div class="section card"><div class="card-head"><div class="card-title">Train / meta-model controls</div></div><div class="run-actions"><button class="btn primary" onclick="runScript('train_meta_model')">Train Meta-Model</button><button class="btn" onclick="runScript('combine_logs')">Combine Trade Logs</button><button class="btn" onclick="runScript('pretest_check')">Run Pre-Test</button></div></div>
    </section>

    <section id="page-experiments" class="page">
      <div class="card"><div class="card-head"><div><div class="card-title">Experiment / hypothesis history</div><div class="card-sub">Recent entries from hypothesis_log.csv.</div></div><button class="btn" onclick="refreshHypothesis()">Refresh</button></div><div id="hypothesisTable"></div></div>
    </section>

    <section id="page-runs" class="page">
      <div class="section card"><div class="card-head"><div><div class="card-title">Process control</div><div class="card-sub">Start / stop the same scripts available in the original dashboard.</div></div></div><div id="runCards" class="grid run-grid"></div></div>
      <div class="section card"><div class="card-head"><div class="card-title">Selected process output</div><span id="selectedRunLabel" class="tag">None</span></div><div id="selectedRunConsole" class="console">Select a running or completed process.</div></div>
    </section>
  </main>
</div>

<div id="toast" class="toast"></div>
<div id="settingsModal" class="modal"><div class="modal-card"><div class="card-head"><div><div class="card-title">Settings</div><div class="card-sub">Credentials and notifications.</div></div><button class="btn" onclick="closeSettings()">Close</button></div><div class="field"><label>Alpaca API Key</label><input id="apiKey" type="password" autocomplete="off" placeholder="Leave blank to keep saved key"></div><div class="field"><label>Alpaca Secret Key</label><input id="secretKey" type="password" autocomplete="off" placeholder="Leave blank to keep saved secret"></div><div class="field"><label>Alpaca Base URL</label><input id="baseUrl" value="https://paper-api.alpaca.markets"></div><div id="credStatus" class="card-sub" style="margin:6px 0 16px">Loading…</div><div class="run-actions"><button class="btn primary" onclick="saveCreds()">Save Credentials</button><button class="btn" onclick="sendTestNotification()">Test Notification</button></div><div class="card-sub" style="margin-top:14px">Secrets are never shown back to the browser in raw form.</div></div></div>

<script>
const state={charts:{},runs:[],selectedRun:null,live:null,backtest:null,optimizer:null,model:null};
const pageMeta={
 overview:['System Overview','A live view of trading, research, and model health.'],
 live:['Live Trading','Portfolio outcomes, process logs, and completed trades.'],
 backtest:['Backtest Lab','Run experiments and compare walk-forward results.'],
 optimizer:['Strategy Optimizer','Search, validate, and promote conservative strategy improvements.'],
 model:['Meta Model','Training data, validation quality, and learned feature importance.'],
 experiments:['Experiments','Review hypothesis and experiment history.'],
 runs:['Runs & Logs','Full process control and console output.']
};
function $(id){return document.getElementById(id)}
function toast(msg){const e=$('toast');e.textContent=msg;e.style.display='block';clearTimeout(window.__toast);window.__toast=setTimeout(()=>e.style.display='none',2800)}
async function api(url,opts){const r=await fetch(url,opts);if(r.status===401){location.href='/login';return null}if(!r.ok){const t=await r.text();throw new Error(t||r.statusText)}return r}
function fmtPct(v){const n=Number(v);if(!Number.isFinite(n))return '—';return `${n>=0?'+':''}${n.toFixed(2)}%`}
function fmt(v,d=2){const n=Number(v);return Number.isFinite(n)?n.toFixed(d):'—'}
function destroyChart(id){if(state.charts[id]){state.charts[id].destroy();delete state.charts[id]}}
function chart(id,type,data,options={}){destroyChart(id);state.charts[id]=new Chart($(id),{type,data,options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#93a0b3',boxWidth:10,font:{size:10}}}},scales:type==='doughnut'?{}:{x:{ticks:{color:'#758399',font:{size:10}},grid:{color:'#192434'}},y:{ticks:{color:'#758399',font:{size:10}},grid:{color:'#192434'}}},...options}});return state.charts[id]}
function setPage(name){document.querySelectorAll('.page').forEach(e=>e.classList.remove('active'));$('page-'+name).classList.add('active');document.querySelectorAll('.nav button').forEach(b=>b.classList.toggle('active',b.dataset.page===name));$('pageTitle').textContent=pageMeta[name][0];$('pageSubtitle').textContent=pageMeta[name][1];location.hash=name;if(name==='live')refreshLive();if(name==='backtest')refreshBacktest();if(name==='optimizer')refreshOptimizer();if(name==='model')refreshModel();if(name==='experiments')refreshHypothesis();if(name==='runs')renderRuns()}
document.querySelectorAll('.nav button').forEach(b=>b.onclick=()=>setPage(b.dataset.page));
function renderTable(container,rows,columns,empty='No data yet.'){if(!rows||!rows.length){$(container).innerHTML=`<div class="empty">${empty}</div>`;return}let h='<div class="table-wrap"><table class="table"><thead><tr>'+columns.map(c=>`<th>${c.label}</th>`).join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+columns.map(c=>`<td>${c.render?c.render(r):formatCell(r[c.key])}</td>`).join('').replace(/<td>undefined<\/td>/g,'<td>—</td>')+'</tr>').join('')+'</tbody></table></div>';$(container).innerHTML=h}
function formatCell(v){if(v===null||v===undefined||v==='')return '—';if(typeof v==='number')return Number.isFinite(v)?v.toFixed(3):'—';return String(v)}
async function loadScripts(){try{const r=await api('/api/scripts');const data=await r.json();state.runs=data;updateSystemDot(data);renderRunCards(data);renderOverviewRuns(data);const b=data.find(x=>x.key==='backtest');if(b){$('backtestBadge').textContent=b.running_id?'RUNNING':'IDLE';$('backtestBadge').className='tag '+(b.running_id?'blue':'');$('backtestTiming').textContent=b.running_id?`Elapsed ${fmt(b.elapsed_seconds,0)}s · ETA ${b.eta_seconds==null?'—':fmt(b.eta_seconds,0)+'s'}`:'No active run.'}}catch(e){toast('Could not load script status')}}
function updateSystemDot(data){const active=data.some(x=>x.running_id);$('systemDot').className='dot '+(active?'ok':'ok')}
function runButton(key,label){return `<button class="btn primary" onclick="runScript('${key}')">${label}</button>`}
function renderOverviewRuns(data){$('overviewRuns').innerHTML=data.filter(x=>x.running_id).map(x=>`<div class="run-card" style="padding:12px"><h3>${x.label}</h3><p>${fmt(x.elapsed_seconds,0)}s elapsed · ETA ${x.eta_seconds==null?'—':fmt(x.eta_seconds,0)+'s'}</p><div class="run-actions"><button class="btn danger" onclick="stopScriptByKey('${x.key}')">Stop</button></div></div>`).join('')||'<div class="empty">No active process.</div>'}
function renderRunCards(data){$('runCards').innerHTML=data.map(x=>`<div class="card run-card"><h3>${x.label}</h3><p>${x.running_id?'Running':'Idle'} · ETA ${x.estimated_seconds==null?'—':fmt(x.estimated_seconds,0)+'s'}</p><div class="run-actions">${x.running_id?`<button class="btn danger" onclick="stopScriptByKey('${x.key}')">Stop</button>`:`<button class="btn primary" onclick="runScript('${x.key}')">Run</button>`}<button class="btn" onclick="selectRun('${x.running_id||''}')">Open</button></div></div>`).join('')}
async function runScript(key){try{const r=await api(`/api/run/${key}`,{method:'POST'});const j=await r.json();toast(`${key} started`);state.selectedRun=j.run_id;await loadScripts();pollRun(j.run_id)}catch(e){toast('Start failed: '+e.message)}}
async function stopScriptByKey(key){const run=state.runs.find(x=>x.key===key);if(!run||!run.running_id){toast('Nothing is running');return}try{await api(`/api/run/${run.running_id}/stop`,{method:'POST'});toast('Stop requested');await loadScripts()}catch(e){toast('Stop failed')}}
async function selectRun(id){state.selectedRun=id||null;if(id)pollRun(id);else renderConsole('',null)}
function renderConsole(text,id){$('selectedRunLabel').textContent=id||'None';$('selectedRunConsole').textContent=text||'Select a running or completed process.'}
async function pollRun(id){try{const r=await api(`/api/run/${id}/status`);const j=await r.json();renderConsole((j.output||[]).join('\n'),id);if(j.status==='running'||j.status==='stopping'){setTimeout(()=>pollRun(id),1200)}else{loadScripts();toast(`${id} ${j.status}`)}}catch(e){renderConsole('Unable to read process output.',id)}}
async function refreshOverview(){await Promise.allSettled([refreshLive(),refreshBacktest(),refreshOptimizer(),refreshModel(),refreshHypothesis(),loadScripts()]);updateKpis()}
async function refreshLive(){try{const r=await api('/api/analytics/portfolio');state.live=await r.json();renderLive()}catch(e){}}
function renderLive(){const d=state.live||{};if(!d.available){$('liveTotal').textContent='—';$('liveAvg').textContent='—';$('liveDD').textContent='—';$('kLivePnl').textContent='—';$('kWinRate').textContent='—';return}$('liveTotal').textContent=fmtPct(d.total_pnl_pct);$('liveAvg').textContent=fmtPct(d.avg_trade_pct);$('liveDD').textContent=fmtPct(d.max_drawdown_pct);$('kLivePnl').textContent=fmtPct(d.total_pnl_pct);$('kWinRate').textContent=fmt(d.win_rate,1)+'%';$('kTrades').textContent=`${d.total_trades} completed trades`;const t=d.timeline||{};chart('livePnlChart','line',{labels:t.timestamps||[],datasets:[{label:'Cumulative P&L %',data:t.cum_pnl||[],borderColor:'#6c8cff',backgroundColor:'rgba(108,140,255,.10)',fill:true,tension:.24,pointRadius:0}]});chart('overviewLiveChart','line',{labels:t.timestamps||[],datasets:[{label:'Cumulative P&L %',data:t.cum_pnl||[],borderColor:'#50c9ff',backgroundColor:'rgba(80,201,255,.08)',fill:true,tension:.24,pointRadius:0}]});const wins=Math.round(d.total_trades*d.win_rate/100),losses=d.total_trades-wins;chart('livePieChart','doughnut',{labels:['Wins','Losses'],datasets:[{data:[wins,losses],backgroundColor:['#2dd78b','#ff6673'],borderWidth:0}]});const rows=(d.trades||[]).slice().reverse();renderTable('liveTradesTable',rows,[{key:'exit_timestamp',label:'Time'},{key:'symbol',label:'Symbol'},{key:'pnl_pct',label:'P&L',render:r=>`<span class="${Number(r.pnl_pct)>=0?'positive':'negative'}">${fmtPct(r.pnl_pct)}</span>`},{key:'exit_reason',label:'Reason'},{key:'entry_confidence',label:'Conf.'},{key:'entry_meta_win_prob',label:'Meta'}]);}
async function refreshBacktest(){try{const [s,l,h]=await Promise.all([api('/api/data/backtest_summary'),api('/api/data/backtest_live'),api('/api/backtests')]);state.backtest={summary:await s.json(),live:await l.json(),history:await h.json()};renderBacktest()}catch(e){}}
function renderBacktest(){const d=state.backtest?.summary||{};if(!d.available){return}$('kBacktest').textContent=d.returns_pct?.length?fmtPct(d.returns_pct.reduce((a,b)=>a+b,0)/(d.returns_pct.length)): '—';const labels=d.symbols||[];chart('backtestReturnChart','bar',{labels,datasets:[{label:'Return %',data:d.returns_pct||[],backgroundColor:(d.returns_pct||[]).map(v=>Number(v)>=0?'#2dd78b':'#ff6673'),borderRadius:5}]});chart('overviewBacktestChart','bar',{labels,datasets:[{label:'Return %',data:d.returns_pct||[],backgroundColor:'#6c8cff',borderRadius:5}]});chart('backtestWinChart','bar',{labels,datasets:[{label:'Win rate %',data:d.win_rates||[],backgroundColor:'#50c9ff',borderRadius:5}]});const live=state.backtest?.live||{};const eq=live.equity||[];const ts=live.timestamps||[];if(eq.length){$('backtestLiveTag').textContent='LIVE';$('backtestLiveTag').className='tag green';const base=Number(eq[0]||0);const ret=eq.map(v=>base?((Number(v)/base)-1)*100:0);chart('backtestLiveChart','line',{labels:ts,datasets:[{label:'Return %',data:ret,borderColor:'#50c9ff',backgroundColor:'rgba(80,201,255,.08)',fill:true,tension:.2,pointRadius:0}]})}const rows=labels.map((s,i)=>({symbol:s,return:d.returns_pct?.[i],win_rate:d.win_rates?.[i],trades:d.num_trades?.[i]}));renderTable('backtestTable',rows,[{key:'symbol',label:'Symbol'},{key:'return',label:'Return',render:r=>`<span class="${Number(r.return)>=0?'positive':'negative'}">${fmtPct(r.return)}</span>`},{key:'win_rate',label:'Win Rate',render:r=>fmt(r.win_rate,1)+'%'},{key:'trades',label:'Trades'}]);$('backtestHistory').innerHTML=(state.backtest.history||[]).slice(0,30).map(r=>`<div style="padding:11px 0;border-bottom:1px solid #1b2636"><div style="display:flex;justify-content:space-between;gap:10px"><strong>${r.run_id}</strong><span class="tag">${r.summary?.status||'run'}</span></div><div class="muted" style="margin-top:5px;font-size:11px">${r.summary?.completed_at||''}</div><div class="run-actions" style="margin-top:8px"><a class="btn" href="/api/backtests/${r.run_id}/chart" target="_blank">Chart</a><a class="btn" href="/api/backtests/${r.run_id}/results" target="_blank">CSV</a></div></div>`).join('')||'<div class="empty">No archived backtests.</div>'}
async function refreshOptimizer(){try{const r=await api('/api/analytics/optimizer');state.optimizer=await r.json();renderOptimizer()}catch(e){}}
function renderOptimizer(){const d=state.optimizer||{};const trials=d.recent_trials||[];const top=trials.slice(0,24);chart('optimizerScoreChart','scatter',{datasets:[{label:'Trials',data:top.map((r,i)=>({x:i+1,y:Number(r.score)||0})),backgroundColor:'#9a7cff',pointRadius:5}]},{scales:{x:{title:{display:true,text:'Rank',color:'#758399'},ticks:{color:'#758399'},grid:{color:'#192434'}},y:{title:{display:true,text:'Score',color:'#758399'},ticks:{color:'#758399'},grid:{color:'#192434'}}}});const cfg=d.best_config||{};$('currentConfig').innerHTML=Object.keys(cfg).length?Object.entries(cfg).map(([k,v])=>`<div class="config-item"><div class="k">${k}</div><div class="v">${v}</div></div>`).join(''):'<div class="empty">No promoted strategy yet.</div>';renderTable('optimizerTable',trials,[{key:'tested_at',label:'Tested'},{key:'BAR_TIMEFRAME',label:'TF'},{key:'MIN_SIGNAL_CONFIDENCE',label:'Signal'},{key:'MIN_META_WIN_PROB',label:'Meta'},{key:'return',label:'Return',render:r=>fmtPct(r.return)},{key:'sharpe',label:'Sharpe',render:r=>fmt(r.sharpe)},{key:'drawdown',label:'DD',render:r=>fmtPct(r.drawdown)},{key:'score',label:'Score',render:r=>fmt(r.score)}]);renderTable('validationTable',d.validation||[],[{key:'config_id',label:'Config'},{key:'validation_offset',label:'Offset'},{key:'return',label:'Return',render:r=>fmtPct(r.return)},{key:'sharpe',label:'Sharpe',render:r=>fmt(r.sharpe)},{key:'drawdown',label:'DD',render:r=>fmtPct(r.drawdown)},{key:'score',label:'Score',render:r=>fmt(r.score)}]);}
async function refreshModel(){try{const r=await api('/api/analytics/model');state.model=await r.json();renderModel()}catch(e){}}
function renderModel(){const d=state.model||{};const m=d.info||{};const vm=m.validation_metrics||{};$('modelStatus').textContent=d.model_exists?'Loaded':'Not trained';$('modelAuc').textContent=vm.auc!=null?fmt(vm.auc,4):'—';$('modelRows').textContent=(d.training_rows||0).toLocaleString();$('kAuc').textContent=vm.auc!=null?fmt(vm.auc,3):'—';$('kTrainRows').textContent=(d.training_rows||0).toLocaleString();const imp=d.feature_importance||[];chart('featureChart','bar',{labels:imp.slice(0,12).map(x=>x.feature),datasets:[{label:'Importance',data:imp.slice(0,12).map(x=>x.importance),backgroundColor:'#6c8cff',borderRadius:4}]},{indexAxis:'y'});const src=d.sources||{};chart('sourceChart','doughnut',{labels:Object.keys(src),datasets:[{data:Object.values(src),backgroundColor:['#6c8cff','#2dd78b','#f2b84b','#9a7cff'],borderWidth:0}]});$('modelDetails').innerHTML=[['Validation accuracy',vm.accuracy],['Balanced accuracy',vm.balanced_accuracy],['Brier score',vm.brier],['Validation rows',m.validation_rows],['Training rows',m.training_rows],['Total completed trades',m.total_completed_trades]].map(([k,v])=>`<div class="config-item"><div class="k">${k}</div><div class="v">${v??'—'}</div></div>`).join('');}
async function refreshHypothesis(){try{const r=await api('/api/data/hypothesis');const d=await r.json();if(!d.available){$('hypothesisTable').innerHTML='<div class=\"empty\">No hypothesis history yet.</div>';return}const rows=d.rows||[];const cols=rows.length?Object.keys(rows[0]).map(k=>({key:k,label:k.replace(/_/g,' ')})):[];renderTable('hypothesisTable',rows,cols,'No experiment history yet.')}catch(e){$('hypothesisTable').innerHTML='<div class=\"empty\">Unable to load experiment history.</div>'}}
async function refreshCreds(){try{const r=await api('/api/credentials');const d=await r.json();$('credStatus').textContent=`API key: ${d.api_key_masked||'not set'} · Secret: ${d.secret_key_masked||'not set'} · ${d.is_paper?'Paper':'Live'} endpoint`;if(d.base_url)$('baseUrl').value=d.base_url}catch(e){}}
function openSettings(){$('settingsModal').classList.add('open');refreshCreds()};function closeSettings(){$('settingsModal').classList.remove('open')}
async function saveCreds(){const body={base_url:$('baseUrl').value.trim()};if($('apiKey').value.trim())body.alpaca_api_key=$('apiKey').value.trim();if($('secretKey').value.trim())body.alpaca_secret_key=$('secretKey').value.trim();try{await api('/api/credentials',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});$('apiKey').value='';$('secretKey').value='';toast('Credentials saved');refreshCreds()}catch(e){toast('Could not save credentials')}}
async function sendTestNotification(){try{await api('/api/notifications/test',{method:'POST'});toast('Test notification sent')}catch(e){toast('Notification test failed')}}
function updateKpis(){const l=state.live||{},b=state.backtest?.summary||{},o=state.optimizer||{};if(b.returns_pct?.length){const avg=b.returns_pct.reduce((a,x)=>a+Number(x||0),0)/b.returns_pct.length;$('kBacktest').textContent=fmtPct(avg)}if(o.best_config&&Object.keys(o.best_config).length){$('kStrategy').textContent=o.best_config.BAR_TIMEFRAME||'Active';$('kStrategySub').textContent=`signal ${o.best_config.MIN_SIGNAL_CONFIDENCE??'—'} · meta ${o.best_config.MIN_META_WIN_PROB??'—'}`}}
setInterval(()=>{const now=new Date();$('lastRefresh').textContent=`Updated ${now.toLocaleTimeString()}`},1000);setInterval(refreshOverview,10000);
(async()=>{const hash=location.hash.replace('#','');if(pageMeta[hash])setPage(hash);else setPage('overview');await refreshOverview();})();
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