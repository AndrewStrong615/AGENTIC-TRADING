"""Conservative strategy optimizer with persistent search history.

Design goals
------------
1. Never repeatedly test the same strategy configuration.
2. Search broadly enough to find real improvements, but keep risk bounded.
3. Compare candidates against the CURRENT promoted strategy on the same
   validation periods.
4. Require consistency across multiple holdouts instead of one lucky period.
5. Use a smaller promotion margin than the original optimizer, but add
   absolute safety gates and a worst-case guardrail.
6. Keep a permanent optimizer history so future runs continue exploring.

Usage
-----
    python strategy_optimizer_conservative.py --trials 24

Optional:
    python strategy_optimizer_conservative.py --trials 24 --seed 123
    python strategy_optimizer_conservative.py --trials 24 --reset-history

The optimizer writes:
    trade_data/optimizer_history.csv
    trade_data/optimizer_results.csv
    trade_data/optimizer_validation.csv
    trade_data/best_strategy_config.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BACKTEST = BASE_DIR / "backtest.py"
DATA_DIR = BASE_DIR / "trade_data"
RUNS_DIR = DATA_DIR / "backtests"
RESULTS_PATH = DATA_DIR / "optimizer_results.csv"
VALIDATION_PATH = DATA_DIR / "optimizer_validation.csv"
HISTORY_PATH = DATA_DIR / "optimizer_history.csv"
BEST_PATH = DATA_DIR / "best_strategy_config.json"

CONFIG_KEYS = (
    "BAR_TIMEFRAME",
    "MIN_SIGNAL_CONFIDENCE",
    "MIN_META_WIN_PROB",
    "MIN_ATR_PCT",
    "MAX_ATR_PCT",
    "MIN_TREND_CHANGE_PCT",
    "MIN_BARS_BETWEEN_TRADES",
    "RISK_PER_TRADE_PCT",
    "MAX_POSITION_PCT",
    "STOP_LOSS_PCT",
    "TAKE_PROFIT_PCT",
    "TRAIL_STOP_PCT",
    "ATR_STOP_MULTIPLIER",
)

# Conservative promotion settings.
PROMOTION_MARGIN = 0.03
MIN_SHARPE = 0.30
MIN_RETURN = 0.0
MAX_DRAWDOWN = -15.0
MIN_TRADES = 20
MIN_HOLDOUTS_TO_BEAT_CURRENT = 2
WORST_CASE_TOLERANCE = 0.05

DEFAULT_SYMBOLS = (
    "AAPL,MSFT,NVDA,AMZN,GOOGL,META,AVGO,TSLA,AMD,MU,JPM,V,MA,BAC,WMT,"
    "COST,XOM,JNJ,LLY,ABBV,HD,CAT,CSCO,ORCL,NFLX"
)


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def config_id(cfg: dict) -> str:
    payload = {k: cfg.get(k) for k in CONFIG_KEYS}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def calculate_metrics(summary: dict) -> dict:
    rows = summary.get("windows") or []
    if not rows:
        return {}

    def avg(key):
        values = [as_float(row.get(key), 0.0) for row in rows]
        return sum(values) / len(values) if values else 0.0

    ret = avg("total_return_pct_mean")
    dd = avg("max_drawdown_pct_mean")
    sharpe = avg("sharpe_mean")
    sortino = avg("sortino_mean")
    calmar = avg("calmar_mean")
    profit_factor = avg("profit_factor_mean")
    trades = avg("num_trades_mean")

    # Same family of metrics as the original optimizer, but retained here as
    # a single source of truth.
    score = (
        1.00 * sharpe
        + 0.40 * sortino
        + 0.20 * calmar
        + 0.06 * ret
        + 0.12 * profit_factor
        - 0.10 * abs(dd)
    )

    return {
        "return": ret,
        "drawdown": dd,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "profit_factor": profit_factor,
        "trades": trades,
        "score": score,
    }


def load_current_config() -> dict:
    if not BEST_PATH.exists():
        return {}
    try:
        data = json.loads(BEST_PATH.read_text(encoding="utf-8"))
        config = data.get("config", {})
        return config if isinstance(config, dict) else {}
    except Exception as exc:
        print(f"WARNING: could not read {BEST_PATH}: {exc}")
        return {}


def load_history() -> dict:
    """Return {config_id: row} for every previously tested config."""
    history = {}
    if not HISTORY_PATH.exists():
        return history
    try:
        with HISTORY_PATH.open("r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                cid = row.get("config_id")
                if cid:
                    history[cid] = row
    except Exception as exc:
        print(f"WARNING: could not read {HISTORY_PATH}: {exc}")
    return history


def save_history(rows: list[dict]) -> None:
    if not rows:
        return

    DATA_DIR.mkdir(exist_ok=True)
    existing = HISTORY_PATH.exists()

    fieldnames = [
        "tested_at",
        "seed",
        "config_id",
        *CONFIG_KEYS,
        "run_id",
        "returncode",
        "return",
        "drawdown",
        "sharpe",
        "sortino",
        "calmar",
        "profit_factor",
        "trades",
        "score",
    ]

    # Append only new rows.
    prior_ids = set(load_history())
    new_rows = [row for row in rows if row["config_id"] not in prior_ids]
    if not new_rows:
        return

    with HISTORY_PATH.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if not existing:
            writer.writeheader()
        writer.writerows(new_rows)


def make_candidate_from_current(current: dict, rng: random.Random) -> dict:
    """Explore a neighborhood around the current config."""
    stop = rng.choice([0.6, 0.8, 1.0, 1.2, 1.4])
    rr = rng.choice([1.8, 2.0, 2.5, 3.0])

    current_conf = as_float(current.get("MIN_SIGNAL_CONFIDENCE"), 0.40)
    current_meta = as_float(current.get("MIN_META_WIN_PROB"), 0.62)

    return {
        "BAR_TIMEFRAME": rng.choice(["5m", "10m", "15m"]),
        "MIN_SIGNAL_CONFIDENCE": round(
            min(0.55, max(0.30, current_conf + rng.choice([-0.05, -0.025, 0.025, 0.05])),),
            3,
        ),
        "MIN_META_WIN_PROB": round(
            min(0.72, max(0.55, current_meta + rng.choice([-0.04, -0.02, 0.02, 0.04])),),
            3,
        ),
        "RISK_PER_TRADE_PCT": rng.choice([0.0020, 0.0025, 0.0030, 0.0035, 0.0040]),
        "MAX_POSITION_PCT": rng.choice([0.08, 0.10, 0.12, 0.15, 0.18]),
        "STOP_LOSS_PCT": stop,
        "TAKE_PROFIT_PCT": round(stop * rr, 2),
        "TRAIL_STOP_PCT": rng.choice([0.0, 0.4, 0.6, 0.8, 1.0]),
        "ATR_STOP_MULTIPLIER": rng.choice([0.5, 0.75, 1.0, 1.25, 1.5]),
        "MIN_ATR_PCT": rng.choice([0.0, 0.03, 0.05, 0.08, 0.10]),
        "MAX_ATR_PCT": rng.choice([1.5, 2.0, 2.5, 3.0]),
        "MIN_TREND_CHANGE_PCT": rng.choice([0.0, 0.01, 0.02, 0.03]),
        "MIN_BARS_BETWEEN_TRADES": rng.choice([0, 1, 2, 3]),
    }


def make_global_candidate(rng: random.Random) -> dict:
    stop = rng.choice([0.6, 0.8, 1.0, 1.2, 1.4])
    rr = rng.choice([1.8, 2.0, 2.5, 3.0])
    return {
        "BAR_TIMEFRAME": rng.choice(["5m", "10m", "15m"]),
        "MIN_SIGNAL_CONFIDENCE": rng.choice([0.30, 0.35, 0.40, 0.425, 0.45, 0.475, 0.50, 0.525]),
        "MIN_META_WIN_PROB": rng.choice([0.55, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70]),
        "RISK_PER_TRADE_PCT": rng.choice([0.0020, 0.0025, 0.0030, 0.0035, 0.0040]),
        "MAX_POSITION_PCT": rng.choice([0.08, 0.10, 0.12, 0.15, 0.18]),
        "STOP_LOSS_PCT": stop,
        "TAKE_PROFIT_PCT": round(stop * rr, 2),
        "TRAIL_STOP_PCT": rng.choice([0.0, 0.4, 0.6, 0.8, 1.0]),
        "ATR_STOP_MULTIPLIER": rng.choice([0.5, 0.75, 1.0, 1.25, 1.5]),
        "MIN_ATR_PCT": rng.choice([0.0, 0.03, 0.05, 0.08, 0.10]),
        "MAX_ATR_PCT": rng.choice([1.5, 2.0, 2.5, 3.0]),
        "MIN_TREND_CHANGE_PCT": rng.choice([0.0, 0.01, 0.02, 0.03]),
        "MIN_BARS_BETWEEN_TRADES": rng.choice([0, 1, 2, 3]),
    }


def generate_candidates(n: int, current: dict, history: dict, seed: int | None) -> list[dict]:
    if seed is None:
        seed = int(time.time_ns() & 0xFFFFFFFF)
    rng = random.Random(seed)

    seen = set(history)
    if current:
        seen.add(config_id(current))

    candidates = []

    # ~65% local search / ~35% global exploration. This lets the optimizer
    # exploit promising regions without collapsing into one tiny neighborhood.
    while len(candidates) < n:
        use_local = bool(current) and rng.random() < 0.65
        cfg = make_candidate_from_current(current, rng) if use_local else make_global_candidate(rng)
        cid = config_id(cfg)
        if cid in seen:
            continue
        seen.add(cid)
        candidates.append(cfg)

    return candidates


def run_backtest(cfg: dict, offset: int, windows: int, days: int, symbols: list[str], tag: str) -> dict:
    run_id = f"opt-{tag}-{uuid.uuid4().hex[:8]}"
    env = os.environ.copy()
    env.update({key: str(value) for key, value in cfg.items()})
    env.update(
        {
            "BACKTEST_RUN_ID": run_id,
            "BACKTEST_END_OFFSET_DAYS": str(offset),
            "NUM_WINDOWS": str(windows),
            "WINDOW_LENGTH_DAYS": str(days),
            "ITERATIONS_PER_WINDOW": "1",
            "BACKTEST_SYMBOLS": ",".join(symbols),
            "SLEEP_BETWEEN_SYMBOLS": "0",
            "NTFY_TOPIC": "",
        }
    )

    process = subprocess.run(
        [sys.executable, str(BACKTEST)],
        cwd=str(BASE_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    summary_path = RUNS_DIR / run_id / "summary.json"
    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}

    result = calculate_metrics(summary)
    result.update(
        {
            "run_id": run_id,
            "returncode": process.returncode,
            "output_tail": process.stdout[-1000:],
        }
    )
    return result


def safe_holdout(result: dict) -> bool:
    return bool(result) and (
        result.get("return", -999.0) > MIN_RETURN
        and result.get("drawdown", -999.0) > MAX_DRAWDOWN
        and result.get("sharpe", -999.0) > MIN_SHARPE
        and result.get("trades", 0.0) >= MIN_TRADES
    )


def evaluate_candidate_vs_current(candidate_results: list[dict], current_results: list[dict]) -> dict:
    if not candidate_results or len(candidate_results) != len(current_results):
        return {"eligible": False, "reason": "missing holdout results"}

    candidate_scores = [as_float(r.get("score"), -999.0) for r in candidate_results]
    current_scores = [as_float(r.get("score"), -999.0) for r in current_results]

    candidate_avg = sum(candidate_scores) / len(candidate_scores)
    current_avg = sum(current_scores) / len(current_scores)
    candidate_min = min(candidate_scores)
    current_min = min(current_scores)

    beats = sum(c > b for c, b in zip(candidate_scores, current_scores))
    avg_gain = candidate_avg - current_avg
    worst_case_change = candidate_min - current_min

    eligible = (
        all(safe_holdout(r) for r in candidate_results)
        and beats >= MIN_HOLDOUTS_TO_BEAT_CURRENT
        and avg_gain >= PROMOTION_MARGIN
        and worst_case_change >= -WORST_CASE_TOLERANCE
    )

    return {
        "eligible": eligible,
        "beats": beats,
        "candidate_avg": candidate_avg,
        "current_avg": current_avg,
        "candidate_min": candidate_min,
        "current_min": current_min,
        "avg_gain": avg_gain,
        "worst_case_change": worst_case_change,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--train-offset", type=int, default=240)
    parser.add_argument("--validation-offsets", default="0,30,60")
    parser.add_argument("--train-windows", type=int, default=4)
    parser.add_argument("--validation-windows", type=int, default=3)
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--reset-history", action="store_true")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    if args.reset_history and HISTORY_PATH.exists():
        HISTORY_PATH.unlink()
        print(f"Reset optimizer history: {HISTORY_PATH}")

    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    history = load_history()
    current = load_current_config()

    candidates = generate_candidates(args.trials, current, history, args.seed)
    if not candidates:
        print("No new configurations available.")
        return

    print(f"Testing {len(candidates)} NEW configurations.")
    print(f"Previously tested: {len(history)}")
    print(f"Current promoted config: {'yes' if current else 'no'}")
    print(f"Validation holdouts: {args.validation_offsets}")

    training_rows = []
    for index, cfg in enumerate(candidates, 1):
        result = run_backtest(
            cfg,
            args.train_offset,
            args.train_windows,
            args.window_days,
            symbols,
            f"train-{index}",
        )
        row = {
            "tested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seed": "" if args.seed is None else args.seed,
            "config_id": config_id(cfg),
            **cfg,
            **{k: v for k, v in result.items() if k != "output_tail"},
        }
        training_rows.append(row)
        print(
            f"train {index}/{len(candidates)} "
            f"score={result.get('score', 0):.2f} "
            f"ret={result.get('return', 0):+.2f}% "
            f"sharpe={result.get('sharpe', 0):.2f} "
            f"dd={result.get('drawdown', 0):.2f}%"
        )

    save_history(training_rows)

    # Latest-run report.
    if training_rows:
        with RESULTS_PATH.open("w", newline="", encoding="utf-8") as fh:
            fields = sorted(set().union(*(r.keys() for r in training_rows)))
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(training_rows)

    # Validate the strongest five candidates instead of only three.
    ranked = sorted(training_rows, key=lambda row: as_float(row.get("score"), -999.0), reverse=True)
    top = ranked[: min(5, len(ranked))]
    offsets = [int(x) for x in args.validation_offsets.split(",") if x.strip()]

    validation_rows = []
    candidate_holdouts: dict[str, list[dict]] = {}

    for candidate in top:
        cfg = {key: candidate[key] for key in CONFIG_KEYS}
        results = []
        for offset in offsets:
            result = run_backtest(
                cfg,
                offset,
                args.validation_windows,
                args.window_days,
                symbols,
                f"val-{offset}",
            )
            results.append(result)
            validation_rows.append(
                {
                    **cfg,
                    "config_id": candidate["config_id"],
                    "validation_offset": offset,
                    **{k: v for k, v in result.items() if k != "output_tail"},
                }
            )
        candidate_holdouts[candidate["config_id"]] = results

    # Always evaluate the current strategy on exactly the same holdouts.
    current_holdouts = []
    if current:
        for offset in offsets:
            current_holdouts.append(
                run_backtest(
                    current,
                    offset,
                    args.validation_windows,
                    args.window_days,
                    symbols,
                    f"current-{offset}",
                )
            )

        for offset, result in zip(offsets, current_holdouts):
            validation_rows.append(
                {
                    **{key: current.get(key, "") for key in CONFIG_KEYS},
                    "config_id": config_id(current),
                    "validation_offset": offset,
                    "is_current": True,
                    **{k: v for k, v in result.items() if k != "output_tail"},
                }
            )

    if validation_rows:
        with VALIDATION_PATH.open("w", newline="", encoding="utf-8") as fh:
            fields = sorted(set().union(*(row.keys() for row in validation_rows)))
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(validation_rows)

    # First promotion if there is no current strategy.
    if not current:
        eligible = [
            row for row in top
            if all(safe_holdout(x) for x in candidate_holdouts.get(row["config_id"], []))
        ]
        if eligible:
            best = max(
                eligible,
                key=lambda row: min(
                    as_float(x.get("score"), -999.0)
                    for x in candidate_holdouts[row["config_id"]]
                ),
            )
            best_cfg = {key: best[key] for key in CONFIG_KEYS}
            BEST_PATH.write_text(
                json.dumps(
                    {
                        "mode": "conservative",
                        "config": best_cfg,
                        "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "reason": "initial candidate cleared all safety holdouts",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"\nPROMOTED initial conservative strategy -> {BEST_PATH}")
        else:
            print("\nNo initial candidate cleared the safety holdouts.")
        return

    comparisons = []
    for candidate in top:
        cid = candidate["config_id"]
        result = evaluate_candidate_vs_current(
            candidate_holdouts.get(cid, []),
            current_holdouts,
        )
        result["config_id"] = cid
        result["candidate"] = candidate
        comparisons.append(result)

    eligible = [x for x in comparisons if x.get("eligible")]
    if eligible:
        winner = max(eligible, key=lambda x: x["candidate_avg"])
        candidate = winner["candidate"]
        payload = {
            "mode": "conservative",
            "config": {key: candidate[key] for key in CONFIG_KEYS},
            "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "promotion": {
                "promotion_margin": PROMOTION_MARGIN,
                "holdouts_beaten": winner["beats"],
                "holdout_count": len(offsets),
                "candidate_average_score": winner["candidate_avg"],
                "current_average_score": winner["current_avg"],
                "candidate_worst_score": winner["candidate_min"],
                "current_worst_score": winner["current_min"],
                "average_gain": winner["avg_gain"],
                "worst_case_change": winner["worst_case_change"],
            },
        }
        BEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nPROMOTED improved conservative strategy -> {BEST_PATH}")
        print(
            f"Beat current on {winner['beats']}/{len(offsets)} holdouts; "
            f"average score gain={winner['avg_gain']:+.3f}; "
            f"worst-case change={winner['worst_case_change']:+.3f}."
        )
    else:
        current_scores = [as_float(x.get("score"), -999.0) for x in current_holdouts]
        current_avg = sum(current_scores) / len(current_scores) if current_scores else -999.0
        current_min = min(current_scores) if current_scores else -999.0
        print("\nKEPT current strategy config.")
        print(
            "No candidate cleared all conservative promotion gates: "
            f"{MIN_HOLDOUTS_TO_BEAT_CURRENT}/{len(offsets)} holdouts, "
            f"+{PROMOTION_MARGIN:.2f} average score, and "
            f"worst-case tolerance {WORST_CASE_TOLERANCE:.2f}."
        )
        print(f"Current holdout average score: {current_avg:.3f}")
        print(f"Current holdout worst score:    {current_min:.3f}")


if __name__ == "__main__":
    main()