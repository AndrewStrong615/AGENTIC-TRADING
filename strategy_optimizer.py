"""
strategy_optimizer.py

Walk-forward optimizer for the existing trading bot.

It does NOT replace Tradingcore.py. Instead it launches the existing
backtest.py repeatedly with different, externally-configurable signal/risk
settings, ranks configurations on risk-adjusted performance, then validates
the best configurations on a different historical period.

Outputs:
    trade_data/optimizer_results.csv
    trade_data/optimizer_validation.csv
    trade_data/best_strategy_config.json

Recommended workflow:
    python strategy_optimizer.py

Use --trials 8 for a quick test or --trials 24 for a broader search.
All optimization runs are backtests only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
BEST_PATH = DATA_DIR / "best_strategy_config.json"



def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)



def flatten_summary(summary):
    rows = summary.get("windows") or []
    if not rows:
        return {}

    def vals(key):
        out = []
        for row in rows:
            try:
                v = float(row.get(key, 0.0))
                if math.isfinite(v):
                    out.append(v)
            except (TypeError, ValueError):
                pass
        return out

    metrics = {}
    for key in (
        "total_return_pct_mean",
        "max_drawdown_pct_mean",
        "win_rate_mean",
        "num_trades_mean",
        "profit_factor_mean",
        "expectancy_pct_mean",
        "sharpe_mean",
        "sortino_mean",
        "calmar_mean",
    ):
        v = vals(key)
        metrics[key] = sum(v) / len(v) if v else 0.0

    # Stability across walk-forward windows.
    returns = vals("total_return_pct_mean")
    sharpes = vals("sharpe_mean")
    metrics["return_std"] = (
        math.sqrt(sum((x - metrics["total_return_pct_mean"]) ** 2 for x in returns) / len(returns))
        if returns else 0.0
    )
    metrics["sharpe_std"] = (
        math.sqrt(sum((x - metrics["sharpe_mean"]) ** 2 for x in sharpes) / len(sharpes))
        if sharpes else 0.0
    )

    # Prefer stable risk-adjusted returns over one spectacular backtest window.
    metrics["score"] = (
        1.00 * metrics["sharpe_mean"]
        + 0.50 * metrics["sortino_mean"]
        + 0.20 * metrics["calmar_mean"]
        + 0.08 * metrics["total_return_pct_mean"]
        + 0.15 * metrics["profit_factor_mean"]
        - 0.12 * metrics["return_std"]
        - 0.08 * metrics["sharpe_std"]
    )
    return metrics



def make_configs(trials, seed=42):
    rng = random.Random(seed)

    # Intentionally moderate search space. The optimizer is supposed to find
    # robust settings, not a bizarre one-off parameter combination.
    timeframes = ["1m", "5m", "15m"]
    confs = [0.0, 0.20, 0.35, 0.50]
    meta_probs = [0.0, 0.55, 0.60, 0.65]
    risk = [0.003, 0.005, 0.0075, 0.01]
    max_pos = [0.10, 0.15, 0.20, 0.25]
    stops = [0.6, 0.8, 1.0, 1.25]
    take_multipliers = [1.5, 2.0, 2.5, 3.0]
    trails = [0.0, 0.6, 0.8]
    atr_mults = [0.0, 1.0, 1.5]
    min_atr = [0.0, 0.05, 0.10]
    max_atr = [1.0, 1.5, 2.5]

    configs = []
    seen = set()

    # Always include sensible baselines.
    baseline = [
        {"BAR_TIMEFRAME": "1m", "MIN_SIGNAL_CONFIDENCE": 0.0, "MIN_META_WIN_PROB": 0.0,
         "RISK_PER_TRADE_PCT": 0.005, "MAX_POSITION_PCT": 0.20,
         "STOP_LOSS_PCT": 1.0, "TAKE_PROFIT_PCT": 2.0, "TRAIL_STOP_PCT": 0.0,
         "ATR_STOP_MULTIPLIER": 0.0, "MIN_ATR_PCT": 0.0, "MAX_ATR_PCT": 1000.0},
        {"BAR_TIMEFRAME": "5m", "MIN_SIGNAL_CONFIDENCE": 0.20, "MIN_META_WIN_PROB": 0.55,
         "RISK_PER_TRADE_PCT": 0.005, "MAX_POSITION_PCT": 0.20,
         "STOP_LOSS_PCT": 0.8, "TAKE_PROFIT_PCT": 2.0, "TRAIL_STOP_PCT": 0.0,
         "ATR_STOP_MULTIPLIER": 1.0, "MIN_ATR_PCT": 0.0, "MAX_ATR_PCT": 2.5},
        {"BAR_TIMEFRAME": "15m", "MIN_SIGNAL_CONFIDENCE": 0.35, "MIN_META_WIN_PROB": 0.60,
         "RISK_PER_TRADE_PCT": 0.005, "MAX_POSITION_PCT": 0.15,
         "STOP_LOSS_PCT": 1.0, "TAKE_PROFIT_PCT": 2.5, "TRAIL_STOP_PCT": 0.8,
         "ATR_STOP_MULTIPLIER": 1.0, "MIN_ATR_PCT": 0.05, "MAX_ATR_PCT": 2.5},
    ]

    for c in baseline:
        configs.append(c)
        seen.add(json.dumps(c, sort_keys=True))

    while len(configs) < trials:
        stop = rng.choice(stops)
        take = round(stop * rng.choice(take_multipliers), 3)
        c = {
            "BAR_TIMEFRAME": rng.choice(timeframes),
            "MIN_SIGNAL_CONFIDENCE": rng.choice(confs),
            "MIN_META_WIN_PROB": rng.choice(meta_probs),
            "RISK_PER_TRADE_PCT": rng.choice(risk),
            "MAX_POSITION_PCT": rng.choice(max_pos),
            "STOP_LOSS_PCT": stop,
            "TAKE_PROFIT_PCT": take,
            "TRAIL_STOP_PCT": rng.choice(trails),
            "ATR_STOP_MULTIPLIER": rng.choice(atr_mults),
            "MIN_ATR_PCT": rng.choice(min_atr),
            "MAX_ATR_PCT": rng.choice(max_atr),
        }
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            configs.append(c)
            seen.add(key)

    return configs[:trials]



def run_backtest(config, offset_days, windows, days, symbols, tag):
    run_id = f"opt-{tag}-{uuid.uuid4().hex[:8]}"
    env = os.environ.copy()
    env.update({k: str(v) for k, v in config.items()})
    env.update({
        "BACKTEST_RUN_ID": run_id,
        "BACKTEST_END_OFFSET_DAYS": str(offset_days),
        "NUM_WINDOWS": str(windows),
        "WINDOW_LENGTH_DAYS": str(days),
        "ITERATIONS_PER_WINDOW": "1",
        "BACKTEST_SYMBOLS": ",".join(symbols),
        "SLEEP_BETWEEN_SYMBOLS": "0",
        "NTFY_TOPIC": "",
    })

    completed = subprocess.run(
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

    metrics = flatten_summary(summary)
    metrics["returncode"] = completed.returncode
    metrics["run_id"] = run_id
    metrics["output_tail"] = completed.stdout[-1200:]
    return metrics



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--train-offset", type=int, default=180)
    parser.add_argument("--validation-offset", type=int, default=0)
    parser.add_argument("--train-windows", type=int, default=4)
    parser.add_argument("--validation-windows", type=int, default=4)
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--symbols", default="AAPL,MSFT,NVDA,SPY")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    configs = make_configs(max(1, args.trials), seed=args.seed)

    rows = []
    print("=" * 70)
    print("STRATEGY OPTIMIZER — TRAINING")
    print("=" * 70)
    print(f"Trials: {len(configs)} | training offset: {args.train_offset} days")
    print(f"Symbols: {', '.join(symbols)}")

    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {json.dumps(config, sort_keys=True)}")
        metrics = run_backtest(
            config,
            args.train_offset,
            args.train_windows,
            args.window_days,
            symbols,
            tag=f"train-{i}",
        )
        row = {**config, **{k: v for k, v in metrics.items() if k != "output_tail"}}
        row["output_tail"] = metrics.get("output_tail", "")
        rows.append(row)
        print(
            f"score={metrics.get('score', 0):.3f} "
            f"return={metrics.get('total_return_pct_mean', 0):+.2f}% "
            f"Sharpe={metrics.get('sharpe_mean', 0):.2f} "
            f"DD={metrics.get('max_drawdown_pct_mean', 0):.2f}%"
        )

    rows.sort(key=lambda r: as_float(r.get("score")), reverse=True)
    top = rows[: max(1, min(args.top_k, len(rows)))]

    fields = sorted({k for r in rows for k in r.keys()})
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 70)
    print("VALIDATING TOP CONFIGURATIONS")
    print("=" * 70)

    validation_rows = []
    for rank, trained in enumerate(top, 1):
        config = {k: trained[k] for k in configs[0].keys()}
        print(f"\nValidation {rank}: score={as_float(trained.get('score')):.3f}")
        metrics = run_backtest(
            config,
            args.validation_offset,
            args.validation_windows,
            args.window_days,
            symbols,
            tag=f"validation-{rank}",
        )
        row = {**config, "training_score": trained.get("score", 0), **{k: v for k, v in metrics.items() if k != "output_tail"}}
        row["output_tail"] = metrics.get("output_tail", "")
        validation_rows.append(row)
        print(
            f"validation score={metrics.get('score', 0):.3f} "
            f"return={metrics.get('total_return_pct_mean', 0):+.2f}% "
            f"Sharpe={metrics.get('sharpe_mean', 0):.2f} "
            f"DD={metrics.get('max_drawdown_pct_mean', 0):.2f}%"
        )

    vfields = sorted({k for r in validation_rows for k in r.keys()})
    with VALIDATION_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=vfields)
        w.writeheader()
        w.writerows(validation_rows)

    # Select on validation score, not training score.
    validation_rows.sort(key=lambda r: as_float(r.get("score")), reverse=True)
    best = validation_rows[0]
    best_config = {k: best[k] for k in configs[0].keys()}
    payload = {
        "selected_on": "validation_score",
        "config": best_config,
        "training_score": as_float(best.get("training_score")),
        "validation_score": as_float(best.get("score")),
        "validation_metrics": {k: best.get(k) for k in (
            "total_return_pct_mean", "max_drawdown_pct_mean", "win_rate_mean",
            "num_trades_mean", "profit_factor_mean", "expectancy_pct_mean",
            "sharpe_mean", "sortino_mean", "calmar_mean",
        )},
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    BEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("BEST VALIDATED CONFIGURATION")
    print("=" * 70)
    print(json.dumps(payload, indent=2))
    print(f"\nSaved: {RESULTS_PATH}")
    print(f"Saved: {VALIDATION_PATH}")
    print(f"Saved: {BEST_PATH}")
    print("\nIMPORTANT: use this configuration only for a new paper-trading test; do not select it based on the training results alone.")


if __name__ == "__main__":
    main()
