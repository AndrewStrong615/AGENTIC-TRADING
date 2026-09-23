"""Read-only visualization of strategy_optimizer.py's search history.

This is a standalone analysis script. It does NOT import or execute
strategy_optimizer.py, does not change optimizer behavior, and does not
affect promotion gates in any way -- it only reads the CSV/JSON files that
the optimizer already writes to `trade_data/` and renders PNG charts.

Inputs (all optional -- missing/empty files are reported and skipped):
    trade_data/optimizer_history.csv     every config ever tested (append-only)
    trade_data/optimizer_validation.csv  the top candidates from the most
                                          recent optimizer run, evaluated on
                                          holdout periods, plus a re-evaluation
                                          of the currently promoted config
    trade_data/best_strategy_config.json the currently promoted config

Outputs (PNG files written to --output-dir, default trade_data/optimizer_plots/):
    score_over_trials.png       candidate score vs. trial index/time, to spot
                                 search progress or stagnation
    param_distributions.png     tested parameter values, split by
                                 promoted vs. rejected configs

Usage:
    python optimizer_history_viz.py
    python optimizer_history_viz.py --output-dir trade_data/optimizer_plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: never try to open a GUI window
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "trade_data"

DEFAULT_HISTORY_PATH = DATA_DIR / "optimizer_history.csv"
DEFAULT_VALIDATION_PATH = DATA_DIR / "optimizer_validation.csv"
DEFAULT_BEST_CONFIG_PATH = DATA_DIR / "best_strategy_config.json"
DEFAULT_OUTPUT_DIR = DATA_DIR / "optimizer_plots"

# Mirrors strategy_optimizer.py's CONFIG_KEYS (kept as a local copy so this
# script stays standalone and never imports/executes strategy_optimizer.py).
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

# Numeric config keys are shown as jittered value distributions; the one
# categorical key (BAR_TIMEFRAME) is shown as grouped counts instead.
NUMERIC_CONFIG_KEYS = tuple(k for k in CONFIG_KEYS if k != "BAR_TIMEFRAME")
CATEGORICAL_CONFIG_KEYS = ("BAR_TIMEFRAME",)

PROMOTED_COLOR = "#2E7D32"   # green
REJECTED_COLOR = "#B0BEC5"   # neutral gray


def compute_config_id(cfg: dict) -> str:
    """Recompute the same config_id strategy_optimizer.py's config_id() would
    produce for a given config dict, so we can match best_strategy_config.json
    against rows in optimizer_validation.csv without importing that module."""
    payload = {k: cfg.get(k) for k in CONFIG_KEYS}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def load_csv_or_none(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[skip] {label}: file not found at {path}")
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        print(f"[skip] {label}: file at {path} is empty (no header/rows)")
        return None
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[skip] {label}: could not read {path}: {exc}")
        return None
    if df.empty:
        print(f"[skip] {label}: file at {path} has a header but no rows yet")
        return None
    return df


def plot_score_over_trials(history_df: pd.DataFrame, output_dir: Path) -> Path | None:
    if "score" not in history_df.columns:
        print("[skip] score_over_trials: optimizer_history.csv has no 'score' column")
        return None

    df = history_df.copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"]).reset_index(drop=True)
    if df.empty:
        print("[skip] score_over_trials: no rows with a numeric score")
        return None

    # optimizer_history.csv is append-only, so row order is already
    # chronological; use it directly as the trial index.
    trial_index = range(1, len(df) + 1)
    rolling = df["score"].rolling(window=min(10, len(df)), min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.scatter(trial_index, df["score"], s=18, alpha=0.55, color="#1565C0", label="trial score")
    ax.plot(trial_index, rolling, color="#E65100", linewidth=2, label="rolling mean (10 trials)")
    ax.axhline(0.0, color="#9E9E9E", linewidth=1, linestyle="--", alpha=0.7)

    ax.set_xlabel("Trial index (chronological)")
    ax.set_ylabel("Score")
    ax.set_title(f"Optimizer candidate score across {len(df)} tested trials")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out_path = output_dir / "score_over_trials.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[ok]   wrote {out_path}")
    return out_path


def label_promoted_vs_rejected(validation_df: pd.DataFrame, best_config_path: Path) -> pd.DataFrame:
    """Return one row per distinct tested config_id from optimizer_validation.csv,
    labeled 'promoted' or 'rejected'.

    Promotion signal, in order of preference:
      1. config_id matches the config currently stored in best_strategy_config.json
         (the ground truth for "what is promoted right now").
      2. Fallback, only if best_strategy_config.json is missing/unreadable: the
         'is_current' column strategy_optimizer.py stamps on the row where it
         re-evaluates the previously-promoted config against the same holdouts.
    """
    if "config_id" not in validation_df.columns:
        print("[skip] param_distributions: optimizer_validation.csv has no 'config_id' column")
        return pd.DataFrame()

    # One row per distinct candidate config (validation.csv has one row per
    # (config, holdout offset) pair).
    dedup_cols = [c for c in ["config_id", *CONFIG_KEYS] if c in validation_df.columns]
    configs = validation_df.drop_duplicates(subset=["config_id"])[dedup_cols].reset_index(drop=True)

    promoted_config_id = None
    if best_config_path.exists():
        try:
            data = json.loads(best_config_path.read_text(encoding="utf-8"))
            cfg = data.get("config", {})
            if isinstance(cfg, dict) and cfg:
                promoted_config_id = compute_config_id(cfg)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[warn] could not read {best_config_path}: {exc}")

    if promoted_config_id is not None:
        configs["status"] = configs["config_id"].apply(
            lambda cid: "promoted" if cid == promoted_config_id else "rejected"
        )
        if not (configs["status"] == "promoted").any():
            print(
                "[note] param_distributions: none of the validated candidates in "
                "optimizer_validation.csv match best_strategy_config.json's current "
                "config (it may predate this run's candidates); all shown as rejected."
            )
    elif "is_current" in validation_df.columns:
        print(
            "[note] best_strategy_config.json unavailable -- falling back to the "
            "'is_current' column as the promoted/rejected signal."
        )
        current_ids = set(
            validation_df.loc[validation_df["is_current"].fillna(False).astype(bool), "config_id"]
        )
        configs["status"] = configs["config_id"].apply(
            lambda cid: "promoted" if cid in current_ids else "rejected"
        )
    else:
        print(
            "[note] no best_strategy_config.json and no 'is_current' column -- "
            "cannot distinguish promoted from rejected; treating all as rejected."
        )
        configs["status"] = "rejected"

    return configs


def plot_param_distributions(
    validation_df: pd.DataFrame, best_config_path: Path, output_dir: Path
) -> Path | None:
    configs = label_promoted_vs_rejected(validation_df, best_config_path)
    if configs.empty:
        return None

    numeric_keys = [k for k in NUMERIC_CONFIG_KEYS if k in configs.columns]
    categorical_keys = [k for k in CATEGORICAL_CONFIG_KEYS if k in configs.columns]
    plot_keys = numeric_keys + categorical_keys
    if not plot_keys:
        print("[skip] param_distributions: no recognized CONFIG_KEYS columns in optimizer_validation.csv")
        return None

    n_promoted = int((configs["status"] == "promoted").sum())
    n_rejected = int((configs["status"] == "rejected").sum())

    n_cols = 3
    n_rows = -(-len(plot_keys) // n_cols)  # ceil div
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.2 * n_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    rng = pd.Series(range(len(configs)))  # deterministic jitter seed source

    for i, key in enumerate(plot_keys):
        ax = axes[i]
        if key in categorical_keys:
            counts = (
                configs.groupby(["status", key]).size().unstack(fill_value=0)
                if not configs.empty
                else pd.DataFrame()
            )
            categories = sorted(configs[key].dropna().unique().tolist())
            width = 0.35
            positions = range(len(categories))
            promoted_counts = [
                int(configs[(configs[key] == c) & (configs["status"] == "promoted")].shape[0])
                for c in categories
            ]
            rejected_counts = [
                int(configs[(configs[key] == c) & (configs["status"] == "rejected")].shape[0])
                for c in categories
            ]
            ax.bar(
                [p - width / 2 for p in positions], rejected_counts, width=width,
                color=REJECTED_COLOR, label="rejected",
            )
            ax.bar(
                [p + width / 2 for p in positions], promoted_counts, width=width,
                color=PROMOTED_COLOR, label="promoted",
            )
            ax.set_xticks(list(positions))
            ax.set_xticklabels(categories, rotation=0)
            ax.set_ylabel("count")
        else:
            values = pd.to_numeric(configs[key], errors="coerce")
            for status, color, y in (("rejected", REJECTED_COLOR, 0.0), ("promoted", PROMOTED_COLOR, 1.0)):
                mask = configs["status"] == status
                x = values[mask]
                if x.empty:
                    continue
                jitter = (rng[mask] % 7 - 3) / 3.0 * 0.08  # small deterministic vertical jitter
                ax.scatter(x, [y] * len(x) + jitter, color=color, s=42, alpha=0.85, edgecolor="white", linewidth=0.5)
            ax.set_yticks([0.0, 1.0])
            ax.set_yticklabels(["rejected", "promoted"])
            ax.set_ylim(-0.5, 1.5)
        ax.set_title(key, fontsize=10)
        ax.grid(True, axis="x" if key not in categorical_keys else "y", alpha=0.25)

    for j in range(len(plot_keys), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Tested parameter values: promoted (n={n_promoted}) vs. rejected (n={n_rejected}) "
        "-- from the most recent optimizer run's validated candidates",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = output_dir / "param_distributions.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[ok]   wrote {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot strategy_optimizer.py search history and promoted/rejected "
        "parameter distributions. Read-only: never modifies optimizer data, "
        "behavior, or promotion gates."
    )
    parser.add_argument(
        "--history-path", type=Path, default=DEFAULT_HISTORY_PATH,
        help=f"Path to optimizer_history.csv (default: {DEFAULT_HISTORY_PATH})",
    )
    parser.add_argument(
        "--validation-path", type=Path, default=DEFAULT_VALIDATION_PATH,
        help=f"Path to optimizer_validation.csv (default: {DEFAULT_VALIDATION_PATH})",
    )
    parser.add_argument(
        "--best-config-path", type=Path, default=DEFAULT_BEST_CONFIG_PATH,
        help=f"Path to best_strategy_config.json (default: {DEFAULT_BEST_CONFIG_PATH})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write PNG charts into (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    history_df = load_csv_or_none(args.history_path, "score_over_trials")
    validation_df = load_csv_or_none(args.validation_path, "param_distributions")

    written = []
    if history_df is not None:
        result = plot_score_over_trials(history_df, args.output_dir)
        if result:
            written.append(result)

    if validation_df is not None:
        result = plot_param_distributions(validation_df, args.best_config_path, args.output_dir)
        if result:
            written.append(result)

    if not written:
        print(
            "\nNo charts were produced. Run strategy_optimizer.py first to generate "
            f"{args.history_path.name} / {args.validation_path.name}, then re-run this script."
        )
    else:
        print(f"\nWrote {len(written)} chart(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
