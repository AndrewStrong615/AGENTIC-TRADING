"""Conservative meta-model trainer.

Collects every completed backtest trade log under trade_data/backtests/, keeps a
single deduplicated training dataset, trains on older trades, validates on newer
unseen trades, and only promotes a new model when it clears conservative gates.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, roc_auc_score, brier_score_loss

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "trade_data"
RUNS_DIR = DATA_DIR / "backtests"
DATASET_PATH = DATA_DIR / "meta_training_data.csv"
MODEL_PATH = DATA_DIR / "meta_model.joblib"
MODEL_INFO_PATH = DATA_DIR / "meta_model_info.json"

MIN_TOTAL_TRADES = 200
MIN_NEW_TRADES_TO_RETRAIN = 100
MIN_VALIDATION_TRADES = 50
MIN_AUC_IMPROVEMENT = 0.02
MAX_BRIER_WORSENING = 0.01

FEATURE_EXCLUDE = {
    "entry_direction", "entry_price", "entry_meta_win_prob", "pnl_pct", "won",
}


def collect_logs() -> pd.DataFrame:
    frames = []
    if RUNS_DIR.exists():
        paths = sorted(RUNS_DIR.glob("**/backtest_trade_log_*.csv"))
        for p in paths:
            try:
                df = pd.read_csv(p)
                if not df.empty:
                    df["_source_file"] = str(p.relative_to(BASE_DIR))
                    frames.append(df)
            except Exception as e:
                print(f"[meta] skipped {p}: {e}")

    # Also include the aggregate file if it exists.
    aggregate = DATA_DIR / "backtest_trade_log.csv"
    if aggregate.exists():
        try:
            df = pd.read_csv(aggregate)
            if not df.empty:
                df["_source_file"] = str(aggregate.relative_to(BASE_DIR))
                frames.append(df)
        except Exception as e:
            print(f"[meta] skipped {aggregate}: {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True, sort=False)
    required = ["symbol", "entry_price", "exit_price", "pnl_pct"]
    df = df.dropna(subset=[c for c in required if c in df.columns])

    # Backtests may overlap or be copied into the aggregate file. Deduplicate by
    # the economic identity of the trade, not by source path.
    keys = [c for c in ["exit_timestamp", "symbol", "entry_price", "exit_price", "pnl_pct", "entry_direction"] if c in df.columns]
    if keys:
        df = df.drop_duplicates(subset=keys, keep="last")

    return df.reset_index(drop=True)


def prepare(df: pd.DataFrame):
    numeric = [
        "entry_confidence", "entry_trend_change_pct", "entry_atr_pct",
        "entry_news_sentiment", "entry_batch_weight", "entry_online_weight",
        "entry_markov_weight", "entry_meanrev_weight", "entry_hmm_regime_prob",
    ]
    for c in numeric:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=numeric + ["pnl_pct"]).copy()
    df["won"] = (pd.to_numeric(df["pnl_pct"], errors="coerce") > 0).astype(int)

    categorical = [c for c in ["entry_volatility_regime", "entry_trend_regime", "entry_hmm_regime"] if c in df.columns]
    prefixes = {"entry_volatility_regime": "vol", "entry_trend_regime": "trend", "entry_hmm_regime": "hmm"}
    if categorical:
        df = pd.get_dummies(df, columns=categorical, prefix=[prefixes[c] for c in categorical])

    feature_cols = [
        c for c in df.columns
        if c.startswith("entry_") and c not in FEATURE_EXCLUDE and c != "entry_direction"
    ] + [c for c in df.columns if c.startswith(("vol_", "trend_", "hmm_"))]
    feature_cols = list(dict.fromkeys(feature_cols))

    return df, feature_cols


def evaluate(model, X, y):
    proba = model.predict_proba(X)[:, list(model.classes_).index(1)] if 1 in model.classes_ else np.full(len(X), 0.5)
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y, proba) if y.nunique() == 2 else 0.5
    return {
        "auc": float(auc),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "brier": float(brier_score_loss(y, proba)),
    }


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_df = collect_logs()
    if all_df.empty:
        print("[meta] No backtest trade logs found.")
        sys.exit(1)

    # Persist the complete historical training set.
    all_df.to_csv(DATASET_PATH, index=False)
    print(f"[meta] Collected {len(all_df)} unique historical trades -> {DATASET_PATH}")

    df, feature_cols = prepare(all_df)
    if len(df) < MIN_TOTAL_TRADES or df["won"].nunique() < 2:
        print(f"[meta] Not promoting: need at least {MIN_TOTAL_TRADES} labeled trades with wins and losses.")
        sys.exit(0)

    if "exit_timestamp" in df.columns:
        timestamps = pd.to_datetime(df["exit_timestamp"], errors="coerce", utc=True)
        df = df.assign(_ts=timestamps).sort_values("_ts")
    else:
        df = df.reset_index(drop=True)

    split = int(len(df) * 0.75)
    train = df.iloc[:split].copy()
    valid = df.iloc[split:].copy()
    if len(valid) < MIN_VALIDATION_TRADES or train["won"].nunique() < 2 or valid["won"].nunique() < 2:
        print("[meta] Not promoting: insufficient time-ordered validation data.")
        sys.exit(0)

    X_train = train[feature_cols].fillna(0)
    y_train = train["won"]
    X_valid = valid[feature_cols].fillna(0)
    y_valid = valid["won"]

    candidate = RandomForestClassifier(
        n_estimators=400,
        max_depth=5,
        min_samples_leaf=6,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    candidate.fit(X_train, y_train)
    candidate_metrics = evaluate(candidate, X_valid, y_valid)

    existing_metrics = None
    existing_model = None
    if MODEL_PATH.exists():
        try:
            saved = joblib.load(MODEL_PATH)
            existing_model = saved.get("model")
            old_features = saved.get("feature_cols", feature_cols)
            if existing_model is not None and list(old_features) == list(feature_cols):
                existing_metrics = evaluate(existing_model, X_valid, y_valid)
        except Exception as e:
            print(f"[meta] Existing model could not be evaluated: {e}")

    baseline = max(float(y_valid.mean()), 1.0 - float(y_valid.mean()))
    print("\n[meta] Time-ordered validation")
    print(f"  train trades: {len(train)}")
    print(f"  validation trades: {len(valid)}")
    print(f"  candidate AUC: {candidate_metrics['auc']:.3f}")
    print(f"  candidate accuracy: {candidate_metrics['accuracy']:.3f}")
    print(f"  candidate Brier: {candidate_metrics['brier']:.4f}")
    print(f"  validation majority baseline: {baseline:.3f}")
    if existing_metrics:
        print(f"  current AUC: {existing_metrics['auc']:.3f}")
        print(f"  current Brier: {existing_metrics['brier']:.4f}")

    # Conservative promotion: the candidate must clearly beat the old model,
    # or there must be no old model. Small noisy improvements are ignored.
    promote = existing_model is None
    reason = "no existing model"
    if existing_model is not None and existing_metrics is not None:
        auc_gain = candidate_metrics["auc"] - existing_metrics["auc"]
        brier_change = candidate_metrics["brier"] - existing_metrics["brier"]
        promote = auc_gain >= MIN_AUC_IMPROVEMENT and brier_change <= MAX_BRIER_WORSENING
        reason = f"AUC gain={auc_gain:+.3f}, Brier change={brier_change:+.4f}"

    if promote:
        joblib.dump({"model": candidate, "feature_cols": feature_cols}, MODEL_PATH)
        info = {
            "version": int(pd.Timestamp.utcnow().timestamp()),
            "trained_at": pd.Timestamp.utcnow().isoformat(),
            "training_trades": int(len(train)),
            "validation_trades": int(len(valid)),
            "candidate_metrics": candidate_metrics,
            "previous_metrics": existing_metrics,
            "promotion_reason": reason,
            "feature_cols": feature_cols,
        }
        MODEL_INFO_PATH.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
        print(f"[meta] PROMOTED new meta-model -> {MODEL_PATH}")
        print(f"[meta] {reason}")
    else:
        print(f"[meta] KEPT existing meta-model. {reason}")

    print("\n" + "=" * 60)
    print(classification_report(y_valid, candidate.predict(X_valid), target_names=["lost", "won"], zero_division=0))


if __name__ == "__main__":
    main()
