"""
train_meta_model.py — Trains a SECOND, separate model on your own trade
history. Reads from and writes to trade_data/ automatically.

Usage:
    python train_meta_model.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

DATA_DIR = "trade_data"
os.makedirs(DATA_DIR, exist_ok=True)

LOG_PATH = os.path.join(DATA_DIR, "backtest_trade_log.csv")
MODEL_OUTPUT_PATH = os.path.join(DATA_DIR, "meta_model.joblib")
MIN_TRADES_REQUIRED = 100


def main():
    try:
        df = pd.read_csv(LOG_PATH)
    except FileNotFoundError:
        print(f"Couldn't find {LOG_PATH}. Run combine_logs.py first (after a backtest) to generate it.")
        sys.exit(1)

    print(f"Loaded {len(df)} logged trades from {LOG_PATH}")

    if len(df) < MIN_TRADES_REQUIRED:
        print(f"\nWARNING: only {len(df)} trades logged. Fewer than {MIN_TRADES_REQUIRED} "
              f"means this meta-model will likely just memorize noise, not find a real pattern. "
              f"Consider running the backtest over more/longer windows first.\n")

    df = df.dropna(subset=["entry_confidence", "entry_trend_change_pct", "entry_atr_pct",
                            "entry_news_sentiment", "pnl_pct"])

    df["won"] = (df["pnl_pct"] > 0).astype(int)

    categorical_cols = [c for c in ["entry_volatility_regime", "entry_trend_regime", "entry_hmm_regime"]
                         if c in df.columns]
    prefixes = {"entry_volatility_regime": "vol", "entry_trend_regime": "trend", "entry_hmm_regime": "hmm"}
    df = pd.get_dummies(df, columns=categorical_cols, prefix=[prefixes[c] for c in categorical_cols])

    # Excluded: text/non-generalizing columns, AND entry_meta_win_prob —
    # that column is a PAST meta-model's own output, so training a new
    # meta-model on it would be circular (learning from its own old guesses
    # instead of the real underlying conditions).
    non_feature_entry_cols = {"entry_direction", "entry_price", "entry_meta_win_prob"}

    feature_cols = [c for c in df.columns if c.startswith("entry_") and c not in non_feature_entry_cols] + \
                   [c for c in df.columns if c.startswith(("vol_", "trend_", "hmm_"))]

    X = df[feature_cols]
    y = df["won"]

    print(f"\nFeatures used: {feature_cols}")
    print(f"Win rate in logged trades: {y.mean() * 100:.1f}%")

    if len(df) < 30 or y.nunique() < 2:
        print("\nNot enough data or not enough variety in outcomes (all wins or all losses) to train on. Stopping.")
        sys.exit(1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n=== Meta-model performance on held-out trades ===")
    print(classification_report(y_test, y_pred, target_names=["lost", "won"]))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))

    importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n=== What the meta-model thinks matters most ===")
    print(importances.to_string())

    joblib.dump({"model": clf, "feature_cols": feature_cols}, MODEL_OUTPUT_PATH)
    print(f"\nSaved trained meta-model to {MODEL_OUTPUT_PATH}")

    baseline_accuracy = max(y_test.mean(), 1 - y_test.mean())
    test_accuracy = (y_pred == y_test).mean()
    print(f"\nBaseline (always guess majority class): {baseline_accuracy * 100:.1f}% accuracy")
    print(f"Meta-model accuracy:                     {test_accuracy * 100:.1f}% accuracy")
    if test_accuracy <= baseline_accuracy + 0.03:
        print("\nHonest read: the meta-model isn't beating a naive guess by much. That's a real, "
              "useful finding — it suggests the logged features don't yet strongly predict which "
              "trades win. Not a failure of this script; a finding about the strategy.")


if __name__ == "__main__":
    main()