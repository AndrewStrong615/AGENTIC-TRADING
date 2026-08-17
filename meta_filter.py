"""
meta_filter.py — Loads a trained meta-model (from train_meta_model.py) and
scores live TradeSignals against it.

Defaults to trade_data/meta_model.joblib — the same folder everything else
now reads/writes to automatically.
"""

import logging

import pandas as pd

log = logging.getLogger("trader")

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False


class MetaModelFilter:
    def __init__(self, model_path="trade_data/meta_model.joblib"):
        self.model_path = model_path
        self.available = False
        self.model = None
        self.feature_cols = None

        if not _JOBLIB_AVAILABLE:
            log.info("joblib not installed — meta-model filter disabled (pip install joblib).")
            return
        try:
            saved = joblib.load(model_path)
            self.model = saved["model"]
            self.feature_cols = saved["feature_cols"]
            self.available = True
            log.info(f"Loaded meta-model from {model_path} ({len(self.feature_cols)} features).")
        except FileNotFoundError:
            log.info(f"No meta-model found at {model_path} — filter disabled "
                      f"(run train_meta_model.py first, once you have enough logged trades).")
        except Exception as e:
            log.error(f"Failed to load meta-model from {model_path}: {e}")

    def score(self, signal) -> float:
        if not self.available:
            return 0.5

        row = {
            "entry_confidence": signal.confidence,
            "entry_trend_change_pct": signal.trend_change_pct,
            "entry_atr_pct": signal.atr_pct,
            "entry_news_sentiment": signal.news_sentiment,
            "entry_batch_weight": signal.batch_weight,
            "entry_online_weight": signal.online_weight,
            "entry_markov_weight": signal.markov_weight,
            "entry_meanrev_weight": signal.meanrev_weight,
            "entry_hmm_regime_prob": signal.hmm_regime_prob,
        }
        row[f"vol_{signal.volatility_regime}"] = 1
        row[f"trend_{signal.trend_regime}"] = 1
        row[f"hmm_{signal.hmm_regime}"] = 1

        X = pd.DataFrame([{c: row.get(c, 0) for c in self.feature_cols}])[self.feature_cols]

        try:
            proba = self.model.predict_proba(X)[0]
            classes = list(self.model.classes_)
            return float(proba[classes.index(1)]) if 1 in classes else 0.5
        except Exception as e:
            log.error(f"Meta-model scoring failed, using neutral 0.5: {e}")
            return 0.5