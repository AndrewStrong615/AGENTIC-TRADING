import logging
import pandas as pd
try:
    import joblib
except ImportError:
    joblib=None
log=logging.getLogger("trader")
class MetaModelFilter:
    def __init__(self,model_path="trade_data/meta_model.joblib"):
        self.model_path=model_path; self.available=False; self.model=None; self.feature_cols=None
        if joblib is None:return
        try:
            saved=joblib.load(model_path); self.model=saved["model"]; self.feature_cols=saved["feature_cols"]; self.available=True
            log.info(f"Loaded meta-model from {model_path} ({len(self.feature_cols)} features).")
        except FileNotFoundError: log.info(f"No meta-model found at {model_path} — filter disabled.")
        except Exception as e: log.error(f"Failed to load meta-model from {model_path}: {e}")
    def score(self,signal):
        if not self.available:return 0.5
        row={"entry_confidence":signal.confidence,"entry_trend_change_pct":signal.trend_change_pct,"entry_atr_pct":signal.atr_pct,"entry_news_sentiment":signal.news_sentiment,"entry_batch_weight":signal.batch_weight,"entry_online_weight":signal.online_weight,"entry_markov_weight":signal.markov_weight,"entry_meanrev_weight":signal.meanrev_weight,"entry_hmm_regime_prob":signal.hmm_regime_prob}
        names=["market_return_pct","qqq_return_pct","market_volatility_proxy","relative_strength_pct","relative_volume_ratio","volume_acceleration","distance_20_high_pct","distance_20_low_pct","distance_50_high_pct","distance_50_low_pct","atr_change_pct","realized_volatility_pct","volatility_percentile","minutes_since_open","minutes_to_close","intraday_return_pct","distance_from_open_pct","opening_range_breakout","gap_pct","cross_section_rank","breadth_pct_up","sector_relative_return_pct","news_count","news_sentiment_change","news_sentiment_dispersion"]
        for n in names:row["entry_"+n]=getattr(signal,n,0.0)
        row["vol_"+signal.volatility_regime]=1.0; row["trend_"+signal.trend_regime]=1.0; row["hmm_"+signal.hmm_regime]=1.0
        X=pd.DataFrame([{c:row.get(c,0.0) for c in self.feature_cols}],columns=self.feature_cols)
        try:
            p=self.model.predict_proba(X)[0]; classes=list(self.model.classes_); return float(p[classes.index(1)]) if 1 in classes else 0.5
        except Exception as e:log.error(f"Meta-model scoring failed, using neutral 0.5: {e}"); return 0.5