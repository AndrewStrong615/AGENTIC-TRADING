"""
Tradingcore.py — The three-layer architecture, shared by live trading and
backtesting so both run the exact same logic.

LAYER 1 — MarketDataLayer (deterministic, + HMM regime detection)
LAYER 2 — SignalLayer (non-deterministic / ML, adaptive 4-expert ensemble
    + optional meta-labeling filter)
LAYER 3 — RiskGate (deterministic, unbypassable, Kelly + vol-targeted sizing)

All trade-log output automatically lands in a `trade_data/` folder next to
this file — RiskGate creates that folder itself the first time it needs it,
so no manual file-moving is required, regardless of what path a caller
passes in (as long as it's under trade_data/.
"""

import time
import threading
import logging
import csv
import os
import json
import uuid
from dataclasses import dataclass, replace
from collections import deque
from datetime import datetime, timezone, timedelta, time as dt_time
from zoneinfo import ZoneInfo
from notifications import notify_order_placed
from typing import Optional, Dict

# Conservative shared strategy configuration. Explicit environment variables
# (including optimizer trials) take precedence over the saved config.
try:
    from strategy_config import load_best_config
    _BEST_CONFIG = load_best_config()
except Exception:
    _BEST_CONFIG = {}


def _cfg_float(name, default):
    try:
        return float(os.environ.get(name, _BEST_CONFIG.get(name, default)))
    except (TypeError, ValueError):
        return float(default)


def _cfg_int(name, default):
    try:
        return int(float(os.environ.get(name, _BEST_CONFIG.get(name, default))))
    except (TypeError, ValueError):
        return int(default)


import numpy as np
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    from hmmlearn.hmm import GaussianHMM
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False

log = logging.getLogger("trader")
logging.getLogger("hmmlearn").setLevel(logging.CRITICAL)
logging.getLogger("hmmlearn.base").setLevel(logging.CRITICAL)
_hmm_warning_logged = False


# =============================================================================
# Shared data structures
# =============================================================================
@dataclass
class TradeSignal:
    symbol: str
    direction: str
    confidence: float
    trend_change_pct: float
    volatility_regime: str
    trend_regime: str
    hmm_regime: str
    hmm_regime_prob: float
    atr_pct: float
    news_sentiment: float
    batch_confidence: float
    online_confidence: float
    markov_confidence: float
    meanrev_confidence: float
    batch_weight: float
    online_weight: float
    markov_weight: float
    meanrev_weight: float
    timestamp: datetime
    meta_win_prob: float = 0.5
    market_return_pct: float = 0.0
    qqq_return_pct: float = 0.0
    market_volatility_proxy: float = 0.0
    relative_strength_pct: float = 0.0
    relative_volume_ratio: float = 1.0
    volume_acceleration: float = 0.0
    distance_20_high_pct: float = 0.0
    distance_20_low_pct: float = 0.0
    distance_50_high_pct: float = 0.0
    distance_50_low_pct: float = 0.0
    atr_change_pct: float = 0.0
    realized_volatility_pct: float = 0.0
    volatility_percentile: float = 0.5
    minutes_since_open: float = 0.0
    minutes_to_close: float = 390.0
    intraday_return_pct: float = 0.0
    distance_from_open_pct: float = 0.0
    opening_range_breakout: float = 0.0
    gap_pct: float = 0.0
    cross_section_rank: float = 0.5
    breadth_pct_up: float = 0.5
    sector_relative_return_pct: float = 0.0
    news_count: float = 0.0
    news_sentiment_change: float = 0.0
    news_sentiment_dispersion: float = 0.0


@dataclass
class GateDecision:
    approved: bool
    action: Optional[str]
    qty: Optional[int]
    reason: str


# =============================================================================
# LAYER 1 — MarketDataLayer
# =============================================================================
class MarketDataLayer:
    CONTEXT_SYMBOLS = ("SPY", "QQQ")
    BAR_SECONDS = 60
    MAX_BARS_KEPT = 200
    ATR_PERIOD = 14
    ATR_HISTORY_FOR_REGIME = 100
    NEWS_REFRESH_SECONDS = 300
    NEWS_LOOKBACK_HOURS = 12
    NEWS_MAX_ARTICLES = 15
    HISTORY_BOOTSTRAP_DAYS = 5

    HMM_N_STATES = 3
    HMM_REFIT_EVERY_N_BARS = 20
    HMM_MIN_BARS = 50
    HMM_RETURN_SCALE = 100.0

    def __init__(self, symbols, api_client=None):
        self.symbols = list(symbols)
        self.context_symbols = [s for s in self.CONTEXT_SYMBOLS if s not in self.symbols]
        self.all_symbols = list(dict.fromkeys(self.symbols + self.context_symbols))
        self.api = api_client
        self.bars = {s: deque(maxlen=self.MAX_BARS_KEPT) for s in self.all_symbols}
        self.current_bar: Dict[str, Optional[dict]] = {s: None for s in self.all_symbols}
        self._vader = SentimentIntensityAnalyzer()
        self.news_sentiment = {s: 0.0 for s in self.all_symbols}
        self.news_count = {s: 0 for s in self.all_symbols}
        self.news_dispersion = {s: 0.0 for s in self.all_symbols}
        self.news_previous_sentiment = {s: 0.0 for s in self.all_symbols}
        self._news_lock = threading.Lock()
        self._bars_lock = threading.Lock()

        self._hmm_models = {}
        self._hmm_bars_since_fit = {}
        self._hmm_state_labels = {}
        self._hmm_last_result = {s: ("unavailable", 0.0) for s in self.all_symbols}

        # Increments each time a bar actually CLOSES.  Keep counters for every
        # symbol in the layer, including context symbols, because the backtest
        # can seed the whole universe while generating a signal for one symbol.
        self.bar_close_count = {s: 0 for s in self.all_symbols}

    def _ensure_symbol(self, symbol):
        """Ensure all per-symbol stores exist before a data operation."""
        if symbol not in self.bars:
            self.bars[symbol] = deque(maxlen=self.MAX_BARS_KEPT)
            self.current_bar[symbol] = None
            self.news_sentiment[symbol] = 0.0
            self.news_count[symbol] = 0
            self.news_dispersion[symbol] = 0.0
            self.news_previous_sentiment[symbol] = 0.0
            self._hmm_last_result[symbol] = ("unavailable", 0.0)
            self.bar_close_count[symbol] = 0

    def update_bar(self, symbol, price, size, ts):
        with self._bars_lock:
            self._ensure_symbol(symbol)
            bucket = int(ts // self.BAR_SECONDS) * self.BAR_SECONDS
            cb = self.current_bar[symbol]
            if cb is None or cb["t"] != bucket:
                if cb is not None:
                    self.bars[symbol].append(cb)
                    self.bar_close_count[symbol] = self.bar_close_count.get(symbol, 0) + 1
                self.current_bar[symbol] = {"t": bucket, "o": price, "h": price,
                                             "l": price, "c": price, "v": size}
            else:
                cb["h"] = max(cb["h"], price)
                cb["l"] = min(cb["l"], price)
                cb["c"] = price
                cb["v"] += size

    def seed_bar(self, symbol, bar_dict):
        # Used by backtesting, where every seeded bar is already "closed" —
        # counts as a bar-close event, same as a live rollover.
        with self._bars_lock:
            self._ensure_symbol(symbol)
            clean = {
                "t": float(bar_dict["t"]),
                "o": float(bar_dict["o"]),
                "h": float(bar_dict["h"]),
                "l": float(bar_dict["l"]),
                "c": float(bar_dict["c"]),
                "v": float(bar_dict.get("v", 0.0)),
            }
            self.bars[symbol].append(clean)
            self.bar_close_count[symbol] = self.bar_close_count.get(symbol, 0) + 1

    def get_bars(self, symbol, include_partial=False):
        """By default returns only COMPLETED bars (unchanged behavior — used
        for ATR/regime/HMM and for training data). Pass include_partial=True
        to also get the still-forming current bar appended as the last
        entry, reflecting the very latest tick — this is what lets live
        predictions actually update between bar closes instead of repeating
        a stale value until the next bar completes."""
        with self._bars_lock:
            self._ensure_symbol(symbol)
            bars = list(self.bars[symbol])
            if include_partial and self.current_bar[symbol] is not None:
                bars = bars + [dict(self.current_bar[symbol])]
            return bars

    def get_bar_close_count(self, symbol):
        with self._bars_lock:
            return self.bar_close_count.get(symbol, 0)

    def bootstrap_history(self, days=None):
        if self.api is None:
            return
        days = days or self.HISTORY_BOOTSTRAP_DAYS
        end = datetime.now(timezone.utc) - timedelta(minutes=16)
        start = end - timedelta(days=days)
        for symbol in self.all_symbols:
            try:
                df = self.api.get_bars(symbol, "1Min", start.strftime("%Y-%m-%d"),
                                        end.strftime("%Y-%m-%d"), adjustment="raw", feed="iex").df
            except Exception as e:
                log.error(f"[{symbol}] Bootstrap history fetch failed: {e}")
                continue
            for idx, row in df.iterrows():
                self.seed_bar(symbol, {"t": idx.timestamp(), "o": float(row["open"]),
                                        "h": float(row["high"]), "l": float(row["low"]),
                                        "c": float(row["close"]), "v": float(row["volume"])})
            log.info(f"[{symbol}] Bootstrapped {len(df)} historical bars.")

    @staticmethod
    def _atr_series(bar_list, period):
        if len(bar_list) < 2:
            return np.array([])
        highs = np.array([b["h"] for b in bar_list], dtype=float)
        lows = np.array([b["l"] for b in bar_list], dtype=float)
        closes = np.array([b["c"] for b in bar_list], dtype=float)
        prev_closes = np.roll(closes, 1)
        prev_closes[0] = closes[0]
        tr = np.maximum.reduce([highs - lows, np.abs(highs - prev_closes), np.abs(lows - prev_closes)])
        n = len(tr)
        atr = np.zeros(n)
        for i in range(n):
            lo = max(0, i - period + 1)
            atr[i] = tr[lo:i + 1].mean()
        return atr

    def current_atr_and_regime(self, symbol):
        bar_list = self.get_bars(symbol)
        if len(bar_list) < self.ATR_PERIOD:
            return None, "normal"
        atr_series = self._atr_series(bar_list, self.ATR_PERIOD)
        current_price = bar_list[-1]["c"]
        current_atr = atr_series[-1]
        atr_pct = (current_atr / current_price) * 100 if current_price else 0.0

        history = atr_series[-self.ATR_HISTORY_FOR_REGIME:]
        if len(history) < 20:
            regime = "normal"
        else:
            p25, p75 = np.percentile(history, [25, 75])
            regime = "high" if current_atr >= p75 else "low" if current_atr <= p25 else "normal"
        return atr_pct, regime

    @staticmethod
    def trend_regime(bar_list, atr_pct):
        if len(bar_list) < 20 or atr_pct is None:
            return "choppy"
        closes = np.array([b["c"] for b in bar_list], dtype=float)
        sma_fast = closes[-5:].mean()
        sma_slow = closes[-20:].mean()
        spread_pct = ((sma_fast - sma_slow) / sma_slow) * 100 if sma_slow else 0
        threshold = max(atr_pct * 0.5, 0.05)
        if spread_pct > threshold:
            return "trending_up"
        if spread_pct < -threshold:
            return "trending_down"
        return "choppy"

    def current_hmm_regime(self, symbol):
        global _hmm_warning_logged
        if not _HMM_AVAILABLE:
            if not _hmm_warning_logged:
                log.warning("hmmlearn not installed — HMM regime detection disabled. Run: pip install hmmlearn")
                _hmm_warning_logged = True
            return "unavailable", 0.0

        bar_list = self.get_bars(symbol)
        if len(bar_list) < self.HMM_MIN_BARS:
            return "unavailable", 0.0

        self._hmm_bars_since_fit.setdefault(symbol, self.HMM_REFIT_EVERY_N_BARS)
        needs_fit = (symbol not in self._hmm_models or
                     self._hmm_bars_since_fit[symbol] >= self.HMM_REFIT_EVERY_N_BARS)

        closes = np.array([b["c"] for b in bar_list], dtype=float)
        returns = (np.diff(closes) / closes[:-1]) * self.HMM_RETURN_SCALE
        returns = returns.reshape(-1, 1)

        if needs_fit:
            try:
                model = GaussianHMM(n_components=self.HMM_N_STATES, covariance_type="diag",
                                     n_iter=100, tol=1e-3, min_covar=1e-3, random_state=42)
                model.fit(returns)
                state_means = model.means_.flatten()
                order = np.argsort(state_means)
                labels = {}
                names = ["bear", "neutral", "bull"] if self.HMM_N_STATES == 3 else \
                        [f"state_{i}" for i in range(self.HMM_N_STATES)]
                for rank, state_idx in enumerate(order):
                    labels[state_idx] = names[rank] if rank < len(names) else f"state_{state_idx}"
                self._hmm_models[symbol] = model
                self._hmm_state_labels[symbol] = labels
                self._hmm_bars_since_fit[symbol] = 0
            except Exception as e:
                log.error(f"[{symbol}] HMM fit failed: {e}")
                self._hmm_bars_since_fit[symbol] = self._hmm_bars_since_fit.get(symbol, 0) + 1
                return self._hmm_last_result[symbol]

        self._hmm_bars_since_fit[symbol] = self._hmm_bars_since_fit.get(symbol, 0) + 1

        model = self._hmm_models.get(symbol)
        if model is None:
            return "unavailable", 0.0
        try:
            state_probs = model.predict_proba(returns)[-1]
            state_idx = int(np.argmax(state_probs))
            label = self._hmm_state_labels[symbol].get(state_idx, "unavailable")
            prob = float(state_probs[state_idx])
            self._hmm_last_result[symbol] = (label, prob)
            return label, prob
        except Exception as e:
            log.error(f"[{symbol}] HMM predict failed: {e}")
            return self._hmm_last_result[symbol]

    def refresh_news_sentiment(self):
        if self.api is None:
            return
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=self.NEWS_LOOKBACK_HOURS)
        for symbol in self.all_symbols:
            try:
                articles=self.api.get_news(symbol,start=start.isoformat(),end=end.isoformat(),limit=self.NEWS_MAX_ARTICLES)
            except Exception as e:
                log.error(f"[{symbol}] News fetch failed: {e}")
                continue
            scores=[]
            for a in (articles or []):
                text=f"{getattr(a,'headline','') or ''}. {getattr(a,'summary','') or ''}".strip()
                if text: scores.append(self._vader.polarity_scores(text)["compound"])
            avg=float(np.mean(scores)) if scores else 0.0
            disp=float(np.std(scores)) if len(scores)>1 else 0.0
            with self._news_lock:
                self.news_previous_sentiment[symbol]=self.news_sentiment.get(symbol,0.0)
                self.news_sentiment[symbol]=avg
                self.news_count[symbol]=len(scores)
                self.news_dispersion[symbol]=disp

    def set_news_sentiment(self, symbol, value):
        with self._news_lock:
            self.news_previous_sentiment[symbol]=self.news_sentiment.get(symbol,0.0)
            self.news_sentiment[symbol]=float(value)

    def get_news_sentiment(self, symbol):
        with self._news_lock:
            return float(self.news_sentiment.get(symbol,0.0))

    def get_news_features(self, symbol):
        with self._news_lock:
            s=float(self.news_sentiment.get(symbol,0.0)); p=float(self.news_previous_sentiment.get(symbol,0.0))
            return {"news_count":float(self.news_count.get(symbol,0)),"news_sentiment_change":s-p,"news_sentiment_dispersion":float(self.news_dispersion.get(symbol,0.0))}

    def news_refresher_loop(self):
        while True:
            self.refresh_news_sentiment()
            time.sleep(self.NEWS_REFRESH_SECONDS)


# =============================================================================
# LAYER 2 — SignalLayer
# =============================================================================
class SignalLayer:
    MIN_BARS_FOR_MODEL = 40
    RETRAIN_EVERY_N_BARS = _cfg_int("RETRAIN_EVERY_N_BARS", 5)
    WEIGHT_LEARNING_RATE = _cfg_float("WEIGHT_LEARNING_RATE", 0.15)

    MARKOV_STATE_WINDOW = 100
    MARKOV_FLAT_THRESHOLD = 0.0005

    MEANREV_LOOKBACK = 20
    MEANREV_Z_SCALE = 1.0

    EXPERTS = ["batch", "online", "markov", "meanrev"]

    def __init__(self, data_layer: MarketDataLayer, retrain_every_n_bars=None, random_state=42,
                 meta_filter=None, weights_path=None):
        self.data = data_layer
        self.random_state = random_state
        self.retrain_every_n_bars = retrain_every_n_bars or self.RETRAIN_EVERY_N_BARS
        self.meta_filter = meta_filter

        self.batch_models = {}
        self.online_models = {}
        self.bars_since_retrain = {}

        # weights_path: persists WHICH EXPERT TO TRUST across restarts. This
        # is genuinely part of price prediction (it directly shapes the
        # blended p_up), unlike the meta_filter above, which only adjusts
        # confidence afterward. None = don't persist (used in backtesting,
        # where each iteration/window intentionally starts blank so the
        # random-seed-averaging methodology stays statistically valid).
        self.weights_path = weights_path
        self._weights_lock = threading.Lock()
        self.weights = self._load_weights() if weights_path else {}
        self.pending = {}

    def _load_weights(self):
        if not self.weights_path or not os.path.exists(self.weights_path):
            return {}
        try:
            with open(self.weights_path, "r") as f:
                loaded = json.load(f)
            log.info(f"SignalLayer: loaded persisted expert weights from {self.weights_path} "
                      f"for {len(loaded)} symbol(s).")
            return loaded
        except Exception as e:
            log.error(f"SignalLayer: failed to load {self.weights_path}, starting uniform: {e}")
            return {}

    def save_weights(self):
        """Persists current per-symbol expert-trust weights to disk. Read-
        modify-write so multiple SignalLayer instances (e.g. one per symbol
        in some setups) don't clobber each other's entries."""
        if not self.weights_path:
            return
        with self._weights_lock:
            existing = {}
            if os.path.exists(self.weights_path):
                try:
                    with open(self.weights_path, "r") as f:
                        existing = json.load(f)
                except Exception:
                    existing = {}
            existing.update(self.weights)
            parent = os.path.dirname(self.weights_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.weights_path, "w") as f:
                json.dump(existing, f, indent=2)
            log.info(f"SignalLayer: saved expert weights for {len(self.weights)} symbol(s) "
                      f"to {self.weights_path}.")

    FEATURE_NAMES = [
        "sma_spread_pct","ema_distance_pct","rsi_centered","momentum_pct","rolling_volatility_pct","atr_pct",
        "market_return_pct","qqq_return_pct","market_volatility_proxy","relative_strength_pct","relative_volume_ratio",
        "volume_acceleration","distance_20_high_pct","distance_20_low_pct","distance_50_high_pct","distance_50_low_pct",
        "atr_change_pct","realized_volatility_pct","volatility_percentile","minutes_since_open","minutes_to_close",
        "intraday_return_pct","distance_from_open_pct","opening_range_breakout","gap_pct","cross_section_rank",
        "breadth_pct_up","sector_relative_return_pct","news_count","news_sentiment_change","news_sentiment_dispersion",
    ]
    SECTOR_GROUPS={
        "technology":{"AAPL","MSFT","NVDA","AVGO","AMD","MU","CSCO","ORCL"},
        "consumer":{"AMZN","TSLA","WMT","COST","HD","NFLX"},"financial":{"JPM","V","MA","BAC"},
        "health":{"JNJ","LLY","ABBV"},"energy":{"XOM"},"industrial":{"CAT"},"communication":{"GOOGL","META"},
    }
    @staticmethod
    def _pct(a,b):
        try:return float((a/b-1)*100) if b else 0.0
        except:return 0.0
    @staticmethod
    def _return(c,lookback=10):
        return SignalLayer._pct(c[-1],c[-lookback-1]) if len(c)>lookback else 0.0
    @staticmethod
    def _session_context(bl):
        if not bl:return {"minutes_since_open":0.0,"minutes_to_close":390.0,"intraday_return_pct":0.0,"distance_from_open_pct":0.0,"opening_range_breakout":0.0,"gap_pct":0.0}
        tz=ZoneInfo("America/New_York"); last=datetime.fromtimestamp(float(bl[-1]["t"]),tz=timezone.utc).astimezone(tz); day=last.date(); op=datetime.combine(day,dt_time(9,30),tzinfo=tz); cl=datetime.combine(day,dt_time(16,0),tzinfo=tz)
        today=[]; prev=[]
        for b in bl:
            t=datetime.fromtimestamp(float(b["t"]),tz=timezone.utc).astimezone(tz)
            if t.date()==day:today.append(b)
            elif t.date()<day:prev.append(b)
        day_open=float(today[0]["o"]) if today else float(bl[-1]["o"]); price=float(bl[-1]["c"]); intraday=SignalLayer._pct(price,day_open)
        opening=[b for b in today if 0<=((datetime.fromtimestamp(float(b["t"]),tz=timezone.utc).astimezone(tz)-op).total_seconds()/60)<15]
        if opening:
            hi=max(float(b["h"]) for b in opening); lo=min(float(b["l"]) for b in opening); orb=1.0 if price>hi else (-1.0 if price<lo else 0.0)
        else:orb=0.0
        gap=SignalLayer._pct(day_open,float(prev[-1]["c"])) if today and prev else 0.0
        return {"minutes_since_open":(last-op).total_seconds()/60.0,"minutes_to_close":(cl-last).total_seconds()/60.0,"intraday_return_pct":intraday,"distance_from_open_pct":intraday,"opening_range_breakout":orb,"gap_pct":gap}
    def _market_features(self, symbol, target_bars):
        """Build market/context features without ever assuming a symbol exists.

        The backtester can run with a full universe while some symbols have no
        bars for a particular timestamp.  Missing context is therefore ignored
        rather than turning into a KeyError or contaminating the feature vector.
        """
        target = np.array([b["c"] for b in target_bars], dtype=float)
        market = 0.0
        qqq = 0.0
        mv = 0.0

        spy = self.data.get_bars("SPY", include_partial=True)
        q = self.data.get_bars("QQQ", include_partial=True)

        if len(spy) > 10:
            market = self._return(np.array([b["c"] for b in spy], dtype=float))
        if len(q) > 10:
            qqq = self._return(np.array([b["c"] for b in q], dtype=float))
        if len(spy) > 20:
            c = np.array([b["c"] for b in spy[-50:]], dtype=float)
            if len(c) > 1:
                mv = float(np.std(np.diff(c) / np.where(c[:-1] == 0, 1.0, c[:-1])) * 100)

        # Only include symbols for which enough bars are actually available.
        universe_returns = []
        for ss in self.data.symbols:
            bs = self.data.get_bars(ss, include_partial=True)
            if len(bs) > 10:
                universe_returns.append(
                    self._return(np.array([b["c"] for b in bs], dtype=float))
                )

        if abs(market) < 1e-12 and universe_returns:
            market = float(np.mean(universe_returns))

        tr = self._return(target)
        rank = (
            sum(r < tr for r in universe_returns) / max(len(universe_returns) - 1, 1)
            if universe_returns else 0.5
        )
        breadth = (
            sum(r > 0 for r in universe_returns) / len(universe_returns)
            if universe_returns else 0.5
        )

        sector = 0.0
        for members in self.SECTOR_GROUPS.values():
            if symbol in members:
                peer = []
                for ss in members - {symbol}:
                    bs = self.data.get_bars(ss, include_partial=True)
                    if len(bs) > 10:
                        peer.append(
                            self._return(np.array([b["c"] for b in bs], dtype=float))
                        )
                if peer:
                    sector = tr - float(np.mean(peer))
                break

        return {
            "market_return_pct": market,
            "qqq_return_pct": qqq,
            "market_volatility_proxy": mv,
            "relative_strength_pct": tr - market,
            "cross_section_rank": float(np.clip(rank, 0.0, 1.0)),
            "breadth_pct_up": float(np.clip(breadth, 0.0, 1.0)),
            "sector_relative_return_pct": sector,
        }
    def _features(self,symbol,bl):
        c=np.array([b["c"] for b in bl],float); h=np.array([b["h"] for b in bl],float); l=np.array([b["l"] for b in bl],float); v=np.array([b.get("v",0) for b in bl],float); n=len(c)
        sma5=self._sma(c,5)[-1]; sma15=self._sma(c,15)[-1]; ema5=self._ema(c,5)[-1]; rsi=self._rsi(c,14)[-1]; mom=self._return(c,5); roll=float(np.std(c[-10:])/np.mean(c[-10:])*100) if n else 0.0
        atrs=MarketDataLayer._atr_series(bl,14); atr=float(atrs[-1]/c[-1]*100) if len(atrs) and c[-1] else 0.0
        avgv=float(np.mean(v[-21:-1])) if n>21 else 0.0; rv=float(v[-1]/avgv) if avgv>0 else 1.0; va=0.0
        if n>=10:
            a=float(np.mean(v[-10:-5])); b=float(np.mean(v[-5:])); va=(b/a-1)*100 if a>0 else 0.0
        d20h=self._pct(c[-1],np.max(h[-20:])); d20l=self._pct(c[-1],np.min(l[-20:])); d50h=self._pct(c[-1],np.max(h[-50:])); d50l=self._pct(c[-1],np.min(l[-50:]))
        atrchg=0.0
        if len(atrs)>5 and c[-6] and atrs[-6]:
            prev=atrs[-6]/c[-6]*100; atrchg=(atr-prev)/prev*100 if prev else 0.0
        ret=np.diff(c)/np.where(c[:-1]==0,1,c[:-1])*100 if n>1 else np.array([]); realized=float(np.std(ret[-20:])) if len(ret) else 0.0; volpct=float(np.mean(ret[-70:]<=realized)) if len(ret) else 0.5
        ses=self._session_context(bl); mk=self._market_features(symbol,bl); nf=self.data.get_news_features(symbol)
        return np.array([self._pct(sma5,sma15),self._pct(ema5,c[-1]),rsi-50,mom,roll,atr,mk["market_return_pct"],mk["qqq_return_pct"],mk["market_volatility_proxy"],mk["relative_strength_pct"],rv,va,d20h,d20l,d50h,d50l,atrchg,realized,volpct,ses["minutes_since_open"],ses["minutes_to_close"],ses["intraday_return_pct"],ses["distance_from_open_pct"],ses["opening_range_breakout"],ses["gap_pct"],mk["cross_section_rank"],mk["breadth_pct_up"],mk["sector_relative_return_pct"],nf["news_count"],nf["news_sentiment_change"],nf["news_sentiment_dispersion"]],float)

    @staticmethod
    def _sma(arr, window):
        return np.array([arr[max(0, i - window + 1):i + 1].mean() for i in range(len(arr))])

    @staticmethod
    def _ema(arr, span):
        alpha = 2 / (span + 1)
        out = np.zeros_like(arr)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
        return out

    @staticmethod
    def _rsi(arr, window):
        out = np.full_like(arr, 50.0)
        deltas = np.diff(arr, prepend=arr[0])
        gains, losses = np.clip(deltas, 0, None), np.clip(-deltas, 0, None)
        for i in range(window, len(arr)):
            ag, al = gains[i - window:i].mean(), losses[i - window:i].mean()
            out[i] = 100.0 if al == 0 else 100 - (100 / (1 + ag / al))
        return out

    def _train_batch(self, symbol, bar_list):
        X = self._features(symbol, bar_list)
        closes = np.array([b["c"] for b in bar_list], dtype=float)
        y = (np.diff(closes) > 0).astype(int)
        X_train = X[:-1]
        if len(np.unique(y)) < 2 or len(X_train) < self.MIN_BARS_FOR_MODEL:
            return None
        clf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=self.random_state, n_jobs=1)
        clf.fit(X_train, y)
        self.bars_since_retrain[symbol] = 0
        return clf

    @staticmethod
    def _p_up_from_classifier(model, X_row):
        if model is None:
            return 0.5
        proba = model.predict_proba(X_row.reshape(1, -1))[0]
        classes = list(model.classes_)
        return proba[classes.index(1)] if 1 in classes else 0.5

    def _markov_p_up(self, bar_list):
        closes = np.array([b["c"] for b in bar_list[-self.MARKOV_STATE_WINDOW:]], dtype=float)
        if len(closes) < 10:
            return 0.5
        returns = np.diff(closes) / closes[:-1]

        def state_of(r):
            if r > self.MARKOV_FLAT_THRESHOLD:
                return "up"
            if r < -self.MARKOV_FLAT_THRESHOLD:
                return "down"
            return "flat"

        states = [state_of(r) for r in returns]
        if len(states) < 2:
            return 0.5

        current_state = states[-1]
        counts = {"up": 1, "down": 1, "flat": 1}
        for a, b in zip(states[:-1], states[1:]):
            if a == current_state:
                counts[b] = counts.get(b, 0) + 1
        total = sum(counts.values())
        p_up = (counts["up"] + 0.5 * counts["flat"]) / total
        return float(np.clip(p_up, 0.01, 0.99))

    def _meanrev_p_up(self, bar_list):
        closes = np.array([b["c"] for b in bar_list[-self.MEANREV_LOOKBACK:]], dtype=float)
        if len(closes) < 5:
            return 0.5
        mean = closes.mean()
        std = closes.std()
        if std == 0:
            return 0.5
        z = (closes[-1] - mean) / std
        p_up = 1 / (1 + np.exp(self.MEANREV_Z_SCALE * z))
        return float(np.clip(p_up, 0.01, 0.99))

    def _default_weights(self):
        return {e: 1.0 / len(self.EXPERTS) for e in self.EXPERTS}

    def _normalized_weights(self, symbol):
        """Return a complete per-symbol weight dict, filling any missing
        experts with the uniform default. Persisted weights files can be
        partially populated (older versions, manual edits, a prior crash
        mid-write), and a missing key would otherwise raise KeyError in
        generate()/_update_online_and_weights()."""
        raw = self.weights.get(symbol) or {}
        result = dict(self._default_weights())
        for expert in self.EXPERTS:
            try:
                result[expert] = float(raw.get(expert, result[expert]))
            except (TypeError, ValueError):
                pass  # keep the uniform default for malformed values
        # Re-normalize in case a stored file had weights that don't sum to 1.
        total = sum(result.values())
        if total > 0:
            result = {e: w / total for e, w in result.items()}
        return result

    def _update_online_and_weights(self, symbol, true_label):
        prev = self.pending.get(symbol)
        if prev is None:
            return

        online_model = self.online_models.get(symbol)
        if online_model is None:
            online_model = SGDClassifier(loss="log_loss", random_state=self.random_state)
            online_model.partial_fit(prev["features"].reshape(1, -1), [true_label], classes=[0, 1])
            self.online_models[symbol] = online_model
        else:
            online_model.partial_fit(prev["features"].reshape(1, -1), [true_label])

        weights = self._normalized_weights(symbol)
        new_weights = {}
        for expert in self.EXPERTS:
            loss = (prev["p_up"][expert] - true_label) ** 2
            new_weights[expert] = weights[expert] * np.exp(-self.WEIGHT_LEARNING_RATE * loss)
        total = sum(new_weights.values())
        if total > 0:
            self.weights[symbol] = {e: w / total for e, w in new_weights.items()}

    def generate(self, symbol) -> Optional[TradeSignal]:
        # completed_bars: only fully-closed bars — used for training labels
        # and batch retraining, exactly as before. live_bars: completed bars
        # PLUS the still-forming current bar — used for the actual prediction,
        # so confidence reflects the latest tick's price, not a stale value
        # repeated until the next bar closes.
        completed_bars = self.data.get_bars(symbol)
        if len(completed_bars) < self.MIN_BARS_FOR_MODEL:
            return None
        closes_completed = np.array([b["c"] for b in completed_bars], dtype=float)

        # --- learn from the outcome of the PREVIOUS prediction, but only
        # when a bar has genuinely closed since last time — not every tick ---
        current_count = self.data.get_bar_close_count(symbol)
        prev = self.pending.get(symbol)
        if prev is not None and current_count > prev.get("bar_close_count", -1):
            true_label = 1 if closes_completed[-1] > closes_completed[-2] else 0
            self._update_online_and_weights(symbol, true_label)
            self.bars_since_retrain[symbol] = self.bars_since_retrain.get(symbol, self.retrain_every_n_bars) + 1

        # --- periodic batch retrain, still counted in real bars, not ticks ---
        self.bars_since_retrain.setdefault(symbol, self.retrain_every_n_bars)
        batch_model = self.batch_models.get(symbol)
        if batch_model is None or self.bars_since_retrain[symbol] >= self.retrain_every_n_bars:
            trained = self._train_batch(symbol, completed_bars)  # _train_batch resets the counter internally
            if trained is not None:
                batch_model = trained
                self.batch_models[symbol] = batch_model

        if batch_model is None:
            return None

        # --- live prediction: reflects the current tick's price ---
        live_bars = self.data.get_bars(symbol, include_partial=True) or completed_bars
        X_live = self._features(symbol, live_bars)
        latest_X = X_live[-1]
        online_model = self.online_models.get(symbol)

        p_up = {
            "batch": self._p_up_from_classifier(batch_model, latest_X),
            "online": self._p_up_from_classifier(online_model, latest_X),
            "markov": self._markov_p_up(live_bars),
            "meanrev": self._meanrev_p_up(live_bars),
        }

        weights = self._normalized_weights(symbol)
        blended_p_up = sum(weights[e] * p_up[e] for e in self.EXPERTS)

        direction = "up" if blended_p_up >= 0.5 else "down"
        confidence = blended_p_up if direction == "up" else (1 - blended_p_up)

        def conf_for(expert):
            p = p_up[expert]
            return p if direction == "up" else (1 - p)

        self.pending[symbol] = {"features": latest_X, "p_up": p_up, "bar_close_count": current_count}

        # Ensure a complete, normalized weight row exists for this symbol so
        # downstream consumers (and the weights autosave) always see all four
        # experts.
        if symbol not in self.weights or set(self.weights.get(symbol, {})) != set(self.EXPERTS):
            self.weights[symbol] = weights

        closes_live = np.array([b["c"] for b in live_bars], dtype=float)
        recent = closes_live[-15:]
        x = np.arange(len(recent)).reshape(-1, 1)
        reg = LinearRegression().fit(x, recent.reshape(-1, 1))
        predicted = reg.predict(np.array([[len(recent) + 3]]))[0][0]
        trend_change_pct = ((predicted - recent[-1]) / recent[-1]) * 100

        atr_pct, vol_regime = self.data.current_atr_and_regime(symbol)
        trend_regime = self.data.trend_regime(completed_bars, atr_pct)
        hmm_regime, hmm_prob = self.data.current_hmm_regime(symbol)
        sentiment = self.data.get_news_sentiment(symbol)
        news_features = self.data.get_news_features(symbol)
        feature_map = dict(zip(self.FEATURE_NAMES, latest_X.tolist()))

        signal = TradeSignal(
            symbol=symbol, direction=direction, confidence=confidence,
            trend_change_pct=trend_change_pct, volatility_regime=vol_regime,
            trend_regime=trend_regime, hmm_regime=hmm_regime, hmm_regime_prob=hmm_prob,
            atr_pct=atr_pct or 0.0, news_sentiment=sentiment,
            batch_confidence=conf_for("batch"), online_confidence=conf_for("online"),
            markov_confidence=conf_for("markov"), meanrev_confidence=conf_for("meanrev"),
            batch_weight=weights["batch"], online_weight=weights["online"],
            markov_weight=weights["markov"], meanrev_weight=weights["meanrev"],
            timestamp=datetime.now(timezone.utc),
            market_return_pct=feature_map.get("market_return_pct",0.0), qqq_return_pct=feature_map.get("qqq_return_pct",0.0),
            market_volatility_proxy=feature_map.get("market_volatility_proxy",0.0), relative_strength_pct=feature_map.get("relative_strength_pct",0.0),
            relative_volume_ratio=feature_map.get("relative_volume_ratio",1.0), volume_acceleration=feature_map.get("volume_acceleration",0.0),
            distance_20_high_pct=feature_map.get("distance_20_high_pct",0.0), distance_20_low_pct=feature_map.get("distance_20_low_pct",0.0),
            distance_50_high_pct=feature_map.get("distance_50_high_pct",0.0), distance_50_low_pct=feature_map.get("distance_50_low_pct",0.0),
            atr_change_pct=feature_map.get("atr_change_pct",0.0), realized_volatility_pct=feature_map.get("realized_volatility_pct",0.0),
            volatility_percentile=feature_map.get("volatility_percentile",0.5), minutes_since_open=feature_map.get("minutes_since_open",0.0),
            minutes_to_close=feature_map.get("minutes_to_close",390.0), intraday_return_pct=feature_map.get("intraday_return_pct",0.0),
            distance_from_open_pct=feature_map.get("distance_from_open_pct",0.0), opening_range_breakout=feature_map.get("opening_range_breakout",0.0),
            gap_pct=feature_map.get("gap_pct",0.0), cross_section_rank=feature_map.get("cross_section_rank",0.5),
            breadth_pct_up=feature_map.get("breadth_pct_up",0.5), sector_relative_return_pct=feature_map.get("sector_relative_return_pct",0.0),
            news_count=news_features.get("news_count",0.0), news_sentiment_change=news_features.get("news_sentiment_change",0.0),
            news_sentiment_dispersion=news_features.get("news_sentiment_dispersion",0.0),
        )

        # --- meta-labeling adjustment (FIXED: symmetric around 0.5) ---
        # meta_prob == 0.5 (untrained/neutral) => no change at all.
        # meta_prob > 0.5 => stretches confidence away from 0.5 (more sure).
        # meta_prob < 0.5 => shrinks confidence toward 0.5 (less sure).
        # (Previously this multiplied confidence by meta_prob directly, which
        # punished almost every trade even when meta_prob was decent — e.g.
        # meta_prob=0.65 cut confidence by 35%, making the 0.62 threshold
        # nearly impossible to clear. That's why no trades were firing.)
        if self.meta_filter is not None and self.meta_filter.available:
            meta_prob = self.meta_filter.score(signal)
            adjustment_factor = min(2 * meta_prob, 2.0)
            adjusted = max(0.01, min(0.99, 0.5 + (signal.confidence - 0.5) * adjustment_factor))
            signal = replace(signal, confidence=adjusted, meta_win_prob=meta_prob)

        return signal


# =============================================================================
# LAYER 3 — RiskGate
# =============================================================================
class RiskGate:
    CONFIDENCE_THRESHOLD = _cfg_float("MIN_SIGNAL_CONFIDENCE", 0.62)
    TREND_AGREEMENT_REQUIRED = True
    NEWS_VETO_THRESHOLD = -0.35
    MAX_POSITION_PER_SYMBOL = 8
    MIN_SECONDS_BETWEEN_ORDERS = 30
    ATR_STOP_MULTIPLIER = _cfg_float("ATR_STOP_MULTIPLIER", 1.5)
    ATR_TARGET_MULTIPLIER = 2.5
    MIN_STOP_PCT = 0.15
    MAX_STOP_PCT = 1.5
    DAILY_LOSS_LIMIT_PCT = 2.0
    MAX_QTY_PER_ORDER = 3

    KELLY_FRACTION_MULTIPLIER = 0.5
    MAX_KELLY_FRACTION = 0.05
    MAX_POSITION_PCT = _cfg_float("MAX_POSITION_PCT", 0.20)
    RISK_PER_TRADE_PCT = _cfg_float("RISK_PER_TRADE_PCT", 0.005)
    FIXED_STOP_PCT = _cfg_float("STOP_LOSS_PCT", 1.0)
    FIXED_TARGET_PCT = _cfg_float("TAKE_PROFIT_PCT", 2.0)
    DEFAULT_EQUITY_ESTIMATE = 10_000.0

    # -------------------------------------------------------------------------
    # Day-trading timing
    # -------------------------------------------------------------------------
    # IMPORTANT:
    # These are NOT loaded from strategy_config.py or optimizer environment
    # variables. Strategy optimization must never be able to change the
    # market-hours safety system.
    MARKET_TIMEZONE_NAME = "America/New_York"
    MARKET_TIMEZONE = ZoneInfo(MARKET_TIMEZONE_NAME)

    # Regular U.S. equity session.
    REGULAR_SESSION_OPEN = dt_time(9, 30)
    REGULAR_SESSION_CLOSE = dt_time(16, 0)

    # Stop entering new positions with 30 minutes left.
    NO_NEW_ENTRIES_MINUTES_BEFORE_CLOSE = 30

    # Begin flattening with 10 minutes left.
    EOD_LIQUIDATION_MINUTES_BEFORE_CLOSE = 10

    # Emergency/defensive settings.
    CLOCK_CACHE_SECONDS = 30
    EOD_CLOSE_RETRIES = 3
    EOD_CLOSE_RETRY_SECONDS = 2

    # Position tapering remains conservative but cannot extend trading past
    # the hard entry cutoff.
    MIN_SIZE_TAPER_MULTIPLIER = 0.4
    TOTAL_SESSION_MINUTES = 390


    def __init__(self, symbols, api_client=None, trade_log_path="trade_data/trade_log.csv"):
        self.symbols = symbols
        self.api = api_client
        self.positions_lock = threading.Lock()
        self.current_positions: Dict[str, int] = {s: 0 for s in symbols}
        self.entry_prices: Dict[str, Optional[float]] = {s: None for s in symbols}
        self.entry_stop_pct: Dict[str, Optional[float]] = {s: None for s in symbols}
        self.entry_target_pct: Dict[str, Optional[float]] = {s: None for s in symbols}
        self.entry_signal_context: Dict[str, Optional[TradeSignal]] = {s: None for s in symbols}
        self.entry_qty: Dict[str, int] = {s: 0 for s in symbols}
        self.last_order_time: Dict[str, float] = {s: 0.0 for s in symbols}
        self._last_known_qty: Dict[str, int] = {s: 0 for s in symbols}
        self.trading_halted = False
        self.starting_equity = None
        self.state_lock = threading.Lock()

        # EOD liquidation state. eod_liquidation_active blocks NEW buy
        # entries once we've flattened for the day; _eod_liquidation_done_date
        # tracks which calendar date (per the exchange clock) we last ran
        # liquidation for, so it only fires once per session and resets
        # automatically the next trading day.
        self.eod_liquidation_active = False
        self._eod_liquidation_done_date = None

        # Cached Alpaca clock. It is used only as a validated source for
        # early-close information and to confirm whether the market is open.
        # Actual entry/liquidation decisions are anchored to America/New_York.
        self._clock_lock = threading.Lock()
        self._clock_cache = None
        self._clock_cache_time = 0.0

        # Session state.
        self._session_date = None
        self._session_close_et = None
        self._session_open_et = None

        self.trade_log_path = trade_log_path
        self._log_lock = threading.Lock()
        self._ensure_trade_log_header()

        try:
            initial_status = self._session_status()
            self._log_timing_snapshot(initial_status, prefix="STARTUP")
        except Exception as e:
            log.error(f"RiskGate: failed initial session timing check: {e}")

    def _ensure_trade_log_header(self):
        # Auto-creates the parent folder (e.g. trade_data/) if it doesn't
        # exist yet — no manual folder setup ever required.
        parent = os.path.dirname(self.trade_log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if not os.path.exists(self.trade_log_path):
            with open(self.trade_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "exit_timestamp", "symbol", "entry_price", "exit_price", "pnl_pct", "exit_reason",
                    "entry_confidence", "entry_direction", "entry_trend_change_pct",
                    "entry_volatility_regime", "entry_trend_regime", "entry_hmm_regime", "entry_hmm_regime_prob",
                    "entry_atr_pct", "entry_news_sentiment",
                    "entry_batch_weight", "entry_online_weight", "entry_markov_weight", "entry_meanrev_weight",
                    "entry_meta_win_prob",
                    'entry_market_return_pct', 'entry_qqq_return_pct', 'entry_market_volatility_proxy', 'entry_relative_strength_pct', 'entry_relative_volume_ratio', 'entry_volume_acceleration', 'entry_distance_20_high_pct', 'entry_distance_20_low_pct', 'entry_distance_50_high_pct', 'entry_distance_50_low_pct', 'entry_atr_change_pct', 'entry_realized_volatility_pct', 'entry_volatility_percentile', 'entry_minutes_since_open', 'entry_minutes_to_close', 'entry_intraday_return_pct', 'entry_distance_from_open_pct', 'entry_opening_range_breakout', 'entry_gap_pct', 'entry_cross_section_rank', 'entry_breadth_pct_up', 'entry_sector_relative_return_pct', 'entry_news_count', 'entry_news_sentiment_change', 'entry_news_sentiment_dispersion'
                ])

    def record_trade_outcome(self, symbol, exit_price, exit_reason):
        entry_price = self.entry_prices[symbol]
        signal = self.entry_signal_context[symbol]
        if entry_price is None:
            return

        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        row = [datetime.now(timezone.utc).isoformat(), symbol, entry_price, exit_price,
               round(pnl_pct, 4), exit_reason]
        if signal is not None:
            row += [round(signal.confidence, 4), signal.direction, round(signal.trend_change_pct, 4),
                    signal.volatility_regime, signal.trend_regime, signal.hmm_regime,
                    round(signal.hmm_regime_prob, 4), round(signal.atr_pct, 4), round(signal.news_sentiment, 4),
                    round(signal.batch_weight, 4), round(signal.online_weight, 4),
                    round(signal.markov_weight, 4), round(signal.meanrev_weight, 4),
                    round(signal.meta_win_prob, 4)]
            row += [getattr(signal, n, None) for n in ['market_return_pct', 'qqq_return_pct', 'market_volatility_proxy', 'relative_strength_pct', 'relative_volume_ratio', 'volume_acceleration', 'distance_20_high_pct', 'distance_20_low_pct', 'distance_50_high_pct', 'distance_50_low_pct', 'atr_change_pct', 'realized_volatility_pct', 'volatility_percentile', 'minutes_since_open', 'minutes_to_close', 'intraday_return_pct', 'distance_from_open_pct', 'opening_range_breakout', 'gap_pct', 'cross_section_rank', 'breadth_pct_up', 'sector_relative_return_pct', 'news_count', 'news_sentiment_change', 'news_sentiment_dispersion']]
        else:
            row += [None] * 14 + [None] * 25

        with self._log_lock:
            with open(self.trade_log_path, "a", newline="") as f:
                csv.writer(f).writerow(row)

    def record_open_order(self, symbol, action, qty, price, reason):
        """
        Append an "open" order row to the trade log so the dashboard can
        surface newly placed buy orders immediately. Uses the same column
        layout as the closed-trade rows but leaves exit fields empty.
        """
        signal = self.entry_signal_context.get(symbol)
        # exit_timestamp left empty to indicate an open order
        row = ["", symbol, price, "", "", reason]
        if signal is not None:
            row += [round(signal.confidence, 4), signal.direction, round(signal.trend_change_pct, 4),
                    signal.volatility_regime, signal.trend_regime, signal.hmm_regime,
                    round(signal.hmm_regime_prob, 4), round(signal.atr_pct, 4), round(signal.news_sentiment, 4),
                    round(signal.batch_weight, 4), round(signal.online_weight, 4),
                    round(signal.markov_weight, 4), round(signal.meanrev_weight, 4),
                    round(signal.meta_win_prob, 4)]
            row += [getattr(signal, n, None) for n in ['market_return_pct', 'qqq_return_pct', 'market_volatility_proxy', 'relative_strength_pct', 'relative_volume_ratio', 'volume_acceleration', 'distance_20_high_pct', 'distance_20_low_pct', 'distance_50_high_pct', 'distance_50_low_pct', 'atr_change_pct', 'realized_volatility_pct', 'volatility_percentile', 'minutes_since_open', 'minutes_to_close', 'intraday_return_pct', 'distance_from_open_pct', 'opening_range_breakout', 'gap_pct', 'cross_section_rank', 'breadth_pct_up', 'sector_relative_return_pct', 'news_count', 'news_sentiment_change', 'news_sentiment_dispersion']]
        else:
            row += [None] * 14 + [None] * 25

        with self._log_lock:
            with open(self.trade_log_path, "a", newline="") as f:
                csv.writer(f).writerow(row)

    def refresh_positions(self):
        """Synchronize local state with Alpaca and record externally closed positions.

        The old implementation silently cleared local entry state whenever
        Alpaca reported qty=0. That meant a trade closed by Alpaca, a manual
        close, or another process never produced a completed row in
        trade_log.csv. This reconciliation records that close before clearing
        the entry context.
        """
        if self.api is None:
            return
        try:
            held = {
                p.symbol: (int(float(p.qty)), float(p.avg_entry_price))
                for p in self.api.list_positions()
            }
        except Exception as e:
            log.error(f"RiskGate: failed to refresh positions: {e}")
            return

        for s in self.symbols:
            current_qty, current_avg = held.get(s, (0, None))

            with self.positions_lock:
                previous_qty = self._last_known_qty.get(s, self.current_positions.get(s, 0))
                previous_entry = self.entry_prices.get(s)
                previous_signal = self.entry_signal_context.get(s)
                previous_stop = self.entry_stop_pct.get(s)
                previous_target = self.entry_target_pct.get(s)

            # If a previously-held long position is now flat, Alpaca (or an
            # external/manual action) closed it. Log the completed outcome using
            # the actual Alpaca average entry price and the current market/exit
            # price when available.
            if previous_qty > 0 and current_qty == 0 and previous_entry:
                try:
                    exit_price = previous_entry

                    # Prefer the latest quote when available. Fall back to the
                    # stored entry price if quote retrieval fails.
                    try:
                        quote = self.api.get_latest_trade(s)
                        quote_price = getattr(quote, "price", None)
                        if quote_price is not None:
                            exit_price = float(quote_price)
                    except Exception:
                        pass

                    pnl_pct = ((exit_price - float(previous_entry)) / float(previous_entry)) * 100.0
                    row = [
                        datetime.now(timezone.utc).isoformat(),
                        s,
                        float(previous_entry),
                        float(exit_price),
                        round(pnl_pct, 4),
                        "position_closed_detected_by_sync",
                    ]

                    if previous_signal is not None:
                        row += [
                            round(previous_signal.confidence, 4),
                            previous_signal.direction,
                            round(previous_signal.trend_change_pct, 4),
                            previous_signal.volatility_regime,
                            previous_signal.trend_regime,
                            previous_signal.hmm_regime,
                            round(previous_signal.hmm_regime_prob, 4),
                            round(previous_signal.atr_pct, 4),
                            round(previous_signal.news_sentiment, 4),
                            round(previous_signal.batch_weight, 4),
                            round(previous_signal.online_weight, 4),
                            round(previous_signal.markov_weight, 4),
                            round(previous_signal.meanrev_weight, 4),
                            round(previous_signal.meta_win_prob, 4),
                            previous_signal.market_return_pct, previous_signal.qqq_return_pct, previous_signal.market_volatility_proxy, previous_signal.relative_strength_pct, previous_signal.relative_volume_ratio, previous_signal.volume_acceleration, previous_signal.distance_20_high_pct, previous_signal.distance_20_low_pct, previous_signal.distance_50_high_pct, previous_signal.distance_50_low_pct, previous_signal.atr_change_pct, previous_signal.realized_volatility_pct, previous_signal.volatility_percentile, previous_signal.minutes_since_open, previous_signal.minutes_to_close, previous_signal.intraday_return_pct, previous_signal.distance_from_open_pct, previous_signal.opening_range_breakout, previous_signal.gap_pct, previous_signal.cross_section_rank, previous_signal.breadth_pct_up, previous_signal.sector_relative_return_pct, previous_signal.news_count, previous_signal.news_sentiment_change, previous_signal.news_sentiment_dispersion,
                        ]
                    else:
                        row += [None] * 14 + [None] * 25

                    with self._log_lock:
                        with open(self.trade_log_path, "a", newline="", encoding="utf-8") as f:
                            csv.writer(f).writerow(row)

                    log.info(
                        f"[{s}] Recorded externally detected closed trade: "
                        f"entry=${float(previous_entry):.2f} exit=${float(exit_price):.2f} "
                        f"PnL={pnl_pct:+.4f}%"
                    )
                except Exception as e:
                    log.error(f"[{s}] Failed to record externally detected close: {e}")

            with self.positions_lock:
                self.current_positions[s] = current_qty
                self.entry_prices[s] = current_avg if current_qty > 0 else None

                if current_qty == 0:
                    self.entry_stop_pct[s] = None
                    self.entry_target_pct[s] = None
                    self.entry_signal_context[s] = None
                    self.entry_qty[s] = 0
                else:
                    self.entry_qty[s] = current_qty
                    if self.entry_stop_pct[s] is None or self.entry_target_pct[s] is None:
                        # A position exists on Alpaca but we have no locally
                        # tracked stop/target for it — this happens for any
                        # position that was already open when this RiskGate
                        # instance was constructed (fresh process start,
                        # restart after a crash, dashboard relaunch, etc.),
                        # since __init__ starts everyone at None and only
                        # evaluate() ever sets a real stop/target, at the
                        # moment THIS process approves a buy.
                        #
                        # Without this fallback, the stop-loss/take-profit
                        # check in evaluate() (`stop_pct is not None and
                        # target_pct is not None`) silently never fires for
                        # this symbol, so a losing position can sit
                        # completely unprotected indefinitely — it only
                        # exits if the model later fires a strongly
                        # confident, trend-confirmed "down" signal, or via
                        # EOD liquidation. Assign the fixed config stop/
                        # target immediately so it's always protected.
                        fallback_stop = float(np.clip(
                            self.FIXED_STOP_PCT, self.MIN_STOP_PCT, self.MAX_STOP_PCT))
                        fallback_target = float(np.clip(
                            self.FIXED_TARGET_PCT, self.MIN_STOP_PCT, self.MAX_STOP_PCT * 2))
                        self.entry_stop_pct[s] = fallback_stop
                        self.entry_target_pct[s] = fallback_target
                        log.warning(
                            f"[{s}] RiskGate: found an open position with no tracked "
                            f"stop/target (likely after a restart) — applying fallback "
                            f"stop={fallback_stop:.2f}% target={fallback_target:.2f}% "
                            f"so it isn't held unprotected."
                        )

                self._last_known_qty[s] = current_qty

    def check_daily_loss_limit(self):
        if self.api is None:
            return
        try:
            equity = float(self.api.get_account().equity)
        except Exception as e:
            log.error(f"RiskGate: failed to fetch account: {e}")
            return
        with self.state_lock:
            self.starting_equity = self.starting_equity or equity
            drawdown_pct = ((self.starting_equity - equity) / self.starting_equity) * 100
            if drawdown_pct >= self.DAILY_LOSS_LIMIT_PCT and not self.trading_halted:
                self.trading_halted = True
                log.error(f"RiskGate: DAILY LOSS LIMIT HIT ({drawdown_pct:.2f}%). Trading halted for the day.")

    def _get_cached_clock(self, force=False):
        """Fetch Alpaca's clock with a short cache.

        The clock is never allowed to redefine timezone semantics. All local
        trading-hour calculations are done in America/New_York.
        """
        if self.api is None:
            return None

        now = time.time()
        with self._clock_lock:
            if (
                not force
                and self._clock_cache is not None
                and (now - self._clock_cache_time) < self.CLOCK_CACHE_SECONDS
            ):
                return self._clock_cache

        try:
            clock = self.api.get_clock()
        except Exception as e:
            log.error(f"RiskGate: failed to fetch Alpaca market clock: {e}")
            return None

        with self._clock_lock:
            self._clock_cache = clock
            self._clock_cache_time = now

        return clock

    @classmethod
    def _now_et(cls):
        return datetime.now(timezone.utc).astimezone(cls.MARKET_TIMEZONE)

    @staticmethod
    def _is_weekday(et_now):
        return et_now.weekday() < 5

    def _refresh_session_schedule(self, et_now=None, force_clock=False):
        """Build today's session boundaries in Eastern Time.

        Regular close is always 16:00 ET. If Alpaca reports a same-day
        earlier close (holiday/early-close session), that earlier close is
        adopted only after sanity-checking it. A malformed/future-day clock
        cannot move today's schedule to an arbitrary hour.
        """
        et_now = et_now or self._now_et()
        date_key = et_now.date()

        regular_open = datetime.combine(
            date_key,
            self.REGULAR_SESSION_OPEN,
            tzinfo=self.MARKET_TIMEZONE,
        )
        regular_close = datetime.combine(
            date_key,
            self.REGULAR_SESSION_CLOSE,
            tzinfo=self.MARKET_TIMEZONE,
        )

        clock = self._get_cached_clock(force=force_clock)

        session_open = regular_open
        session_close = regular_close

        if clock is not None:
            try:
                clock_open = clock.next_open.astimezone(self.MARKET_TIMEZONE)
                clock_close = clock.next_close.astimezone(self.MARKET_TIMEZONE)
            except Exception:
                clock_open = None
                clock_close = None

            # If today's market is open, prefer the actual current session's
            # close when it is today and earlier than the standard 16:00 close.
            if getattr(clock, "is_open", False) and clock_close is not None:
                if (
                    clock_close.date() == date_key
                    and regular_open <= clock_close <= regular_close
                ):
                    session_close = clock_close

            # On a closed day, don't infer today's schedule from tomorrow's
            # next_open/next_close.
            #
            # We intentionally do not adopt arbitrary clock times here.

        self._session_date = date_key
        self._session_open_et = session_open
        self._session_close_et = session_close

        return session_open, session_close

    def _session_status(self, et_now=None):
        """Return authoritative local session status and time-to-close.

        Returns:
            dict with:
              et_now
              is_weekday
              session_open
              session_close
              market_open
              minutes_to_open
              minutes_to_close
              can_enter
              liquidation_due
        """
        et_now = et_now or self._now_et()
        open_et, close_et = self._refresh_session_schedule(et_now)

        weekday = self._is_weekday(et_now)

        # Use Alpaca's is_open as a confirmation when available, but never
        # allow an invalid clock to create a bizarre cutoff hour.
        clock = self._get_cached_clock()
        alpaca_open = getattr(clock, "is_open", None) if clock is not None else None

        inside_local_session = open_et <= et_now < close_et

        if not weekday:
            market_open = False
        elif alpaca_open is True:
            market_open = inside_local_session
        elif alpaca_open is False:
            market_open = False
        else:
            # If the API clock is temporarily unavailable, the local
            # Eastern-time schedule remains the fallback.
            market_open = inside_local_session

        minutes_to_close = None
        minutes_to_open = None

        if et_now < open_et:
            minutes_to_open = (open_et - et_now).total_seconds() / 60.0
        elif et_now < close_et:
            minutes_to_close = max(
                0.0,
                (close_et - et_now).total_seconds() / 60.0,
            )

        no_new_entries = (
            market_open
            and minutes_to_close is not None
            and minutes_to_close <= self.NO_NEW_ENTRIES_MINUTES_BEFORE_CLOSE
        )

        liquidation_due = (
            market_open
            and minutes_to_close is not None
            and minutes_to_close <= self.EOD_LIQUIDATION_MINUTES_BEFORE_CLOSE
        )

        return {
            "et_now": et_now,
            "is_weekday": weekday,
            "session_open": open_et,
            "session_close": close_et,
            "market_open": market_open,
            "minutes_to_open": minutes_to_open,
            "minutes_to_close": minutes_to_close,
            "no_new_entries": no_new_entries,
            "liquidation_due": liquidation_due,
        }

    def _minutes_to_close(self):
        status = self._session_status()
        return status["minutes_to_close"]

    def _log_timing_snapshot(self, status, prefix="RiskGate"):
        """Useful diagnostic output without relying on ambiguous local time."""
        et_now = status["et_now"]
        close_et = status["session_close"]
        minutes_to_close = status["minutes_to_close"]

        log.info(
            f"{prefix}: session clock ET={et_now:%Y-%m-%d %H:%M:%S %Z}, "
            f"open={status['market_open']}, "
            f"session_close={close_et:%Y-%m-%d %H:%M:%S %Z}, "
            f"minutes_to_close="
            f"{'n/a' if minutes_to_close is None else f'{minutes_to_close:.1f}'}, "
            f"new_entries_blocked={status['no_new_entries']}, "
            f"liquidation_due={status['liquidation_due']}"
        )

    def check_eod_liquidation(self):
        """Flatten every non-zero equity position before the session ends.

        Timing is based on America/New_York. Alpaca's clock can shorten the
        session for legitimate early closes, but it cannot arbitrarily move
        the close to 8 AM or another unrelated hour.
        """
        if self.api is None:
            return

        status = self._session_status()

        # Outside the session, reset the completed-day flag when the ET date
        # changes. This guarantees the next trading day gets a fresh window.
        today = status["et_now"].date()

        if not status["market_open"]:
            with self.state_lock:
                if (
                    self._eod_liquidation_done_date is not None
                    and self._eod_liquidation_done_date != today
                ):
                    self.eod_liquidation_active = False
                    self._eod_liquidation_done_date = None

            return

        with self.state_lock:
            already_done_today = (
                self._eod_liquidation_done_date == today
            )

        if already_done_today:
            return

        if not status["liquidation_due"]:
            return

        self._log_timing_snapshot(status, prefix="EOD")

        # Include BOTH long and short positions. A day-trading system must
        # finish completely flat.
        with self.positions_lock:
            open_symbols = [
                s for s in self.symbols
                if self.current_positions.get(s, 0) != 0
            ]

        if not open_symbols:
            log.info(
                f"RiskGate: {status['minutes_to_close']:.1f} min to close, "
                "no open positions to liquidate."
            )
        else:
            log.warning(
                f"RiskGate: {status['minutes_to_close']:.1f} min to close — "
                f"flattening ALL {len(open_symbols)} open position(s): "
                f"{', '.join(open_symbols)}"
            )

            # Best-effort cancel of outstanding orders first, then close
            # every position. Retry because an EOD liquidation is safety-
            # critical and a transient broker/API failure should not end the
            # attempt after one call.
            close_error = None
            for attempt in range(1, self.EOD_CLOSE_RETRIES + 1):
                try:
                    try:
                        self.api.cancel_all_orders()
                    except Exception:
                        pass

                    self.api.close_all_positions(cancel_orders=True)
                    close_error = None
                    break
                except Exception as e:
                    close_error = e
                    log.error(
                        f"RiskGate: EOD close attempt {attempt}/"
                        f"{self.EOD_CLOSE_RETRIES} failed: {e}"
                    )
                    if attempt < self.EOD_CLOSE_RETRIES:
                        time.sleep(self.EOD_CLOSE_RETRY_SECONDS)

            if close_error is not None:
                # Final per-symbol fallback.
                for s in open_symbols:
                    try:
                        self.api.close_position(s)
                    except Exception as e2:
                        log.error(
                            f"[{s}] RiskGate: final EOD close attempt failed: {e2}"
                        )

            # Do NOT invent a filled exit price or record a completed trade
            # before the broker confirms the position is actually gone.
            self.refresh_positions()

        with self.state_lock:
            self.eod_liquidation_active = True
            self._eod_liquidation_done_date = today

        # One more synchronization pass to capture any fills that arrived
        # during the liquidation request.
        self.refresh_positions()

        # Safety check: if anything remains open, continue treating the bot as
        # flattened for NEW entries but make the problem explicit in logs.
        with self.positions_lock:
            remaining = [
                s for s in self.symbols
                if self.current_positions.get(s, 0) != 0
            ]

        if remaining:
            log.error(
                "RiskGate: EOD liquidation requested but positions remain open: "
                + ", ".join(remaining)
            )
        else:
            log.info(
                f"RiskGate: EOD liquidation complete; account is flat for "
                f"{today}."
            )

    def background_sync_loop(self, interval=30):
        while True:
            try:
                self.refresh_positions()
                self.check_daily_loss_limit()
                self.check_eod_liquidation()
            except Exception as e:
                # Timing/safety loop must never die silently.
                log.exception(f"RiskGate: background safety loop error: {e}")
            time.sleep(interval)

    def _size_position(self, confidence, stop_pct, target_pct, price, size_multiplier=1.0):
        if stop_pct <= 0:
            return 0

        b = target_pct / stop_pct
        kelly_f = confidence - (1 - confidence) / b
        kelly_f = max(0.0, kelly_f) * self.KELLY_FRACTION_MULTIPLIER
        kelly_f = min(kelly_f, self.MAX_KELLY_FRACTION)

        equity_estimate = self.starting_equity or self.DEFAULT_EQUITY_ESTIMATE
        dollar_risk = equity_estimate * kelly_f
        stop_distance = price * (stop_pct / 100)
        if stop_distance <= 0:
            return 0

        # Conservative hard risk budget from shared strategy config.
        configured_risk = max(0.0, min(self.RISK_PER_TRADE_PCT, 0.02))
        dollar_risk = min(dollar_risk, equity_estimate * configured_risk)

        # Intraday size taper: scales the risk dollars down (never up) based
        # on how much of the session is left, so late-day entries — which
        # have less runway before EOD liquidation forces them closed — are
        # sized smaller than the same signal would be earlier in the day.
        size_multiplier = max(0.0, min(1.0, size_multiplier))
        dollar_risk *= size_multiplier

        qty_by_risk = int(dollar_risk / stop_distance)
        qty_by_position = int((equity_estimate * self.MAX_POSITION_PCT) / price) if price > 0 else 0
        qty = min(qty_by_risk, qty_by_position)
        return max(0, min(qty, self.MAX_QTY_PER_ORDER))

    def _intraday_size_multiplier(self, minutes_to_close):
        """Conservative size taper during the final 30 minutes."""
        if minutes_to_close is None:
            return 1.0

        if minutes_to_close <= 0:
            return 0.0

        cutoff = float(self.NO_NEW_ENTRIES_MINUTES_BEFORE_CLOSE)

        # Full size before taper window.
        if minutes_to_close >= cutoff:
            return 1.0

        # At/under the no-entry cutoff, entries are blocked by evaluate().
        usable = max(cutoff, 1.0)
        progress = max(0.0, min(1.0, (cutoff - minutes_to_close) / usable))

        return 1.0 - progress * (1.0 - self.MIN_SIZE_TAPER_MULTIPLIER)

    def evaluate(self, signal: TradeSignal, current_price: float, now: Optional[float] = None) -> GateDecision:
        if now is None:
            now = time.time()
        symbol = signal.symbol

        # Market-hours safety checks must use the evaluation time, not the
        # caller's wall clock. Backtests pass a historical bar timestamp for
        # `now`; if we ignored it and used the real current time, every gate
        # (market open, entry cutoff, EOD taper) would be evaluated against
        # the moment the backtest RUNS instead of the moment the bar formed —
        # on weekends/holidays/after-hours runs this silently blocks every
        # single trade.
        session = self._session_status(
            et_now=datetime.fromtimestamp(now, tz=self.MARKET_TIMEZONE)
        ) if now is not None else self._session_status()

        with self.positions_lock:
            held = self.current_positions[symbol]
            entry = self.entry_prices[symbol]
            stop_pct = self.entry_stop_pct[symbol]
            target_pct = self.entry_target_pct[symbol]

        if held > 0 and entry and stop_pct is not None and target_pct is not None:
            change_from_entry = ((current_price - entry) / entry) * 100
            if change_from_entry <= -stop_pct:
                return self._approve(symbol, "sell", held, now, f"stop-loss hit ({stop_pct:.2f}%)")
            if change_from_entry >= target_pct:
                return self._approve(symbol, "sell", held, now, f"take-profit hit ({target_pct:.2f}%)")

        if self.trading_halted:
            return GateDecision(False, None, None, "daily loss limit halted trading")

        # Hard market-hours guard for NEW entries. This is deliberately
        # independent from strategy confidence and optimizer settings.
        if not session["market_open"]:
            return GateDecision(
                False,
                None,
                None,
                "market session closed — no new entries"
            )

        if self.eod_liquidation_active:
            return GateDecision(
                False,
                None,
                None,
                "EOD liquidation completed — no new positions until next session"
            )

        if now - self.last_order_time[symbol] < self.MIN_SECONDS_BETWEEN_ORDERS:
            return GateDecision(False, None, None, "order cooldown active")

        if signal.confidence < self.CONFIDENCE_THRESHOLD:
            return GateDecision(False, None, None,
                                 f"confidence {signal.confidence:.2f} below threshold {self.CONFIDENCE_THRESHOLD}")

        if self.TREND_AGREEMENT_REQUIRED:
            if signal.direction == "up" and signal.trend_change_pct <= 0:
                return GateDecision(False, None, None, "trend regression disagrees with 'up' signal")
            if signal.direction == "down" and signal.trend_change_pct >= 0:
                return GateDecision(False, None, None, "trend regression disagrees with 'down' signal")

        if signal.direction == "up" and signal.news_sentiment <= self.NEWS_VETO_THRESHOLD:
            return GateDecision(False, None, None,
                                 f"news sentiment {signal.news_sentiment:+.2f} vetoes new long")

        if signal.direction == "up":
            if held >= self.MAX_POSITION_PER_SYMBOL:
                return GateDecision(False, None, None, "at max position size")

            # Day-trading awareness: don't open or add to a position once
            # there isn't enough runway left in the session for it to
            # plausibly develop before EOD liquidation force-closes
            # everything. This is checked here (not just at liquidation
            # time) so the bot never enters a trade it already knows will
            # be swept up in minutes.
            minutes_to_close = session["minutes_to_close"]

            if (
                minutes_to_close is None
                or minutes_to_close <= self.NO_NEW_ENTRIES_MINUTES_BEFORE_CLOSE
            ):
                return GateDecision(
                    False,
                    None,
                    None,
                    "day-trading entry cutoff active — no new positions "
                    f"with {self.NO_NEW_ENTRIES_MINUTES_BEFORE_CLOSE} min "
                    "or less remaining"
                )

            if self.ATR_STOP_MULTIPLIER > 0:
                stop_pct = float(np.clip(signal.atr_pct * self.ATR_STOP_MULTIPLIER, self.MIN_STOP_PCT, self.MAX_STOP_PCT))
                target_pct = float(np.clip(signal.atr_pct * self.ATR_TARGET_MULTIPLIER, self.MIN_STOP_PCT, self.MAX_STOP_PCT * 2))
            else:
                stop_pct = float(np.clip(self.FIXED_STOP_PCT, self.MIN_STOP_PCT, self.MAX_STOP_PCT))
                target_pct = float(np.clip(self.FIXED_TARGET_PCT, self.MIN_STOP_PCT, self.MAX_STOP_PCT * 2))

            # Size tapers down the later in the session we are (full size
            # early, smaller near the no-new-entries cutoff) — see
            # _intraday_size_multiplier for the exact schedule.
            size_multiplier = self._intraday_size_multiplier(minutes_to_close)
            qty = self._size_position(signal.confidence, stop_pct, target_pct, current_price,
                                       size_multiplier=size_multiplier)
            if qty <= 0:
                return GateDecision(False, None, None, "Kelly-sized position rounded to zero shares — edge too small")

            with self.positions_lock:
                self.entry_stop_pct[symbol] = stop_pct
                self.entry_target_pct[symbol] = target_pct
                self.entry_signal_context[symbol] = signal
                self.entry_qty[symbol] = qty
            return self._approve(symbol, "buy", qty, now,
                                  f"signal conf={signal.confidence:.2f} qty={qty} (Kelly-sized, "
                                  f"intraday size x{size_multiplier:.2f}) "
                                  f"experts=batch:{signal.batch_weight:.2f}/online:{signal.online_weight:.2f}/"
                                  f"markov:{signal.markov_weight:.2f}/meanrev:{signal.meanrev_weight:.2f} "
                                  f"regime={signal.trend_regime}/{signal.volatility_regime}/hmm:{signal.hmm_regime} "
                                  f"stop={stop_pct:.2f}% target={target_pct:.2f}% "
                                  f"mins_to_close={'n/a' if minutes_to_close is None else f'{minutes_to_close:.0f}'}")

        if signal.direction == "down" and held > 0:
            qty = held
            return self._approve(symbol, "sell", qty, now, f"signal conf={signal.confidence:.2f}")

        return GateDecision(False, None, None, "no qualifying action")

    def _approve(self, symbol, action, qty, now, reason):
        self.last_order_time[symbol] = now
        return GateDecision(True, action, qty, reason)

    def execute(self, decision: GateDecision, symbol: str, price: float):
        if not decision.approved:
            log.info(f"[{symbol}] VETOED: {decision.reason}")
            return
        if self.api is None:
            log.error(f"[{symbol}] No api client configured — cannot execute.")
            return
        try:
            self.api.submit_order(
                symbol=symbol,
                qty=decision.qty,
                side=decision.action,
                type="market",
                time_in_force="day",
            )
            log.info(f"[{symbol}] ORDER APPROVED & SENT: {decision.action.upper()} "
                      f"{decision.qty} @ ~${price:.2f} — {decision.reason}")
            # Send an immediate notification about the order placement so users
            # know when an order has actually been sent from the system.
            try:
                notify_order_placed(symbol, decision.action, decision.qty, price, decision.reason)
            except Exception:
                pass

            # Also write a small per-order JSON file into the same data folder so
            # dashboards/watchers that avoid reading a single locked file can
            # pick up this event reliably. This is a best-effort, non-blocking
            # write — failures are ignored so live trading is not disrupted.
            try:
                orders_dir = os.path.join(os.path.dirname(self.trade_log_path) or '.', 'orders')
                os.makedirs(orders_dir, exist_ok=True)
                order_payload = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'symbol': symbol,
                    'action': decision.action,
                    'qty': decision.qty,
                    'price': price,
                    'reason': decision.reason,
                }
                order_fname = os.path.join(orders_dir, f"{int(time.time())}_{uuid.uuid4().hex}.json")
                with open(order_fname, 'w', encoding='utf-8') as of:
                    json.dump(order_payload, of, default=str)
            except Exception:
                pass

            if decision.action == "sell":
                self.record_trade_outcome(symbol, price, decision.reason)
            elif decision.action == "buy":
                # Log open buy orders so they appear immediately on the dashboard
                try:
                    self.record_open_order(symbol, decision.action, decision.qty, price, decision.reason)
                except Exception:
                    pass
            with self.positions_lock:
                delta = decision.qty if decision.action == "buy" else -decision.qty
                self.current_positions[symbol] += delta
                if decision.action == "buy":
                    self._last_known_qty[symbol] = self.current_positions[symbol]
                    self.entry_qty[symbol] = self.current_positions[symbol]
                else:
                    self._last_known_qty[symbol] = self.current_positions[symbol]
                    if self.current_positions[symbol] <= 0:
                        self.entry_qty[symbol] = 0
        except Exception as e:
            log.error(f"[{symbol}] Order failed: {e}")