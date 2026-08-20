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
passes in (as long as it's under trade_data/, which backtest.py and
trader_v3.py both do by default now).
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
from datetime import datetime, timezone, timedelta
from notifications import notify_order_placed
from typing import Optional, Dict

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
        self.symbols = symbols
        self.api = api_client
        self.bars = {s: deque(maxlen=self.MAX_BARS_KEPT) for s in symbols}
        self.current_bar = {s: None for s in symbols}
        self._vader = SentimentIntensityAnalyzer()
        self.news_sentiment = {s: 0.0 for s in symbols}
        self._news_lock = threading.Lock()
        self._bars_lock = threading.Lock()

        self._hmm_models = {}
        self._hmm_bars_since_fit = {}
        self._hmm_state_labels = {}
        self._hmm_last_result = {s: ("unavailable", 0.0) for s in symbols}

        # Increments each time a bar actually CLOSES (a new time bucket
        # starts), as opposed to every tick. Used by SignalLayer to tell
        # "a genuinely new bar closed" apart from "another tick arrived
        # within the same still-forming bar" — ticks now update predictions
        # live, but training/retraining still happens per real bar, not per tick.
        self.bar_close_count = {s: 0 for s in symbols}

    def update_bar(self, symbol, price, size, ts):
        with self._bars_lock:
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
            self.bars[symbol].append(bar_dict)
            self.bar_close_count[symbol] = self.bar_close_count.get(symbol, 0) + 1

    def get_bars(self, symbol, include_partial=False):
        """By default returns only COMPLETED bars (unchanged behavior — used
        for ATR/regime/HMM and for training data). Pass include_partial=True
        to also get the still-forming current bar appended as the last
        entry, reflecting the very latest tick — this is what lets live
        predictions actually update between bar closes instead of repeating
        a stale value until the next bar completes."""
        with self._bars_lock:
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
        for symbol in self.symbols:
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
        for symbol in self.symbols:
            try:
                articles = self.api.get_news(symbol, start=start.isoformat(), end=end.isoformat(),
                                              limit=self.NEWS_MAX_ARTICLES)
            except Exception as e:
                log.error(f"[{symbol}] News fetch failed: {e}")
                continue
            scores = []
            for a in (articles or []):
                text = f"{getattr(a, 'headline', '') or ''}. {getattr(a, 'summary', '') or ''}".strip()
                if text:
                    scores.append(self._vader.polarity_scores(text)["compound"])
            avg = float(np.mean(scores)) if scores else 0.0
            self.set_news_sentiment(symbol, avg)

    def set_news_sentiment(self, symbol, value):
        with self._news_lock:
            self.news_sentiment[symbol] = value

    def get_news_sentiment(self, symbol):
        with self._news_lock:
            return self.news_sentiment[symbol]

    def news_refresher_loop(self):
        while True:
            self.refresh_news_sentiment()
            time.sleep(self.NEWS_REFRESH_SECONDS)


# =============================================================================
# LAYER 2 — SignalLayer
# =============================================================================
class SignalLayer:
    MIN_BARS_FOR_MODEL = 40
    RETRAIN_EVERY_N_BARS = 5
    WEIGHT_LEARNING_RATE = 0.15

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

    @staticmethod
    def _features(bar_list):
        closes = np.array([b["c"] for b in bar_list], dtype=float)
        n = len(closes)
        sma_fast = SignalLayer._sma(closes, 5)
        sma_slow = SignalLayer._sma(closes, 15)
        ema_fast = SignalLayer._ema(closes, 5)
        rsi = SignalLayer._rsi(closes, 14)
        momentum = np.zeros(n)
        momentum[5:] = closes[5:] - closes[:-5]
        vol = np.array([closes[max(0, i - 10 + 1):i + 1].std() for i in range(n)])
        atr_series = MarketDataLayer._atr_series(bar_list, MarketDataLayer.ATR_PERIOD)
        atr_pct = np.divide(atr_series, closes, out=np.zeros_like(closes), where=closes != 0) * 100
        return np.column_stack([sma_fast - sma_slow, ema_fast - closes, rsi - 50, momentum, vol, atr_pct])

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
        X = self._features(bar_list)
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

        weights = self.weights.setdefault(symbol, {e: 1.0 / len(self.EXPERTS) for e in self.EXPERTS})
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
        X_live = self._features(live_bars)
        latest_X = X_live[-1]
        online_model = self.online_models.get(symbol)

        p_up = {
            "batch": self._p_up_from_classifier(batch_model, latest_X),
            "online": self._p_up_from_classifier(online_model, latest_X),
            "markov": self._markov_p_up(live_bars),
            "meanrev": self._meanrev_p_up(live_bars),
        }

        weights = self.weights.setdefault(symbol, {e: 1.0 / len(self.EXPERTS) for e in self.EXPERTS})
        blended_p_up = sum(weights[e] * p_up[e] for e in self.EXPERTS)

        direction = "up" if blended_p_up >= 0.5 else "down"
        confidence = blended_p_up if direction == "up" else (1 - blended_p_up)

        def conf_for(expert):
            p = p_up[expert]
            return p if direction == "up" else (1 - p)

        self.pending[symbol] = {"features": latest_X, "p_up": p_up, "bar_close_count": current_count}

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
    CONFIDENCE_THRESHOLD = 0.62
    TREND_AGREEMENT_REQUIRED = True
    NEWS_VETO_THRESHOLD = -0.35
    MAX_POSITION_PER_SYMBOL = 8
    MIN_SECONDS_BETWEEN_ORDERS = 30
    ATR_STOP_MULTIPLIER = 1.5
    ATR_TARGET_MULTIPLIER = 2.5
    MIN_STOP_PCT = 0.15
    MAX_STOP_PCT = 1.5
    DAILY_LOSS_LIMIT_PCT = 2.0
    MAX_QTY_PER_ORDER = 3

    KELLY_FRACTION_MULTIPLIER = 0.5
    MAX_KELLY_FRACTION = 0.05
    DEFAULT_EQUITY_ESTIMATE = 10_000.0

    def __init__(self, symbols, api_client=None, trade_log_path="trade_data/trade_log.csv"):
        self.symbols = symbols
        self.api = api_client
        self.positions_lock = threading.Lock()
        self.current_positions: Dict[str, int] = {s: 0 for s in symbols}
        self.entry_prices: Dict[str, Optional[float]] = {s: None for s in symbols}
        self.entry_stop_pct: Dict[str, Optional[float]] = {s: None for s in symbols}
        self.entry_target_pct: Dict[str, Optional[float]] = {s: None for s in symbols}
        self.entry_signal_context: Dict[str, Optional[TradeSignal]] = {s: None for s in symbols}
        self.last_order_time: Dict[str, float] = {s: 0.0 for s in symbols}
        self.trading_halted = False
        self.starting_equity = None
        self.state_lock = threading.Lock()

        self.trade_log_path = trade_log_path
        self._log_lock = threading.Lock()
        self._ensure_trade_log_header()

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
        else:
            row += [None] * 14

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
        else:
            row += [None] * 14

        with self._log_lock:
            with open(self.trade_log_path, "a", newline="") as f:
                csv.writer(f).writerow(row)

    def refresh_positions(self):
        if self.api is None:
            return
        try:
            held = {p.symbol: (int(float(p.qty)), float(p.avg_entry_price)) for p in self.api.list_positions()}
        except Exception as e:
            log.error(f"RiskGate: failed to refresh positions: {e}")
            return
        with self.positions_lock:
            for s in self.symbols:
                qty, avg_price = held.get(s, (0, None))
                self.current_positions[s] = qty
                self.entry_prices[s] = avg_price if qty > 0 else None
                if qty == 0:
                    self.entry_stop_pct[s] = None
                    self.entry_target_pct[s] = None
                    self.entry_signal_context[s] = None

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

    def background_sync_loop(self, interval=30):
        while True:
            self.refresh_positions()
            self.check_daily_loss_limit()
            time.sleep(interval)

    def _size_position(self, confidence, stop_pct, target_pct, price):
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

        qty = int(dollar_risk / stop_distance)
        return max(0, min(qty, self.MAX_QTY_PER_ORDER))

    def evaluate(self, signal: TradeSignal, current_price: float, now: Optional[float] = None) -> GateDecision:
        if now is None:
            now = time.time()
        symbol = signal.symbol

        with self.positions_lock:
            held = self.current_positions[symbol]
            entry = self.entry_prices[symbol]
            stop_pct = self.entry_stop_pct[symbol]
            target_pct = self.entry_target_pct[symbol]

        if held > 0 and entry and stop_pct is not None:
            change_from_entry = ((current_price - entry) / entry) * 100
            if change_from_entry <= -stop_pct:
                return self._approve(symbol, "sell", held, now, f"stop-loss hit ({stop_pct:.2f}%)")
            if change_from_entry >= target_pct:
                return self._approve(symbol, "sell", held, now, f"take-profit hit ({target_pct:.2f}%)")

        if self.trading_halted:
            return GateDecision(False, None, None, "daily loss limit halted trading")

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

            stop_pct = float(np.clip(signal.atr_pct * self.ATR_STOP_MULTIPLIER, self.MIN_STOP_PCT, self.MAX_STOP_PCT))
            target_pct = float(np.clip(signal.atr_pct * self.ATR_TARGET_MULTIPLIER, self.MIN_STOP_PCT, self.MAX_STOP_PCT * 2))
            qty = self._size_position(signal.confidence, stop_pct, target_pct, current_price)
            if qty <= 0:
                return GateDecision(False, None, None, "Kelly-sized position rounded to zero shares — edge too small")

            with self.positions_lock:
                self.entry_stop_pct[symbol] = stop_pct
                self.entry_target_pct[symbol] = target_pct
                self.entry_signal_context[symbol] = signal
            return self._approve(symbol, "buy", qty, now,
                                  f"signal conf={signal.confidence:.2f} qty={qty} (Kelly-sized) "
                                  f"experts=batch:{signal.batch_weight:.2f}/online:{signal.online_weight:.2f}/"
                                  f"markov:{signal.markov_weight:.2f}/meanrev:{signal.meanrev_weight:.2f} "
                                  f"regime={signal.trend_regime}/{signal.volatility_regime}/hmm:{signal.hmm_regime} "
                                  f"stop={stop_pct:.2f}% target={target_pct:.2f}%")

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
            self.api.submit_order(symbol=symbol, qty=decision.qty, side=decision.action,
                                   type="market", time_in_force="gtc")
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
        except Exception as e:
            log.error(f"[{symbol}] Order failed: {e}")