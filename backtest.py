"""
backtest.py — Walk-forward backtest using the ACTUAL three-layer system
from Tradingcore.py.

Dashboard features added:
    - Persistent backtest job state
    - Live progress: AAPL 3/8, etc.
    - Live elapsed / estimated remaining time
    - Live equity / return curve
    - Live drawdown curve
    - Live SPY comparison
    - Running / Complete / Failed status
    - Clear completion message
    - Automatic saving of every backtest
    - Historical backtest archive
    - Safe dashboard reconnect after leaving the page

All generated backtest files are stored under trade_data/.

The backtest itself is an independent Python process. The dashboard only
reads its state, so navigating away from the dashboard does NOT stop it.
"""

import os
import json
import socket
import time
import logging
import csv
import traceback
import uuid
import shutil
from datetime import datetime, timedelta, timezone
from notifications import notify_error
socket.setdefaulttimeout(20)

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from typing import Any, TYPE_CHECKING, cast

try:
    from typing import Any, TYPE_CHECKING, cast

    try:
        import alpaca_trade_api as tradeapi
        from alpaca_trade_api.rest import TimeFrame
    except Exception:
        tradeapi = None  # type: Any
        TimeFrame: Any = None

    # yfinance optional import
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        yf = None  # type: Any

except Exception:
    tradeapi = None  # type: Any
    TimeFrame: Any = None


from Tradingcore import (
    MarketDataLayer,
    SignalLayer,
    RiskGate,
    GateDecision,
)
from meta_filter import MetaModelFilter


# ============================================================================
# DIRECTORIES / FILES
# ============================================================================

DATA_DIR = "trade_data"
os.makedirs(DATA_DIR, exist_ok=True)

BACKTEST_RUNS_DIR = os.path.join(DATA_DIR, "backtests")
os.makedirs(BACKTEST_RUNS_DIR, exist_ok=True)

WEIGHTS_PATH = os.path.join(
    DATA_DIR,
    "expert_weights.json",
)

JOB_STATE_PATH = os.path.join(
    DATA_DIR,
    "backtest_job.json",
)

LIVE_DATA_PATH = os.path.join(
    DATA_DIR,
    "backtest_live.json",
)

LATEST_RESULTS_PATH = os.path.join(
    DATA_DIR,
    "backtest_results.csv",
)

LATEST_CHART_PATH = os.path.join(
    DATA_DIR,
    "backtest_vs_spy.png",
)


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.WARNING
)

log = logging.getLogger("backtest")


# ============================================================================
# ALPACA
# ============================================================================

API_KEY = os.environ.get(
    "ALPACA_API_KEY"
)

SECRET_KEY = os.environ.get(
    "ALPACA_SECRET_KEY"
)

BASE_URL = os.environ.get(
    "ALPACA_BASE_URL",
    "https://paper-api.alpaca.markets",
)

api = None

try:
    if API_KEY and SECRET_KEY and 'tradeapi' in globals() and tradeapi is not None:
        try:
            api = tradeapi.REST(
                API_KEY,
                SECRET_KEY,
                cast(Any, BASE_URL),
                api_version="v2",
            )
        except Exception as e:
            print(f"[autofix] Failed to initialize Alpaca REST client: {e}")
            api = None
    else:
        print(
            "[autofix] ALPACA credentials not found or Alpaca SDK unavailable; "
            "running in fallback mode (yfinance/synthetic)."
        )

except Exception as e:
    print(f"[autofix] Failed to initialize Alpaca API: {e}")
    api = None


# ============================================================================
# META MODEL
# ============================================================================

meta_filter = MetaModelFilter(
    os.path.join(
        DATA_DIR,
        "meta_model.joblib",
    )
)


# ============================================================================
# CONFIGURATION
# ============================================================================

SYMBOLS = [
    "AAPL",
    "TSLA",
    "MSFT",
    "NVDA",
    "JPM",
    "XOM",
    "JNJ",
    "SPY",
]

BENCHMARK_SYMBOL = "SPY"

WINDOW_LENGTH_DAYS = 30
NUM_WINDOWS = 4

ITERATIONS_PER_WINDOW = 4
BASE_RANDOM_SEED = 1

RETRAIN_EVERY_N_BARS = 20

STARTING_CASH_PER_SYMBOL = 10_000.0

SLEEP_BETWEEN_SYMBOLS = 2.0

COMMISSION_PER_SHARE = 0.005

SLIPPAGE_PCT = 0.02


# ============================================================================
# FAST MODE
# ============================================================================

FAST_BACKTEST = str(
    os.environ.get(
        "FAST_BACKTEST",
        "0",
    )
).lower() in (
    "1",
    "true",
    "yes",
    "y",
)

if FAST_BACKTEST:

    print(
        "[autofix] FAST_BACKTEST enabled: "
        "shortening backtest for CI/agents"
    )

    SYMBOLS = [
        "AAPL",
        "SPY",
    ]

    WINDOW_LENGTH_DAYS = 7
    NUM_WINDOWS = 1
    ITERATIONS_PER_WINDOW = 1
    RETRAIN_EVERY_N_BARS = 50
    STARTING_CASH_PER_SYMBOL = 5_000.0
    SLEEP_BETWEEN_SYMBOLS = 0.1


# ============================================================================
# JOB STATE
# ============================================================================

RUN_ID = os.environ.get(
    "BACKTEST_RUN_ID"
) or uuid.uuid4().hex[:12]

RUN_DIR = os.path.join(
    BACKTEST_RUNS_DIR,
    RUN_ID,
)

os.makedirs(
    RUN_DIR,
    exist_ok=True,
)


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json_write(path, data):

    tmp = path + ".tmp"

    with open(
        tmp,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            default=str,
        )

    os.replace(
        tmp,
        path,
    )


def update_job(
    status=None,
    current_symbol=None,
    symbol_index=None,
    total_symbols=None,
    window_index=None,
    total_windows=None,
    iteration=None,
    total_iterations=None,
    progress=None,
    elapsed_seconds=None,
    estimated_seconds=None,
    eta_seconds=None,
    message=None,
    error=None,
    completed=False,
):

    state = {}

    if os.path.exists(JOB_STATE_PATH):

        try:

            with open(
                JOB_STATE_PATH,
                "r",
                encoding="utf-8",
            ) as f:

                state = json.load(f)

        except Exception:

            state = {}

    state.update({

        "run_id": RUN_ID,

        "status": (
            status
            if status is not None
            else state.get("status", "running")
        ),

        "current_symbol": (
            current_symbol
            if current_symbol is not None
            else state.get("current_symbol")
        ),

        "symbol_index": (
            symbol_index
            if symbol_index is not None
            else state.get("symbol_index", 0)
        ),

        "total_symbols": (
            total_symbols
            if total_symbols is not None
            else state.get(
                "total_symbols",
                len(SYMBOLS),
            )
        ),

        "window_index": (
            window_index
            if window_index is not None
            else state.get("window_index", 1)
        ),

        "total_windows": (
            total_windows
            if total_windows is not None
            else state.get(
                "total_windows",
                NUM_WINDOWS,
            )
        ),

        "iteration": (
            iteration
            if iteration is not None
            else state.get("iteration", 0)
        ),

        "total_iterations": (
            total_iterations
            if total_iterations is not None
            else state.get(
                "total_iterations",
                ITERATIONS_PER_WINDOW,
            )
        ),

        "progress": (
            float(progress)
            if progress is not None
            else float(
                state.get(
                    "progress",
                    0,
                )
            )
        ),

        "elapsed_seconds": (
            float(elapsed_seconds)
            if elapsed_seconds is not None
            else float(
                state.get(
                    "elapsed_seconds",
                    0,
                )
            )
        ),

        "estimated_seconds": (
            float(estimated_seconds)
            if estimated_seconds is not None
            else float(
                state.get(
                    "estimated_seconds",
                    0,
                )
            )
        ),

        "eta_seconds": (
            float(eta_seconds)
            if eta_seconds is not None
            else float(
                state.get(
                    "eta_seconds",
                    0,
                )
            )
        ),

        "message": (
            message
            if message is not None
            else state.get("message", "")
        ),

        "error": (
            error
            if error is not None
            else state.get("error")
        ),

        "completed": (
            bool(completed)
            if completed is not None
            else bool(
                state.get(
                    "completed",
                    False,
                )
            )
        ),

        "updated_at": utc_now(),

    })

    atomic_json_write(
        JOB_STATE_PATH,
        state,
    )

    atomic_json_write(
        os.path.join(
            RUN_DIR,
            "job.json",
        ),
        state,
    )


# ============================================================================
# LIVE DATA
# ============================================================================

def write_live_data(
    equity_curve=None,
    timestamps=None,
    spy_curve=None,
    drawdown_curve=None,
    symbol=None,
    symbol_progress=None,
):

    payload = {

        "run_id": RUN_ID,

        "symbol": symbol,

        "symbol_progress": symbol_progress,

        "timestamps": (
            timestamps
            if timestamps is not None
            else []
        ),

        "equity": (
            equity_curve
            if equity_curve is not None
            else []
        ),

        "return_pct": [],

        "drawdown_pct": (
            drawdown_curve
            if drawdown_curve is not None
            else []
        ),

        "spy_return_pct": (
            spy_curve
            if spy_curve is not None
            else []
        ),

        "updated_at": utc_now(),

    }

    equity = payload["equity"]

    if equity:

        starting = float(
            equity[0]
        )

        if starting:

            payload["return_pct"] = [
                (
                    (
                        float(x) /
                        starting
                    ) - 1
                ) * 100
                for x in equity
            ]

    atomic_json_write(
        LIVE_DATA_PATH,
        payload,
    )

    atomic_json_write(
        os.path.join(
            RUN_DIR,
            "live.json",
        ),
        payload,
    )


# ============================================================================
# PROGRESS ESTIMATION
# ============================================================================

JOB_STARTED_AT = time.time()


def update_progress(
    completed_units,
    total_units,
    current_symbol,
    symbol_index,
    iteration,
    message,
):

    elapsed = (
        time.time()
        - JOB_STARTED_AT
    )

    if completed_units > 0:

        seconds_per_unit = (
            elapsed /
            completed_units
        )

        estimated = (
            seconds_per_unit *
            total_units
        )

        remaining = max(
            0,
            estimated - elapsed,
        )

    else:

        estimated = 0
        remaining = 0

    progress = (
        completed_units /
        total_units *
        100
        if total_units
        else 0
    )

    update_job(
        status="running",
        current_symbol=current_symbol,
        symbol_index=symbol_index,
        iteration=iteration,
        progress=progress,
        elapsed_seconds=elapsed,
        estimated_seconds=estimated,
        eta_seconds=remaining,
        message=message,
        completed=False,
    )


# ============================================================================
# RISK GATE
# ============================================================================

class BacktestRiskGate(RiskGate):

    def __init__(
        self,
        symbols,
        trade_log_path,
    ):

        super().__init__(
            symbols,
            api_client=None,
            trade_log_path=trade_log_path,
        )

        self.cash = {
            s: STARTING_CASH_PER_SYMBOL
            for s in symbols
        }

        self.trades = {
            s: []
            for s in symbols
        }

        self.starting_equity = (
            STARTING_CASH_PER_SYMBOL
        )

        self.entry_cost_per_share = {
            s: None
            for s in symbols
        }


    def refresh_positions(self):
        pass


    def check_daily_loss_limit(self):
        pass


    def execute(
        self,
        decision: GateDecision,
        symbol: str,
        price: float,
        bar_index=None,
    ):

        if not decision.approved:
            return

        # Normalize qty to an integer to avoid Optional[int] arithmetic warnings
        qty = int(decision.qty or 0)

        slippage_frac = (
            SLIPPAGE_PCT /
            100.0
        )

        if decision.action == "buy":

            eff_price = (
                price *
                (
                    1 +
                    slippage_frac
                )
            )

            commission = (
                COMMISSION_PER_SHARE *
                qty
            )

            cost = (
                qty *
                eff_price
                +
                commission
            )

            if self.cash[symbol] < cost:
                return

            self.cash[symbol] -= cost

            with self.positions_lock:

                self.current_positions[
                    symbol
                ] += qty

                self.entry_prices[
                    symbol
                ] = eff_price

                self.entry_cost_per_share[
                    symbol
                ] = (
                    eff_price +
                    COMMISSION_PER_SHARE
                )


        elif decision.action == "sell":

            eff_price = (
                price *
                (
                    1 -
                    slippage_frac
                )
            )

            commission = (
                COMMISSION_PER_SHARE *
                            qty
            )

            proceeds = (
                            qty *
                eff_price
                -
                commission
            )

            self.cash[symbol] += proceeds

            entry_cost = (
                self.entry_cost_per_share.get(
                    symbol
                )
                or
                self.entry_prices.get(
                    symbol
                )
            )

            if entry_cost:

                net_exit_per_share = (
                    eff_price -
                    COMMISSION_PER_SHARE
                )

                pnl_pct = (
                    (
                        net_exit_per_share -
                        entry_cost
                    )
                    /
                    entry_cost
                ) * 100

            else:

                pnl_pct = 0.0

            self.trades[symbol].append({

                "pnl_pct": pnl_pct,

                "price": eff_price,

                "bar": bar_index,

                "reason": decision.reason,

            })

            self.record_trade_outcome(
                symbol,
                eff_price,
                decision.reason,
            )

            with self.positions_lock:

                self.current_positions[
                    symbol
                ] = max(
                    0,
                    self.current_positions[
                        symbol
                                        ] - qty,
                )

                if (
                    self.current_positions[
                        symbol
                    ] == 0
                ):

                    self.entry_prices[
                        symbol
                    ] = None

                    self.entry_cost_per_share[
                        symbol
                    ] = None

                    self.entry_stop_pct[
                        symbol
                    ] = None

                    self.entry_target_pct[
                        symbol
                    ] = None

                    self.entry_signal_context[
                        symbol
                    ] = None


    def equity(
        self,
        symbol,
        current_price,
    ):

        return (
            self.cash[symbol]
            +
            self.current_positions[
                symbol
            ] *
            current_price
        )


    def force_close(
        self,
        symbol,
        price,
        bar_index,
    ):

        held = self.current_positions[
            symbol
        ]

        if held > 0:

            decision = GateDecision(
                True,
                "sell",
                held,
                "end-of-backtest close",
            )

            self.execute(
                decision,
                symbol,
                price,
                bar_index,
            )


    def record_trade_outcome(
        self,
        symbol,
        exit_price,
        exit_reason,
    ):

        entry_cost = (
            self.entry_cost_per_share.get(
                symbol
            )
            or
            self.entry_prices.get(
                symbol
            )
        )

        if entry_cost is None:
            return

        net_exit_per_share = (
            exit_price -
            COMMISSION_PER_SHARE
        )

        pnl_pct = (
            (
                net_exit_per_share -
                entry_cost
            )
            /
            entry_cost
        ) * 100

        signal = (
            self.entry_signal_context[
                symbol
            ]
        )

        if signal is not None:

            row = [

                datetime.now().isoformat(),

                symbol,

                round(
                    entry_cost,
                    6,
                ),

                round(
                    exit_price,
                    6,
                ),

                round(
                    pnl_pct,
                    4,
                ),

                exit_reason,

                round(
                    signal.confidence,
                    4,
                ),

                signal.direction,

                round(
                    signal.trend_change_pct,
                    4,
                ),

                signal.volatility_regime,

                signal.trend_regime,

                signal.hmm_regime,

                round(
                    signal.hmm_regime_prob,
                    4,
                ),

                round(
                    signal.atr_pct,
                    4,
                ),

                round(
                    signal.news_sentiment,
                    4,
                ),

                round(
                    signal.batch_weight,
                    4,
                ),

                round(
                    signal.online_weight,
                    4,
                ),

                round(
                    signal.markov_weight,
                    4,
                ),

                round(
                    signal.meanrev_weight,
                    4,
                ),

                round(
                    signal.meta_win_prob,
                    4,
                ),

            ]

        else:

            row = [

                datetime.now().isoformat(),

                symbol,

                round(
                    entry_cost,
                    6,
                ),

                round(
                    exit_price,
                    6,
                ),

                round(
                    pnl_pct,
                    4,
                ),

                exit_reason,

            ] + [None] * 14

        with self._log_lock:

            with open(
                self.trade_log_path,
                "a",
                newline="",
            ) as f:

                csv.writer(f).writerow(
                    row
                )


# ============================================================================
# WEIGHTS
# ============================================================================

def load_persisted_weights(
    weights_path,
):

    if not os.path.exists(
        weights_path
    ):

        return {}

    try:

        with open(
            weights_path,
            "r",
        ) as f:

            return json.load(f)

    except Exception as e:

        print(
            f" Couldn't read persisted weights "
            f"from {weights_path}, starting uniform: {e}"
        )

        return {}


# ============================================================================
# MARKET DATA
# ============================================================================

def fetch_bars(
    symbol,
    days,
    end_offset_days=0,
    max_retries=3,
):

    end = (
        datetime.now()
        -
        timedelta(minutes=16)
        -
        timedelta(days=end_offset_days)
    )

    start = (
        end -
        timedelta(days=days)
    )

    if api is None:

        try:
                        if yf is None:
                            raise ImportError('yfinance not installed')

                        try:
                            df = yf.download(
                                symbol,
                                start=start.strftime("%Y-%m-%d"),
                                end=end.strftime("%Y-%m-%d"),
                                interval="1m",
                                progress=False,
                            )
                        except Exception:
                            df = pd.DataFrame()

                        if not df.empty:
                            if isinstance(df.columns, pd.MultiIndex):
                                df.columns = [c[0] for c in df.columns]

                            cols = {c: c.lower() for c in df.columns}
                            df = df.rename(columns=cols)

                            required = ["open", "high", "low", "close", "volume"]
                            if all(c in df.columns for c in required):
                                df.index = pd.DatetimeIndex(df.index)
                                return df[required]

                        print(f"  yfinance returned no minute bars for {symbol}; falling back to synthetic data.")

        except ImportError:
                        print("  yfinance not installed; falling back to synthetic data.")

        except Exception as e:
                        print(f"  yfinance fetch failed for {symbol}: {e}; falling back to synthetic data.")

        rng = pd.date_range(
            start=start,
            end=end,
            freq="1min",
        )

        n = len(rng)

        if n <= 0:

            raise ValueError(
                "Requested zero-length period "
                "for fetch_bars"
            )

        base = 100.0

        prices = (
            base +
            np.arange(n) *
            0.01
        )

        return pd.DataFrame(
            {
                "open": prices,
                "high": prices,
                "low": prices,
                "close": prices,
                "volume": 1,
            },
            index=rng,
        )

    last_error = None

    for attempt in range(
        1,
        max_retries + 1,
    ):

        try:

            df = api.get_bars(
                symbol,
                TimeFrame.Minute,
                start.strftime(
                    "%Y-%m-%d"
                ),
                end.strftime(
                    "%Y-%m-%d"
                ),
                adjustment="raw",
                feed="iex",
            ).df

            if df.empty:

                raise ValueError(
                    f"No bar data for {symbol}."
                )

            return df

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  {symbol}: bars fetch failed ({e.__class__.__name__}), retrying in {wait}s...")
                notify_error(f"Backtest failed (Run ID: {RUN_ID})", e)
                time.sleep(wait)

    if last_error is not None:
        raise last_error
    else:
        raise RuntimeError("fetch_bars failed without capturing an exception")


# ============================================================================
# SENTIMENT
# ============================================================================

def fetch_daily_sentiment(
    symbol,
    start_date,
    end_date,
    max_retries=3,
):

    from vaderSentiment.vaderSentiment import (
        SentimentIntensityAnalyzer,
    )

    vader = (
        SentimentIntensityAnalyzer()
    )

    last_error = None
    articles = []

    if api is None:

        return {}

    for attempt in range(
        1,
        max_retries + 1,
    ):

        try:

            articles = api.get_news(
                symbol,
                start=start_date.strftime(
                    "%Y-%m-%d"
                ),
                end=end_date.strftime(
                    "%Y-%m-%d"
                ),
                limit=1000,
            )

            break

        except Exception as e:

            last_error = e

            if attempt < max_retries:

                wait = 2 ** attempt

                print(
                    f"  {symbol}: news fetch failed "
                    f"({e.__class__.__name__}), "
                    f"retrying in {wait}s..."
                )

                time.sleep(
                    wait
                )

            else:

                print(
                    f"  {symbol}: news fetch failed "
                    f"after {max_retries} attempts, "
                    f"proceeding with neutral sentiment "
                    f"({last_error})."
                )

                return {}

    day_buckets = {}

    for article in (
        articles or []
    ):

        text = (
            f"{getattr(article, 'headline', '') or ''}. "
            f"{getattr(article, 'summary', '') or ''}"
        ).strip()

        created = getattr(
            article,
            "created_at",
            None,
        )

        if not text or created is None:
            continue

        day = (
            created.date()
            if hasattr(
                created,
                "date",
            )
            else pd.Timestamp(
                created
            ).date()
        )

        day_buckets.setdefault(
            day,
            [],
        ).append(
            vader.polarity_scores(
                text
            )["compound"]
        )

    return {
        day: float(
            np.mean(scores)
        )
        for day, scores
        in day_buckets.items()
    }


# ============================================================================
# DRAWDOWN
# ============================================================================

def calculate_drawdown(
    equity_curve,
):

    if not equity_curve:
        return []

    equity = np.asarray(
        equity_curve,
        dtype=float,
    )

    running_max = np.maximum.accumulate(
        equity
    )

    drawdown = (
        (
            equity -
            running_max
        )
        /
        running_max
    ) * 100

    return drawdown.tolist()


# ============================================================================
# LIVE SPY
# ============================================================================

def build_live_spy_curve(
    timestamps,
    symbol,
    df,
    portfolio_start,
):

    if not timestamps or not portfolio_start:
        return []

    try:

        spy_df = fetch_bars(
            BENCHMARK_SYMBOL,
            WINDOW_LENGTH_DAYS,
        )

        spy = (
            spy_df["close"]
            .astype(float)
        )

        spy.index = pd.DatetimeIndex(
            spy_df.index
        )

        target_index = pd.DatetimeIndex(
            timestamps
        )

        aligned = (
            spy.reindex(
                spy.index.union(
                    target_index
                )
            )
            .ffill()
            .bfill()
            .reindex(
                target_index
            )
        )

        if aligned.empty:
            return []

        first = float(
            aligned.iloc[0]
        )

        if not first:
            return []

        return (
            (
                aligned /
                first
                - 1
            ) * 100
        ).tolist()

    except Exception:

        return []


# ============================================================================
# SINGLE SYMBOL SIMULATION
# ============================================================================

def simulate_symbol(
    symbol,
    df,
    daily_sentiment,
    random_state,
    trade_log_path,
    initial_weights=None,
    persist_weights_path=None,
    global_completed_units=0,
    total_units=1,
    symbol_index=1,
    iteration=1,
):

    data_layer = MarketDataLayer(
        [symbol]
    )

    # Seed numpy and python random for deterministic iterations when a random
    # seed is provided. This reduces non-deterministic failures during
    # repeated backtests and makes 'pretest' behavior reproducible.
    try:
        import random as _random_mod
        np.random.seed(int(random_state))
        _random_mod.seed(int(random_state))
    except Exception:
        pass

    signal_layer = SignalLayer(
        data_layer,
        retrain_every_n_bars=(
            RETRAIN_EVERY_N_BARS
        ),
        random_state=random_state,
        meta_filter=meta_filter,
    )

    if initial_weights:

        signal_layer.weights[
            symbol
        ] = dict(
            initial_weights
        )

    risk_gate = BacktestRiskGate(
        [symbol],
        trade_log_path=trade_log_path,
    )

    closes = (
        df["close"]
        .values
        .astype(float)
    )

    highs = (
        df["high"]
        .values
        .astype(float)
    )

    lows = (
        df["low"]
        .values
        .astype(float)
    )

    dates = (
        df.index
        .to_pydatetime()
    )

    n = len(closes)

    equity_curve = []

    timestamps = []

    last_live_write = 0

    for i in range(n):

        bar = {

            "t": i,

            "o": closes[i],

            "h": highs[i],

            "l": lows[i],

            "c": closes[i],

            "v": 1,

        }

        data_layer.seed_bar(
            symbol,
            bar,
        )

        bar_date = (
            dates[i].date()
        )

        sentiment = (
            daily_sentiment.get(
                bar_date,
                0.0,
            )
        )

        data_layer.set_news_sentiment(
            symbol,
            sentiment,
        )

        current_price = closes[i]

        equity_curve.append(
            risk_gate.equity(
                symbol,
                current_price,
            )
        )

        timestamps.append(
            dates[i].isoformat()
        )

        signal = (
            signal_layer.generate(
                symbol
            )
        )

        if signal is not None:

            bar_time = (
                dates[i].timestamp()
            )

            decision = (
                risk_gate.evaluate(
                    signal,
                    current_price,
                    now=bar_time,
                )
            )

            risk_gate.execute(
                decision,
                symbol,
                current_price,
                bar_index=i,
            )

        now = time.time()

        if (
            now -
            last_live_write
        ) >= 0.75 or i == n - 1:

            last_live_write = now

            completed_symbol_fraction = (
                i + 1
            ) / max(
                1,
                n,
            )

            completed_units = (
                global_completed_units
                +
                completed_symbol_fraction
            )

            update_progress(
                completed_units,
                total_units,
                symbol,
                symbol_index,
                iteration,
                f"{symbol} "
                f"{i + 1}/{n}",
            )

            write_live_data(
                equity_curve=equity_curve,
                timestamps=timestamps,
                drawdown_curve=calculate_drawdown(
                    equity_curve
                ),
                symbol=symbol,
                symbol_progress={
                    "current": i + 1,
                    "total": n,
                },
            )

    risk_gate.force_close(
        symbol,
        closes[-1],
        n - 1,
    )

    equity_curve.append(
        risk_gate.equity(
            symbol,
            closes[-1],
        )
    )

    timestamps.append(
        dates[-1].isoformat()
    )

    if persist_weights_path:

        signal_layer.weights_path = (
            persist_weights_path
        )

        signal_layer.save_weights()

    return (
        equity_curve,
        timestamps,
        risk_gate.trades[symbol],
    )


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(
    equity_curve,
    trades,
):

    equity = np.array(
        equity_curve
    )

    total_return_pct = (
        (
            equity[-1] -
            STARTING_CASH_PER_SYMBOL
        )
        /
        STARTING_CASH_PER_SYMBOL
    ) * 100

    running_max = (
        np.maximum.accumulate(
            equity
        )
    )

    drawdowns = (
        (
            equity -
            running_max
        )
        /
        running_max
    ) * 100

    max_drawdown_pct = (
        drawdowns.min()
    )

    wins = [
        t
        for t in trades
        if t["pnl_pct"] > 0
    ]

    losses = [
        t
        for t in trades
        if t["pnl_pct"] <= 0
    ]

    win_rate = (
        len(wins) /
        len(trades) *
        100
        if trades
        else 0.0
    )

    avg_win = (
        np.mean(
            [
                t["pnl_pct"]
                for t in wins
            ]
        )
        if wins
        else 0.0
    )

    avg_loss = (
        np.mean(
            [
                t["pnl_pct"]
                for t in losses
            ]
        )
        if losses
        else 0.0
    )

    return {

        "total_return_pct":
            total_return_pct,

        "max_drawdown_pct":
            max_drawdown_pct,

        "num_trades":
            len(trades),

        "win_rate":
            win_rate,

        "avg_win_pct":
            avg_win,

        "avg_loss_pct":
            avg_loss,

    }


# ============================================================================
# ITERATIONS
# ============================================================================

def simulate_symbol_iterations(
    symbol,
    df,
    daily_sentiment,
    trade_log_path,
    global_completed_units,
    total_units,
    symbol_index,
):

    initial_weights = (
        load_persisted_weights(
            WEIGHTS_PATH
        ).get(symbol)
    )

    iter_metrics = []
    equity_runs = []

    first_live_curve = None
    first_live_timestamps = None

    for it in range(
        ITERATIONS_PER_WINDOW
    ):

        seed = (
            BASE_RANDOM_SEED +
            it
        )

        persist_path = (
            WEIGHTS_PATH
            if it == 0
            else None
        )

        equity_curve, timestamps, trades = (
            simulate_symbol(
                symbol,
                df,
                daily_sentiment,
                seed,
                trade_log_path,
                initial_weights=initial_weights,
                persist_weights_path=persist_path,
                global_completed_units=(
                    global_completed_units +
                    it
                ),
                total_units=total_units,
                symbol_index=symbol_index,
                iteration=it + 1,
            )
        )

        metrics = compute_metrics(
            equity_curve,
            trades,
        )

        iter_metrics.append(
            metrics
        )

        equity_runs.append(
            equity_curve
        )

        if it == 0:

            first_live_curve = (
                equity_curve
            )

            first_live_timestamps = (
                timestamps
            )

    return (
        iter_metrics,
        equity_runs,
        first_live_curve,
        first_live_timestamps,
    )


# ============================================================================
# AGGREGATION
# ============================================================================

def aggregate_metrics(
    symbol,
    iter_metrics,
):

    keys = [
        "total_return_pct",
        "max_drawdown_pct",
        "num_trades",
        "win_rate",
    ]

    agg = {
        "symbol": symbol
    }

    for key in keys:

        vals = [
            m[key]
            for m in iter_metrics
        ]

        agg[
            f"{key}_mean"
        ] = float(
            np.mean(vals)
        )

        agg[
            f"{key}_std"
        ] = float(
            np.std(vals)
        )

    return agg


def print_symbol_summary(
    agg,
):

    print(
        f"\n=== {agg['symbol']} "
        f"(avg of "
        f"{ITERATIONS_PER_WINDOW} runs) ==="
    )

    print(
        f"Total return:     "
        f"{agg['total_return_pct_mean']:+.2f}%  "
        f"(std {agg['total_return_pct_std']:.2f})"
    )

    print(
        f"Max drawdown:     "
        f"{agg['max_drawdown_pct_mean']:.2f}%  "
        f"(std {agg['max_drawdown_pct_std']:.2f})"
    )

    print(
        f"Trades:           "
        f"{agg['num_trades_mean']:.0f}  "
        f"(std {agg['num_trades_std']:.1f})"
    )

    print(
        f"Win rate:         "
        f"{agg['win_rate_mean']:.1f}%  "
        f"(std {agg['win_rate_std']:.1f})"
    )


# ============================================================================
# PORTFOLIO
# ============================================================================

def build_portfolio_series(
    symbol_avg_equity,
    symbol_dates,
):

    per_symbol_series = {}

    for symbol, avg_curve in (
        symbol_avg_equity.items()
    ):

        dates = symbol_dates[
            symbol
        ]

        n = min(
            len(avg_curve),
            len(dates),
        )

        per_symbol_series[
            symbol
        ] = pd.Series(
            avg_curve[:n],
            index=pd.DatetimeIndex(
                dates[:n]
            ),
        ).sort_index()

    union_index = None

    for series in (
        per_symbol_series.values()
    ):

        union_index = (
            series.index
            if union_index is None
            else
            union_index.union(
                series.index
            )
        )

    aligned = pd.DataFrame({

        symbol:
            series.reindex(
                union_index
            ).ffill().bfill()

        for symbol, series
        in per_symbol_series.items()

    })

    return aligned.sum(
        axis=1
    )


# ============================================================================
# BENCHMARK
# ============================================================================

def fetch_benchmark_series(
    days,
    end_offset_days,
):

    df = fetch_bars(
        BENCHMARK_SYMBOL,
        days,
        end_offset_days=end_offset_days,
    )

    closes = (
        df["close"]
        .astype(float)
    )

    closes.index = (
        pd.DatetimeIndex(
            df.index
            .to_pydatetime()
        )
    )

    return closes.sort_index()


# ============================================================================
# CHART
# ============================================================================

def plot_windows(
    window_charts,
    out_path=None,
):

    if out_path is None:

        out_path = (
            LATEST_CHART_PATH
        )

    if not window_charts:
        return

    n = len(
        window_charts
    )

    fig, axes = plt.subplots(
        n,
        1,
        figsize=(
            11,
            3.2 * n,
        ),
        squeeze=False,
    )

    for ax, (
        label,
        portfolio_pct,
        spy_pct,
    ) in zip(
        axes[:, 0],
        window_charts,
    ):

        ax.plot(
            portfolio_pct.index,
            portfolio_pct.values,
            label=(
                "Strategy "
                f"(avg of "
                f"{ITERATIONS_PER_WINDOW} runs)"
            ),
            color="#1f77b4",
            linewidth=1.4,
        )

        ax.plot(
            spy_pct.index,
            spy_pct.values,
            label=(
                "S&P 500 "
                "(SPY, buy & hold)"
            ),
            color="#888888",
            linewidth=1.2,
            linestyle="--",
        )

        ax.axhline(
            0,
            color="#dddddd",
            linewidth=0.8,
            zorder=0,
        )

        ax.set_title(
            label,
            fontsize=10,
        )

        ax.set_ylabel(
            "Return (%)"
        )

        ax.legend(
            loc="upper left",
            fontsize=8,
        )

        ax.tick_params(
            axis="x",
            labelsize=8,
        )

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=140,
    )

    plt.close(
        fig
    )

    print(
        f"\nSaved comparison chart to "
        f"{out_path}"
    )


# ============================================================================
# WINDOWS
# ============================================================================

def build_windows():

    windows = []

    for i in range(
        NUM_WINDOWS
    ):

        end_offset_days = (
            i *
            WINDOW_LENGTH_DAYS
        )

        approx_end = (
            datetime.now()
            -
            timedelta(
                days=end_offset_days
            )
        )

        approx_start = (
            approx_end
            -
            timedelta(
                days=WINDOW_LENGTH_DAYS
            )
        )

        label = (
            f"Window {i + 1}: "
            f"{approx_start:%Y-%m-%d} "
            f"to "
            f"{approx_end:%Y-%m-%d}"
        )

        windows.append(
            (
                label,
                end_offset_days,
            )
        )

    return windows


# ============================================================================
# SAVE COMPLETED RUN
# ============================================================================

def save_completed_run(
    all_window_results,
    window_charts,
):

    results_path = os.path.join(
        RUN_DIR,
        "backtest_results.csv",
    )

    pd.DataFrame(
        all_window_results
    ).to_csv(
        results_path,
        index=False,
    )

    chart_path = os.path.join(
        RUN_DIR,
        "backtest_vs_spy.png",
    )

    plot_windows(
        window_charts,
        chart_path,
    )

    shutil.copy2(
        results_path,
        LATEST_RESULTS_PATH,
    )

    if os.path.exists(
        chart_path
    ):

        shutil.copy2(
            chart_path,
            LATEST_CHART_PATH,
        )

    summary = {

        "run_id":
            RUN_ID,

        "completed_at":
            utc_now(),

        "status":
            "complete",

        "results_file":
            results_path,

        "chart_file":
            chart_path,

        "windows":
            all_window_results,

    }

    atomic_json_write(
        os.path.join(
            RUN_DIR,
            "summary.json",
        ),
        summary,
    )

    return (
        results_path,
        chart_path,
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    total_units = (
        NUM_WINDOWS *
        len(SYMBOLS) *
        ITERATIONS_PER_WINDOW
    )

    update_job(
        status="running",
        symbol_index=0,
        total_symbols=len(
            SYMBOLS
        ),
        window_index=1,
        total_windows=NUM_WINDOWS,
        iteration=0,
        total_iterations=(
            ITERATIONS_PER_WINDOW
        ),
        progress=0,
        elapsed_seconds=0,
        estimated_seconds=0,
        eta_seconds=0,
        message="Backtest starting...",
        completed=False,
    )

    windows = build_windows()

    all_window_results = []

    window_charts = []

    completed_units = 0

    try:

        for window_number, (
            window_label,
            end_offset_days,
        ) in enumerate(
            windows,
            start=1,
        ):

            print(
                f"\n{'#' * 60}\n"
                f"# {window_label}\n"
                f"{'#' * 60}"
            )

            update_job(
                status="running",
                window_index=window_number,
                message=(
                    f"Starting {window_label}"
                ),
            )

            symbol_avg_equity = {}

            symbol_dates = {}

            for symbol_number, symbol in enumerate(
                SYMBOLS,
                start=1,
            ):

                progress_message = (
                    f"{symbol} "
                    f"1/{len(SYMBOLS)}"
                )

                update_progress(
                    completed_units,
                    total_units,
                    symbol,
                    symbol_number,
                    1,
                    progress_message,
                )

                print(
                    f"Fetching bars for "
                    f"{symbol}..."
                )

                df = fetch_bars(
                    symbol,
                    WINDOW_LENGTH_DAYS,
                    end_offset_days=(
                        end_offset_days
                    ),
                )

                end_date = (
                    datetime.now()
                    -
                    timedelta(
                        days=end_offset_days
                    )
                )

                start_date = (
                    end_date
                    -
                    timedelta(
                        days=WINDOW_LENGTH_DAYS
                    )
                )

                print(
                    f"Fetching news sentiment "
                    f"for {symbol}..."
                )

                daily_sentiment = (
                    fetch_daily_sentiment(
                        symbol,
                        start_date,
                        end_date,
                    )
                )

                trade_log_path = os.path.join(
                    RUN_DIR,
                    f"backtest_trade_log_"
                    f"{symbol}_"
                    f"{end_offset_days}.csv",
                )

                print(
                    f"Running "
                    f"{ITERATIONS_PER_WINDOW} "
                    f"iterations for {symbol}..."
                )

                (
                    iter_metrics,
                    equity_runs,
                    live_curve,
                    live_timestamps,
                ) = simulate_symbol_iterations(
                    symbol,
                    df,
                    daily_sentiment,
                    trade_log_path,
                    completed_units,
                    total_units,
                    symbol_number,
                )

                agg = aggregate_metrics(
                    symbol,
                    iter_metrics,
                )

                agg["window"] = (
                    window_label
                )

                all_window_results.append(
                    agg
                )

                print_symbol_summary(
                    agg
                )

                symbol_avg_equity[
                    symbol
                ] = np.mean(
                    equity_runs,
                    axis=0,
                )

                symbol_dates[
                    symbol
                ] = df.index.to_pydatetime()

                completed_units += (
                    ITERATIONS_PER_WINDOW
                )

                update_progress(
                    completed_units,
                    total_units,
                    symbol,
                    symbol_number,
                    ITERATIONS_PER_WINDOW,
                    (
                        f"{symbol} complete "
                        f"— "
                        f"{symbol_number}/"
                        f"{len(SYMBOLS)}"
                    ),
                )

                write_live_data(
                    equity_curve=live_curve,
                    timestamps=live_timestamps,
                    drawdown_curve=calculate_drawdown(
                        live_curve
                    ),
                    symbol=symbol,
                    symbol_progress={
                        "current":
                            len(live_curve)
                            if live_curve
                            else 0,

                        "total":
                            len(df),
                    },
                )

                time.sleep(
                    SLEEP_BETWEEN_SYMBOLS
                )

            rows = [
                r
                for r in all_window_results
                if r["window"]
                == window_label
            ]

            avg_return = np.mean(
                [
                    r[
                        "total_return_pct_mean"
                    ]
                    for r in rows
                ]
            )

            total_trades = sum(
                r[
                    "num_trades_mean"
                ]
                for r in rows
            )

            print(
                f"\n=== {window_label} "
                f"Combined "
                f"(avg of "
                f"{ITERATIONS_PER_WINDOW} "
                f"runs each) ==="
            )

            print(
                f"Average return across "
                f"symbols: "
                f"{avg_return:+.2f}%"
            )

            print(
                f"Total trades across "
                f"symbols: "
                f"{total_trades:.0f}"
            )

            portfolio = (
                build_portfolio_series(
                    symbol_avg_equity,
                    symbol_dates,
                )
            )

            portfolio_pct = (
                (
                    portfolio /
                    portfolio.iloc[0]
                    - 1
                ) * 100
            )

            try:

                spy_closes = (
                    fetch_benchmark_series(
                        WINDOW_LENGTH_DAYS,
                        end_offset_days,
                    )
                )

                spy_aligned = (
                    spy_closes
                    .reindex(
                        spy_closes.index.union(
                            portfolio_pct.index
                        )
                    )
                    .ffill()
                    .bfill()
                    .reindex(
                        portfolio_pct.index
                    )
                )

                spy_pct = (
                    (
                        spy_aligned /
                        spy_aligned.iloc[0]
                        - 1
                    ) * 100
                )

                window_charts.append(
                    (
                        window_label,
                        portfolio_pct,
                        spy_pct,
                    )
                )

                write_live_data(
                    equity_curve=(
                        portfolio.tolist()
                    ),
                    timestamps=[
                        x.isoformat()
                        for x in portfolio.index
                    ],
                    drawdown_curve=calculate_drawdown(
                        portfolio.tolist()
                    ),
                    spy_curve=(
                        spy_pct.tolist()
                    ),
                    symbol="PORTFOLIO",
                    symbol_progress={
                        "current":
                            len(portfolio),

                        "total":
                            len(portfolio),
                    },
                )

            except Exception as e:

                print(
                    "  Benchmark fetch failed "
                    f"for this window, "
                    f"skipping it in the chart "
                    f"({e})."
                )

        # ====================================================================
        # FINAL CONSOLE SUMMARY
        # ====================================================================

        print(
            f"\n{'=' * 60}\n"
            "CROSS-WINDOW COMPARISON "
            f"(avg of "
            f"{ITERATIONS_PER_WINDOW} "
            f"runs per window)\n"
            f"{'=' * 60}"
        )

        for window_label, _ in windows:

            rows = [
                r
                for r in all_window_results
                if r["window"]
                == window_label
            ]

            avg_return = np.mean(
                [
                    r[
                        "total_return_pct_mean"
                    ]
                    for r in rows
                ]
            )

            avg_drawdown = np.mean(
                [
                    r[
                        "max_drawdown_pct_mean"
                    ]
                    for r in rows
                ]
            )

            total_trades = sum(
                r[
                    "num_trades_mean"
                ]
                for r in rows
            )

            print(
                f"{window_label:45s}  "
                f"avg return "
                f"{avg_return:+.2f}%   "
                f"avg max drawdown "
                f"{avg_drawdown:.2f}%   "
                f"trades "
                f"{total_trades:.0f}"
            )

        # ====================================================================
        # SAVE EVERYTHING
        # ====================================================================

        results_path, chart_path = (
            save_completed_run(
                all_window_results,
                window_charts,
            )
        )

        elapsed = (
            time.time() -
            JOB_STARTED_AT
        )

        update_job(
            status="complete",
            current_symbol=None,
            symbol_index=len(
                SYMBOLS
            ),
            progress=100,
            elapsed_seconds=elapsed,
            estimated_seconds=elapsed,
            eta_seconds=0,
            message=(
                "Backtest complete. "
                "Results automatically saved."
            ),
            error=None,
            completed=True,
        )

        write_live_data(
            equity_curve=[],
            timestamps=[],
            spy_curve=[],
            drawdown_curve=[],
            symbol=None,
            symbol_progress=None,
        )

        print()
        print(
            "=" * 60
        )

        print(
            "BACKTEST COMPLETE"
        )

        print(
            "=" * 60
        )

        print(
            f"Run ID:       {RUN_ID}"
        )

        print(
            f"Saved to:     {RUN_DIR}"
        )

        print(
            f"Results:      {results_path}"
        )

        print(
            f"SPY chart:    {chart_path}"
        )

        print(
            "=" * 60
        )

    except Exception as e:

        elapsed = (
            time.time() -
            JOB_STARTED_AT
        )

        error_text = (
            f"{e.__class__.__name__}: "
            f"{e}"
        )

        print(
            "\n"
            + "=" * 60
        )

        print(
            "BACKTEST FAILED"
        )

        print(
            "=" * 60
        )

        print(
            error_text
        )

        traceback.print_exc()

        update_job(
            status="failed",
            progress=(
                min(
                    99,
                    (
                        completed_units /
                        total_units *
                        100
                    )
                    if total_units
                    else 0,
                )
            ),
            elapsed_seconds=elapsed,
            eta_seconds=0,
            message=(
                "Backtest failed."
            ),
            error=(
                traceback.format_exc()
            ),
            completed=False,
        )

        raise


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    print()
    print(
        "=" * 60
    )

    print(
        "TRADING BOT BACKTEST"
    )

    print(
        "=" * 60
    )

    print(
        f"Run ID: {RUN_ID}"
    )

    print(
        f"Symbols: {', '.join(SYMBOLS)}"
    )

    print(
        f"Windows: {NUM_WINDOWS}"
    )

    print(
        f"Iterations per window: "
        f"{ITERATIONS_PER_WINDOW}"
    )

    print(
        f"Run directory: {RUN_DIR}"
    )

    print(
        "=" * 60
    )

    main()