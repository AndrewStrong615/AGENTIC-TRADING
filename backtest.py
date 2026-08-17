"""
backtest.py — Walk-forward backtest running the ACTUAL three-layer system
from Tradingcore.py. All trade logs, the combined results CSV, the meta-
model file, the learned expert-weights file, and the comparison chart now
automatically land in trade_data/ — that folder is created on first run if
it doesn't exist, no manual setup.

Expert weights (SignalLayer's per-symbol trust in batch/online/markov/
meanrev) now persist across runs in trade_data/expert_weights.json, the
same way live trading (trader_v3.py) persists them. Within a single run,
every iteration in ITERATIONS_PER_WINDOW still starts from the SAME
snapshot of those weights (loaded once per symbol before the iteration
loop) so the random-seed-averaging methodology stays statistically valid —
only the first iteration's final weights get written back to disk.

Usage:
    python backtest.py
"""

import os
import json
import socket
import time
import logging
from datetime import datetime, timedelta

socket.setdefaulttimeout(20)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame

from Tradingcore import MarketDataLayer, SignalLayer, RiskGate, GateDecision
from meta_filter import MetaModelFilter

DATA_DIR = "trade_data"
os.makedirs(DATA_DIR, exist_ok=True)

WEIGHTS_PATH = os.path.join(DATA_DIR, "expert_weights.json")

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("trader")

API_KEY = os.environ["ALPACA_API_KEY"]
SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")
meta_filter = MetaModelFilter(os.path.join(DATA_DIR, "meta_model.joblib"))

SYMBOLS = ["AAPL", "TSLA", "MSFT", "NVDA", "JPM", "XOM", "JNJ", "SPY"]
BENCHMARK_SYMBOL = "SPY"

WINDOW_LENGTH_DAYS = 30
NUM_WINDOWS = 1

ITERATIONS_PER_WINDOW = 2
BASE_RANDOM_SEED = 0

RETRAIN_EVERY_N_BARS = 20
STARTING_CASH_PER_SYMBOL = 10_000.0


class BacktestRiskGate(RiskGate):
    def __init__(self, symbols, trade_log_path):
        super().__init__(symbols, api_client=None, trade_log_path=trade_log_path)
        self.cash = {s: STARTING_CASH_PER_SYMBOL for s in symbols}
        self.trades = {s: [] for s in symbols}
        self.starting_equity = STARTING_CASH_PER_SYMBOL

    def refresh_positions(self):
        pass

    def check_daily_loss_limit(self):
        pass

    def execute(self, decision: GateDecision, symbol: str, price: float, bar_index=None):
        if not decision.approved:
            return
        if decision.action == "buy":
            cost = decision.qty * price
            if self.cash[symbol] < cost:
                return
            self.cash[symbol] -= cost
            with self.positions_lock:
                self.current_positions[symbol] += decision.qty
                self.entry_prices[symbol] = price
        elif decision.action == "sell":
            proceeds = decision.qty * price
            self.cash[symbol] += proceeds
            entry_price = self.entry_prices[symbol]
            pnl_pct = ((price - entry_price) / entry_price) * 100 if entry_price else 0.0
            self.trades[symbol].append({"pnl_pct": pnl_pct, "price": price, "bar": bar_index,
                                         "reason": decision.reason})
            self.record_trade_outcome(symbol, price, decision.reason)
            with self.positions_lock:
                self.current_positions[symbol] = max(0, self.current_positions[symbol] - decision.qty)
                if self.current_positions[symbol] == 0:
                    self.entry_prices[symbol] = None
                    self.entry_stop_pct[symbol] = None
                    self.entry_target_pct[symbol] = None
                    self.entry_signal_context[symbol] = None

    def equity(self, symbol, current_price):
        return self.cash[symbol] + self.current_positions[symbol] * current_price

    def force_close(self, symbol, price, bar_index):
        held = self.current_positions[symbol]
        if held > 0:
            decision = GateDecision(True, "sell", held, "end-of-backtest close")
            self.execute(decision, symbol, price, bar_index)


def load_persisted_weights(weights_path):
    """Read the whole persisted expert-weights file (all symbols). Returns
    {} if it doesn't exist yet or can't be parsed — same fallback behavior
    as SignalLayer._load_weights uses for live trading."""
    if not os.path.exists(weights_path):
        return {}
    try:
        with open(weights_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"  Couldn't read persisted weights from {weights_path}, starting uniform: {e}")
        return {}


def fetch_bars(symbol, days, end_offset_days=0, max_retries=3):
    end = datetime.now() - timedelta(minutes=16) - timedelta(days=end_offset_days)
    start = end - timedelta(days=days)
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            df = api.get_bars(symbol, TimeFrame.Minute, start.strftime("%Y-%m-%d"),
                               end.strftime("%Y-%m-%d"), adjustment="raw", feed="iex").df
            if df.empty:
                raise ValueError(f"No bar data for {symbol}.")
            return df
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  {symbol}: bars fetch failed ({e.__class__.__name__}), retrying in {wait}s...")
                time.sleep(wait)
    raise last_error


def fetch_daily_sentiment(symbol, start_date, end_date, max_retries=3):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
    last_error = None
    articles = []
    for attempt in range(1, max_retries + 1):
        try:
            articles = api.get_news(symbol, start=start_date.strftime("%Y-%m-%d"),
                                     end=end_date.strftime("%Y-%m-%d"), limit=1000)
            break
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  {symbol}: news fetch failed ({e.__class__.__name__}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  {symbol}: news fetch failed after {max_retries} attempts, "
                      f"proceeding with neutral sentiment ({last_error}).")
                return {}

    day_buckets = {}
    for a in (articles or []):
        text = f"{getattr(a, 'headline', '') or ''}. {getattr(a, 'summary', '') or ''}".strip()
        created = getattr(a, "created_at", None)
        if not text or created is None:
            continue
        day = created.date() if hasattr(created, "date") else pd.Timestamp(created).date()
        day_buckets.setdefault(day, []).append(vader.polarity_scores(text)["compound"])

    return {day: float(np.mean(scores)) for day, scores in day_buckets.items()}


def simulate_symbol(symbol, df, daily_sentiment, random_state, trade_log_path,
                     initial_weights=None, persist_weights_path=None):
    """Runs one iteration for one symbol.

    initial_weights: the expert-trust weights (dict of expert -> weight)
        this iteration should START from. Passed in explicitly (rather than
        via SignalLayer's own weights_path loading) so every iteration in
        ITERATIONS_PER_WINDOW starts from the exact same snapshot, keeping
        the random-seed-averaging methodology valid — none of them read
        each other's in-progress updates.
    persist_weights_path: if set, this iteration's FINAL learned weights
        for this symbol get written here after the run. Only pass this for
        one iteration (we use the first) or later iterations will overwrite
        it with an equally-arbitrary-seeded result and the write becomes
        wasted work.
    """
    data_layer = MarketDataLayer([symbol])
    signal_layer = SignalLayer(data_layer, retrain_every_n_bars=RETRAIN_EVERY_N_BARS,
                                random_state=random_state, meta_filter=meta_filter)
    if initial_weights:
        signal_layer.weights[symbol] = dict(initial_weights)

    risk_gate = BacktestRiskGate([symbol], trade_log_path=trade_log_path)

    closes = df["close"].values.astype(float)
    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    dates = df.index.to_pydatetime()
    n = len(closes)

    equity_curve = []

    for i in range(n):
        bar = {"t": i, "o": closes[i], "h": highs[i], "l": lows[i], "c": closes[i], "v": 1}
        data_layer.seed_bar(symbol, bar)

        bar_date = dates[i].date()
        sentiment = daily_sentiment.get(bar_date, 0.0)
        data_layer.set_news_sentiment(symbol, sentiment)

        current_price = closes[i]
        equity_curve.append(risk_gate.equity(symbol, current_price))

        signal = signal_layer.generate(symbol)
        if signal is None:
            continue

        bar_time = dates[i].timestamp()
        decision = risk_gate.evaluate(signal, current_price, now=bar_time)
        risk_gate.execute(decision, symbol, current_price, bar_index=i)

    risk_gate.force_close(symbol, closes[-1], n - 1)
    equity_curve.append(risk_gate.equity(symbol, closes[-1]))

    if persist_weights_path:
        signal_layer.weights_path = persist_weights_path
        signal_layer.save_weights()

    return equity_curve, risk_gate.trades[symbol]


def compute_metrics(equity_curve, trades):
    equity = np.array(equity_curve)
    total_return_pct = ((equity[-1] - STARTING_CASH_PER_SYMBOL) / STARTING_CASH_PER_SYMBOL) * 100
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max * 100
    max_drawdown_pct = drawdowns.min()

    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    win_rate = (len(wins) / len(trades) * 100) if trades else 0.0
    avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0.0
    avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0.0

    return {"total_return_pct": total_return_pct, "max_drawdown_pct": max_drawdown_pct,
            "num_trades": len(trades), "win_rate": win_rate, "avg_win_pct": avg_win, "avg_loss_pct": avg_loss}


def simulate_symbol_iterations(symbol, df, daily_sentiment, trade_log_path):
    # Load once per symbol, before the iteration loop, so every iteration
    # starts from the identical persisted snapshot rather than drifting
    # across iterations (which would break seed-variance comparability).
    initial_weights = load_persisted_weights(WEIGHTS_PATH).get(symbol)

    iter_metrics = []
    equity_runs = []
    for it in range(ITERATIONS_PER_WINDOW):
        seed = BASE_RANDOM_SEED + it
        # Only the first iteration's resulting weights get persisted —
        # later iterations use different seeds and would otherwise clobber
        # that write with an equally-arbitrary result.
        persist_path = WEIGHTS_PATH if it == 0 else None
        equity_curve, trades = simulate_symbol(
            symbol, df, daily_sentiment, seed, trade_log_path,
            initial_weights=initial_weights, persist_weights_path=persist_path,
        )
        iter_metrics.append(compute_metrics(equity_curve, trades))
        equity_runs.append(equity_curve)
    return iter_metrics, equity_runs


def aggregate_metrics(symbol, iter_metrics):
    keys = ["total_return_pct", "max_drawdown_pct", "num_trades", "win_rate"]
    agg = {"symbol": symbol}
    for k in keys:
        vals = [m[k] for m in iter_metrics]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))
    return agg


def print_symbol_summary(agg):
    print(f"\n=== {agg['symbol']} (avg of {ITERATIONS_PER_WINDOW} runs) ===")
    print(f"Total return:     {agg['total_return_pct_mean']:+.2f}%  (std {agg['total_return_pct_std']:.2f})")
    print(f"Max drawdown:     {agg['max_drawdown_pct_mean']:.2f}%  (std {agg['max_drawdown_pct_std']:.2f})")
    print(f"Trades:           {agg['num_trades_mean']:.0f}  (std {agg['num_trades_std']:.1f})")
    print(f"Win rate:         {agg['win_rate_mean']:.1f}%  (std {agg['win_rate_std']:.1f})")


def build_portfolio_series(symbol_avg_equity, symbol_dates):
    per_symbol_series = {}
    for symbol, avg_curve in symbol_avg_equity.items():
        dates = symbol_dates[symbol]
        n = len(dates)
        per_symbol_series[symbol] = pd.Series(avg_curve[:n], index=pd.DatetimeIndex(dates)).sort_index()

    union_index = None
    for s in per_symbol_series.values():
        union_index = s.index if union_index is None else union_index.union(s.index)

    aligned = pd.DataFrame({sym: s.reindex(union_index).ffill().bfill() for sym, s in per_symbol_series.items()})
    portfolio = aligned.sum(axis=1)
    return portfolio


def fetch_benchmark_series(days, end_offset_days):
    df = fetch_bars(BENCHMARK_SYMBOL, days, end_offset_days=end_offset_days)
    closes = df["close"].astype(float)
    closes.index = pd.DatetimeIndex(df.index.to_pydatetime())
    return closes.sort_index()


def plot_windows(window_charts, out_path=None):
    if out_path is None:
        out_path = os.path.join(DATA_DIR, "backtest_vs_spy.png")
    n = len(window_charts)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.2 * n), squeeze=False)
    for ax, (label, portfolio_pct, spy_pct) in zip(axes[:, 0], window_charts):
        ax.plot(portfolio_pct.index, portfolio_pct.values, label=f"Strategy (avg of {ITERATIONS_PER_WINDOW} runs)",
                color="#1f77b4", linewidth=1.4)
        ax.plot(spy_pct.index, spy_pct.values, label="S&P 500 (SPY, buy & hold)",
                color="#888888", linewidth=1.2, linestyle="--")
        ax.axhline(0, color="#dddddd", linewidth=0.8, zorder=0)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Return (%)")
        ax.legend(loc="upper left", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"\nSaved comparison chart to {out_path}")


def build_windows():
    windows = []
    for i in range(NUM_WINDOWS):
        end_offset_days = i * WINDOW_LENGTH_DAYS
        approx_end = datetime.now() - timedelta(days=end_offset_days)
        approx_start = approx_end - timedelta(days=WINDOW_LENGTH_DAYS)
        label = f"Window {i + 1}: {approx_start:%Y-%m-%d} to {approx_end:%Y-%m-%d}"
        windows.append((label, end_offset_days))
    return windows


def main():
    windows = build_windows()
    all_window_results = []
    window_charts = []

    for window_label, end_offset_days in windows:
        print(f"\n{'#' * 60}\n# {window_label}\n{'#' * 60}")

        symbol_avg_equity = {}
        symbol_dates = {}

        for symbol in SYMBOLS:
            print(f"Fetching bars for {symbol}...")
            df = fetch_bars(symbol, WINDOW_LENGTH_DAYS, end_offset_days=end_offset_days)

            end_date = datetime.now() - timedelta(days=end_offset_days)
            start_date = end_date - timedelta(days=WINDOW_LENGTH_DAYS)
            print(f"Fetching news sentiment for {symbol}...")
            daily_sentiment = fetch_daily_sentiment(symbol, start_date, end_date)

            trade_log_path = os.path.join(DATA_DIR, f"backtest_trade_log_{symbol}_{end_offset_days}.csv")
            print(f"Running {ITERATIONS_PER_WINDOW} iterations for {symbol}...")
            iter_metrics, equity_runs = simulate_symbol_iterations(symbol, df, daily_sentiment, trade_log_path)

            agg = aggregate_metrics(symbol, iter_metrics)
            agg["window"] = window_label
            all_window_results.append(agg)
            print_symbol_summary(agg)

            symbol_avg_equity[symbol] = np.mean(equity_runs, axis=0)
            symbol_dates[symbol] = df.index.to_pydatetime()

            time.sleep(2)

        avg_return = np.mean([r["total_return_pct_mean"] for r in all_window_results if r["window"] == window_label])
        total_trades = sum(r["num_trades_mean"] for r in all_window_results if r["window"] == window_label)
        print(f"\n=== {window_label} Combined (avg of {ITERATIONS_PER_WINDOW} runs each) ===")
        print(f"Average return across symbols: {avg_return:+.2f}%")
        print(f"Total trades across symbols:   {total_trades:.0f}")

        portfolio = build_portfolio_series(symbol_avg_equity, symbol_dates)
        portfolio_pct = (portfolio / portfolio.iloc[0] - 1) * 100

        try:
            spy_closes = fetch_benchmark_series(WINDOW_LENGTH_DAYS, end_offset_days)
            spy_aligned = spy_closes.reindex(spy_closes.index.union(portfolio_pct.index)).ffill().bfill().reindex(portfolio_pct.index)
            spy_pct = (spy_aligned / spy_aligned.iloc[0] - 1) * 100
            window_charts.append((window_label, portfolio_pct, spy_pct))
        except Exception as e:
            print(f"  Benchmark fetch failed for this window, skipping it in the chart ({e}).")

    print(f"\n{'=' * 60}\nCROSS-WINDOW COMPARISON (avg of {ITERATIONS_PER_WINDOW} runs per window)\n{'=' * 60}")
    for window_label, _ in windows:
        rows = [r for r in all_window_results if r["window"] == window_label]
        avg_return = np.mean([r["total_return_pct_mean"] for r in rows])
        avg_drawdown = np.mean([r["max_drawdown_pct_mean"] for r in rows])
        total_trades = sum(r["num_trades_mean"] for r in rows)
        print(f"{window_label:45s}  avg return {avg_return:+.2f}%   "
              f"avg max drawdown {avg_drawdown:.2f}%   trades {total_trades:.0f}")

    results_path = os.path.join(DATA_DIR, "backtest_results.csv")
    pd.DataFrame(all_window_results).to_csv(results_path, index=False)
    print(f"\nSaved all results to {results_path}")

    plot_windows(window_charts)


if __name__ == "__main__":
    main()