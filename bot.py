"""
trader_v3.py — Live trading entry point.

All trade logs and the meta-model file automatically land in trade_data/,
same as backtest.py — that folder is created on first run if needed.
"""

import os
import time
import threading
import logging

import alpaca_trade_api as tradeapi

from Tradingcore import MarketDataLayer, SignalLayer, RiskGate
from meta_filter import MetaModelFilter

DATA_DIR = "trade_data"
os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(DATA_DIR, "trader.log")), logging.StreamHandler()],
)
log = logging.getLogger("trader")

API_KEY = os.environ.get("ALPACA_API_KEY")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

api = None
stream = None
try:
    if API_KEY and SECRET_KEY:
        api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")
        stream = tradeapi.Stream(API_KEY, SECRET_KEY, BASE_URL, data_feed="iex")
    else:
        log.info("ALPACA credentials not found; running in offline/fallback mode (no live stream).")
except Exception as e:
    log.error(f"Failed to initialize Alpaca API/stream: {e}")
    api = None
    stream = None

SYMBOLS = ["AAPL", "TSLA", "MSFT", "NVDA", "JPM", "XOM", "JNJ", "SPY"]

data_layer = MarketDataLayer(SYMBOLS, api_client=api)
meta_filter = MetaModelFilter(os.path.join(DATA_DIR, "meta_model.joblib"))
signal_layer = SignalLayer(data_layer, meta_filter=meta_filter,
                            weights_path=os.path.join(DATA_DIR, "expert_weights.json"))
risk_gate = RiskGate(SYMBOLS, api_client=api, trade_log_path=os.path.join(DATA_DIR, "trade_log.csv"))

WEIGHTS_AUTOSAVE_INTERVAL_SECONDS = 300  # persist learned expert-trust every 5 minutes, not just on shutdown


def weights_autosave_loop():
    while True:
        time.sleep(WEIGHTS_AUTOSAVE_INTERVAL_SECONDS)
        signal_layer.save_weights()


async def on_trade(trade):
    symbol = trade.symbol
    price = trade.price
    size = getattr(trade, "size", 1)
    ts = trade.timestamp.timestamp() if hasattr(trade.timestamp, "timestamp") else time.time()

    data_layer.update_bar(symbol, price, size, ts)
    signal = signal_layer.generate(symbol)
    if signal is None:
        return
    decision = risk_gate.evaluate(signal, price)
    risk_gate.execute(decision, symbol, price)


def main():
    log.info("Bootstrapping historical bars...")
    data_layer.bootstrap_history()

    risk_gate.refresh_positions()
    risk_gate.check_daily_loss_limit()
    data_layer.refresh_news_sentiment()

    for symbol in SYMBOLS:
        stream.subscribe_trades(on_trade, symbol)

    threading.Thread(target=risk_gate.background_sync_loop, daemon=True).start()
    threading.Thread(target=data_layer.news_refresher_loop, daemon=True).start()
    threading.Thread(target=weights_autosave_loop, daemon=True).start()
    threading.Thread(target=stream.run, daemon=True).start()

    log.info("Three-layer trading system online: MarketDataLayer -> SignalLayer (adaptive) -> RiskGate (fixed).")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Shutting down — saving expert weights before exit.")
        signal_layer.save_weights()


if __name__ == "__main__":
    main()