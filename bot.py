import os
import time
import threading
import logging

import alpaca_trade_api as tradeapi

from credentials_store import load_credentials
from Tradingcore import MarketDataLayer, SignalLayer, RiskGate
from meta_filter import MetaModelFilter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "trade_data")
os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(DATA_DIR, "trader.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("trader")

try:
    creds = load_credentials()
except Exception as e:
    log.error(f"Failed to load credentials_store.py: {e}")
    creds = {}

API_KEY = creds.get("alpaca_api_key") or os.environ.get("ALPACA_API_KEY", "")
SECRET_KEY = creds.get("alpaca_secret_key") or os.environ.get("ALPACA_SECRET_KEY", "")
BASE_URL = (
    creds.get("alpaca_base_url")
    or os.environ.get("ALPACA_BASE_URL")
    or "https://paper-api.alpaca.markets"
).rstrip("/")

log.info(
    "Alpaca credentials loaded: api_key=%s secret_key=%s base_url=%s",
    "yes" if API_KEY else "NO",
    "yes" if SECRET_KEY else "NO",
    BASE_URL,
)

api = None
stream = None
try:
    if API_KEY and SECRET_KEY:
        api = tradeapi.REST(
            API_KEY,
            SECRET_KEY,
            BASE_URL,
            api_version="v2",
        )
        stream = tradeapi.Stream(
            API_KEY,
            SECRET_KEY,
            BASE_URL,
            data_feed="iex",
        )
    else:
        log.error("Alpaca credentials not found; live trading stream disabled.")
except Exception as e:
    log.error(f"Failed to initialize Alpaca API/stream: {e}")
    api = None
    stream = None

SYMBOLS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "AVGO",
    "TSLA",
    "AMD",
    "MU",
    "JPM",
    "V",
    "MA",
    "BAC",
    "WMT",
    "COST",
    "XOM",
    "JNJ",
    "LLY",
    "ABBV",
    "HD",
    "CAT",
    "CSCO",
    "ORCL",
    "NFLX",
]

data_layer = MarketDataLayer(SYMBOLS, api_client=api)
CONTEXT_SYMBOLS = getattr(data_layer, "context_symbols", [])
STREAM_SYMBOLS = list(dict.fromkeys(SYMBOLS + CONTEXT_SYMBOLS))
meta_filter = MetaModelFilter(os.path.join(DATA_DIR, "meta_model.joblib"))
signal_layer = SignalLayer(
    data_layer,
    meta_filter=meta_filter,
    weights_path=os.path.join(DATA_DIR, "expert_weights.json"),
)
risk_gate = RiskGate(
    SYMBOLS,
    api_client=api,
    trade_log_path=os.path.join(DATA_DIR, "trade_log.csv"),
)

WEIGHTS_AUTOSAVE_INTERVAL_SECONDS = 300


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
    if symbol not in SYMBOLS:
        return
    signal = signal_layer.generate(symbol)
    if signal is None:
        return
    decision = risk_gate.evaluate(signal, price)
    risk_gate.execute(decision, symbol, price)


def main():
    if api is None or stream is None:
        raise RuntimeError(
            "Alpaca API/stream is not initialized. Check trade_data/credentials.json "
            "and make sure the credentials belong to the selected Alpaca endpoint."
        )

    log.info("Bootstrapping historical bars...")
    data_layer.bootstrap_history()

    risk_gate.refresh_positions()
    risk_gate.check_daily_loss_limit()
    data_layer.refresh_news_sentiment()

    for symbol in STREAM_SYMBOLS:
        stream.subscribe_trades(on_trade, symbol)

    threading.Thread(target=risk_gate.background_sync_loop, daemon=True).start()
    threading.Thread(target=data_layer.news_refresher_loop, daemon=True).start()
    threading.Thread(target=weights_autosave_loop, daemon=True).start()
    threading.Thread(target=stream.run, daemon=True).start()

    log.info(
        "Three-layer trading system online: "
        "MarketDataLayer -> SignalLayer (adaptive) -> RiskGate (fixed)."
    )
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Shutting down — saving expert weights before exit.")
        signal_layer.save_weights()


if __name__ == "__main__":
    main()