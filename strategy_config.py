from __future__ import annotations
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "trade_data" / "best_strategy_config.json"

CONFIG_KEYS = (
    "BAR_TIMEFRAME", "MIN_SIGNAL_CONFIDENCE", "MIN_META_WIN_PROB",
    "MIN_ATR_PCT", "MAX_ATR_PCT", "MIN_TREND_CHANGE_PCT",
    "MIN_BARS_BETWEEN_TRADES", "RISK_PER_TRADE_PCT", "MAX_POSITION_PCT",
    "STOP_LOSS_PCT", "TAKE_PROFIT_PCT", "TRAIL_STOP_PCT",
    "ATR_STOP_MULTIPLIER",
)

def load_best_config() -> dict:
    try:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cfg = payload.get("config", payload) if isinstance(payload, dict) else {}
        return {k: cfg[k] for k in CONFIG_KEYS if k in cfg}
    except Exception:
        return {}

def apply_best_config_to_env() -> dict:
    cfg = load_best_config()
    applied = {}
    for key, value in cfg.items():
        if key not in os.environ:
            os.environ[key] = str(value)
            applied[key] = value
    return applied

APPLIED_BEST_CONFIG = apply_best_config_to_env()
