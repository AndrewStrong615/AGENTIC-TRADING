"""Shared pytest configuration and fixtures for the unit test suite.

Ensures the repo root is importable (so ``import strategy_optimizer`` etc.
work regardless of how pytest is invoked) and provides small shared
fixtures. Nothing here imports bot.py, Tradingcore.py, or
credentials_store.py, and nothing here performs network access.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def sample_config():
    """A representative strategy config dict, matching
    strategy_optimizer.CONFIG_KEYS, for optimizer tests."""
    return {
        "BAR_TIMEFRAME": "5m",
        "MIN_SIGNAL_CONFIDENCE": 0.40,
        "MIN_META_WIN_PROB": 0.62,
        "MIN_ATR_PCT": 0.05,
        "MAX_ATR_PCT": 3.0,
        "MIN_TREND_CHANGE_PCT": 0.1,
        "MIN_BARS_BETWEEN_TRADES": 5,
        "RISK_PER_TRADE_PCT": 1.0,
        "MAX_POSITION_PCT": 20.0,
        "STOP_LOSS_PCT": 2.0,
        "TAKE_PROFIT_PCT": 4.0,
        "TRAIL_STOP_PCT": 1.5,
        "ATR_STOP_MULTIPLIER": 2.0,
    }
