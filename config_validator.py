#!/usr/bin/env python3
"""config_validator.py - Standalone sanity checker for the strategy config.

WHY THIS EXISTS
----------------
strategy_config.py's ``load_best_config()`` deliberately swallows *any* error
(missing file, bad JSON, missing keys) and returns ``{}``, so every caller
silently falls back to hard-coded defaults. That's a reasonable safety
behavior for the bot at runtime, but it also means a corrupted config file,
or one with an out-of-range value (e.g. TAKE_PROFIT_PCT smaller than
STOP_LOSS_PCT, or MAX_POSITION_PCT > 1) can go unnoticed indefinitely -
the bot just quietly runs on defaults or on a nonsensical value instead of
telling anyone.

This script re-reads trade_data/best_strategy_config.json on its own
(reusing strategy_config.py's CONFIG_KEYS/CONFIG_PATH constants) and prints
loud, human-readable warnings for anything missing, malformed, out of range,
or internally inconsistent - instead of failing silently.

SCOPE (intentionally standalone)
---------------------------------
This is a standalone, manually-run checker only. It is NOT wired into
backtest.py or bot.py - bot.py is a protected file that must never be
modified, and keeping this self-contained keeps the change minimal and
avoids touching any order-execution or risk-gate code. Run it by hand (or
from a pre-flight shell step / CI job you add yourself) whenever you want to
confirm the saved best-strategy config looks sane. A human can wire it into
the bot's startup sequence later if desired.

USAGE
-----
    python config_validator.py                # checks the real config
    python config_validator.py path/to/other.json   # checks any JSON file
                                                      # (for manual testing
                                                      # against a scratch
                                                      # copy only - never
                                                      # point this at a file
                                                      # you intend to edit
                                                      # in place)

EXIT CODE
---------
    0   config file was found, parsed, and passed every check (no warnings)
    1   the file could not be found/read/parsed, OR one or more warnings
        were raised (missing key, out-of-range value, inconsistent
        settings). Non-zero on any warning so this doubles as a pre-flight
        check in a shell pipeline (e.g. `python config_validator.py || echo
        "review config before trading"`).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

import strategy_config

# ---------------------------------------------------------------------------
# Loading (does NOT swallow errors - that's the whole point of this script)
# ---------------------------------------------------------------------------


def load_raw_payload(path: Path) -> tuple[Optional[dict], list[str]]:
    """Read + JSON-parse `path`, returning (payload, warnings).

    Unlike strategy_config.load_best_config(), any problem is reported as a
    warning instead of being swallowed into an empty dict.
    """
    warnings: list[str] = []

    if not path.exists():
        warnings.append(
            f"Config file not found at {path} -- callers will silently fall "
            "back to built-in defaults for EVERY setting."
        )
        return None, warnings

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        warnings.append(f"Could not read config file {path}: {exc}")
        return None, warnings

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        warnings.append(
            f"Config file {path} is not valid JSON ({exc}) -- callers will "
            "silently fall back to built-in defaults for EVERY setting."
        )
        return None, warnings

    if not isinstance(payload, dict):
        warnings.append(
            f"Config file {path} does not contain a JSON object at the top "
            "level (got a JSON value of a different type)."
        )
        return None, warnings

    return payload, warnings


def extract_config(payload: dict) -> dict:
    """Mirror strategy_config.load_best_config()'s unwrap-then-filter logic,
    but keep every key present (not just the ones in CONFIG_KEYS) so callers
    can spot typos/extra keys too."""
    cfg = payload.get("config", payload)
    return cfg if isinstance(cfg, dict) else {}


# ---------------------------------------------------------------------------
# Small local numeric coercion helpers.
#
# Tradingcore.py has its own _cfg_float/_cfg_int helpers, but those are part
# of the protected risk-gate module and must not be imported from here.
# These are trivial, read-only equivalents used purely for validating types.
# ---------------------------------------------------------------------------


def as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_BAR_TIMEFRAME_RE = re.compile(r"^\d+[mhd]$")


def validate_config(cfg: dict) -> list[str]:
    """Return a list of human-readable warning strings for `cfg`.

    An empty list means the config looks internally consistent. Range
    checks are applied only to keys that are present and numeric so a
    missing-key warning isn't duplicated by a spurious range warning.
    """
    warnings: list[str] = []

    # --- missing / unexpected keys -----------------------------------
    missing = [k for k in strategy_config.CONFIG_KEYS if k not in cfg]
    if missing:
        warnings.append(
            "Missing key(s) "
            + ", ".join(missing)
            + " -- these will silently fall back to built-in defaults "
            "wherever they're used."
        )

    extra = [k for k in cfg if k not in strategy_config.CONFIG_KEYS]
    if extra:
        warnings.append(
            "Unrecognized key(s) in config (ignored by strategy_config.py): "
            + ", ".join(extra)
            + " -- possible typo?"
        )

    def get_float(key: str) -> Optional[float]:
        if key not in cfg:
            return None
        val = as_float(cfg[key])
        if val is None:
            warnings.append(f"{key} = {cfg[key]!r} is not a number.")
        return val

    # --- BAR_TIMEFRAME --------------------------------------------------
    if "BAR_TIMEFRAME" in cfg:
        tf = cfg["BAR_TIMEFRAME"]
        if not isinstance(tf, str) or not _BAR_TIMEFRAME_RE.match(tf.strip().lower()):
            warnings.append(
                f"BAR_TIMEFRAME = {tf!r} does not look like a valid timeframe "
                "(expected a string like '5m', '15m', '1h', '1d')."
            )

    # --- probability-like fields in (0, 1) ------------------------------
    for key in ("MIN_SIGNAL_CONFIDENCE", "MIN_META_WIN_PROB"):
        val = get_float(key)
        if val is not None and not (0.0 <= val <= 1.0):
            warnings.append(f"{key} = {val} is outside the expected [0, 1] range.")

    # --- ATR band: MIN_ATR_PCT < MAX_ATR_PCT, both non-negative --------
    min_atr = get_float("MIN_ATR_PCT")
    max_atr = get_float("MAX_ATR_PCT")
    if min_atr is not None and min_atr < 0:
        warnings.append(f"MIN_ATR_PCT = {min_atr} is negative.")
    if max_atr is not None and max_atr <= 0:
        warnings.append(f"MAX_ATR_PCT = {max_atr} should be positive.")
    if min_atr is not None and max_atr is not None and min_atr >= max_atr:
        warnings.append(
            f"MIN_ATR_PCT ({min_atr}) should be smaller than MAX_ATR_PCT "
            f"({max_atr}) -- the ATR filter band is empty or inverted."
        )

    # --- MIN_TREND_CHANGE_PCT: non-negative -----------------------------
    trend = get_float("MIN_TREND_CHANGE_PCT")
    if trend is not None and trend < 0:
        warnings.append(f"MIN_TREND_CHANGE_PCT = {trend} is negative.")

    # --- MIN_BARS_BETWEEN_TRADES: non-negative integer ------------------
    if "MIN_BARS_BETWEEN_TRADES" in cfg:
        bars = as_int(cfg["MIN_BARS_BETWEEN_TRADES"])
        if bars is None:
            warnings.append(
                f"MIN_BARS_BETWEEN_TRADES = {cfg['MIN_BARS_BETWEEN_TRADES']!r} "
                "is not an integer."
            )
        elif bars < 0:
            warnings.append(f"MIN_BARS_BETWEEN_TRADES = {bars} is negative.")

    # --- RISK_PER_TRADE_PCT: sane, positive, and small ------------------
    risk = get_float("RISK_PER_TRADE_PCT")
    if risk is not None:
        if risk <= 0:
            warnings.append(f"RISK_PER_TRADE_PCT = {risk} must be positive.")
        elif risk > 0.05:
            warnings.append(
                f"RISK_PER_TRADE_PCT = {risk} risks more than 5% of the "
                "account per trade, which looks unusually high."
            )

    # --- MAX_POSITION_PCT: (0, 1] ----------------------------------------
    max_pos = get_float("MAX_POSITION_PCT")
    if max_pos is not None and not (0.0 < max_pos <= 1.0):
        warnings.append(
            f"MAX_POSITION_PCT = {max_pos} is outside the expected (0, 1] "
            "range (fraction of the portfolio per position)."
        )

    # --- STOP_LOSS_PCT / TAKE_PROFIT_PCT: positive, TP > SL --------------
    stop_loss = get_float("STOP_LOSS_PCT")
    take_profit = get_float("TAKE_PROFIT_PCT")
    if stop_loss is not None and stop_loss <= 0:
        warnings.append(f"STOP_LOSS_PCT = {stop_loss} must be positive.")
    if take_profit is not None and take_profit <= 0:
        warnings.append(f"TAKE_PROFIT_PCT = {take_profit} must be positive.")
    if stop_loss is not None and take_profit is not None and take_profit <= stop_loss:
        warnings.append(
            f"TAKE_PROFIT_PCT ({take_profit}) should exceed STOP_LOSS_PCT "
            f"({stop_loss}) -- as configured, a losing trade risks more than "
            "a winning trade gains."
        )

    # --- TRAIL_STOP_PCT: non-negative (0 = disabled) ---------------------
    trail = get_float("TRAIL_STOP_PCT")
    if trail is not None and trail < 0:
        warnings.append(f"TRAIL_STOP_PCT = {trail} is negative.")

    # --- ATR_STOP_MULTIPLIER: non-negative (0 = disabled, per Tradingcore.py)
    atr_mult = get_float("ATR_STOP_MULTIPLIER")
    if atr_mult is not None and atr_mult < 0:
        warnings.append(f"ATR_STOP_MULTIPLIER = {atr_mult} is negative.")

    return warnings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    path = Path(argv[1]).expanduser().resolve() if len(argv) > 1 else strategy_config.CONFIG_PATH

    print(f"Validating strategy config: {path}")

    payload, warnings = load_raw_payload(path)
    if payload is not None:
        cfg = extract_config(payload)
        warnings.extend(validate_config(cfg))

    if not warnings:
        print("OK: no issues found.")
        return 0

    print(f"Found {len(warnings)} warning(s):")
    for w in warnings:
        print(f"  [WARNING] {w}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
