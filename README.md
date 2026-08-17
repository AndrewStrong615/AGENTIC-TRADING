# Trading Bot

An automated stock-trading system built in Python using a three-layer architecture:

1. **Market Data Layer** — collects and processes market data.
2. **Signal Layer** — uses machine-learning models to generate trading signals.
3. **Risk Gate** — applies deterministic risk rules before any trade can be executed.

The same core trading logic is designed to be shared between live trading and backtesting.

> **Disclaimer:** This project is for educational and research purposes. It is not financial advice and does not guarantee profits. Automated trading involves substantial risk of loss.

---

## Features

- Real-time market data processing
- Historical backtesting
- Machine-learning trading signals
- Adaptive batch + online ML model
- Random Forest batch model
- SGD online learning model
- Technical indicators and features
- ATR-based volatility measurement
- Trend-regime detection
- News sentiment analysis using VADER
- Deterministic risk-management rules
- Stop-loss and take-profit protection
- Position-size limits
- Confidence-based trade filtering
- News sentiment veto
- Trade logging
- SPY benchmark comparison
- Multiple historical backtest windows
- Multiple ML random seeds per window
- Equity-curve visualization

---

## Architecture

```text
                    ┌─────────────────────┐
                    │     Market Data     │
                    │       Layer 1       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Signal Layer    │
                    │       Layer 2       │
                    │                     │
                    │ Random Forest       │
                    │        +            │
                    │ Online SGD Model    │
                    │        ↓            │
                    │ Adaptive Blend      │
                    └──────────┬──────────┘
                               │
                         TradeSignal
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Risk Gate      │
                    │       Layer 3       │
                    │                     │
                    │ Confidence          │
                    │ Trend Agreement     │
                    │ News Veto           │
                    │ Position Limits     │
                    │ Stop Loss           │
                    │ Take Profit         │
                    │ Cooldown            │
                    └──────────┬──────────┘
                               │
                         GateDecision
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
              Live Execution         Backtest
