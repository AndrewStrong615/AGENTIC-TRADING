# Adaptive Trading Bot

A Python trading and backtesting system built around a three-layer architecture, historical meta-learning, online learning, and conservative strategy optimization.

> **Status:** Experimental / paper-trading focused. This project is not financial advice and is not guaranteed to be profitable. Do not use live money until the strategy has been thoroughly validated.

## What it does

The system combines:

- **Market data and regime detection** — ATR, volatility regimes, trend regimes, HMM regime detection, and news sentiment.
- **Adaptive signal generation** — a four-expert ensemble using batch learning, online learning, Markov logic, and mean-reversion logic.
- **Meta-model filtering** — a separate Random Forest learns which historical trade conditions were associated with profitable outcomes.
- **Risk management** — confidence gates, trend/news vetoes, ATR-based stops and targets, Kelly-style sizing, position limits, and daily loss protection.
- **Walk-forward backtesting** — historical performance is evaluated over rolling windows and archived under `trade_data/backtests/`.
- **Conservative strategy optimization** — candidate configurations are tested on newer holdout data before they can replace the current configuration.
- **Persistent strategy configuration** — the promoted configuration is stored in `trade_data/best_strategy_config.json` and automatically consumed by the backtest and trading logic.
- **Dashboard support** — the included Flask dashboard can launch the backtest, model training, and trading bot as separate processes.

## Architecture

```text
                         Historical Backtests
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │ Meta-model training │
                       └──────────┬──────────┘
                                  │
                       time-ordered validation
                                  │
                                  ▼
                         meta_model.joblib
                                  │
                                  ▼
 Market Data ──────► SignalLayer ──────► Meta Filter ──────► RiskGate
                         │                    │                │
                         │                    │                ▼
                         │                    │             Trade
                         ▼                    ▼
                 Online learning       Probability score
                         │
                         └──────────────┬─────────────────────┘
                                        ▼
                              Historical outcomes
                                        │
                                        ▼
                           Conservative optimizer
                                        │
                                        ▼
                         best_strategy_config.json
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
                    backtest.py                  Tradingcore.py
```

## Repository structure

```text
.
├── Tradingcore.py
├── backtest.py
├── train_meta_model.py
├── strategy_optimizer.py
├── strategy_config.py
├── dashboard_server.py
├── credentials_store.py
├── meta_filter.py
├── bot.py
├── notifications.py
├── trade_data/
│   ├── backtests/
│   ├── meta_training_data.csv
│   ├── meta_model.joblib
│   ├── meta_model_info.json
│   └── best_strategy_config.json
└── README.md
```

Files such as `meta_model.joblib`, historical logs, credentials, and other generated data are runtime artifacts and should normally **not** be committed to Git.

## Requirements

Use a recent Python 3 installation. The project uses packages including:

- `numpy`
- `pandas`
- `scikit-learn`
- `joblib`
- `alpaca-trade-api`
- `matplotlib`
- `vaderSentiment`
- `hmmlearn` (optional; HMM regime detection is disabled if unavailable)
- `Flask`
- `yfinance` (optional, depending on the configured data path)

Install the packages required by your local project, for example:

```bash
python -m pip install numpy pandas scikit-learn joblib alpaca-trade-api matplotlib vaderSentiment flask yfinance
python -m pip install hmmlearn
```

## Alpaca credentials

The project stores Alpaca credentials locally through `credentials_store.py` instead of depending on a terminal's temporary environment variables.

Credentials are stored at:

```text
trade_data/credentials.json
```

### Security

**Never commit `trade_data/credentials.json`.** Add it to `.gitignore`.

Example:

```gitignore
trade_data/credentials.json
trade_data/*.joblib
trade_data/*.csv
trade_data/*.json
trade_data/backtests/
__pycache__/
*.pyc
```

If an API key or secret is ever exposed, revoke/regenerate it before continuing.

The default endpoint is the Alpaca paper-trading endpoint:

```text
https://paper-api.alpaca.markets
```

## Conservative learning workflow

The intended workflow is deliberately conservative. The system does **not** change strategy settings just because one backtest looks good.

### 1. Run a backtest

```bash
python backtest.py
```

Backtest runs are archived under:

```text
trade_data/backtests/
```

### 2. Train the historical meta-model

```bash
python train_meta_model.py
```

The trainer collects completed backtest trade data, builds a deduplicated training set, trains on older observations, and validates against newer observations.

The conservative promotion gates require substantial evidence before replacing an existing meta-model.

### 3. Optimize the strategy

Start with a small search:

```bash
python strategy_optimizer.py --trials 8
```

The optimizer tests a deliberately limited parameter space and evaluates candidates on newer holdout periods.

If a candidate passes the conservative gates, its settings are written to:

```text
trade_data/best_strategy_config.json
```

### 4. Run the backtest again

```bash
python backtest.py
```

The promoted strategy configuration is loaded automatically. You should not need to manually copy optimizer settings into `backtest.py` or `Tradingcore.py`.

### 5. Repeat

The intended loop is:

```text
Backtest
   ↓
Collect historical outcomes
   ↓
Train / validate Meta Model
   ↓
Conservative strategy optimization
   ↓
Promote only if validation improves
   ↓
Backtest with promoted config
   ↓
Paper trade
   ↓
Collect more outcomes
   ↓
Repeat
```

## Conservative promotion rules

The current conservative design includes gates such as:

- At least **200 historical labeled trades** before meta-model promotion.
- At least **50 newer validation trades**.
- Meta-model AUC must improve by at least **0.02** when an existing model is available.
- Meta-model Brier score cannot worsen by more than **0.01**.
- Strategy candidates need positive return and at least **0.30 Sharpe** on each validation holdout.
- A candidate must materially outperform the currently promoted configuration before it can replace it.

These rules are intended to reduce overfitting and unnecessary parameter churn. They do not eliminate the possibility of overfitting.

## Online learning

`Tradingcore.py` also contains an online-learning component that updates from new bar outcomes. The adaptive signal layer uses:

- Batch Random Forest predictions
- Online `SGDClassifier` updates
- Markov-state probabilities
- Mean-reversion probabilities
- Learned expert weights

The separate meta-model is trained from historical trade outcomes and is used as an additional filter rather than replacing the signal ensemble.

## Dashboard

Start the dashboard with:

```bash
python dashboard_server.py
```

The dashboard can launch and monitor the project scripts, including:

- Backtest
- Meta-model training
- Trading bot
- Other project utilities included in the dashboard configuration

If dashboard credentials are configured, the dashboard should pass the saved Alpaca credentials to launched subprocesses.

## Notifications

`notifications.py` supports push notifications through [ntfy](https://ntfy.sh/).

Set a private topic before starting the dashboard, for example on Windows:

```bat
setx NTFY_TOPIC "your-private-topic"
```

Then open a new terminal before launching the dashboard.

Do not publish your private ntfy topic in a public repository.

## Results and evaluation

Do not judge the system from a single week of returns.

Track at least:

- Total return
- Maximum drawdown
- Sharpe ratio
- Sortino ratio
- Win rate
- Profit factor
- Trade expectancy
- Number of trades
- Performance versus SPY or another benchmark
- Out-of-sample validation performance

A configuration with lower raw return but substantially lower drawdown can be preferable to a configuration that produces a higher return by taking much more risk.

## Paper trading first

Before using live money:

1. Run historical backtests over multiple market regimes.
2. Confirm that the meta-model improves on genuinely newer validation data.
3. Confirm that promoted strategy configurations remain stable across additional holdout periods.
4. Paper trade the promoted configuration.
5. Compare live/paper behavior with backtest assumptions, including slippage and execution behavior.
6. Only consider live deployment after independent validation and risk review.

## Development notes

The project intentionally separates:

- **credentials** — `credentials_store.py`
- **strategy configuration** — `strategy_config.py` and `best_strategy_config.json`
- **strategy/model logic** — `Tradingcore.py`
- **historical simulation** — `backtest.py`
- **meta-model training** — `train_meta_model.py`
- **configuration search** — `strategy_optimizer.py`
- **dashboard/process control** — `dashboard_server.py`

This makes it easier to test each component independently and prevents a single backtest from automatically changing live behavior without passing the conservative validation gates.


## Disclaimer

This software is experimental and may lose money. Historical or backtested performance does not guarantee future results. Use paper trading and appropriate risk controls before considering live trading.
