# Backtest Engine

A simple, event-driven backtesting engine for trading strategies in Python. This engine allows you to backtest custom trading strategies on historical OHLC data, view trades, and evaluate performance.

## Features

- Event-driven architecture
- Support for market orders, stop-loss, and take-profit orders
- Commission and Foreign exchange support
- Visualise PnL, Balance, Equity, Drawdown, Winning and Losing Trades
- Evaluate performance with Profit Factor, Drawdown, Sharpe Ratio, Sortino Ratio etc.
- Time based Trade logging

## Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for package management.

1. **Clone the repository**

2. **Create a virtual environment and install dependencies:**
   `uv` will create a virtual environment and install the dependencies from `pyproject.toml`.

   ```bash
   uv venv
   source .venv/bin/activate
   uv sync
   ```

## TODO

- [X] Add support for edit stop loss and edit limit orders using signals.
- [ ] Add starter example strategies.
- [ ] Add support to export plots as HTML or PNG.
- [ ] Generate a starter signal dataframe to allow users to backtest instantly.
