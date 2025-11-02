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

## Usage

1. **Prepare your OHLC data**: Follow the data preprocessing logic in the `examples/crossover.ipynb` notebook.

2. **Define your strategy**: Create entry, exit and edit signals based on your strategy logic. Refer to `models.py` for signal definitions.

3. **Run the backtest**: Use the `Backtester` class to run your backtest with the prepared data and signals.

## TODO

- [X] Add support for edit stop loss and edit limit orders using signals.
- [X] Add starter example strategies.
- [X] Generate a starter signal dataframe to allow users to backtest instantly.
- [ ] Add support to export plots as HTML or PNG.
- [ ] Improve error handling and logging.
