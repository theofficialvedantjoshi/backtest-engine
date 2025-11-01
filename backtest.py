from typing import Any, List, cast

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import display
from pandera.errors import SchemaErrors

from models import OHLCSchema, Order, OrderAction, OrderType, Signal, TradesSchema

pio.templates.default = "plotly_dark"


class Orders:

    def __init__(self):
        self.orders: List[Order] = []

    def open_trade(
        self,
        volume: int,
        order_type: OrderType,
        stop_loss: float = 0.0,
        limit: float = 0.0,
        info: str = "",
        trade_id: Any = None,
    ) -> None:
        order = Order(
            action=OrderAction.ENTRY,
            trade_id=trade_id,
            volume=volume,
            order_type=order_type,
            stop_loss=stop_loss,
            limit=limit,
            info=info,
        )
        self.orders.append(order)

    def close_trade(self, trade_id: Any) -> None:
        order = Order(
            action=OrderAction.EXIT,
            trade_id=trade_id,
        )
        self.orders.append(order)

    def edit_sl(self, trade_id: Any, stop_loss: float) -> None:
        order = Order(
            action=OrderAction.EDIT_SL,
            trade_id=trade_id,
            stop_loss=stop_loss,
        )
        self.orders.append(order)

    def edit_limit(self, trade_id: Any, limit: float) -> None:
        order = Order(
            action=OrderAction.EDIT_LIMIT,
            trade_id=trade_id,
            limit=limit,
        )
        self.orders.append(order)


class Backtester:

    def __init__(
        self,
        starting_balance: float,
        ohlc_data: pd.DataFrame,
        currency: str = "INR",
        exchange_rate: float = 1.0,
        commission: float = 0.0,
    ):
        self.exchange_rate: float
        self.commission: float
        self.currency: str
        self.starting_balance: float
        self.ohlc_data: pd.DataFrame

        self.set_starting_balance(starting_balance, currency)
        self.set_exchange_rate(exchange_rate)
        self.set_commission(commission)
        self.set_ohlc_data(ohlc_data)

        self.orders: Orders = Orders()
        self.trades: pd.DataFrame = cast(pd.DataFrame, TradesSchema.example(size=0))

    def set_starting_balance(
        self, starting_balance: float, currency: str = "INR"
    ) -> None:
        if starting_balance <= 0.0:
            raise ValueError("Starting balance must be greater than 0.0")
        self.starting_balance = starting_balance
        self.currency = currency

    def set_exchange_rate(self, exchange_rate: float) -> None:
        if exchange_rate <= 0.0:
            raise ValueError("Exchange rate must be greater than 0.0")
        self.exchange_rate = exchange_rate

    def set_commission(self, commission: float) -> None:
        self.commission = -1.0 * commission

    def set_ohlc_data(self, ohlc_data: pd.DataFrame) -> None:
        try:
            validated_data = OHLCSchema.validate(ohlc_data.copy())
            self.ohlc_data = validated_data
        except SchemaErrors as e:
            raise ValueError(f"OHLC data validation failed: {e}") from e

    def run_backtest(self) -> pd.DataFrame:
        for i in self.ohlc_data.index:
            data = self.ohlc_data.loc[i]
            print("DATA", data)
            self.orders = Orders()

            open_trades = self.trades[self.trades["State"] == "Open"]

            if (
                data["Signal"] == Signal.BUY.value
                and pd.notna(data["Signal_Volume"])
                and open_trades.empty
            ):  # Open Long Position
                self.orders.open_trade(
                    volume=data["Signal_Volume"],
                    order_type=OrderType.BUY,
                    stop_loss=float(data["Signal_Stop_Loss"]),
                    limit=float(data["Signal_Limit"]),
                    info="",
                )
            elif (
                data["Signal"] == Signal.SELL.value
                and pd.notna(data["Signal_Volume"])
                and open_trades.empty
            ):  # Open Short Position
                self.orders.open_trade(
                    volume=data["Signal_Volume"],
                    order_type=OrderType.SELL,
                    stop_loss=float(data["Signal_Stop_Loss"]),
                    limit=float(data["Signal_Limit"]),
                    info="",
                )
            elif (
                data["Signal"] == Signal.CLOSE_BUY.value
                and not open_trades[
                    open_trades["Order_Type"] == OrderType.BUY.value
                ].empty
            ):  # Close Long Position
                for trade_id, trade in open_trades[
                    open_trades["Order_Type"] == OrderType.BUY.value
                ].iterrows():
                    self.orders.close_trade(trade_id=trade_id)
            elif (
                data["Signal"] == Signal.CLOSE_SELL.value
                and not open_trades[
                    open_trades["Order_Type"] == OrderType.SELL.value
                ].empty
            ):  # Close Short Position
                for trade_id, trade in open_trades[
                    open_trades["Order_Type"] == OrderType.SELL.value
                ].iterrows():
                    self.orders.close_trade(trade_id=trade_id)
            elif (
                data["Signal"] == Signal.CLOSE_ALL.value and not open_trades.empty
            ):  # Close All Positions
                for trade_id, trade in open_trades.iterrows():
                    self.orders.close_trade(trade_id=trade_id)
            elif data["Signal"] == Signal.EDIT_SL.value and not open_trades.empty:
                for trade_id, trade in open_trades.iterrows():
                    self.orders.edit_sl(
                        trade_id=trade_id, stop_loss=float(data["Signal_Stop_Loss"])
                    )
            elif data["Signal"] == Signal.EDIT_LIMIT.value and not open_trades.empty:
                for trade_id, trade in open_trades.iterrows():
                    self.orders.edit_limit(
                        trade_id=trade_id, limit=float(data["Signal_Limit"])
                    )

            for order in self.orders.orders:
                if order.action == OrderAction.ENTRY:
                    if order.volume is None or order.order_type is None:
                        raise ValueError(
                            "Volume and Order_Type must be provided for opening a trade."
                        )
                    row = {
                        "State": "Open",
                        "Order_Type": order.order_type.value,
                        "Volume": order.volume,
                        "Open_Time": i,
                        "Open_Price": data["Open"],
                        "Close_Time": np.nan,
                        "Close_Price": np.nan,
                        "Stop_Loss": order.stop_loss,
                        "Limit": order.limit,
                        "Info": order.info,
                        "Profit": np.nan,
                    }
                    self.trades.loc[len(self.trades)] = pd.Series(row)
                elif order.action == OrderAction.EXIT:
                    trade_id = order.trade_id
                    if trade_id is None:
                        raise ValueError(
                            "Trade ID must be provided for closing a trade."
                        )
                    self.trades.loc[
                        trade_id,
                        ["State", "Close_Time", "Close_Price"],
                    ] = ["Closed", i, data["Open"]]
                elif order.action == OrderAction.EDIT_SL:
                    trade_id = order.trade_id
                    if trade_id is None:
                        raise ValueError(
                            "Trade ID must be provided for closing a trade."
                        )
                    self.trades.loc[trade_id, ["Stop_Loss"]] = [order.stop_loss]
                elif order.action == OrderAction.EDIT_LIMIT:
                    trade_id = order.trade_id
                    if trade_id is None:
                        raise ValueError(
                            "Trade ID must be provided for closing a trade."
                        )
                    self.trades.loc[trade_id, ["Limit"]] = [order.limit]

            open_trades = self.trades[self.trades["State"] == "Open"]

            for idx in open_trades.index:
                trade = open_trades.loc[idx]
                if trade["Order_Type"] == OrderType.BUY.value:
                    if trade["Stop_Loss"] >= data["Low"] and trade["Stop_Loss"] >= 0.0:
                        self.trades.loc[idx, ["State", "Close_Time", "Close_Price"]] = [
                            "Closed",
                            i,
                            trade["Stop_Loss"],
                        ]
                    elif trade["Limit"] <= data["High"] and trade["Limit"] >= 0.0:
                        self.trades.loc[idx, ["State", "Close_Time", "Close_Price"]] = [
                            "Closed",
                            i,
                            trade["Limit"],
                        ]
                elif trade["Order_Type"] == OrderType.SELL.value:
                    if trade["Stop_Loss"] <= data["High"] and trade["Stop_Loss"] >= 0.0:
                        self.trades.loc[idx, ["State", "Close_Time", "Close_Price"]] = [
                            "Closed",
                            i,
                            trade["Stop_Loss"],
                        ]
                    elif trade["Limit"] >= data["Low"] and trade["Limit"] >= 0.0:
                        self.trades.loc[idx, ["State", "Close_Time", "Close_Price"]] = [
                            "Closed",
                            i,
                            trade["Limit"],
                        ]

        final_time = self.ohlc_data.index[-1]
        final_close = self.ohlc_data.iloc[-1]["Close"]

        # Close all open trades at the end of the backtest.
        self.trades.loc[
            self.trades["State"] == "Open",
            ["State", "Close_Time", "Close_Price"],
        ] = [
            "Closed",
            final_time,
            final_close,
        ]

        def get_profit(x: pd.Series):
            if x["Order_Type"] == OrderType.BUY.value:
                return (
                    (x["Close_Price"] - x["Open_Price"])
                    * x["Volume"]
                    * self.exchange_rate
                )
            elif x["Order_Type"] == OrderType.SELL.value:
                return (
                    (x["Open_Price"] - x["Close_Price"])
                    * x["Volume"]
                    * self.exchange_rate
                )
            else:
                raise ValueError("Invalid Order_Type")

        self.trades["Profit"] = self.trades.apply(get_profit, axis=1).round(2)
        self.trades["Commission"] = self.commission * self.trades["Volume"]
        self.trades["Net_Profit"] = self.trades["Profit"] + self.trades["Commission"]
        self.trades["Cumulative_Profit"] = self.trades["Net_Profit"].cumsum()
        self.trades["Balance"] = (
            self.starting_balance + self.trades["Cumulative_Profit"]
        )

        return self.trades

    def evaluate_backtest(self, periods_per_year: int = 252) -> dict:
        results: dict = dict()

        print("RESULTS")
        print("=" * 40)

        biggest_win = self.trades["Net_Profit"].max()
        results["biggest_win"] = biggest_win
        print(f"Biggest Win: {biggest_win:.2f} {self.currency}")

        biggest_loss = self.trades["Net_Profit"].min()
        results["biggest_loss"] = biggest_loss
        print(f"Biggest Loss: {biggest_loss:.2f} {self.currency}")

        win_trades = self.trades[self.trades["Net_Profit"] > 0]
        display(win_trades)

        loss_trades = self.trades[self.trades["Net_Profit"] <= 0]
        display(loss_trades)

        avg_win = win_trades["Net_Profit"].mean()
        results["avg_win"] = avg_win
        print(f"Average Win: {avg_win:.2f} {self.currency}")

        avg_loss = loss_trades["Net_Profit"].mean()
        results["avg_loss"] = avg_loss
        print(f"Average Loss: {avg_loss:.2f} {self.currency}")

        total_win_trades = win_trades.shape[0]
        results["total_win_trades"] = total_win_trades
        print(f"Total Winning Trades: {total_win_trades}")

        total_loss_trades = loss_trades.shape[0]
        results["total_loss_trades"] = total_loss_trades
        print(f"Total Losing Trades: {total_loss_trades}")

        if (total_win_trades + total_loss_trades) == 0:
            win_rate = 0.0
        else:
            win_rate = total_win_trades / (total_win_trades + total_loss_trades) * 100
            print(f"Win Rate: {win_rate:.2f}%")
        results["win_rate"] = win_rate

        gross_profit = win_trades["Net_Profit"].sum()
        gross_loss = abs(loss_trades["Net_Profit"].sum())
        results["gross_profit"] = gross_profit
        results["gross_loss"] = gross_loss
        if gross_loss > 0.0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = float("inf")

        results["profit_factor"] = profit_factor
        print(f"Profit Factor: {profit_factor:.2f}")

        if avg_loss == 0.0:
            risk_reward_ratio = float("inf")
        else:
            risk_reward_ratio = abs(avg_win / avg_loss)
            print(f"Risk-Reward Ratio: {risk_reward_ratio:.2f}")
        results["risk_reward_ratio"] = risk_reward_ratio

        trades_by_ordertype = self.trades.groupby("Order_Type", as_index=False)[
            "Net_Profit"
        ].sum()
        display(trades_by_ordertype)

        plot_ordertype = px.bar(trades_by_ordertype, x="Order_Type", y="Net_Profit")
        display(plot_ordertype)

        self.trades["Drawdown"] = (
            self.trades["Cumulative_Profit"].cummax() - self.trades["Cumulative_Profit"]
        )

        plot_drawdown = px.line(
            self.trades, x="Close_Time", y="Drawdown", title="Drawdown Over Time"
        )
        display(plot_drawdown)

        max_drawdown = self.trades["Drawdown"].max()
        results["max_drawdown"] = max_drawdown
        print(f"Maximum Drawdown: {max_drawdown:.2f} {self.currency}")

        shifted_balance = self.trades["Balance"].shift(1).fillna(self.starting_balance)
        returns = self.trades["Net_Profit"] / shifted_balance
        if returns.std() == 0.0:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(periods_per_year)
        results["sharpe_ratio"] = sharpe_ratio
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

        negative_returns = returns[returns < 0]
        if negative_returns.std() == 0.0:
            sortino_ratio = 0.0
        else:
            sortino_ratio = (
                returns.mean() / negative_returns.std() * np.sqrt(periods_per_year)
            )
        results["sortino_ratio"] = sortino_ratio
        print(f"Sortino Ratio: {sortino_ratio:.2f}")

        total_net_profit = self.trades["Net_Profit"].sum()
        results["total_net_profit"] = total_net_profit
        print(f"Total Net Profit: {total_net_profit:.2f} {self.currency}")

        final_balance = self.starting_balance + total_net_profit
        results["final_balance"] = final_balance
        print(f"Final Balance: {final_balance:.2f} {self.currency}")

        print("=" * 40)

        return results

    def visualize_backtest(self, num_trades: int = 0) -> go.Figure:
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=self.ohlc_data.index,
                    open=self.ohlc_data["Open"],
                    high=self.ohlc_data["High"],
                    low=self.ohlc_data["Low"],
                    close=self.ohlc_data["Close"],
                    name="OHLC Data",
                )
            ]
        )

        fig.update_layout(height=600, title="Backtest Trades")
        fig.update_layout(xaxis_rangeslider_visible=False)
        if num_trades:
            for _, trade in self.trades.tail(num_trades).iterrows():
                color = "green" if trade["Net_Profit"] > 0 else "red"
                fig.add_shape(
                    type="line",
                    x0=trade["Open_Time"],
                    y0=trade["Open_Price"],
                    x1=trade["Close_Time"],
                    y1=trade["Close_Price"],
                    line=dict(
                        color=color,
                        width=5,
                    ),
                )
        else:
            for _, trade in self.trades.iterrows():
                color = "green" if trade["Net_Profit"] > 0 else "red"
                fig.add_shape(
                    type="line",
                    x0=trade["Open_Time"],
                    y0=trade["Open_Price"],
                    x1=trade["Close_Time"],
                    y1=trade["Close_Price"],
                    line=dict(
                        color=color,
                        width=5,
                    ),
                )

        return fig

    def plot_pnl(self) -> go.Figure:
        fig = px.line(
            self.trades, x="Open_Time", y="Cumulative_Profit", title="PnL Graph"
        )
        return fig

    def plot_balance(self) -> go.Figure:
        fig = px.line(self.trades, x="Close_Time", y="Balance", title="Balance Graph")
        return fig

    def export_to_json(self, filename: str, symbol: str = "") -> bool:
        import ujson

        ohlc_data = self.ohlc_data.copy()
        ohlc_data.index = ohlc_data.index.astype(str)

        trades = self.trades.copy()
        trades["Open_Time"] = trades["Open_Time"].astype(str)
        trades["Close_Time"] = trades["Close_Time"].astype(str)
        results = self.evaluate_backtest()

        data = {
            "symbol": symbol,
            "starting_balance": self.starting_balance,
            "exchange_rate": self.exchange_rate,
            "ohlc_history": ohlc_data.to_dict("records"),
            "trade_history": trades.to_dict("records"),
            "results": results,
        }

        with open(filename, "w") as jsonfile:
            ujson.dump(data, jsonfile)
        return True
