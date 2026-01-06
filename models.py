from enum import Enum
from typing import Any, Optional

import pandera.pandas as pa
from pydantic import BaseModel
import pandas as pd

HOLD = 2


class OrderType(Enum):
    BUY = "Buy"
    SELL = "Sell"


class OrderAction(Enum):
    ENTRY = "Entry"
    EXIT = "Exit"
    EDIT_SL = "Edit_Stop_Loss"
    EDIT_LIMIT = "Edit_Limit"


class EntrySignal(Enum):
    BUY = 1
    SELL = -1


class ExitSignal(Enum):
    CLOSE_BUY = 1
    CLOSE_SELL = -1
    CLOSE_ALL = 0


class EditSignal(Enum):
    EDIT_SL = 1
    EDIT_LIMIT = -1


class Order(BaseModel):
    action: OrderAction
    trade_id: Optional[Any] = None
    volume: Optional[int] = None
    order_type: Optional[OrderType] = None
    stop_loss: Optional[float] = None
    limit: Optional[float] = None
    info: Optional[str] = None


class Results(BaseModel):
    biggest_win: float
    biggest_loss: float
    avg_win: float
    avg_loss: float
    total_win_trades: int
    total_loss_trades: int
    win_rate: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    risk_reward_ratio: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    total_net_profit: float
    final_balance: float


class OHLCSchema(pa.DataFrameModel):
    Date: pa.typing.Index[pd.Timestamp] = pa.Field(coerce=True, unique=True)
    Open: pa.typing.Series[float] = pa.Field(coerce=True, check_name=True)
    High: pa.typing.Series[float] = pa.Field(coerce=True, check_name=True)
    Low: pa.typing.Series[float] = pa.Field(coerce=True, check_name=True)
    Close: pa.typing.Series[float] = pa.Field(coerce=True, check_name=True)
    Entry_Signal: pa.typing.Series[int] = pa.Field(
        nullable=True, default=HOLD, check_name=True
    )
    Exit_Signal: pa.typing.Series[int] = pa.Field(
        nullable=True, default=HOLD, check_name=True
    )
    Edit_Signal: pa.typing.Series[int] = pa.Field(
        nullable=True, default=HOLD, check_name=True
    )
    Signal_Volume: pa.typing.Series[int] = pa.Field(
        coerce=True, nullable=True, default=0, check_name=True
    )
    Signal_Stop_Loss: pa.typing.Series[float] = pa.Field(
        coerce=True, nullable=True, default=0.0, check_name=True
    )
    Signal_Limit: pa.typing.Series[float] = pa.Field(
        coerce=True, nullable=True, default=0.0, check_name=True
    )


class TradesSchema(pa.DataFrameModel):
    Trade_ID: pa.typing.Index[int] = pa.Field(unique=True, check_name=True)
    State: pa.typing.Series[str] = pa.Field(isin=["Open", "Closed"], check_name=True)
    Order_Type: pa.typing.Series[str] = pa.Field(
        isin=[OrderType.BUY.value, OrderType.SELL.value], check_name=True
    )
    Volume: pa.typing.Series[int] = pa.Field(gt=0, check_name=True)
    Open_Time: pa.typing.Series[pd.Timestamp] = pa.Field(check_name=True)
    Open_Price: pa.typing.Series[float] = pa.Field(check_name=True)
    Close_Time: pa.typing.Series[pd.Timestamp] = pa.Field(
        nullable=True, check_name=True
    )
    Close_Price: pa.typing.Series[float] = pa.Field(nullable=True, check_name=True)
    Stop_Loss: pa.typing.Series[float] = pa.Field(check_name=True)
    Limit: pa.typing.Series[float] = pa.Field(check_name=True)
    Info: pa.typing.Series[str] = pa.Field(nullable=True, default="", check_name=True)
    Profit: pa.typing.Series[float] = pa.Field(
        nullable=True, coerce=True, check_name=True
    )
    Commission: pa.typing.Series[float] = pa.Field(
        nullable=True, coerce=True, default=0.0, check_name=True
    )
    Net_Profit: pa.typing.Series[float] = pa.Field(
        nullable=True, coerce=True, check_name=True
    )
    Cumulative_Profit: pa.typing.Series[float] = pa.Field(
        nullable=True, coerce=True, check_name=True
    )
    Balance: pa.typing.Series[float] = pa.Field(
        nullable=True, coerce=True, check_name=True
    )
    Drawdown: pa.typing.Series[float] = pa.Field(
        nullable=True, coerce=True, check_name=True
    )
