from enum import Enum
from typing import Any, Optional

import pandera as pa
from pydantic import BaseModel
import pandas as pd


class OrderType(Enum):
    BUY = "Buy"
    SELL = "Sell"


class OrderAction(Enum):
    ENTRY = "Entry"
    EXIT = "Exit"
    EDIT_SL = "Edit_Stop_Loss"
    EDIT_LIMIT = "Edit_Limit"


class Signal(Enum):
    BUY = 1
    SELL = -1
    CLOSE_BUY = 2
    CLOSE_SELL = -2
    CLOSE_ALL = 0
    EDIT_SL = 3
    EDIT_LIMIT = -3


class Order(BaseModel):
    action: OrderAction
    trade_id: Optional[Any] = None
    volume: Optional[int] = None
    order_type: Optional[OrderType] = None
    stop_loss: Optional[float] = None
    limit: Optional[float] = None
    info: Optional[str] = None


class OHLCSchema(pa.DataFrameModel):
    Date: pa.typing.Index[pd.Timestamp] = pa.Field(coerce=True)
    Open: pa.typing.Series[float] = pa.Field(coerce=True)
    High: pa.typing.Series[float] = pa.Field(coerce=True)
    Low: pa.typing.Series[float] = pa.Field(coerce=True)
    Close: pa.typing.Series[float] = pa.Field(coerce=True)
    Signal: pa.typing.Series[int] = pa.Field(nullable=True)
    Signal_Volume: pa.typing.Series[int] = pa.Field(coerce=True, nullable=True)
    Signal_Stop_Loss: pa.typing.Series[float] = pa.Field(
        coerce=True, nullable=True, default=0.0
    )
    Signal_Limit: pa.typing.Series[float] = pa.Field(
        coerce=True, nullable=True, default=0.0
    )


class TradesSchema(pa.DataFrameModel):
    Trade_ID: pa.typing.Index[int] = pa.Field(unique=True)
    State: pa.typing.Series[str] = pa.Field(isin=["Open", "Closed"])
    Order_Type: pa.typing.Series[str] = pa.Field(
        isin=[OrderType.BUY.value, OrderType.SELL.value]
    )
    Volume: pa.typing.Series[int] = pa.Field(gt=0)
    Open_Time: pa.typing.Series[pd.Timestamp]
    Open_Price: pa.typing.Series[float]
    Close_Time: pa.typing.Series[pd.Timestamp] = pa.Field(nullable=True)
    Close_Price: pa.typing.Series[float] = pa.Field(nullable=True)
    Stop_Loss: pa.typing.Series[float]
    Limit: pa.typing.Series[float]
    Info: pa.typing.Series[str] = pa.Field(nullable=True, default="")
    Profit: pa.typing.Series[float] = pa.Field(nullable=True, coerce=True)
    Commission: pa.typing.Series[float] = pa.Field(
        nullable=True, coerce=True, default=0.0
    )
    Net_Profit: pa.typing.Series[float] = pa.Field(nullable=True, coerce=True)
    Cumulative_Profit: pa.typing.Series[float] = pa.Field(nullable=True, coerce=True)
    Balance: pa.typing.Series[float] = pa.Field(nullable=True, coerce=True)
    Drawdown: pa.typing.Series[float] = pa.Field(nullable=True, coerce=True)
