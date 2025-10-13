from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel


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
