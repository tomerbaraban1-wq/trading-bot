from pydantic import BaseModel, Field
from enum import Enum


class TradeAction(str, Enum):
    BUY = "buy"
    SELL = "sell"


class WebhookPayload(BaseModel):
    ticker: str
    action: TradeAction
    price: float
    close: float | None = None
    volume: float | None = None
    rsi: float | None = None
    macd: float | None = None
    interval: str | None = None
    secret: str


class SentimentResult(BaseModel):
    ticker: str
    score: int = Field(ge=1, le=10)
    headlines: list[str]
    reasoning: str
    timestamp: float


class TradeRecord(BaseModel):
    id: int | None = None
    ticker: str
    action: TradeAction
    qty: int
    entry_price: float
    entry_time: str
    sentiment_score: int | None = None
    rsi: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    bb_position: str | None = None
    volume_ratio: float | None = None
    trailing_stop_pct: float | None = None
    exit_price: float | None = None
    exit_time: str | None = None
    pnl_gross: float | None = None
    pnl_net: float | None = None
    tax_reserved: float | None = None
    fees: float = 0.0
    status: str = "open"


class TaxSummary(BaseModel):
    realized_pnl_gross: float = 0.0
    tax_reserved: float = 0.0
    tax_credit: float = 0.0
    realized_pnl_net: float = 0.0


class BudgetStatus(BaseModel):
    total_budget: float
    cash_available: float
    positions_value: float
    open_pnl: float
    realized_pnl_gross: float
    realized_pnl_net: float
    tax_reserved: float
    tax_credit: float


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    last_heartbeat: str
    open_positions: int
    budget_utilization_pct: float


class BrokerSwitch(BaseModel):
    broker: str  # alpaca_paper | alpaca_live | ibkr | oanda | tradier | tradier_live
