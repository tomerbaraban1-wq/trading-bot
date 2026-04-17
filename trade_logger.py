import logging
from models import WebhookPayload, SentimentResult
from database import save_trade, close_trade, save_learning_entry

logger = logging.getLogger(__name__)


def log_trade_open(payload: WebhookPayload, sentiment: SentimentResult | None,
                   order_result: dict, qty: int) -> int:
    """Log a new trade opening. Returns the trade_id."""
    trade = {
        "ticker": payload.ticker.upper(),
        "action": payload.action.value,
        "qty": qty,
        "entry_price": payload.price,
        "trailing_stop_pct": None,
        "rsi": payload.rsi,
        "macd": payload.macd,
        "macd_signal": None,
        "bb_position": None,
        "volume_ratio": None,
        "sentiment_score": sentiment.score if sentiment else None,
        "sentiment_reasoning": sentiment.reasoning if sentiment else None,
    }
    trade_id = save_trade(trade)
    logger.info(f"Trade #{trade_id} opened: BUY {payload.ticker} x{qty} @ ${payload.price:.2f}")
    return trade_id


def log_trade_close(trade_id: int, exit_price: float, pnl_gross: float,
                    pnl_net: float, tax_reserved: float, fees: float = 0.0,
                    status: str = "closed"):
    """Log a trade closing."""
    close_trade(trade_id, exit_price, pnl_gross, pnl_net, tax_reserved, fees, status)
    logger.info(
        f"Trade #{trade_id} closed: exit=${exit_price:.2f} | "
        f"PnL gross=${pnl_gross:+.2f} | net=${pnl_net:+.2f} | tax=${tax_reserved:.2f}"
    )


def log_learning(trade_id: int, description: str, pattern_type: str,
                 indicators: dict, outcome: str, pnl: float):
    """Log a learning entry after trade close."""
    entry = {
        "trade_id": trade_id,
        "pattern_type": pattern_type,
        "description": description,
        "indicators_snapshot": indicators,
        "outcome": outcome,
        "pnl": pnl,
    }
    save_learning_entry(entry)
    logger.info(f"Learning logged for trade #{trade_id}: {pattern_type} - {outcome}")
