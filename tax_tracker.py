import logging
from config import settings
from database import save_tax_event, get_tax_balance, get_tax_summary

logger = logging.getLogger(__name__)


def process_trade_close(trade_id: int, pnl_gross: float) -> dict:
    """
    Process tax implications when a trade closes.
    - Profitable: reserve TAX_RATE (25%) of profit, offset with any tax credits
    - Loss: add absolute loss as tax credit for future offsets

    Returns: {"tax_amount": float, "credit_used": float, "new_credit": float}
    """
    if pnl_gross > 0:
        return _handle_profit(trade_id, pnl_gross)
    elif pnl_gross < 0:
        return _handle_loss(trade_id, pnl_gross)
    else:
        return {"tax_amount": 0.0, "credit_used": 0.0, "new_credit": 0.0}


def _handle_profit(trade_id: int, pnl_gross: float) -> dict:
    """Reserve tax on profit, offset with available credits."""
    raw_tax = pnl_gross * settings.TAX_RATE
    tax_balance = get_tax_balance()
    available_credit = tax_balance["tax_credit"]

    credit_used = 0.0
    actual_tax = raw_tax

    # Offset with available tax credits
    if available_credit > 0:
        credit_used = min(available_credit, raw_tax)
        actual_tax = raw_tax - credit_used

        # Deduct from credit pool (negative credit event)
        if credit_used > 0:
            save_tax_event(trade_id, "tax_credit", -credit_used)
            logger.info(f"Trade #{trade_id}: Used ${credit_used:.2f} tax credit (offset)")

    # Reserve the remaining tax
    if actual_tax > 0:
        save_tax_event(trade_id, "tax_reserved", actual_tax)

    logger.info(
        f"Trade #{trade_id} PROFIT: PnL=${pnl_gross:.2f} | "
        f"Tax=${actual_tax:.2f} (credit used=${credit_used:.2f})"
    )
    return {"tax_amount": actual_tax, "credit_used": credit_used, "new_credit": 0.0}


def _handle_loss(trade_id: int, pnl_gross: float) -> dict:
    """Add loss as tax credit for future offset."""
    credit_amount = abs(pnl_gross) * settings.TAX_RATE
    save_tax_event(trade_id, "tax_credit", credit_amount)

    logger.info(
        f"Trade #{trade_id} LOSS: PnL=${pnl_gross:.2f} | "
        f"Tax credit added=${credit_amount:.2f}"
    )
    return {"tax_amount": 0.0, "credit_used": 0.0, "new_credit": credit_amount}


def get_report() -> dict:
    """Get full tax report."""
    summary = get_tax_summary()
    balance = get_tax_balance()
    return {
        **summary,
        "tax_rate": settings.TAX_RATE,
        "effective_tax": summary["tax_reserved"] - balance["tax_credit"],
    }
