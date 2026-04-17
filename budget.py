import logging
from config import settings
import broker

logger = logging.getLogger(__name__)


def check_can_buy(price: float) -> tuple[bool, int, str]:
    """
    Check if we can buy at this price within budget.
    Returns (can_buy, max_qty, reason).
    """
    if price <= 0:
        return False, 0, "Invalid price"

    try:
        account = broker.get_account()
        positions = broker.get_positions()
    except Exception as e:
        return False, 0, f"Failed to get account info: {e}"

    # Total invested in positions
    positions_value = sum(p["market_value"] for p in positions)

    # Budget remaining
    budget_remaining = settings.MAX_BUDGET - positions_value
    if budget_remaining <= 0:
        return False, 0, f"Budget exhausted: ${positions_value:,.2f} / ${settings.MAX_BUDGET:,.2f} used"

    # Max per single position (% of total budget)
    max_position_value = settings.MAX_BUDGET * (settings.MAX_POSITION_PCT / 100)

    # Available for this trade (min of budget remaining and max position size)
    available = min(budget_remaining, max_position_value)

    # Also limited by actual cash
    cash = account["cash"]
    available = min(available, cash)

    if available < price:
        return False, 0, f"Not enough available (${available:,.2f}) for 1 share at ${price:,.2f}"

    max_qty = int(available / price)
    if max_qty <= 0:
        return False, 0, f"Cannot afford any shares at ${price:,.2f}"

    return True, max_qty, f"Can buy up to {max_qty} shares (${available:,.2f} available)"


def calculate_position_size(price: float) -> int:
    """Calculate optimal position size within budget constraints."""
    can_buy, max_qty, reason = check_can_buy(price)
    if not can_buy:
        return 0
    return max_qty


def get_budget_status() -> dict:
    """Get full budget utilization status."""
    try:
        account = broker.get_account()
        positions = broker.get_positions()
    except Exception as e:
        logger.error(f"Failed to get budget status: {e}")
        return {
            "total_budget": settings.MAX_BUDGET,
            "cash_available": 0,
            "positions_value": 0,
            "open_pnl": 0,
            "budget_used_pct": 0,
        }

    positions_value = sum(p["market_value"] for p in positions)
    open_pnl = sum(p["unrealized_pl"] for p in positions)

    from database import get_tax_summary
    tax = get_tax_summary()

    return {
        "total_budget": settings.MAX_BUDGET,
        "cash_available": account["cash"],
        "positions_value": positions_value,
        "open_pnl": open_pnl,
        "realized_pnl_gross": tax["realized_pnl_gross"],
        "realized_pnl_net": tax["realized_pnl_net"],
        "tax_reserved": tax["tax_reserved"],
        "tax_credit": tax["tax_credit"],
        "budget_used_pct": (positions_value / settings.MAX_BUDGET * 100) if settings.MAX_BUDGET > 0 else 0,
    }
