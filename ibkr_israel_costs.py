"""
Interactive Brokers Israel — Cost Calculator
=============================================

Calculates real commissions for trades through Interactive Israel
(MEXEM LTD - subsidiary regulated by CySEC).

Fee structure (2026):
- US Stocks/ETFs: $0.01 per share, min $2.50
- ILS → USD conversion: ₪10 (one-time per conversion, up to 500K ILS)
- No account/management/inactivity fees
- Options: $2 per contract
- Futures: $3 per contract
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CommissionResult:
    commission_usd: float
    breakeven_pct: float          # % gain needed to cover commission
    minimum_profitable_pct: float  # % gain to be worthwhile (commission < 25% of profit)
    is_economical: bool            # Worth the trade?
    note: str


def calculate_commission(shares: float, price: float) -> CommissionResult:
    """
    Calculate IBKR Israel commission for US stock/ETF trade.

    Returns full analysis including breakeven needed.
    """
    notional = shares * price

    # IBKR Israel: $0.01/share, min $2.50
    commission_one_way = max(2.50, shares * 0.01)
    commission_round_trip = commission_one_way * 2

    # Breakeven: how much price must move to cover commission
    breakeven_pct = (commission_round_trip / notional * 100) if notional > 0 else 100

    # Minimum profitable: commission should be < 25% of expected profit
    # Bot expects ~3% gain → commission should be < 0.75%
    minimum_profitable_pct = breakeven_pct * 4

    # Economical if commission < 1% of position
    is_economical = breakeven_pct < 1.0

    if breakeven_pct > 2.0:
        note = "❌ עמלה גבוהה — לא כדאי"
    elif breakeven_pct > 1.0:
        note = "⚠️  עמלה גבולית — דרוש tp גבוה"
    elif breakeven_pct > 0.5:
        note = "🟡 עמלה סבירה"
    else:
        note = "✅ עמלה נמוכה — מצוין"

    return CommissionResult(
        commission_usd=commission_round_trip,
        breakeven_pct=breakeven_pct,
        minimum_profitable_pct=minimum_profitable_pct,
        is_economical=is_economical,
        note=note,
    )


def get_minimum_position_size() -> float:
    """
    Returns minimum position size in USD for economical trading.
    Round-trip commission is $5 minimum.
    We want commission < 0.5% of notional → minimum $1000 position.
    """
    return 1000.0  # $1,000 minimum to keep commission < 0.5%


def get_recommended_settings_ibkr_israel() -> dict:
    """
    Recommended bot settings for IBKR Israel.
    Optimized to make commissions economical.
    """
    return {
        "MAX_BUDGET": "10000",         # Need at least $10K to have meaningful positions
        "MAX_POSITION_PCT": "20",      # 20% = $2,000 → ~0.25% commission impact
        "MAX_OPEN_POSITIONS": "4",     # 4 positions × $2,000 each
        "MIN_BUY_SCORE": "65",         # Must be confident — paying real commission
        "TAKE_PROFIT_PCT": "15",       # 15% TP vs ~0.5% commission = 30:1
        "STOP_LOSS_PCT": "3.5",        # 3.5% SL vs 0.5% commission = 7:1
        "MIN_HOLD_MINUTES": "30",      # Don't flip-flop (each flip costs $5)
        "MIN_NOTIONAL_USD": "1000",    # Skip trades below $1,000
    }


def cost_warning_for_small_account(budget: float) -> str:
    """Warn if budget too small for commission structure."""
    if budget < 2000:
        return (
            "⚠️ <b>אזהרה — תקציב קטן ל-IBKR Israel</b>\n"
            f"תקציב: ${budget:.0f}\n"
            f"עמלה מינימלית round-trip: $5\n"
            f"זה {5/budget*100:.1f}% מהתקציב!\n\n"
            f"מומלץ: תקציב מינימלי $2,000\n"
            f"אופטימלי: $10,000+ למקסם רווחיות"
        )
    elif budget < 5000:
        return (
            "🟡 תקציב $2K-$5K עובד אבל פוזיציות קטנות.\n"
            "עמלה ~0.25% לכל קנייה+מכירה"
        )
    else:
        return ""


def estimate_monthly_commission_cost(budget: float, trades_per_month: int = 30) -> dict:
    """
    Estimate monthly trading costs at IBKR Israel.
    """
    # Average position = MAX_POSITION_PCT * BUDGET = 20% * BUDGET
    avg_position = budget * 0.20

    # Average shares per trade (varies by stock price; assume avg $100/share)
    avg_shares = avg_position / 100

    # Commission per trade (one way)
    commission_per_trade = max(2.50, avg_shares * 0.01)

    # Monthly cost (round trip = 2x)
    monthly_commission = commission_per_trade * 2 * trades_per_month

    # As % of budget
    monthly_pct = monthly_commission / budget * 100

    return {
        "budget": budget,
        "trades_per_month": trades_per_month,
        "avg_position_size": avg_position,
        "commission_per_trade_one_way": commission_per_trade,
        "monthly_total_commission": monthly_commission,
        "monthly_commission_pct": monthly_pct,
        "annual_total": monthly_commission * 12,
        "recommendation": (
            "✅ עמלות נמוכות מאוד" if monthly_pct < 0.5
            else "🟡 עמלות סבירות" if monthly_pct < 1.5
            else "⚠️ עמלות גבוהות — שקול להגדיל תקציב או להקטין מסחר"
        ),
    }
