"""
Profit Maximizer
==================

Aggressive profit optimization tuned for IBKR Israel commission structure.

Key strategies:
1. Scale-in on winners (+3%, +6% add more)
2. Staged profit taking (sell 33% at +5%, 33% at +10%, 25% at +15%)
3. Asymmetric stop/target (3.5% SL : 15% TP = 4.3:1)
4. Avoid small positions (<$1,000) due to commission impact
5. Reinvest profits (compound growth)
6. Tax-aware: hold short-term winners 12 months for LTCG
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class ProfitOptimizationConfig:
    """Maximum profit settings — calibrated for IBKR Israel."""

    # Position sizing — biggish to amortize commissions
    min_position_usd: float = 1500.0       # Below this, commissions eat profit
    max_position_pct: float = 25.0          # Up to 25% for high-conviction
    base_position_pct: float = 18.0         # Default 18% (was 15%)

    # Entry — selective but not too tight
    min_buy_score: int = 70                 # High bar
    pro_grade_required: str = "B"           # B or A only

    # Profit taking — STAGED
    profit_stage_1_pct: float = 5.0         # Sell 33% at +5%
    profit_stage_2_pct: float = 10.0        # Sell 33% at +10%
    profit_stage_3_pct: float = 18.0        # Sell 25% at +18%
    final_runner_pct: float = 9.0           # Last 9% rides with tight trail

    # Scale-in on winners
    scale_in_trigger_pct: float = 3.0       # When position is up 3%
    scale_in_size_pct: float = 50.0         # Add 50% more
    max_scale_ins: int = 1                  # Only once per position

    # Stop loss — tight but not too tight
    initial_stop_pct: float = 3.0           # 3% initial (was 3.5%)
    breakeven_at_pct: float = 1.0           # Move to BE at +1%
    trail_after_pct: float = 3.0            # Start trailing at +3%

    # Time management — let winners run, cut losers fast
    max_hold_winners: int = 14 * 24         # 14 days for winners
    max_hold_neutral: int = 48              # 2 days for flat positions
    max_hold_losers: int = 8                # 8 hours for losers

    # Risk management
    max_daily_loss_pct: float = 2.0
    max_consecutive_losses: int = 3
    pause_after_drawdown_pct: float = 5.0   # Pause if down 5% from peak


def get_profit_max_config() -> ProfitOptimizationConfig:
    """Get current profit maximizer config."""
    return ProfitOptimizationConfig()


def calculate_optimal_position_size(
    score: float,
    available_cash: float,
    total_budget: float,
    current_drawdown_pct: float = 0,
) -> dict:
    """
    Calculate optimal position size for max profit.

    Logic:
    - Higher score → larger position
    - Drawdown → smaller positions
    - Always above $1,500 minimum (commission economics)
    """
    config = get_profit_max_config()

    # Base sizing by score
    if score >= 85:
        size_pct = config.max_position_pct        # 25%
    elif score >= 78:
        size_pct = config.max_position_pct * 0.85  # 21%
    elif score >= 72:
        size_pct = config.base_position_pct        # 18%
    elif score >= 68:
        size_pct = config.base_position_pct * 0.75 # 13.5%
    else:
        size_pct = config.base_position_pct * 0.55 # 10%

    # Drawdown protection
    if current_drawdown_pct < -3:
        size_pct *= 0.7
    if current_drawdown_pct < -5:
        size_pct *= 0.5

    notional = total_budget * (size_pct / 100)
    notional = min(notional, available_cash * 0.95)  # 5% buffer

    # Minimum check — commission economics
    if notional < config.min_position_usd:
        if available_cash >= config.min_position_usd:
            notional = config.min_position_usd
        else:
            return {
                "notional": 0,
                "size_pct": 0,
                "skip": True,
                "reason": f"Too small — minimum ${config.min_position_usd}",
            }

    return {
        "notional": notional,
        "size_pct": notional / total_budget * 100,
        "skip": False,
        "reason": f"Score {score:.0f} → {size_pct:.1f}% sizing",
    }


def get_staged_profit_targets(entry_price: float) -> dict:
    """
    Generate staged profit-taking levels for max profit.

    Strategy:
    - +5%:  Sell 33%, move stop to entry (locked profit)
    - +10%: Sell 33%, tight trail
    - +18%: Sell 25%, ride final 9% with very tight trail
    """
    config = get_profit_max_config()

    return {
        "stage_1": {
            "price": entry_price * (1 + config.profit_stage_1_pct / 100),
            "qty_pct": 33,
            "new_stop_pct": 0.5,
            "description": f"+{config.profit_stage_1_pct}% — sell 33%, BE+0.5%",
        },
        "stage_2": {
            "price": entry_price * (1 + config.profit_stage_2_pct / 100),
            "qty_pct": 33,
            "new_stop_pct": config.profit_stage_1_pct + 1,
            "description": f"+{config.profit_stage_2_pct}% — sell 33%, lock +6%",
        },
        "stage_3": {
            "price": entry_price * (1 + config.profit_stage_3_pct / 100),
            "qty_pct": 25,
            "new_stop_pct": config.profit_stage_2_pct + 2,
            "description": f"+{config.profit_stage_3_pct}% — sell 25%, lock +12%",
        },
        "runner": {
            "qty_pct": 9,
            "trail_pct": 1.5,
            "description": "Final 9% rides with 1.5% trail",
        },
    }


def should_scale_in(
    ticker: str,
    current_plpc: float,
    score_now: float,
    score_at_entry: float,
    scale_ins_done: int = 0,
) -> tuple[bool, str]:
    """
    Decide if we should add to a winning position.

    Conditions:
    1. Position up at least 3% (config.scale_in_trigger_pct)
    2. Score still strong (>= entry score - 5)
    3. Haven't scaled in too many times
    """
    config = get_profit_max_config()

    if scale_ins_done >= config.max_scale_ins:
        return False, "Already scaled in maximum times"

    if current_plpc < config.scale_in_trigger_pct:
        return False, f"Position only +{current_plpc:.1f}% — need +{config.scale_in_trigger_pct}%"

    if score_now < score_at_entry - 5:
        return False, f"Score weakened {score_at_entry:.0f} → {score_now:.0f}"

    return True, f"✅ Scale in! Position +{current_plpc:.1f}%, score still strong"


def get_dynamic_stop_loss(
    entry_price: float,
    current_price: float,
    high_watermark: float,
    days_held: float,
) -> float:
    """
    Dynamic stop loss that tightens as profit grows.

    Logic:
    - Below +1%:   Initial stop (3%)
    - +1% to +3%:  Breakeven
    - +3% to +5%:  Lock +1%
    - +5% to +10%: Lock +3.5%
    - +10%+:       Trail with 1.5% from high
    """
    config = get_profit_max_config()
    plpc = (current_price - entry_price) / entry_price * 100

    if plpc < 1.0:
        return entry_price * (1 - config.initial_stop_pct / 100)
    elif plpc < 3.0:
        return entry_price * 1.001  # Breakeven + 0.1%
    elif plpc < 5.0:
        return entry_price * 1.01   # Lock +1%
    elif plpc < 10.0:
        return entry_price * 1.035  # Lock +3.5%
    elif plpc < 15.0:
        return high_watermark * 0.985  # Trail 1.5%
    else:
        return high_watermark * 0.99   # Tighter trail at high profit


def estimate_monthly_profit(
    budget: float,
    trades_per_month: int = 25,
    win_rate: float = 55.0,
    avg_win_pct: float = 5.0,
    avg_loss_pct: float = -2.5,
    avg_position_pct: float = 18.0,
) -> dict:
    """
    Estimate monthly profit with current settings.
    """
    avg_position = budget * (avg_position_pct / 100)

    wins_per_month = trades_per_month * (win_rate / 100)
    losses_per_month = trades_per_month - wins_per_month

    gross_profit = wins_per_month * (avg_position * avg_win_pct / 100)
    gross_loss = losses_per_month * (avg_position * abs(avg_loss_pct) / 100)
    net_profit = gross_profit - gross_loss

    # IBKR Israel commission
    avg_shares = avg_position / 100  # assume avg $100/share
    commission_per_trade = max(2.50, avg_shares * 0.01) * 2  # round trip
    total_commissions = commission_per_trade * trades_per_month

    after_commission = net_profit - total_commissions
    monthly_return_pct = after_commission / budget * 100
    annual_return_pct = monthly_return_pct * 12

    return {
        "budget": budget,
        "trades_per_month": trades_per_month,
        "win_rate": win_rate,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_before_commission": net_profit,
        "total_commissions": total_commissions,
        "net_after_commission": after_commission,
        "monthly_return_pct": monthly_return_pct,
        "annual_return_pct": annual_return_pct,
        "interpretation": (
            "🚀 רווחי מעולה" if monthly_return_pct > 5
            else "✅ רווחי טוב" if monthly_return_pct > 2
            else "🟡 רווחי גבולי" if monthly_return_pct > 0
            else "❌ הפסד צפוי"
        ),
    }


def get_optimal_env_settings() -> dict:
    """Get optimal .env settings for max profit at IBKR Israel."""
    return {
        # Budget
        "MAX_BUDGET": "10000",
        "MAX_POSITION_PCT": "18",
        "MAX_OPEN_POSITIONS": "4",

        # Entry — selective for max conviction
        "MIN_BUY_SCORE": "70",
        "MIN_VOLUME_RATIO": "0.85",
        "REQUIRE_ABOVE_SMA50": "true",

        # Exit — staged profit taking
        "TAKE_PROFIT_PCT": "15",
        "STOP_LOSS_PCT": "3.0",
        "TRAILING_STOP_PCT": "1.5",
        "BREAKEVEN_TRIGGER_PCT": "1.0",
        "PROFIT_PROTECT_ENABLED": "true",
        "PROFIT_PROTECT_PEAK_PCT": "5.0",
        "PROFIT_PROTECT_FLOOR_PCT": "2.0",

        # Time — let winners run, cut losers fast
        "MAX_HOLD_HOURS": "336",  # 14 days for winners
        "MIN_HOLD_MINUTES": "20",

        # Drawdown
        "MAX_DAILY_LOSS_PCT": "2.0",
        "MAX_WEEKLY_LOSS_PCT": "5.0",
        "MAX_CONSECUTIVE_LOSSES": "3",

        # IBKR Israel
        "ACTIVE_BROKER": "ibkr",
        "IBKR_HOST": "127.0.0.1",
        "IBKR_PORT": "7497",
    }
