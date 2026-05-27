"""
TradingView Advanced Signal Handler
=====================================

Processes TradingView webhook signals with bot's validation layer.
Allows TradingView to suggest trades, but our bot ALWAYS validates
with its own filters before executing.

Supports:
- Basic buy/sell signals
- Stop loss / take profit overrides from TradingView
- Strategy name tracking (which TV strategy fired)
- Confidence level
- Bot validation override (optional)
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TVSignal:
    """Enriched TradingView signal."""
    ticker: str
    action: str             # buy / sell / close
    price: float
    secret: str
    strategy_name: str = "manual"
    confidence: int = 50    # 0-100
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timeframe: str = "1d"
    notes: str = ""
    received_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class SignalProcessResult:
    accepted: bool
    reason: str
    final_action_taken: str = ""
    bot_validation_passed: bool = False
    score_at_processing: float = 0
    pro_grade: str = ""


def parse_tv_payload(payload: dict) -> TVSignal:
    """Parse TradingView webhook payload into structured signal."""
    return TVSignal(
        ticker=payload.get("ticker", "").upper().strip(),
        action=payload.get("action", "buy").lower().strip(),
        price=float(payload.get("price", 0) or 0),
        secret=payload.get("secret", ""),
        strategy_name=payload.get("strategy", "manual"),
        confidence=int(payload.get("confidence", 50)),
        stop_loss=float(payload.get("stop_loss", 0)) if payload.get("stop_loss") else None,
        take_profit=float(payload.get("take_profit", 0)) if payload.get("take_profit") else None,
        timeframe=payload.get("timeframe", "1d"),
        notes=payload.get("notes", ""),
    )


async def validate_tv_signal(signal: TVSignal) -> SignalProcessResult:
    """
    Validate TradingView signal with bot's filters.

    Returns whether to execute and why.
    """
    # 1. Auth check
    if signal.secret != os.getenv("WEBHOOK_SECRET", ""):
        return SignalProcessResult(
            accepted=False,
            reason="Invalid secret",
        )

    # 2. Bot validation toggle
    require_bot_validation = os.getenv("TV_REQUIRE_BOT_VALIDATION", "true").lower() == "true"

    if not require_bot_validation:
        # Auto-trust TradingView (NOT recommended)
        return SignalProcessResult(
            accepted=True,
            reason="Auto-trusted (TV_REQUIRE_BOT_VALIDATION=false)",
            bot_validation_passed=False,
        )

    # 3. Buy signals — run through bot's full validation
    if signal.action == "buy":
        try:
            # Get composite score
            from scoring import get_composite_score
            score_result = await asyncio.to_thread(
                get_composite_score, signal.ticker, signal.confidence // 10
            )
            score = score_result.get("composite_score", 0)

            if not score_result.get("should_buy"):
                return SignalProcessResult(
                    accepted=False,
                    reason=f"Bot score {score:.0f} < MIN_BUY_SCORE",
                    bot_validation_passed=False,
                    score_at_processing=score,
                )

            # Run Pro Entry analysis
            try:
                from pro_entry_system import pro_entry_gate
                pro_result = await asyncio.wait_for(
                    pro_entry_gate(signal.ticker, score),
                    timeout=20,
                )
                if not pro_result.get("should_enter"):
                    return SignalProcessResult(
                        accepted=False,
                        reason=f"Pro Entry rejected: {pro_result.get('skip_reason', 'Grade ' + pro_result.get('grade', '?'))}",
                        bot_validation_passed=False,
                        score_at_processing=score,
                        pro_grade=pro_result.get("grade", "?"),
                    )
                grade = pro_result.get("grade", "?")
            except Exception:
                grade = "?"

            return SignalProcessResult(
                accepted=True,
                reason=f"Bot approved: score={score:.0f}, grade={grade}",
                bot_validation_passed=True,
                score_at_processing=score,
                pro_grade=grade,
            )

        except Exception as e:
            logger.error(f"TV signal validation error: {e}")
            return SignalProcessResult(
                accepted=False,
                reason=f"Validation error: {e}",
            )

    # 4. Sell signals — always allow (closing risk = good)
    elif signal.action in ("sell", "close"):
        return SignalProcessResult(
            accepted=True,
            reason="Sell/close signal accepted",
            bot_validation_passed=True,
        )

    return SignalProcessResult(
        accepted=False,
        reason=f"Unknown action: {signal.action}",
    )


async def notify_tv_signal_received(signal: TVSignal, result: SignalProcessResult) -> None:
    """Send Telegram notification about TradingView signal."""
    try:
        from telegram_bot import send_message

        emoji = "🟢" if result.accepted else "🔴"
        action_emoji = "🛒" if signal.action == "buy" else "💸"

        lines = [
            f"{emoji} <b>TradingView Signal Received</b>",
            "━━━━━━━━━━━━━━━━",
            f"{action_emoji} <b>{signal.action.upper()}</b> {signal.ticker} @ ${signal.price:.2f}",
            f"📊 Strategy: {signal.strategy_name}",
            f"⏰ Timeframe: {signal.timeframe}",
            "",
            f"<b>Bot Validation:</b>",
        ]

        if result.accepted:
            lines.extend([
                f"  ✅ ACCEPTED",
                f"  📌 {result.reason}",
            ])
            if result.score_at_processing:
                lines.append(f"  📊 Bot Score: {result.score_at_processing:.0f}/100")
            if result.pro_grade:
                lines.append(f"  🏆 Pro Grade: {result.pro_grade}")
        else:
            lines.extend([
                f"  ❌ REJECTED",
                f"  📌 {result.reason}",
            ])

        if signal.stop_loss:
            lines.append(f"\n🛑 TV Stop Loss: ${signal.stop_loss:.2f}")
        if signal.take_profit:
            lines.append(f"🎯 TV Take Profit: ${signal.take_profit:.2f}")

        await send_message("\n".join(lines))

    except Exception as e:
        logger.debug(f"TV signal notification failed: {e}")


def get_tradingview_stats() -> dict:
    """Get statistics on TradingView signal usage."""
    try:
        import database
        conn = database.get_connection()
        # Get TV signals from last 30 days
        rows = conn.execute("""
            SELECT COUNT(*),
                   SUM(CASE WHEN pnl_gross > 0 THEN 1 ELSE 0 END),
                   AVG(pnl_gross)
            FROM trade_log
            WHERE sentiment_reasoning LIKE '%TradingView%'
            AND entry_time >= datetime('now','-30 days')
        """).fetchone()

        if not rows or not rows[0]:
            return {"tv_trades_30d": 0, "tv_win_rate": 0, "tv_avg_pnl": 0}

        total, wins, avg_pnl = rows
        return {
            "tv_trades_30d": total or 0,
            "tv_win_rate": (wins / total * 100) if total > 0 else 0,
            "tv_avg_pnl": avg_pnl or 0,
        }
    except Exception:
        return {"error": "Unable to fetch stats"}


# Sample Pine Script that pairs with this bot (for /tv_pine command)
PINE_SCRIPT_TEMPLATE = '''//@version=5
// Bot-compatible TradingView strategy
// Replace YOUR_SECRET with your WEBHOOK_SECRET from .env
strategy("My Bot Strategy", overlay=true)

// === Inputs ===
rsiPeriod    = input.int(14, "RSI Period")
rsiBuyMax    = input.int(42, "RSI Buy Zone Max")
rsiBuyMin    = input.int(28, "RSI Buy Zone Min")
volMinRatio  = input.float(0.85, "Min Volume Ratio")
requireSMA50 = input.bool(true, "Require above SMA50")
secretKey    = input.string("YOUR_SECRET", "Webhook Secret")

// === Indicators ===
rsi    = ta.rsi(close, rsiPeriod)
sma20  = ta.sma(close, 20)
sma50  = ta.sma(close, 50)
sma200 = ta.sma(close, 200)
volAvg = ta.sma(volume, 20)
volRatio = volume / volAvg

// === Buy Conditions ===
trend_ok    = (not requireSMA50) or close > sma50
volume_ok   = volRatio >= volMinRatio
rsi_in_zone = rsi >= rsiBuyMin and rsi <= rsiBuyMax

buySignal = trend_ok and volume_ok and rsi_in_zone

// === Plot ===
plotshape(buySignal, "BUY", shape.triangleup, location.belowbar, color.green)

// === Webhook Alert ===
if buySignal
    alert('{"secret":"' + secretKey + '","ticker":"' + syminfo.ticker + '","action":"buy","price":' + str.tostring(close) + ',"strategy":"pullback_v1"}', alert.freq_once_per_bar)
'''
