"""
Startup Validator & Reporter
=============================

Validates bot configuration on startup and sends a comprehensive
Telegram report of the current settings and health status.

CRITICAL: This module runs AFTER startup_checklist has verified
all critical configuration. This module ALWAYS assumes startup was successful.
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def validate_and_report() -> dict:
    """
    Run all startup validation checks.
    Returns dict with results and recommendations.

    NOTE: This assumes startup_checklist has already run and passed.
    """
    issues = []
    warnings = []
    info = []

    # ── Critical Settings ────────────────────────────────────────────
    critical = {
        "TELEGRAM_BOT_TOKEN":  ("🔔 Telegram", "הודעות לא יישלחו"),
        "TELEGRAM_CHAT_ID":    ("🔔 Telegram", "הודעות לא יישלחו"),
        "ALPACA_API_KEY":      ("📈 Broker", "לא ניתן לסחור"),
        "ALPACA_SECRET_KEY":   ("📈 Broker", "לא ניתן לסחור"),
        "WEBHOOK_SECRET":      ("🔐 Security", "webhook פתוח לכולם"),
    }
    for var, (category, impact) in critical.items():
        val = os.getenv(var, "")
        if not val or val == "your-value-here":
            issues.append(f"❌ {category}: {var} חסר → {impact}")

    # ── Trading Parameters ────────────────────────────────────────────
    params = {
        "MIN_BUY_SCORE":         ("65",   "ציון כניסה מינימלי"),
        "MAX_OPEN_POSITIONS":    ("4",    "מקסימום פוזיציות"),
        "MAX_HOLD_HOURS":        ("24",   "שעות מקסימום החזקה"),
        "STOP_LOSS_PCT":         ("3.5",  "Stop Loss %"),
        "TAKE_PROFIT_PCT":       ("15.0", "Take Profit %"),
        "MAX_POSITION_PCT":      ("15",   "% מקסימום לפוזיציה"),
        "BREAKEVEN_TRIGGER_PCT": ("0.5",  "% לActivate Break-even"),
        "MAX_DAILY_LOSS_PCT":    ("2.0",  "% הפסד יומי מקסימלי"),
    }
    for var, (recommended, desc) in params.items():
        val = os.getenv(var, "")
        if not val:
            warnings.append(f"⚠️  {var} לא מוגדר — משתמש בברירת מחדל: {recommended}")
        else:
            info.append(f"  ✅ {desc}: {val}")

    # ── Security Keys ─────────────────────────────────────────────────
    sec_keys = ["ADMIN_API_KEY", "ENCRYPTION_KEY", "JWT_SECRET"]
    for key in sec_keys:
        if not os.getenv(key):
            warnings.append(f"⚠️  {key} לא מוגדר (אופציונלי לאבטחה מתקדמת)")

    # ── Deployment Configuration ──────────────────────────────────────
    render_url = os.getenv("RENDER_EXTERNAL_URL", "")
    if not render_url:
        warnings.append("⚠️  RENDER_EXTERNAL_URL לא מוגדר (תצריך זה עבור Render deployment)")

    # ── Summary ───────────────────────────────────────────────────────
    status = "critical" if issues else "warning" if warnings else "ok"

    return {
        "status": status,
        "issues": issues,
        "warnings": warnings,
        "info": info,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def send_startup_report() -> None:
    """Send startup configuration report to Telegram with comprehensive health info."""
    try:
        from telegram_bot import send_message
        from config import settings
        import time

        result = validate_and_report()

        # ── "I'm awake!" header — detect how long the bot was DOWN ────────────
        # The supervisor writes data/last_alive.txt every health check. If the
        # gap is large, the PC was asleep/off (e.g. the 25h Shabbat sleep).
        awake_header = "🟢 <b>אני ער ופעיל!</b>"
        try:
            import os as _os_a, time as _time_a
            _alive_path = _os_a.path.join(_os_a.path.dirname(__file__), "data", "last_alive.txt")
            if _os_a.path.exists(_alive_path):
                with open(_alive_path, "r", encoding="utf-8") as _f:
                    _last = float(_f.read().strip() or 0)
                _down_min = (_time_a.time() - _last) / 60.0
                if _down_min >= 5:
                    if _down_min >= 90:
                        _dur = f"{_down_min/60:.1f} שעות"
                    else:
                        _dur = f"{_down_min:.0f} דקות"
                    awake_header = f"🟢 <b>אני ער! חזרתי לפעולה</b>\n😴 הייתי כבוי במשך <b>{_dur}</b>"
        except Exception:
            pass

        # Build the message
        lines = [
            awake_header,
            f"━━━━━━━━━━━━━━━━━━━━━━━━",
            f"📅 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "<b>⚙️ הגדרות מסחר:</b>",
            f"  📊 Min Score: <b>{settings.MIN_BUY_SCORE}</b>",
            f"  📂 Max Positions: <b>{settings.MAX_OPEN_POSITIONS}</b>",
            f"  ⏰ Max Hold: <b>{settings.MAX_HOLD_HOURS:.0f}h</b>",
            f"  🛑 Stop Loss: <b>{settings.STOP_LOSS_PCT}%</b>",
            f"  🎯 Take Profit: <b>{settings.TAKE_PROFIT_PCT}%</b>",
            f"  💰 Max Position: <b>{settings.MAX_POSITION_PCT}%</b>",
            f"  🔒 Breakeven: <b>+{settings.BREAKEVEN_TRIGGER_PCT}%</b>",
            f"  📉 Daily Loss Limit: <b>{settings.MAX_DAILY_LOSS_PCT}%</b>",
            "",
            "<b>📝 סטטוס ערוצים:</b>",
            f"  🔔 Telegram Chat: <b>{settings.TELEGRAM_CHAT_ID}</b>",
            f"  📈 Active Broker: <b>{settings.ACTIVE_BROKER or 'Alpaca'}</b>",
            f"  💾 Database: <b>SQLite</b>",
            f"  🤖 LLM: <b>Groq</b>",
        ]

        # Add broker info
        try:
            import broker
            acct = broker.get_account()
            cash = float(acct.get("cash", 0))
            equity = float(acct.get("equity", 0))
            lines.extend([
                "",
                "<b>💼 חשבון:</b>",
                f"  💰 Cash: <b>${cash:,.0f}</b>",
                f"  📊 Equity: <b>${equity:,.0f}</b>",
            ])
        except Exception as e:
            logger.warning(f"Could not get account info: {e}")

        if result["issues"]:
            lines.extend(["", "<b>❌ בעיות קריטיות:</b>"])
            for issue in result["issues"][:5]:
                lines.append(f"  {issue}")

        if result["warnings"]:
            lines.extend(["", "<b>⚠️ אזהרות:</b>"])
            for warning in result["warnings"][:3]:
                lines.append(f"  {warning}")

        status_emoji = "🔴" if result["issues"] else "🟡" if result["warnings"] else "🟢"
        lines.extend([
            "",
            f"{status_emoji} סטטוס: {'בעיות קריטיות' if result['issues'] else 'הגדרות חלקיות' if result['warnings'] else 'הכל תקין'}",
            "",
            "💡 <b>הדוקומנטציה:</b>",
            "<code>/health</code> — בדוק בריאות בוט",
            "<code>/help</code> — הראה כל הפקודות",
        ])

        await send_message("\n".join(lines))
        logger.info("Startup report sent to Telegram")

    except Exception as e:
        logger.error(f"Startup report failed: {e}")


async def get_runtime_diagnostics() -> dict:
    """Get comprehensive runtime diagnostics without Telegram.

    Useful for debugging issues via /health endpoint.
    """
    try:
        from config import settings
        import database
        import broker
        import time

        # Collect diagnostics
        diags = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "min_buy_score": settings.MIN_BUY_SCORE,
                "max_positions": settings.MAX_OPEN_POSITIONS,
                "max_hold_hours": settings.MAX_HOLD_HOURS,
                "stop_loss_pct": settings.STOP_LOSS_PCT,
                "take_profit_pct": settings.TAKE_PROFIT_PCT,
                "active_broker": settings.ACTIVE_BROKER,
                "telegram_chat_id": settings.TELEGRAM_CHAT_ID,
            },
            "database": {},
            "broker": {},
        }

        # Database info
        try:
            open_trades = database.get_open_trades()
            diags["database"] = {
                "open_positions": len(open_trades),
                "tickers": [t["ticker"] for t in open_trades],
            }
        except Exception as e:
            diags["database"]["error"] = str(e)

        # Broker info
        try:
            acct = broker.get_account()
            diags["broker"] = {
                "cash": float(acct.get("cash", 0)),
                "equity": float(acct.get("equity", 0)),
                "buying_power": float(acct.get("buying_power", 0)),
            }
        except Exception as e:
            diags["broker"]["error"] = str(e)

        return diags

    except Exception as e:
        return {"error": str(e)}
