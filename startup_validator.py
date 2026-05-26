"""
Startup Validator
==================

Validates bot configuration on startup and sends
a Telegram report of the current settings.

Helps catch misconfigured bots BEFORE they trade.
"""

import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def validate_and_report() -> dict:
    """
    Run all startup validation checks.
    Returns dict with results and recommendations.
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
    """Send startup configuration report to Telegram."""
    try:
        from telegram_bot import send_message
        from config import settings

        result = validate_and_report()

        # Build the message
        lines = [
            f"🚀 <b>בוט הופעל</b>",
            f"━━━━━━━━━━━━━━━━",
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
        ]

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
        ])

        await send_message("\n".join(lines))

    except Exception as e:
        logger.error(f"Startup report failed: {e}")
