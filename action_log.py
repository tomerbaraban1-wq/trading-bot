"""
Action Logger — הבוט מדווח על כל פעולה
=========================================

כל פעולה שהבוט עושה → הודעת טלגרם.

פעולות שמדווחות:
  🔍  סריקה התחילה
  📊  ציון מניה (כל מניה שנבדקה)
  ✅  מניה עברה את כל הפילטרים → קונה
  ❌  מניה נחסמה — עם סיבה
  🛡️  SPY Gate — שוק יורד
  ⏰  שוק סגור
  🛑  Stop Loss הופעל
  💰  Take Profit הושג
  🔒  Stop הוחמר
  ⚠️  מניה מתקרבת לסטופ
  💰  יציאה חלקית
  📰  חדשות חריגות
  🧠  למידה — תובנה חדשה
"""

import asyncio
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# מאגר פעולות לשליחה מצוברת (batch) — נשלח פעם אחת בסוף סריקה
_scan_actions: list[str] = []
_scan_start_time: float = 0
_last_sent: float = 0
_MIN_BATCH_INTERVAL = 30   # שניות מינימום בין הודעות מאגר (was 60)


def start_scan(n_candidates: int, cash: float) -> None:
    """קורא לפני כל סריקה."""
    global _scan_actions, _scan_start_time
    _scan_actions = []
    _scan_start_time = time.time()
    _scan_actions.append(f"🔍 סורק {n_candidates} מניות | 💵 ${cash:,.0f}")


def log_ticker(
    ticker: str,
    score: float,
    passed: bool,
    reason: str = "",
    extra: str = "",
) -> None:
    """
    מוסיף שורה לדוח הסריקה עבור מניה אחת.
    """
    if passed:
        bar = "🟩" * round(score / 10) + "⬜" * (10 - round(score / 10))
        _scan_actions.append(f"✅ <b>{ticker}</b>  {bar} {score:.0f}/100{(' — ' + extra) if extra else ''}")
    else:
        short_reason = _shorten(reason)
        _scan_actions.append(f"❌ {ticker}  {score:.0f}/100 — {short_reason}")


def log_event(icon: str, text: str) -> None:
    """פעולה כללית — מוסיף לדוח."""
    _scan_actions.append(f"{icon} {text}")


def _shorten(reason: str) -> str:
    """קצר סיבה לפורמט קצר."""
    mappings = {
        "should_buy": "ציון נמוך",
        "composite": "ציון נמוך",
        "below SMA50": "מתחת SMA50",
        "Earnings": "דוח קרוב",
        "earnings": "דוח קרוב",
        "ranging": "שוק בצד",
        "ADX": "ADX נמוך",
        "volume": "נפח נמוך",
        "knife": "יורד היום",
        "Buffett": "איכות נמוכה",
        "checklist": "נכשל בדיקות",
        "learning": "דפוס הפסד",
        "Pro gate": "Pro Gate",
        "pro": "Pro Gate",
        "drawdown": "Drawdown",
        "max.*positions": "מלא",
        "skip_trade": "AI מבטל",
        "blackout": "דוח קרוב",
        "downgrade": "Downgrade",
    }
    import re
    r = reason[:50]
    for key, val in mappings.items():
        if re.search(key, reason, re.IGNORECASE):
            return val
    return r


async def flush_scan_report() -> None:
    """
    שולח את כל הדוח שנאסף מהסריקה — הודעה אחת מרוכזת.
    אם עברו פחות מ-_MIN_BATCH_INTERVAL מאז הדוח הקודם — שולחים גרסה תמציתית בלבד.
    """
    global _last_sent

    if not _scan_actions:
        return

    now = time.time()
    _too_soon = (now - _last_sent < _MIN_BATCH_INTERVAL) if _last_sent > 0 else False
    _last_sent = now

    elapsed = now - _scan_start_time if _scan_start_time else 0

    try:
        from telegram_bot import send_message
        from datetime import datetime, timezone, timedelta

        now_il = datetime.now(timezone.utc) + timedelta(hours=3)
        time_str = now_il.strftime("%H:%M")

        # מפריד בין עברו לנכשלו
        passed   = [a for a in _scan_actions if a.startswith("✅")]
        failed   = [a for a in _scan_actions if a.startswith("❌")]
        events   = [a for a in _scan_actions if not a.startswith(("✅", "❌"))]

        lines = [f"📡 <b>דוח סריקה</b> | {time_str}  ({elapsed:.0f}s)", "━━━━━━━━━━━━━━━━"]

        # אירועים כלליים
        if events:
            lines.extend(events)

        # מניות שעברו
        if passed:
            lines.append("")
            lines.extend(passed)

        # מניות שנחסמו (עד 5)
        if failed:
            lines.append("")
            lines.extend(failed[:6])
            if len(failed) > 6:
                lines.append(f"  ...ועוד {len(failed)-6} נחסמו")

        # אם הכל ריק
        if not passed and not failed and not events:
            lines.append("😴 אין פעילות בסריקה זו")

        # If too soon since last report — send only a brief summary (1 line)
        if _too_soon:
            n_pass = len(passed)
            n_fail = len(failed)
            msg = (
                f"📡 <b>סריקה {time_str}</b> ({elapsed:.0f}s) — "
                f"✅ {n_pass} עברו | ❌ {n_fail} נחסמו"
            )
        else:
            msg = "\n".join(lines)
            if len(msg) > 4000:
                msg = msg[:3900] + "\n...(קוצר)"

        await send_message(msg)

    except Exception as e:
        logger.debug(f"flush_scan_report failed: {e}")
    finally:
        _scan_actions.clear()


async def notify_action(icon: str, title: str, lines: list[str]) -> None:
    """
    שולח הודעת פעולה מיידית (לא מחכה לדוח).
    לשימוש עבור פעולות חשובות: stop, partial exit, וכו'.
    """
    try:
        from telegram_bot import send_message
        msg = f"{icon} <b>{title}</b>\n" + "\n".join(lines)
        await send_message(msg)
    except Exception as e:
        logger.debug(f"notify_action failed: {e}")
