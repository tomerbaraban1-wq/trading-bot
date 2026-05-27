"""
Live Reporter — הבוט מדווח על כל מה שהוא עושה
================================================

מה נשלח:
- כל שעה: "אני חי + מצב תיק"
- כל סריקה: "סרקתי X מניות, הכי טוב AAPL 78/100"
- כניסה לשוק / יציאה מהשוק
- כשהבוט מחליט לא לקנות — למה
- כשהבוט מחכה (שוק סגור, אין הזדמנויות)
"""

import asyncio
import logging
import time
import os
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Rate limiting — לא לשלוח יותר מדי
_last_hourly: float = 0
_last_scan_report: float = 0
_scan_report_interval = 300   # דוח סריקה כל 5 דקות לכל היותר
_hourly_interval = 3600       # דוח חיות כל שעה
_last_no_buy_reason: str = ""
_no_buy_count: int = 0        # כמה סריקות ברצף בלי קנייה


async def send_hourly_pulse() -> None:
    """
    💓 שולח "אני חי" כל שעה עם מצב תיק עדכני.
    """
    global _last_hourly
    if time.time() - _last_hourly < _hourly_interval:
        return
    _last_hourly = time.time()

    try:
        import broker, database, budget
        from telegram_bot import send_message

        # מידע תיק
        positions = await asyncio.to_thread(broker.get_positions)
        b = await asyncio.to_thread(budget.get_budget_status)
        open_trades = await asyncio.to_thread(database.get_open_trades)

        cash = float(b.get("cash_available", 0))
        equity = float(b.get("equity", 0))
        open_pnl = sum(float(p.unrealized_pl) for p in (positions or []))
        n_pos = len(positions or [])

        now_il = datetime.now(timezone.utc) + timedelta(hours=3)
        time_str = now_il.strftime("%H:%M")

        pnl_em = "🟢" if open_pnl >= 0 else "🔴"
        pos_lines = ""
        if positions:
            for p in sorted(positions, key=lambda x: float(x.unrealized_plpc), reverse=True)[:3]:
                plpc = float(p.unrealized_plpc) * 100
                em = "📈" if plpc >= 0 else "📉"
                tv = f'<a href="https://www.tradingview.com/chart/?symbol={p.symbol}">{p.symbol}</a>'
                pos_lines += f"\n  {em} {tv} {plpc:+.1f}%"

        await send_message(
            f"💓 <b>הבוט פועל</b> | {time_str}\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"📊 פוזיציות: <b>{n_pos}</b>  |  💵 מזומן: <b>${cash:,.0f}</b>\n"
            f"{pnl_em} P&L פתוח: <b>${open_pnl:+,.2f}</b>"
            + (pos_lines if pos_lines else "\n  📭 אין פוזיציות פתוחות")
            + f"\n💼 שווי תיק: ${equity:,.0f}"
        )
    except Exception as e:
        logger.debug(f"hourly_pulse failed: {e}")


async def send_scan_report(
    scanned: int,
    best_ticker: Optional[str],
    best_score: float,
    no_buy_reason: str,
    candidates_found: int,
    cash: float,
) -> None:
    """
    📡 שולח דוח סריקה — מה נסרק ומה הוחלט.
    נשלח לא יותר מפעם ב-5 דקות.
    """
    global _last_scan_report, _last_no_buy_reason, _no_buy_count

    now = time.time()
    if now - _last_scan_report < _scan_report_interval:
        return

    # אם הסיבה זהה לסריקה הקודמת — אל תחזור על עצמך, רק ספור
    if no_buy_reason and no_buy_reason == _last_no_buy_reason:
        _no_buy_count += 1
        if _no_buy_count < 3:   # שלח רק אחרי 3 פעמים זהות
            return
        _no_buy_count = 0
    else:
        _no_buy_count = 0
        _last_no_buy_reason = no_buy_reason

    _last_scan_report = now

    try:
        from telegram_bot import send_message
        from trading_hours import is_ok_to_trade

        now_il = datetime.now(timezone.utc) + timedelta(hours=3)
        time_str = now_il.strftime("%H:%M")

        if best_ticker and candidates_found > 0:
            # נמצאו מועמדים
            tv = f'https://www.tradingview.com/chart/?symbol={best_ticker}'
            score_bar = "🟩" * round(best_score / 10) + "⬜" * (10 - round(best_score / 10))
            msg = (
                f"🔍 <b>סריקה</b> | {time_str}\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📊 נסרקו: {scanned} מניות | 🎯 מועמדים: {candidates_found}\n"
                f"🏆 הכי טוב: <a href=\"{tv}\"><b>{best_ticker}</b></a>\n"
                f"   {score_bar} <b>{best_score:.0f}/100</b>\n"
                f"💵 מזומן זמין: ${cash:,.0f}"
                + (f"\n✅ קונה!" if not no_buy_reason else f"\n⏸️ {no_buy_reason}")
            )
        elif no_buy_reason:
            # אין קנייה עם סיבה
            msg = (
                f"🔍 <b>סריקה</b> | {time_str}\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📊 נסרקו: {scanned} מניות\n"
                f"⏸️ <b>לא קונה</b>: {no_buy_reason}\n"
                f"💵 מזומן: ${cash:,.0f}"
            )
        else:
            # סריקה רגילה ללא מועמדים
            msg = (
                f"🔍 <b>סריקה</b> | {time_str}\n"
                f"📊 נסרקו: {scanned} | אין הזדמנויות כרגע\n"
                f"💵 מזומן: ${cash:,.0f}"
            )

        await send_message(msg)
    except Exception as e:
        logger.debug(f"scan_report failed: {e}")


async def send_market_status_change(is_open: bool) -> None:
    """🔔 שולח הודעה כשהשוק נפתח או נסגר."""
    try:
        from telegram_bot import send_message
        now_il = datetime.now(timezone.utc) + timedelta(hours=3)
        time_str = now_il.strftime("%H:%M")

        if is_open:
            await send_message(
                f"🔔 <b>השוק נפתח!</b> | {time_str}\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"🤖 הבוט מתחיל לסרוק הזדמנויות...\n"
                f"📡 סריקה ראשונה תוך דקה"
            )
        else:
            await send_message(
                f"🔔 <b>השוק נסגר</b> | {time_str}\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"🤖 הבוט עובר למצב ניטור\n"
                f"📊 סיכום יומי יישלח בקרוב"
            )
    except Exception as e:
        logger.debug(f"market_status_change failed: {e}")


async def send_waiting_message(reason: str) -> None:
    """⏳ שולח מה הבוט מחכה לו — פעם בשעה לכל היותר."""
    global _last_no_buy_reason, _no_buy_count
    # מוכלל ב-send_scan_report — לא צריך פונקציה נפרדת
    pass


_market_was_open: Optional[bool] = None


async def check_and_report_market_change() -> None:
    """
    בודק אם הסטטוס של השוק השתנה ושולח התראה.
    קורא ל-send_market_status_change אם צריך.
    """
    global _market_was_open
    try:
        from trading_hours import is_ok_to_trade
        ok, _ = is_ok_to_trade()
        if _market_was_open is None:
            _market_was_open = ok
            return
        if ok != _market_was_open:
            _market_was_open = ok
            await send_market_status_change(ok)
    except Exception:
        pass
