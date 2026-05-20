"""
Event Memory — הבוט זוכר מה השוק עשה באירועים כלכליים גדולים.

כשיצא CPI גבוה פעם שעברה → שוק ירד 2% → הפעם הבאה שיצא CPI, הבוט זהיר יותר.

אירועים שנזכרים:
  - CPI (מדד מחירים)
  - NFP (Non-Farm Payroll)
  - FOMC (החלטת ריבית)
  - Earnings (דוחות רווחים)

שימוש:
  record_event(event_type, event_date, spy_pct_change, notes)
  get_event_signal(event_type) → (caution_level, reason)
"""

import logging
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from config import settings

logger = logging.getLogger(__name__)

_DB_PATH = Path(settings.DATABASE_PATH)


def _get_conn():
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS event_memory (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type   TEXT NOT NULL,       -- CPI | NFP | FOMC | EARNINGS
            event_date   TEXT NOT NULL,       -- YYYY-MM-DD
            spy_pct      REAL,               -- SPY % change on event day
            nasdaq_pct   REAL,               -- Nasdaq % change
            notes        TEXT,               -- תיאור קצר
            recorded_at  TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def record_event(event_type: str, event_date: str,
                 spy_pct: float, nasdaq_pct: float = 0.0,
                 notes: str = "") -> None:
    """
    שמור תגובת שוק לאירוע כלכלי.

    Args:
        event_type: "CPI" | "NFP" | "FOMC" | "EARNINGS"
        event_date: "2026-05-13"
        spy_pct:    % שינוי ב-SPY באותו יום (למשל -2.1)
        nasdaq_pct: % שינוי ב-QQQ
        notes:      תיאור קצר ("CPI גבוה מהצפי")
    """
    try:
        conn = _get_conn()
        # Don't double-record same event
        existing = conn.execute(
            "SELECT id FROM event_memory WHERE event_type=? AND event_date=?",
            (event_type.upper(), event_date)
        ).fetchone()
        if existing:
            logger.debug(f"[EVENT] {event_type} {event_date} already recorded")
            conn.close()
            return

        conn.execute(
            "INSERT INTO event_memory (event_type, event_date, spy_pct, nasdaq_pct, notes) "
            "VALUES (?,?,?,?,?)",
            (event_type.upper(), event_date, spy_pct, nasdaq_pct, notes)
        )
        conn.commit()
        conn.close()
        direction = "עלה" if spy_pct > 0 else "ירד"
        logger.info(
            f"[EVENT MEMORY] שמרתי: {event_type} {event_date} | "
            f"SPY {direction} {spy_pct:+.1f}% | {notes}"
        )
    except Exception as e:
        logger.warning(f"[EVENT MEMORY] שגיאה בשמירה: {e}")


def get_event_signal(event_type: str, lookback_events: int = 5) -> tuple[str, str]:
    """
    קבל המלצת זהירות לפני אירוע כלכלי בהתבסס על זיכרון.

    Returns:
        (caution_level, reason)
        caution_level: "none" | "light" | "medium" | "high"
    """
    try:
        conn = _get_conn()
        rows = conn.execute(
            """SELECT spy_pct, nasdaq_pct, event_date, notes
               FROM event_memory
               WHERE event_type = ?
               ORDER BY event_date DESC
               LIMIT ?""",
            (event_type.upper(), lookback_events)
        ).fetchall()
        conn.close()

        if not rows:
            return "none", f"אין זיכרון של {event_type} — פועל כרגיל"

        spy_changes  = [r[0] for r in rows if r[0] is not None]
        if not spy_changes:
            return "none", "אין מספיק נתונים"

        avg_spy   = sum(spy_changes) / len(spy_changes)
        neg_count = sum(1 for x in spy_changes if x < -0.5)
        pos_count = sum(1 for x in spy_changes if x > 0.5)
        n         = len(spy_changes)

        # Build memory summary
        last_events = ", ".join(
            f"{r[2]}: {r[0]:+.1f}%" for r in rows[:3]
        )

        if avg_spy <= -1.5 or neg_count / n >= 0.8:
            return "high", (
                f"{event_type}: {neg_count}/{n} פעמים השוק ירד | "
                f"ממוצע {avg_spy:+.1f}% | {last_events}"
            )
        elif avg_spy <= -0.5 or neg_count / n >= 0.6:
            return "medium", (
                f"{event_type}: {neg_count}/{n} פעמים השוק ירד | "
                f"ממוצע {avg_spy:+.1f}% | {last_events}"
            )
        elif avg_spy >= 1.0 or pos_count / n >= 0.7:
            return "positive", (
                f"{event_type}: {pos_count}/{n} פעמים השוק עלה | "
                f"ממוצע {avg_spy:+.1f}% | {last_events}"
            )
        else:
            return "light", (
                f"{event_type}: תוצאות מעורבות | ממוצע {avg_spy:+.1f}% | {last_events}"
            )

    except Exception as e:
        logger.warning(f"[EVENT MEMORY] שגיאה בקריאה: {e}")
        return "none", f"שגיאה: {e}"


def auto_record_today() -> None:
    """
    בדוק אם היום יש אירוע כלכלי ותעד את תגובת השוק.
    קורא אוטומטית מ-yfinance.
    """
    from trading_hours import _ECONOMIC_DATES, _FOMC_DATES
    import yfinance as yf

    today = date.today()
    today_str = today.isoformat()

    # Identify today's event
    event_type = None
    notes = ""
    if today in _FOMC_DATES:
        event_type = "FOMC"
        notes = "יום החלטת ריבית Fed"
    elif today in _ECONOMIC_DATES:
        notes = _ECONOMIC_DATES[today]
        if "CPI" in notes:
            event_type = "CPI"
        elif "NFP" in notes:
            event_type = "NFP"
        else:
            event_type = "ECONOMIC"

    if not event_type:
        return  # no event today

    # Fetch SPY & QQQ % change
    try:
        spy_hist = yf.Ticker("SPY").history(period="5d", auto_adjust=True)
        qqq_hist = yf.Ticker("QQQ").history(period="5d", auto_adjust=True)

        if len(spy_hist) < 2:
            return

        spy_today = float(spy_hist["Close"].iloc[-1])
        spy_prev  = float(spy_hist["Close"].iloc[-2])
        spy_pct   = (spy_today - spy_prev) / spy_prev * 100

        qqq_today = float(qqq_hist["Close"].iloc[-1]) if not qqq_hist.empty else spy_today
        qqq_prev  = float(qqq_hist["Close"].iloc[-2]) if len(qqq_hist) >= 2 else spy_prev
        qqq_pct   = (qqq_today - qqq_prev) / qqq_prev * 100

        record_event(event_type, today_str, spy_pct, qqq_pct, notes)

    except Exception as e:
        logger.debug(f"[EVENT MEMORY] auto_record error: {e}")


def get_memory_summary() -> str:
    """Return Hebrew summary of all recorded events for Telegram."""
    try:
        conn = _get_conn()
        rows = conn.execute(
            """SELECT event_type, event_date, spy_pct, notes
               FROM event_memory
               ORDER BY event_date DESC
               LIMIT 10"""
        ).fetchall()
        conn.close()

        if not rows:
            return "📭 אין זיכרון אירועים עדיין."

        lines = ["📚 <b>זיכרון אירועים כלכליים</b>\n━━━━━━━━━━━━━━━━"]
        for evt, dt, pct, note in rows:
            icon = "📈" if (pct or 0) > 0 else ("📉" if (pct or 0) < 0 else "➡️")
            lines.append(
                f"{icon} <b>{evt}</b> {dt} | SPY {pct:+.1f}%\n"
                f"   <i>{note}</i>"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"שגיאה: {e}"
