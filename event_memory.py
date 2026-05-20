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


# ══════════════════════════════════════════════════════════════════════════════
# Market Scenario Memory — הבוט זוכר מה השוק עשה במצבים דומים
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_scenario_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS market_scenarios (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            date         TEXT NOT NULL,
            spy_pct_1d   REAL,   -- שינוי SPY אותו יום
            spy_pct_3d   REAL,   -- שינוי SPY 3 ימים אחר כך
            spy_pct_5d   REAL,   -- שינוי SPY 5 ימים אחר כך
            vix          REAL,   -- VIX אותו יום
            spy_rsi      REAL,   -- RSI של SPY
            above_sma50  INTEGER, -- SPY מעל SMA50? 1/0
            leading_sector TEXT, -- איזה סקטור הוביל
            market_label TEXT,   -- "rally" | "selloff" | "flat" | "volatile"
            notes        TEXT,
            recorded_at  TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def record_market_scenario() -> None:
    """
    תעד את מצב השוק היום + מה קרה אחר כך (retroactively after 5 days).
    קורא אוטומטית מ-yfinance כל יום.
    """
    try:
        import yfinance as yf
        import numpy as np

        today_str = date.today().isoformat()
        conn = _get_conn()
        _ensure_scenario_table(conn)

        # Don't re-record same day
        if conn.execute("SELECT id FROM market_scenarios WHERE date=?", (today_str,)).fetchone():
            conn.close()
            return

        # Fetch SPY history
        spy = yf.Ticker("SPY").history(period="60d", auto_adjust=True)
        if spy.empty or len(spy) < 10:
            conn.close()
            return

        closes = spy["Close"].dropna()
        today_close = float(closes.iloc[-1])
        prev_close  = float(closes.iloc[-2])
        spy_1d = (today_close - prev_close) / prev_close * 100

        # SPY RSI
        delta = closes.diff()
        gain  = delta.where(delta > 0, 0).ewm(alpha=1/14, min_periods=14).mean()
        loss  = (-delta).where(delta < 0, 0).ewm(alpha=1/14, min_periods=14).mean()
        rs    = gain / (loss + 1e-10)
        rsi_series = 100 - (100 / (1 + rs))
        spy_rsi = float(rsi_series.iloc[-1]) if len(rsi_series) >= 14 else 50.0

        # SMA50
        sma50 = float(closes.rolling(50).mean().iloc[-1]) if len(closes) >= 50 else today_close
        above_sma50 = 1 if today_close > sma50 else 0

        # VIX
        vix_val = None
        try:
            vix = yf.Ticker("^VIX").history(period="5d", auto_adjust=True)
            if not vix.empty:
                vix_val = float(vix["Close"].iloc[-1])
        except Exception:
            pass

        # Market label
        if spy_1d > 1.5:
            label = "rally"
        elif spy_1d < -1.5:
            label = "selloff"
        elif (vix_val or 20) > 25:
            label = "volatile"
        else:
            label = "flat"

        conn.execute("""
            INSERT INTO market_scenarios
                (date, spy_pct_1d, vix, spy_rsi, above_sma50, market_label)
            VALUES (?,?,?,?,?,?)
        """, (today_str, round(spy_1d, 2), vix_val, round(spy_rsi, 1), above_sma50, label))
        conn.commit()

        # Retroactively fill 3d/5d returns for scenarios from 3 & 5 days ago
        for lag, col in [(3, "spy_pct_3d"), (5, "spy_pct_5d")]:
            target_date = (date.today() - timedelta(days=lag*1 + 1)).isoformat()
            row = conn.execute(
                f"SELECT id, spy_pct_1d FROM market_scenarios WHERE date <= ? ORDER BY date DESC LIMIT 1",
                (target_date,)
            ).fetchone()
            if row:
                # find close on target date from spy history
                try:
                    idx = list(closes.index)
                    # find price lag days ago
                    target_close = float(closes.iloc[-(lag+1)])
                    target_prev  = float(closes.iloc[-(lag+2)])
                    cumret = (today_close - target_close) / target_close * 100
                    conn.execute(
                        f"UPDATE market_scenarios SET {col}=? WHERE id=?",
                        (round(cumret, 2), row[0])
                    )
                except Exception:
                    pass

        conn.commit()
        conn.close()
        logger.info(f"[SCENARIO] תועד: {today_str} | SPY {spy_1d:+.1f}% | VIX={vix_val} | RSI={spy_rsi:.0f} | {label}")

    except Exception as e:
        logger.debug(f"[SCENARIO] record error: {e}")


def get_scenario_signal() -> tuple[str, str]:
    """
    בדוק מה השוק עשה בעבר כשהתנאים היו דומים לעכשיו.
    מחזיר: (signal, explanation)
    signal: "bullish" | "bearish" | "neutral" | "caution"
    """
    try:
        import yfinance as yf

        conn = _get_conn()
        _ensure_scenario_table(conn)

        # Current conditions
        spy = yf.Ticker("SPY").history(period="60d", auto_adjust=True)
        if spy.empty or len(spy) < 15:
            conn.close()
            return "neutral", "אין מספיק נתוני SPY"

        closes = spy["Close"].dropna()
        today_close = float(closes.iloc[-1])
        prev_close  = float(closes.iloc[-2])
        today_pct = (today_close - prev_close) / prev_close * 100

        # RSI
        delta = closes.diff()
        gain  = delta.where(delta > 0, 0).ewm(alpha=1/14).mean()
        loss  = (-delta).where(delta < 0, 0).ewm(alpha=1/14).mean()
        rs    = gain / (loss + 1e-10)
        rsi_now = float((100 - 100/(1+rs)).iloc[-1])

        sma50     = float(closes.rolling(50).mean().iloc[-1]) if len(closes) >= 50 else today_close
        above_now = 1 if today_close > sma50 else 0

        vix_now = None
        try:
            vix = yf.Ticker("^VIX").history(period="5d", auto_adjust=True)
            if not vix.empty:
                vix_now = float(vix["Close"].iloc[-1])
        except Exception:
            pass

        # Find similar historical scenarios
        rows = conn.execute("""
            SELECT spy_pct_1d, spy_pct_5d, market_label, date
            FROM market_scenarios
            WHERE above_sma50 = ?
              AND spy_rsi BETWEEN ? AND ?
              AND spy_pct_5d IS NOT NULL
            ORDER BY date DESC
            LIMIT 20
        """, (above_now, rsi_now - 10, rsi_now + 10)).fetchall()
        conn.close()

        if len(rows) < 3:
            return "neutral", f"מעט מדי תרחישים דומים (RSI≈{rsi_now:.0f}, {'מעל' if above_now else 'מתחת'} SMA50)"

        forward_5d  = [r[1] for r in rows if r[1] is not None]
        if not forward_5d:
            return "neutral", "אין נתוני תשואה קדימה"

        avg_5d  = sum(forward_5d) / len(forward_5d)
        pos_pct = sum(1 for x in forward_5d if x > 0.5) / len(forward_5d) * 100
        n       = len(forward_5d)

        summary = (
            f"RSI≈{rsi_now:.0f}, {'מעל' if above_now else 'מתחת'} SMA50 | "
            f"{n} תרחישים דומים | "
            f"ממוצע 5 ימים: {avg_5d:+.1f}% | "
            f"חיובי {pos_pct:.0f}% מהפעמים"
        )

        if avg_5d > 1.0 and pos_pct >= 65:
            return "bullish", summary
        elif avg_5d < -1.0 or pos_pct <= 35:
            return "bearish", summary
        elif (vix_now or 20) > 28:
            return "caution", f"VIX גבוה ({vix_now:.1f}) + {summary}"
        else:
            return "neutral", summary

    except Exception as e:
        logger.debug(f"[SCENARIO] signal error: {e}")
        return "neutral", f"שגיאה: {e}"
