"""
Trading Hours Guard — Liquidity-Aware Session Filter

Why this matters:
  - Trading during low-liquidity windows (pre-market, lunch lull, after-hours)
    produces wider spreads, higher slippage, and more erratic fills.
  - Fed rate announcements (FOMC) cause instant 1-3% spikes; entering a
    position 30 minutes before is pure gambling.
  - NYSE holidays have zero liquidity and many brokers reject orders silently.

Architecture:
  is_ok_to_trade() → (allowed: bool, reason: str)

  Called by:
    - signal_validator.validate_signal()  (webhook BUY path)
    - heartbeat.auto_invest_loop()        (scheduled scan)

  SELL orders are NEVER blocked — we always allow exits.

Configuration (env vars):
    TRADING_HOURS_ENABLED        bool   default True
    TRADING_HIGH_LIQUIDITY_ONLY  bool   default True
        True  → only open rush (9:30–11:00) + close rush (14:30–16:00) ET
        False → full NYSE session (9:30–16:00) ET
    TRADING_FOMC_BLACKOUT_MIN    int    default 30  (minutes before FOMC)
    TRADING_FOMC_POSTBLACKOUT_MIN int   default 60  (minutes after FOMC)

All times in US/Eastern (ET), including DST handling via zoneinfo / pytz fallback.
"""

import os
import logging
from datetime import date, time, datetime, timedelta

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
ENABLED:               bool = os.getenv("TRADING_HOURS_ENABLED",       "true").lower() == "true"
HIGH_LIQUIDITY_ONLY:   bool = os.getenv("TRADING_HIGH_LIQUIDITY_ONLY", "false").lower() == "true"
FOMC_BLACKOUT_MIN:     int  = int(os.getenv("TRADING_FOMC_BLACKOUT_MIN",     "30"))
FOMC_POST_MIN:         int  = int(os.getenv("TRADING_FOMC_POSTBLACKOUT_MIN", "60"))

# ── Timezone setup ────────────────────────────────────────────────────────────
try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
    def _now_et() -> datetime:
        return datetime.now(_ET)
except ImportError:
    try:
        import pytz
        _ET = pytz.timezone("America/New_York")
        def _now_et() -> datetime:
            return datetime.now(pytz.utc).astimezone(_ET)
    except ImportError:
        # Fallback: UTC-5 (no DST — conservative)
        logger.warning("[TRADING_HOURS] zoneinfo/pytz not found — using UTC-5 offset (no DST)")
        def _now_et() -> datetime:
            return datetime.utcnow() - timedelta(hours=5)


# ── NYSE market session ───────────────────────────────────────────────────────
_NYSE_OPEN  = time(9, 30)
_NYSE_CLOSE = time(16, 0)

# High-liquidity windows: open rush + close rush
_LIQOPEN_START  = time(9, 30)
_LIQOPEN_END    = time(11, 0)
_LIQCLOSE_START = time(14, 30)
_LIQCLOSE_END   = time(16, 0)


# ── NYSE Holidays 2025 – 2027 ─────────────────────────────────────────────────
# Source: NYSE annual market holiday schedule
_NYSE_HOLIDAYS: set[date] = {
    # 2025
    date(2025, 1,  1),   # New Year's Day
    date(2025, 1, 20),   # Martin Luther King Jr. Day
    date(2025, 2, 17),   # Presidents' Day
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 26),   # Memorial Day
    date(2025, 6, 19),   # Juneteenth
    date(2025, 7,  4),   # Independence Day
    date(2025, 9,  1),   # Labor Day
    date(2025, 11, 27),  # Thanksgiving
    date(2025, 12, 25),  # Christmas
    # 2026
    date(2026, 1,  1),   # New Year's Day
    date(2026, 1, 19),   # MLK Day
    date(2026, 2, 16),   # Presidents' Day
    date(2026, 4,  3),   # Good Friday
    date(2026, 5, 25),   # Memorial Day
    date(2026, 6, 19),   # Juneteenth
    date(2026, 7,  3),   # Independence Day (observed)
    date(2026, 9,  7),   # Labor Day
    date(2026, 11, 26),  # Thanksgiving
    date(2026, 12, 25),  # Christmas
    # 2027
    date(2027, 1,  1),   # New Year's Day
    date(2027, 1, 18),   # MLK Day
    date(2027, 2, 15),   # Presidents' Day
    date(2027, 3, 26),   # Good Friday
    date(2027, 5, 31),   # Memorial Day
    date(2027, 6, 18),   # Juneteenth (observed)
    date(2027, 7,  5),   # Independence Day (observed)
    date(2027, 9,  6),   # Labor Day
    date(2027, 11, 25),  # Thanksgiving
    date(2027, 12, 24),  # Christmas (observed)
}

# ── FOMC Rate Announcement Schedule ─────────────────────────────────────────
# Announcement time: 14:00 ET (2:00 PM Eastern) on decision day
# Blackout: [14:00 - FOMC_BLACKOUT_MIN ... 14:00 + FOMC_POST_MIN]
_FOMC_ANNOUNCEMENT_HOUR   = 14
_FOMC_ANNOUNCEMENT_MINUTE = 0

# FOMC decision dates 2025–2026 (day of rate announcement, not meeting start)
_FOMC_DATES: set[date] = {
    # 2025
    date(2025, 1, 29),
    date(2025, 3, 19),
    date(2025, 5,  7),
    date(2025, 6, 18),
    date(2025, 7, 30),
    date(2025, 9, 17),
    date(2025, 10, 29),
    date(2025, 12, 10),
    # 2026
    date(2026, 1, 28),
    date(2026, 3, 18),
    date(2026, 4, 29),
    date(2026, 6, 17),
    date(2026, 7, 29),
    date(2026, 9, 16),
    date(2026, 10, 28),
    date(2026, 12,  9),
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def is_ok_to_trade() -> tuple[bool, str]:
    """
    Check whether it is safe and liquid to open a new position RIGHT NOW.

    Returns
    -------
    (allowed: bool, reason: str)
    allowed=True  → go ahead
    allowed=False → skip; reason explains why
    """
    if not ENABLED:
        return True, "trading_hours_disabled"

    now = _now_et()
    today = now.date()
    current_time = now.time()

    # 1. Weekend
    if now.weekday() >= 5:  # 5=Sat, 6=Sun
        return False, f"Weekend — NYSE closed ({now.strftime('%A')})"

    # 2. Market holiday
    if today in _NYSE_HOLIDAYS:
        return False, f"NYSE holiday — {today.isoformat()}"

    # 3. Outside NYSE session (9:30–16:00 ET)
    if current_time < _NYSE_OPEN:
        return False, f"Pre-market — NYSE opens at 09:30 ET (now {current_time.strftime('%H:%M')} ET)"
    if current_time >= _NYSE_CLOSE:
        return False, f"After-hours — NYSE closed at 16:00 ET (now {current_time.strftime('%H:%M')} ET)"

    # 4. FOMC blackout window
    if today in _FOMC_DATES:
        announcement = now.replace(
            hour=_FOMC_ANNOUNCEMENT_HOUR,
            minute=_FOMC_ANNOUNCEMENT_MINUTE,
            second=0, microsecond=0,
        )
        blackout_start = announcement - timedelta(minutes=FOMC_BLACKOUT_MIN)
        blackout_end   = announcement + timedelta(minutes=FOMC_POST_MIN)
        if blackout_start <= now <= blackout_end:
            return False, (
                f"FOMC blackout — rate announcement at 14:00 ET | "
                f"window {blackout_start.strftime('%H:%M')}–{blackout_end.strftime('%H:%M')} ET"
            )

    # 5. High-liquidity window filter
    if HIGH_LIQUIDITY_ONLY:
        in_open_rush  = _LIQOPEN_START  <= current_time < _LIQOPEN_END
        in_close_rush = _LIQCLOSE_START <= current_time < _LIQCLOSE_END
        if not (in_open_rush or in_close_rush):
            return False, (
                f"Low-liquidity window — trading only 09:30–11:00 and 14:30–16:00 ET "
                f"(now {current_time.strftime('%H:%M')} ET)"
            )

    return True, "ok"


def get_status() -> dict:
    """
    Return current trading hours status dict (for /status endpoint).
    """
    ok, reason = is_ok_to_trade()
    now = _now_et()
    today = now.date()
    next_fomc = _next_fomc_date(today)

    return {
        "ok_to_trade":          ok,
        "reason":               reason,
        "current_time_et":      now.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "is_holiday":           today in _NYSE_HOLIDAYS,
        "is_fomc_day":          today in _FOMC_DATES,
        "next_fomc_date":       next_fomc.isoformat() if next_fomc else None,
        "high_liquidity_only":  HIGH_LIQUIDITY_ONLY,
        "fomc_blackout_min":    FOMC_BLACKOUT_MIN,
        "fomc_post_min":        FOMC_POST_MIN,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _next_fomc_date(from_date: date) -> date | None:
    """Return the next FOMC announcement date on or after from_date."""
    upcoming = sorted(d for d in _FOMC_DATES if d >= from_date)
    return upcoming[0] if upcoming else None
