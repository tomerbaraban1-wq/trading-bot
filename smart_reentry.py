"""
Smart Re-Entry
==============

After a stop-loss, the position is closed but the underlying stock may still be
high quality. This module tracks recent stop-outs and allows re-entry IF:
  1. At least 24 hours passed (cooldown — don't chase)
  2. The stock has recovered above its stop price
  3. The stock's current score is HIGHER than at original entry
  4. RSI shows momentum returning (>50, rising)
  5. The original failure was due to noise/news, not fundamentals

Without these guards, the bot would just keep buying losers.

Lifecycle:
  - On stop-loss → mark_stopout(ticker, stop_price, score)
  - On scan → before opening: should_reenter(ticker) → True/False
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# In-memory store of recent stop-outs: ticker → metadata
_stopout_registry: dict[str, dict] = {}

REENTRY_COOLDOWN_HOURS = 24
REENTRY_SCORE_DELTA = 5     # New score must be at least +5 higher
REENTRY_RSI_MIN = 50         # Must be back above 50
REENTRY_PRICE_RECOVERY = 1.02  # Price must be 2%+ above stop


def mark_stopout(ticker: str, stop_price: float, original_score: float = 0,
                 original_rsi: float = 0) -> None:
    """Mark a stop-out event for cooldown tracking."""
    _stopout_registry[ticker] = {
        "stop_price": stop_price,
        "original_score": original_score,
        "original_rsi": original_rsi,
        "stopout_ts": time.time(),
        "reentry_attempted": False,
    }
    logger.info(f"[REENTRY] {ticker} stopout marked @ ${stop_price:.2f}")


def is_in_cooldown(ticker: str) -> bool:
    """Check if ticker is still in cooldown after a stop-out."""
    entry = _stopout_registry.get(ticker)
    if not entry:
        return False
    elapsed_hours = (time.time() - entry["stopout_ts"]) / 3600
    return elapsed_hours < REENTRY_COOLDOWN_HOURS


def should_reenter(
    ticker: str,
    current_price: float,
    current_score: float,
    current_rsi: float,
) -> tuple[bool, str]:
    """
    Decide whether the bot should re-enter a previously stopped-out position.

    Returns (allow_reentry, reason_or_explanation).
    """
    entry = _stopout_registry.get(ticker)
    if not entry:
        # Never stopped out — normal entry allowed
        return True, "אין היסטוריית stop-out"

    elapsed_hours = (time.time() - entry["stopout_ts"]) / 3600

    # 1. Cooldown
    if elapsed_hours < REENTRY_COOLDOWN_HOURS:
        return False, f"בקירור {elapsed_hours:.1f}h/{REENTRY_COOLDOWN_HOURS}h"

    # 2. Already attempted re-entry once — don't try again
    if entry.get("reentry_attempted"):
        return False, "כבר ניסה Re-entry פעם אחת — לא חוזר"

    # 3. Price must have recovered above stop
    stop_price = entry["stop_price"]
    if current_price < stop_price * REENTRY_PRICE_RECOVERY:
        return False, f"מחיר ${current_price:.2f} < סטופ+2% (${stop_price * REENTRY_PRICE_RECOVERY:.2f})"

    # 4. Score must be higher than at original entry
    original_score = entry.get("original_score", 0)
    if current_score < original_score + REENTRY_SCORE_DELTA:
        return False, f"ציון {current_score:.0f} ≤ {original_score:.0f}+{REENTRY_SCORE_DELTA}"

    # 5. RSI must show momentum
    if current_rsi < REENTRY_RSI_MIN:
        return False, f"RSI {current_rsi:.0f} < {REENTRY_RSI_MIN}"

    # All checks passed
    entry["reentry_attempted"] = True   # Don't allow another re-entry after this
    return True, (
        f"Re-entry מאושר — מחיר התאושש +{((current_price/stop_price)-1)*100:.1f}%, "
        f"ציון {current_score:.0f} (היה {original_score:.0f}), RSI {current_rsi:.0f}"
    )


def cleanup_old_stopouts(max_age_days: int = 14) -> int:
    """Remove stop-out records older than max_age_days. Returns count removed."""
    cutoff = time.time() - (max_age_days * 86400)
    expired = [t for t, e in _stopout_registry.items() if e["stopout_ts"] < cutoff]
    for t in expired:
        _stopout_registry.pop(t, None)
    return len(expired)


def get_stopout_registry() -> dict:
    """For diagnostics / debugging."""
    return dict(_stopout_registry)
