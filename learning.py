import json
import logging
import time as _time
from database import get_loss_trades, get_learning_entries, get_win_trades

logger = logging.getLogger(__name__)

# Throttle: update thresholds at most once per hour (not on every scan)
_last_threshold_update: float = 0.0
_THRESHOLD_UPDATE_INTERVAL: float = 3600.0   # 1 hour

# Dynamic thresholds — defaults (overridden by DB on startup)
_DEFAULT_THRESHOLDS = {
    "min_sentiment": 5,
    "max_rsi": 72,
    "min_rsi": 35,
    "max_bb_position": 0.90,
    "min_volume_ratio": 0.8,
}

# In-memory state — persisted to DB on every update
_dynamic_thresholds = _DEFAULT_THRESHOLDS.copy()


def _save_thresholds_to_db() -> None:
    """Persist current thresholds to DB so they survive restarts."""
    try:
        from database import get_connection
        conn = get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS learning_thresholds (
                key TEXT PRIMARY KEY,
                value REAL,
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)
        for k, v in _dynamic_thresholds.items():
            conn.execute(
                "INSERT OR REPLACE INTO learning_thresholds(key, value, updated_at) VALUES(?,?,datetime('now'))",
                (k, float(v))
            )
        conn.commit()
        logger.debug("[LEARNING] Thresholds saved to DB")
    except Exception as e:
        logger.debug(f"[LEARNING] Failed to save thresholds: {e}")


def _load_thresholds_from_db() -> None:
    """Load thresholds from DB on startup (persists across restarts)."""
    global _dynamic_thresholds
    try:
        from database import get_connection
        conn = get_connection()
        rows = conn.execute(
            "SELECT key, value FROM learning_thresholds"
        ).fetchall()
        if rows:
            loaded = {row[0]: float(row[1]) for row in rows}
            _dynamic_thresholds.update(loaded)
            logger.info(f"[LEARNING] Thresholds loaded from DB: {loaded}")
        else:
            logger.info("[LEARNING] No saved thresholds in DB — using defaults")
    except Exception as e:
        logger.debug(f"[LEARNING] Failed to load thresholds: {e}")


# Load on module import — happens once at bot startup
_load_thresholds_from_db()


def get_dynamic_thresholds() -> dict:
    """Return current dynamic thresholds based on learned patterns."""
    _update_thresholds()
    return _dynamic_thresholds.copy()


def _update_thresholds():
    """Analyze trade history and auto-adjust thresholds (at most once per hour)."""
    global _last_threshold_update
    now = _time.time()
    if now - _last_threshold_update < _THRESHOLD_UPDATE_INTERVAL:
        return  # throttled — don't hammer DB on every scan
    _last_threshold_update = now

    losses = get_loss_trades(limit=30)
    wins = get_win_trades(limit=30)

    if len(losses) < 3:
        return  # not enough data yet

    # --- Auto-adjust sentiment threshold ---
    loss_sentiments = [t["sentiment_score"] for t in losses if t.get("sentiment_score") is not None]
    if loss_sentiments:
        avg_loss_sentiment = sum(loss_sentiments) / len(loss_sentiments)
        # If most losses had sentiment <= 5, raise the bar
        if avg_loss_sentiment <= 5:
            new_min = min(7, round(avg_loss_sentiment) + 1)
            if new_min > _dynamic_thresholds["min_sentiment"]:
                logger.info(f"LEARNING: Raising min_sentiment {_dynamic_thresholds['min_sentiment']} → {new_min} (avg loss sentiment was {avg_loss_sentiment:.1f})")
                _dynamic_thresholds["min_sentiment"] = new_min

    # Relaxation: if wins are clustering at sentiment < current threshold, loosen it
    # Prevents the threshold from permanently locking all trades after a bad streak
    win_sentiments = [t["sentiment_score"] for t in wins if t.get("sentiment_score") is not None]
    if win_sentiments and len(wins) >= 3:
        avg_win_sentiment = sum(win_sentiments) / len(win_sentiments)
        # Relax threshold if profitable trades are happening well below the current bar
        relaxed = max(4, round(avg_win_sentiment) - 1)
        if relaxed < _dynamic_thresholds["min_sentiment"]:
            logger.info(
                f"LEARNING: Relaxing min_sentiment {_dynamic_thresholds['min_sentiment']} → {relaxed} "
                f"(wins avg sentiment={avg_win_sentiment:.1f} — threshold too strict)"
            )
            _dynamic_thresholds["min_sentiment"] = relaxed

    # --- Auto-adjust RSI threshold ---
    loss_rsis = [t["rsi"] for t in losses if t.get("rsi") is not None]
    if loss_rsis:
        avg_loss_rsi = sum(loss_rsis) / len(loss_rsis)
        # If losses happen at high RSI, lower the max allowed RSI
        high_rsi_losses = [r for r in loss_rsis if r > 65]
        if len(high_rsi_losses) >= 2:
            new_max_rsi = max(55, round(avg_loss_rsi) - 5)
            if new_max_rsi < _dynamic_thresholds["max_rsi"]:
                logger.info(f"LEARNING: Lowering max_rsi {_dynamic_thresholds['max_rsi']} → {new_max_rsi}")
                _dynamic_thresholds["max_rsi"] = new_max_rsi

    # --- Auto-adjust BB position ---
    loss_bbs = [t["bb_position"] for t in losses if t.get("bb_position") is not None]
    if loss_bbs:
        high_bb_losses = [b for b in loss_bbs if b > 0.75]
        if len(high_bb_losses) >= 2:
            new_max_bb = max(0.6, min(high_bb_losses) - 0.05)
            if new_max_bb < _dynamic_thresholds["max_bb_position"]:
                logger.info(f"LEARNING: Lowering max_bb_position → {new_max_bb:.2f}")
                _dynamic_thresholds["max_bb_position"] = round(new_max_bb, 2)

    # --- Auto-adjust volume ratio ---
    loss_vols = [t["volume_ratio"] for t in losses if t.get("volume_ratio") is not None]
    if loss_vols:
        low_vol_losses = [v for v in loss_vols if v < 0.7]
        if len(low_vol_losses) >= 2:
            new_min_vol = min(0.95, max(low_vol_losses) + 0.1)
            if new_min_vol > _dynamic_thresholds["min_volume_ratio"]:
                logger.info(f"LEARNING: Raising min_volume_ratio → {new_min_vol:.2f}")
                _dynamic_thresholds["min_volume_ratio"] = round(new_min_vol, 2)

    # Relaxation: if winning trades had low volume, loosen the threshold
    win_vols = [t["volume_ratio"] for t in wins if t.get("volume_ratio") is not None]
    if win_vols and len(wins) >= 3:
        avg_win_vol = sum(win_vols) / len(win_vols)
        if avg_win_vol < _dynamic_thresholds["min_volume_ratio"]:
            relaxed_vol = max(0.3, round(avg_win_vol - 0.05, 2))
            if relaxed_vol < _dynamic_thresholds["min_volume_ratio"]:
                logger.info(
                    f"LEARNING: Relaxing min_volume_ratio {_dynamic_thresholds['min_volume_ratio']} → {relaxed_vol} "
                    f"(wins avg vol={avg_win_vol:.2f} — threshold too strict)"
                )
                _dynamic_thresholds["min_volume_ratio"] = relaxed_vol

    # ✅ Persist learned thresholds to DB — survives restarts
    _save_thresholds_to_db()


def should_override_buy(ticker: str, indicators: dict) -> tuple[bool, str]:
    """
    Check if this buy matches known losing patterns.
    Uses dynamically learned thresholds.
    Returns (should_block, reason).
    """
    _update_thresholds()
    patterns = analyze_patterns()

    # Check for ticker-specific loss streaks
    for p in patterns:
        if p["type"] == "ticker_loss_streak" and p["ticker"] == ticker.upper():
            return True, f"{ticker} has {p['count']} consecutive losses - blocking buy"

    # Check RSI against dynamic threshold
    rsi = indicators.get("rsi")
    if rsi is not None:
        if rsi > _dynamic_thresholds["max_rsi"]:
            return True, f"RSI {rsi:.1f} > learned max {_dynamic_thresholds['max_rsi']} (overbought)"
        if rsi < _dynamic_thresholds["min_rsi"]:
            return True, f"RSI {rsi:.1f} < learned min {_dynamic_thresholds['min_rsi']} (oversold crash risk)"

    # Check Bollinger Band position
    bb = indicators.get("bb_position")
    if bb is not None and bb > _dynamic_thresholds["max_bb_position"]:
        return True, f"BB position {bb:.2f} > learned max {_dynamic_thresholds['max_bb_position']} (near top)"

    # Volume ratio — adjust threshold by time of day
    # Intraday volume ratio compares accumulated volume to full-day MA20.
    # At 10:00 ET (~30min in), only ~10% of daily volume has traded — ratio is always low.
    # Scale threshold by % of trading day elapsed to avoid false blocks.
    import os as _os
    from datetime import datetime, timezone
    _extended = _os.getenv("EXTENDED_HOURS_TRADING", "false").lower() == "true"
    _now_utc  = datetime.now(timezone.utc)
    _h, _m    = _now_utc.hour, _now_utc.minute
    _minutes_utc = _h * 60 + _m

    # NYSE session in UTC: 13:30–20:00 (EDT) or 14:30–21:00 (EST)
    # DST: EDT = March 2nd Sunday – November 1st Sunday
    _month = _now_utc.month
    _is_edt = 4 <= _month <= 10 or (_month == 3 and _now_utc.day >= 8) or (_month == 11 and _now_utc.day < 7)
    _session_start_utc = (13 * 60 + 30) if _is_edt else (14 * 60 + 30)
    _session_end_utc   = (20 * 60) if _is_edt else (21 * 60)
    _session_length    = _session_end_utc - _session_start_utc  # 390 minutes

    _is_extended_session = _extended and (
        (8 * 60 <= _minutes_utc < _session_start_utc) or (_minutes_utc >= _session_end_utc)
    )

    if _is_extended_session:
        _vol_threshold = 0.05
    elif _session_start_utc <= _minutes_utc < _session_end_utc:
        _elapsed = _minutes_utc - _session_start_utc
        _day_pct = _elapsed / _session_length
        _vol_threshold = _dynamic_thresholds["min_volume_ratio"] * min(1.0, _day_pct * 1.5)
        _vol_threshold = max(0.05, _vol_threshold)
    else:
        _vol_threshold = 0.05

    vol = indicators.get("volume_ratio")
    if vol is not None and vol < _vol_threshold:
        if _is_extended_session:
            session = "extended"
        elif _session_start_utc <= _minutes_utc < _session_end_utc:
            _elapsed = _minutes_utc - _session_start_utc
            session = f"regular, {_elapsed}min into session"
        else:
            session = "closed"
        return True, f"Volume ratio {vol:.2f} < min {_vol_threshold:.2f} ({session})"

    return False, ""


def analyze_patterns() -> list[dict]:
    """Analyze recent losing trades for common patterns."""
    losses = get_loss_trades(limit=20)
    if len(losses) < 3:
        return []

    patterns = []

    # Pattern 1: Ticker loss streak
    ticker_losses: dict = {}
    for trade in losses:
        t = trade["ticker"]
        ticker_losses[t] = ticker_losses.get(t, 0) + 1
    for ticker, count in ticker_losses.items():
        if count >= 3:
            patterns.append({
                "type": "ticker_loss_streak",
                "ticker": ticker,
                "count": count,
                "description": f"{ticker}: {count} losses in recent history",
            })

    # Pattern 2: Low sentiment losses
    low_sent_losses = [t for t in losses if t.get("sentiment_score") is not None and t["sentiment_score"] <= 4]
    if len(low_sent_losses) >= 2:
        patterns.append({
            "type": "low_sentiment_losses",
            "count": len(low_sent_losses),
            "threshold": 4,
            "description": f"{len(low_sent_losses)} losses had sentiment <= 4",
        })

    # Pattern 3: High RSI entries
    high_rsi_losses = [t for t in losses if t.get("rsi") is not None and t["rsi"] > 65]
    if len(high_rsi_losses) >= 2:
        avg_rsi = sum(t["rsi"] for t in high_rsi_losses) / len(high_rsi_losses)
        patterns.append({
            "type": "high_rsi_entry_losses",
            "count": len(high_rsi_losses),
            "avg_rsi": round(avg_rsi, 1),
            "description": f"{len(high_rsi_losses)} losses entered with RSI > 65 (avg: {avg_rsi:.1f})",
        })

    # Pattern 4: High BB position losses
    high_bb_losses = [t for t in losses if t.get("bb_position") is not None and t["bb_position"] > 0.8]
    if len(high_bb_losses) >= 2:
        patterns.append({
            "type": "high_bb_entry_losses",
            "count": len(high_bb_losses),
            "description": f"{len(high_bb_losses)} losses bought near top of Bollinger Band",
        })

    # Pattern 5: Low volume losses
    low_vol_losses = [t for t in losses if t.get("volume_ratio") is not None and t["volume_ratio"] < 0.6]
    if len(low_vol_losses) >= 2:
        patterns.append({
            "type": "low_volume_losses",
            "count": len(low_vol_losses),
            "description": f"{len(low_vol_losses)} losses happened on low volume days",
        })

    # Pattern 6: Overall loss rate
    all_entries = get_learning_entries(limit=50)
    total = len(all_entries)
    loss_count = sum(1 for e in all_entries if e.get("outcome") == "loss")
    if total >= 5:
        loss_rate = loss_count / total * 100
        patterns.append({
            "type": "overall_loss_rate",
            "rate": round(loss_rate, 1),
            "total_trades": total,
            "description": f"Overall loss rate: {loss_rate:.1f}% ({loss_count}/{total} trades)",
        })

    return patterns


def get_learning_report() -> dict:
    """Generate a human-readable learning report."""
    _update_thresholds()
    patterns = analyze_patterns()
    losses = get_loss_trades(limit=10)

    return {
        "patterns_found": len(patterns),
        "dynamic_thresholds": _dynamic_thresholds.copy(),
        "patterns": patterns,
        "recent_losses": [
            {
                "ticker": t["ticker"],
                "pnl": t["pnl_gross"],
                "sentiment": t.get("sentiment_score"),
                "rsi": t.get("rsi"),
                "bb_position": t.get("bb_position"),
                "volume_ratio": t.get("volume_ratio"),
                "status": t["status"],
            }
            for t in losses
        ],
    }
