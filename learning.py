import logging
from database import get_loss_trades, get_learning_entries, get_win_trades

logger = logging.getLogger(__name__)

# Dynamic thresholds — updated automatically based on trade history
_dynamic_thresholds = {
    "min_sentiment": 4,       # minimum sentiment score to buy
    "max_rsi": 70,            # max RSI to enter (avoid overbought)
    "min_rsi": 25,            # min RSI to enter (avoid oversold crash)
    "max_bb_position": 0.85,  # avoid buying near top of Bollinger Band
    "min_volume_ratio": 0.5,  # avoid low-volume days
}


def get_dynamic_thresholds() -> dict:
    """Return current dynamic thresholds based on learned patterns."""
    _update_thresholds()
    return _dynamic_thresholds.copy()


def _update_thresholds():
    """Analyze trade history and auto-adjust thresholds."""
    losses = get_loss_trades(limit=30)
    wins = get_win_trades(limit=30) if hasattr(__import__('database'), 'get_win_trades') else []

    if len(losses) < 3:
        return  # not enough data yet

    # --- Auto-adjust sentiment threshold ---
    loss_sentiments = [t["sentiment_score"] for t in losses if t.get("sentiment_score") is not None]
    if loss_sentiments:
        avg_loss_sentiment = sum(loss_sentiments) / len(loss_sentiments)
        # If most losses had sentiment <= 5, raise the bar
        if avg_loss_sentiment <= 5:
            new_min = min(7, round(avg_loss_sentiment) + 1)
            if new_min != _dynamic_thresholds["min_sentiment"]:
                logger.info(f"LEARNING: Raising min_sentiment {_dynamic_thresholds['min_sentiment']} → {new_min} (avg loss sentiment was {avg_loss_sentiment:.1f})")
                _dynamic_thresholds["min_sentiment"] = new_min

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
            new_min_vol = min(0.9, max(low_vol_losses) + 0.1)
            if new_min_vol > _dynamic_thresholds["min_volume_ratio"]:
                logger.info(f"LEARNING: Raising min_volume_ratio → {new_min_vol:.2f}")
                _dynamic_thresholds["min_volume_ratio"] = round(new_min_vol, 2)


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

    # Check volume ratio
    vol = indicators.get("volume_ratio")
    if vol is not None and vol < _dynamic_thresholds["min_volume_ratio"]:
        return True, f"Volume ratio {vol:.2f} < learned min {_dynamic_thresholds['min_volume_ratio']} (low volume)"

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
