import logging
from database import get_loss_trades, get_learning_entries

logger = logging.getLogger(__name__)


def should_override_buy(ticker: str, indicators: dict) -> tuple[bool, str]:
    """
    Check if this buy matches known losing patterns.
    Returns (should_block, reason).
    """
    patterns = analyze_patterns()
    if not patterns:
        return False, ""

    # Check for ticker-specific loss streaks
    for p in patterns:
        if p["type"] == "ticker_loss_streak" and p["ticker"] == ticker.upper():
            return True, f"{ticker} has {p['count']} consecutive losses - blocking buy"

    # Check for low-sentiment loss pattern
    sentiment_score = indicators.get("sentiment_score")
    if sentiment_score is not None:
        for p in patterns:
            if p["type"] == "low_sentiment_losses" and sentiment_score <= p.get("threshold", 4):
                return False, ""  # Already handled by sentiment guardrail

    return False, ""


def analyze_patterns() -> list[dict]:
    """
    Analyze recent losing trades for common patterns.
    Returns list of pattern observations.
    """
    losses = get_loss_trades(limit=20)
    if len(losses) < 3:
        return []

    patterns = []

    # Pattern 1: Ticker loss streak (3+ consecutive losses on same ticker)
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

    # Pattern 2: Low sentiment trades that lost money
    low_sent_losses = [
        t for t in losses
        if t.get("sentiment_score") is not None and t["sentiment_score"] <= 4
    ]
    if len(low_sent_losses) >= 2:
        patterns.append({
            "type": "low_sentiment_losses",
            "count": len(low_sent_losses),
            "threshold": 4,
            "description": f"{len(low_sent_losses)} losses had sentiment <= 4",
        })

    # Pattern 3: High RSI entries that lost (overbought entries)
    high_rsi_losses = [
        t for t in losses
        if t.get("rsi") is not None and t["rsi"] > 65
    ]
    if len(high_rsi_losses) >= 2:
        avg_rsi = sum(t["rsi"] for t in high_rsi_losses) / len(high_rsi_losses)
        patterns.append({
            "type": "high_rsi_entry_losses",
            "count": len(high_rsi_losses),
            "avg_rsi": round(avg_rsi, 1),
            "description": f"{len(high_rsi_losses)} losses entered with RSI > 65 (avg RSI: {avg_rsi:.1f})",
        })

    # Pattern 4: Total loss rate
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
    patterns = analyze_patterns()
    losses = get_loss_trades(limit=10)

    return {
        "patterns_found": len(patterns),
        "patterns": patterns,
        "recent_losses": [
            {
                "ticker": t["ticker"],
                "pnl": t["pnl_gross"],
                "sentiment": t.get("sentiment_score"),
                "rsi": t.get("rsi"),
                "status": t["status"],
            }
            for t in losses
        ],
    }
