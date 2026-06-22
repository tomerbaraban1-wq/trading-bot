"""
Feature Edge Learner  (deepest learning — "what actually predicts?")
====================================================================

The deepest, most honest thing the bot can learn: out of every indicator it
records at entry (RSI, MACD, Bollinger position, volume ratio, sentiment),
which ones ACTUALLY correlate with winning trades — and which are just noise.

For each indicator it measures, from REAL closed trades:
  • point-biserial correlation between the indicator value and win/loss
  • win-rate of the TOP third vs the BOTTOM third of that indicator
  • the "edge" = win-rate spread (high third − low third)

Then it ranks indicators by real predictive power and gives an honest verdict
per indicator: predictive / weak / noise. This tells the strategy where to
focus — instead of assuming every indicator matters equally.

Honesty note: with a few dozen trades most indicators will show NEAR-ZERO
edge. That is the truth, and surfacing it is the point — it stops the bot
(and you) from trusting signals that don't actually work.

Public API
----------
  learn_feature_edges(min_samples) -> dict
  get_feature_edge_summary()       -> str   (Hebrew, for Telegram/logs)
"""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# (column in trade_log, Hebrew label, higher-is-better hint for readability)
FEATURES = [
    ("sentiment_score", "סנטימנט"),
    ("rsi",             "RSI"),
    ("macd",            "MACD"),
    ("bb_position",     "מיקום בולינגר"),
    ("volume_ratio",    "יחס נפח"),
]

MIN_SAMPLES = int(os.getenv("FEATURE_EDGE_MIN_SAMPLES", "15"))


def learn_feature_edges(min_samples: int = MIN_SAMPLES) -> dict:
    """Measure the real predictive power of each entry indicator."""
    try:
        import database
        conn = database.get_connection()

        cols = ", ".join(f for f, _ in FEATURES)
        rows = conn.execute(f"""
            SELECT {cols}, pnl_gross
            FROM trade_log
            WHERE status != 'open' AND pnl_gross IS NOT NULL
        """).fetchall()

        n = len(rows)
        if n < min_samples:
            return {"status": "insufficient_data", "samples": n, "need": min_samples}

        edges = []
        for idx, (feat, label) in enumerate(FEATURES):
            pairs = [(float(r[idx]), 1.0 if float(r[-1]) > 0 else 0.0)
                     for r in rows if r[idx] is not None]
            if len(pairs) < min_samples:
                continue

            x   = np.array([p[0] for p in pairs], dtype=float)
            win = np.array([p[1] for p in pairs], dtype=float)

            corr = (float(np.corrcoef(x, win)[0, 1])
                    if x.std() > 0 and win.std() > 0 else 0.0)

            # Win-rate of the top third vs bottom third of this indicator.
            order = np.argsort(x)
            third = max(1, len(order) // 3)
            wr_low  = float(win[order[:third]].mean() * 100)
            wr_high = float(win[order[-third:]].mean() * 100)

            ac = abs(corr)
            if ac >= 0.15:
                verdict = "✅ מנבא"
            elif ac < 0.07:
                verdict = "⚪ רעש"
            else:
                verdict = "🟡 חלש"

            edges.append({
                "feature":       feat,
                "label":         label,
                "n":             len(pairs),
                "correlation":   round(corr, 3),
                "winrate_high":  round(wr_high, 1),
                "winrate_low":   round(wr_low, 1),
                "edge":          round(wr_high - wr_low, 1),
                "verdict":       verdict,
            })

        # Rank by real predictive power (absolute correlation).
        edges.sort(key=lambda d: abs(d["correlation"]), reverse=True)

        predictive = [e for e in edges if e["verdict"].startswith("✅")]
        logger.info(
            f"[FEATURE-EDGE] {n} trades | "
            + " | ".join(f"{e['label']}={e['correlation']:+.2f}" for e in edges)
            + f" | predictive={len(predictive)}"
        )
        return {"status": "ok", "samples": n, "features": edges,
                "predictive_count": len(predictive)}

    except Exception as e:
        logger.error(f"[FEATURE-EDGE] learn failed: {e}")
        return {"status": "error", "error": str(e)}


def get_feature_edge_summary() -> str:
    """Human-readable Hebrew summary for Telegram / logs."""
    data = learn_feature_edges()
    status = data.get("status")

    if status == "insufficient_data":
        return (
            "🔬 <b>לימוד עומק: מה באמת מנבא?</b>\n"
            "━━━━━━━━━━━━━━━━\n"
            f"⏳ צריך עוד עסקאות סגורות: {data['samples']}/{data['need']}."
        )
    if status != "ok":
        return f"🔬 לימוד עומק — שגיאה: {data.get('error', 'לא ידוע')}"

    lines = [
        "🔬 <b>לימוד עומק: מה באמת מנבא הצלחה?</b>",
        "━━━━━━━━━━━━━━━━",
        f"📊 מבוסס על <b>{data['samples']}</b> עסקאות סגורות אמיתיות",
        "<i>(מדורג לפי כוח ניבוי אמיתי)</i>",
        "",
    ]
    for e in data["features"]:
        lines.append(
            f"{e['verdict']} <b>{e['label']}</b> — מתאם {e['correlation']:+.2f} | "
            f"הצלחה גבוה/נמוך: {e['winrate_high']:.0f}% מול {e['winrate_low']:.0f}% "
            f"(n={e['n']})"
        )

    if data["predictive_count"] == 0:
        lines.append("")
        lines.append("⚪ <b>האמת:</b> אף אינדיקטור לא מנבא באופן אמין — "
                     "עדות נוספת שתזמון קצר-טווח קשה מאוד.")
    else:
        top = data["features"][0]
        lines.append("")
        lines.append(f"🎯 הכי חזק: <b>{top['label']}</b> (מתאם {top['correlation']:+.2f})")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    print(get_feature_edge_summary())
