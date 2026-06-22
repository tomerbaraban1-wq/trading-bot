"""
News Reaction Learner  (Phase 2 — real, unbiased news learning)
================================================================

Learns the REAL relationship between news sentiment and what the price
actually did *afterwards*, by:

  1. RECORD   — periodically snapshot (sentiment_score, price) for watchlist
                stocks while the bot is idle / market closed.
  2. EVALUATE — once a snapshot is 1/5/10 trading days old, measure the
                actual forward % price move and store it.
  3. LEARN    — aggregate matured snapshots into honest statistics:
                "positive news → avg +X% after 5d (win rate Y%)" + correlation.

Why this is different from continuous_learner's sentiment correlation:
  - continuous_learner only sees stocks we actually BOUGHT (selection bias).
  - This observes the ENTIRE watchlist, bought or not — unbiased.

Honesty note: reliable *historical* news is not freely available, so this
builds knowledge FORWARD from when it first runs (rolling). It will say
"insufficient data" until enough snapshots mature — and it will honestly
report if news has NO measurable predictive edge.

Public API
----------
  record_news_snapshot(max_tickers)   -> int   (snapshots recorded)
  evaluate_matured_snapshots()        -> int   (snapshots updated)
  learn_news_reaction()               -> dict  (honest statistics)
  get_news_reaction_summary()         -> str   (Hebrew, for Telegram/logs)
  run_news_reaction_cycle()  (async)  -> dict  (record + evaluate + learn)
"""

import logging
import os
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
REACTION_HORIZONS = (1, 5, 10)                                    # trading days
BULLISH_THRESHOLD = float(os.getenv("SENTIMENT_BULLISH_THRESHOLD", "6"))  # >=6/10
SNAPSHOT_MAX_TICKERS = int(os.getenv("NEWS_REACTION_MAX_TICKERS", "12"))
MIN_SAMPLES = int(os.getenv("NEWS_REACTION_MIN_SAMPLES", "10"))


# ── Schema ──────────────────────────────────────────────────────────────────
def _ensure_table(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS news_reactions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker          TEXT NOT NULL,
            observed_at     TEXT NOT NULL,        -- ISO datetime UTC
            sentiment_score REAL NOT NULL,        -- 1-10 at observation
            price_at_obs    REAL NOT NULL,
            reaction_1d     REAL,                 -- % change ~1 trading day later
            reaction_5d     REAL,
            reaction_10d    REAL,
            evaluated_at    TEXT                  -- set when 10d horizon filled
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_news_react_ticker ON news_reactions(ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_news_react_obs ON news_reactions(observed_at)")
    conn.commit()


# ── 1. RECORD ───────────────────────────────────────────────────────────────
def record_news_snapshot(max_tickers: int = SNAPSHOT_MAX_TICKERS) -> int:
    """Snapshot (sentiment, price) for a rotating sample of the watchlist.

    One snapshot per ticker per day (skips tickers already captured today).
    Sentiment uses the existing scorer (LLM + cache + rate-limit aware).
    """
    try:
        import database
        import random
        import scanner
        import broker
        import sentiment

        conn = database.get_connection()
        _ensure_table(conn)

        watchlist = scanner.get_watchlist() or []
        if not watchlist:
            logger.debug("[NEWS-REACT] empty watchlist — nothing to record")
            return 0

        sample = (random.sample(watchlist, max_tickers)
                  if len(watchlist) > max_tickers else list(watchlist))
        today = datetime.now(timezone.utc).date().isoformat()
        recorded = 0

        for ticker in sample:
            try:
                already = conn.execute(
                    "SELECT 1 FROM news_reactions "
                    "WHERE ticker=? AND substr(observed_at,1,10)=? LIMIT 1",
                    (ticker, today),
                ).fetchone()
                if already:
                    continue

                sr = sentiment.score_sentiment(ticker)
                score = float(getattr(sr, "score", 5) or 5)
                price = broker.get_price(ticker)
                if price is None or float(price) <= 0:
                    continue

                conn.execute(
                    "INSERT INTO news_reactions "
                    "(ticker, observed_at, sentiment_score, price_at_obs) "
                    "VALUES (?,?,?,?)",
                    (ticker, datetime.now(timezone.utc).isoformat(),
                     score, float(price)),
                )
                recorded += 1
            except Exception as e:
                logger.debug(f"[NEWS-REACT] snapshot {ticker} failed: {e}")

        conn.commit()
        logger.info(
            f"[NEWS-REACT] Recorded {recorded} snapshots "
            f"(sampled {len(sample)} of {len(watchlist)})"
        )
        return recorded

    except Exception as e:
        logger.error(f"[NEWS-REACT] record failed: {e}")
        return 0


# ── 2. EVALUATE ─────────────────────────────────────────────────────────────
def _reaction_after(df, obs_date, baseline: float, horizon_td: int):
    """% move from baseline to the close `horizon_td` trading days after obs_date."""
    try:
        closes = df["Close"]
        dates = [d.date() if hasattr(d, "date") else d for d in df.index]
        after = [(d, c) for d, c in zip(dates, closes) if d > obs_date]
        if len(after) < horizon_td:
            return None  # not enough trading days have passed yet
        target = float(after[horizon_td - 1][1])
        if baseline and baseline > 0 and target > 0:
            return round((target - baseline) / baseline * 100, 3)
    except Exception:
        pass
    return None


def evaluate_matured_snapshots(max_rows: int = 300) -> int:
    """Fill in reaction_1d/5d/10d for snapshots old enough to measure."""
    try:
        import database
        from yfinance_cache import get_ohlcv

        conn = database.get_connection()
        _ensure_table(conn)

        rows = conn.execute("""
            SELECT id, ticker, observed_at, price_at_obs,
                   reaction_1d, reaction_5d, reaction_10d
            FROM news_reactions
            WHERE reaction_10d IS NULL
            ORDER BY observed_at ASC
            LIMIT ?
        """, (max_rows,)).fetchall()

        today = datetime.now(timezone.utc).date()
        updated = 0

        for rid, ticker, obs_iso, baseline, r1, r5, r10 in rows:
            try:
                obs_date = datetime.fromisoformat(obs_iso).date()
            except Exception:
                continue
            if (today - obs_date).days < 1:
                continue  # too fresh — even the 1d horizon hasn't passed

            df = get_ohlcv(ticker, days=40)
            if df is None or df.empty:
                continue

            new1 = r1 if r1 is not None else _reaction_after(df, obs_date, baseline, 1)
            new5 = r5 if r5 is not None else _reaction_after(df, obs_date, baseline, 5)
            new10 = r10 if r10 is not None else _reaction_after(df, obs_date, baseline, 10)
            evaluated_at = (datetime.now(timezone.utc).isoformat()
                            if new10 is not None else None)

            conn.execute(
                "UPDATE news_reactions "
                "SET reaction_1d=?, reaction_5d=?, reaction_10d=?, evaluated_at=? "
                "WHERE id=?",
                (new1, new5, new10, evaluated_at, rid),
            )
            updated += 1

        conn.commit()
        if updated:
            logger.info(f"[NEWS-REACT] Evaluated {updated} maturing snapshots")
        return updated

    except Exception as e:
        logger.error(f"[NEWS-REACT] evaluate failed: {e}")
        return 0


# ── 3. LEARN ────────────────────────────────────────────────────────────────
def learn_news_reaction() -> dict:
    """Aggregate matured snapshots into honest sentiment→reaction statistics."""
    try:
        import database
        conn = database.get_connection()
        _ensure_table(conn)

        rows = conn.execute("""
            SELECT sentiment_score, reaction_1d, reaction_5d, reaction_10d
            FROM news_reactions
            WHERE reaction_5d IS NOT NULL
        """).fetchall()

        n = len(rows)
        if n < MIN_SAMPLES:
            return {"status": "insufficient_data", "samples": n, "need": MIN_SAMPLES}

        result: dict = {"status": "ok", "samples": n, "horizons": {}}
        # column index per horizon label inside each row tuple
        for col_idx, label in ((1, "1d"), (2, "5d"), (3, "10d")):
            pairs = [(float(r[0]), float(r[col_idx]))
                     for r in rows if r[col_idx] is not None]
            if len(pairs) < MIN_SAMPLES:
                continue
            sents = np.array([p[0] for p in pairs], dtype=float)
            moves = np.array([p[1] for p in pairs], dtype=float)
            bull = moves[sents >= BULLISH_THRESHOLD]
            other = moves[sents < BULLISH_THRESHOLD]
            corr = (float(np.corrcoef(sents, moves)[0, 1])
                    if sents.std() > 0 and moves.std() > 0 else 0.0)

            result["horizons"][label] = {
                "avg_return_bullish": round(float(bull.mean()), 2) if len(bull) else None,
                "avg_return_other":   round(float(other.mean()), 2) if len(other) else None,
                "winrate_bullish":    round(float((bull > 0).mean() * 100), 1) if len(bull) else None,
                "correlation":        round(corr, 3),
                "n_bullish":          int(len(bull)),
                "n_other":            int(len(other)),
            }

        h5 = result["horizons"].get("5d")
        if h5:
            c = h5["correlation"]
            if c >= 0.2:
                result["verdict"] = "✅ חדשות חיוביות אכן קשורות לעלייה במחיר"
            elif c <= -0.2:
                result["verdict"] = "🔴 חדשות חיוביות דווקא קשורות לירידה — להיזהר"
            elif abs(c) < 0.1:
                result["verdict"] = "⚪ לחדשות אין יתרון חיזוי מדיד"
            else:
                result["verdict"] = "🟡 קשר חלש — צריך עוד נתונים"

        logger.info(
            f"[NEWS-REACT] Learned from {n} matured observations | "
            f"5d corr={h5['correlation'] if h5 else 'NA'}"
        )
        return result

    except Exception as e:
        logger.error(f"[NEWS-REACT] learn failed: {e}")
        return {"status": "error", "error": str(e)}


def get_news_reaction_summary() -> str:
    """Human-readable Hebrew summary for Telegram / logs."""
    data = learn_news_reaction()
    status = data.get("status")

    if status == "insufficient_data":
        return (
            "📰 <b>לימוד תגובת חדשות</b>\n"
            "━━━━━━━━━━━━━━━━\n"
            f"⏳ אוסף נתונים... {data['samples']}/{data['need']} תצפיות בשלו.\n"
            "הבוט מתעד חדשות+מחיר בזמן סרק ומודד מה קרה אחרי 1/5/10 ימים."
        )
    if status != "ok":
        return f"📰 לימוד תגובת חדשות — שגיאה: {data.get('error', 'לא ידוע')}"

    lines = [
        "📰 <b>לימוד תגובת חדשות (מדידה אמיתית)</b>",
        "━━━━━━━━━━━━━━━━",
        f"📊 תצפיות שנמדדו: <b>{data['samples']}</b>",
    ]
    for label in ("1d", "5d", "10d"):
        h = data["horizons"].get(label)
        if not h:
            continue
        lines.append(f"\n⏱️ <b>אחרי {label}:</b>")
        if h["avg_return_bullish"] is not None:
            lines.append(
                f"  📈 חדשות חיוביות → {h['avg_return_bullish']:+.2f}% "
                f"(הצלחה {h['winrate_bullish']:.0f}%, n={h['n_bullish']})"
            )
        if h["avg_return_other"] is not None:
            lines.append(
                f"  ➖ שאר → {h['avg_return_other']:+.2f}% (n={h['n_other']})"
            )
        lines.append(f"  🔗 מתאם: {h['correlation']:+.2f}")

    if data.get("verdict"):
        lines.append(f"\n<b>{data['verdict']}</b>")
    return "\n".join(lines)


# ── Combined async entrypoint (for the idle training loop) ───────────────────
async def run_news_reaction_cycle() -> dict:
    """Record fresh snapshots, evaluate matured ones, and re-learn."""
    import asyncio
    recorded = await asyncio.to_thread(record_news_snapshot)
    evaluated = await asyncio.to_thread(evaluate_matured_snapshots)
    insights = await asyncio.to_thread(learn_news_reaction)
    return {"recorded": recorded, "evaluated": evaluated, "insights": insights}


# ── Standalone smoke test ────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    print("evaluate:", evaluate_matured_snapshots())
    print(get_news_reaction_summary())
