"""
Volume Surge Detector
=====================

Detects unusual volume spikes that indicate institutional activity.

When a stock's volume is 3×+ above its 20-day average:
→ Something big is happening (earnings leak, news, institutional buy/sell)
→ Bot sends an alert with score + TV link

Thresholds:
  3×  normal → 🟡 Notable
  5×  normal → 🔴 MAJOR surge (act fast)
  10× normal → 🚨 Extreme (possible halt/halt recovery)
"""

import asyncio
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

_last_alert_time: dict[str, float] = {}   # ticker → last alert timestamp
ALERT_COOLDOWN_SEC = 60 * 60              # 1 hour between alerts per ticker


async def check_volume_surges(watchlist: list[str], top_n: int = 5) -> list[dict]:
    """
    Scan watchlist for volume surges.
    Returns list of surge events sorted by severity.
    """
    try:
        import yfinance as yf
        import pandas as pd
        import numpy as np

        surges = []

        # Batch download for speed
        tickers_str = " ".join(watchlist[:50])
        hist = await asyncio.to_thread(
            yf.download,
            tickers_str,
            period="22d",
            interval="1d",
            progress=False,
            auto_adjust=True,
        )

        if hist is None or hist.empty:
            return []

        # Handle both single and multi-ticker downloads
        import pandas as pd
        if isinstance(hist.columns, pd.MultiIndex):
            vol_data = hist["Volume"]
        else:
            vol_data = hist[["Volume"]]

        for ticker in watchlist[:50]:
            try:
                if ticker not in vol_data.columns:
                    continue
                vols = vol_data[ticker].dropna()
                if len(vols) < 5:
                    continue

                today_vol = float(vols.iloc[-1])
                avg_vol   = float(vols.iloc[-21:-1].mean())  # 20-day avg (excluding today)

                if avg_vol < 1000 or today_vol < 1000:
                    continue

                ratio = today_vol / avg_vol

                if ratio >= 3.0:
                    level = "extreme" if ratio >= 10 else "major" if ratio >= 5 else "notable"
                    surges.append({
                        "ticker": ticker,
                        "ratio": ratio,
                        "today_vol": today_vol,
                        "avg_vol": avg_vol,
                        "level": level,
                    })

            except Exception:
                pass

        surges.sort(key=lambda x: x["ratio"], reverse=True)
        return surges[:top_n]

    except Exception as e:
        logger.debug(f"Volume surge scan failed: {e}")
        return []


async def run_volume_surge_alert(watchlist: list[str]) -> None:
    """
    Check for volume surges and send Telegram alerts.
    Rate-limited per ticker.
    """
    try:
        surges = await check_volume_surges(watchlist, top_n=5)

        for surge in surges:
            ticker = surge["ticker"]

            # Rate limit
            if time.time() - _last_alert_time.get(ticker, 0) < ALERT_COOLDOWN_SEC:
                continue

            ratio = surge["ratio"]
            level = surge["level"]

            # Get score quickly
            score = 0
            try:
                from scoring import get_composite_score
                _sr = await asyncio.wait_for(
                    asyncio.to_thread(get_composite_score, ticker, 5),
                    timeout=20,
                )
                score = _sr.get("composite_score", 0)
            except Exception:
                pass

            # Build alert
            emoji = "🚨" if level == "extreme" else "🔴" if level == "major" else "🟡"
            tv_url = f"https://www.tradingview.com/chart/?symbol={ticker}"
            score_bar = "🟩" * round(score / 10) + "⬜" * (10 - round(score / 10))

            from telegram_bot import send_message
            await send_message(
                f"{emoji} <b>Volume Surge — {ticker}!</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📊 נפח היום: <b>{ratio:.1f}×</b> מהממוצע\n"
                f"  ({surge['today_vol']/1e6:.1f}M vs {surge['avg_vol']/1e6:.1f}M avg)\n"
                f"📈 ציון: {score_bar} <b>{score:.0f}/100</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"💡 מישהו גדול קונה/מוכר!\n"
                f'🔗 <a href="{tv_url}">פתח גרף ב-TradingView</a>',
                reply_markup={"inline_keyboard": [[
                    {"text": f"📊 גרף {ticker}", "url": tv_url},
                    {"text": f"⚡ AI ניתוח", "callback_data": f"ai:{ticker}"},
                ]]}
            )

            _last_alert_time[ticker] = time.time()
            logger.info(f"[VOLUME SURGE] {ticker}: {ratio:.1f}× — alert sent")

    except Exception as e:
        logger.debug(f"Volume surge alert failed: {e}")
