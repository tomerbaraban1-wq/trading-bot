"""
Discord Notification Module
============================
Sends trading alerts to a Discord channel.
Uses Discord REST API directly — no library needed (aiohttp already installed).

Setup:
  DISCORD_BOT_TOKEN  — bot token from Discord Developer Portal
  DISCORD_CHANNEL_ID — channel ID (right-click → Copy Channel ID)
"""

import logging
import os

import aiohttp

logger = logging.getLogger(__name__)

DISCORD_BOT_TOKEN:    str = os.getenv("DISCORD_BOT_TOKEN",      "")
DISCORD_CHANNEL_ID:   str = os.getenv("DISCORD_CHANNEL_ID",    "")   # channel to SEND alerts
DISCORD_READ_CHANNEL: str = os.getenv("DISCORD_READ_CHANNEL_ID","1460937711396589634")  # #ישראל SKIL

_DISCORD_API = "https://discord.com/api/v10"

# Cache for community sentiment
_community_sentiment: dict[str, tuple[float, float]] = {}  # ticker → (score, timestamp)
_COMMUNITY_TTL = 1800  # 30 min


def _enabled() -> bool:
    return bool(DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID)


async def fetch_community_sentiment(ticker: str) -> float | None:
    """
    Read recent messages from the SKIL #ישראל channel and extract
    community sentiment for a given ticker.
    Returns sentiment score 1-10 or None if not mentioned.
    Cached 30 minutes.
    """
    import time, re
    now = time.time()
    cached = _community_sentiment.get(ticker.upper())
    if cached and now - cached[1] < _COMMUNITY_TTL:
        return cached[0]

    if not DISCORD_BOT_TOKEN or not DISCORD_READ_CHANNEL:
        return None

    try:
        url = f"{_DISCORD_API}/channels/{DISCORD_READ_CHANNEL}/messages?limit=50"
        headers = {"Authorization": f"Bot {DISCORD_BOT_TOKEN}"}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return None
                messages = await resp.json()

        # Extract messages mentioning the ticker
        t = ticker.upper()
        _BULLISH = {"קנה","קנייה","עולה","חיובי","buy","bull","long","שורי","מומנטום","פריצה","ירוק"}
        _BEARISH = {"מכור","מכירה","יורד","שלילי","sell","bear","short","דובי","נפילה","אדום","סכנה"}

        bull = bear = mentions = 0
        for msg in messages:
            content = msg.get("content", "")
            if t not in content.upper():
                continue
            mentions += 1
            words = set(content.lower().split())
            bull += len(words & _BULLISH)
            bear += len(words & _BEARISH)

        if mentions == 0:
            return None

        # Score 1-10
        net = bull - bear
        score = max(1, min(10, 5 + net))
        _community_sentiment[t] = (float(score), now)
        logger.info(f"[DISCORD SENTIMENT] {t}: {mentions} mentions, bull={bull}, bear={bear} → score={score}")
        return float(score)

    except Exception as exc:
        logger.debug(f"[DISCORD SENTIMENT] {ticker}: {exc}")
        return None


async def get_trending_tickers() -> list[str]:
    """
    Scan the SKIL #ישראל channel and find most-mentioned tickers.
    Returns list of tickers ordered by mention count.
    """
    import re
    if not DISCORD_BOT_TOKEN or not DISCORD_READ_CHANNEL:
        return []
    try:
        url = f"{_DISCORD_API}/channels/{DISCORD_READ_CHANNEL}/messages?limit=100"
        headers = {"Authorization": f"Bot {DISCORD_BOT_TOKEN}"}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return []
                messages = await resp.json()

        # Find ticker-like patterns: 2-5 uppercase letters
        counts: dict[str, int] = {}
        for msg in messages:
            content = msg.get("content", "")
            found = re.findall(r"\b[A-Z]{2,5}\b", content)
            # Filter common English words
            skip = {"THE","AND","FOR","ARE","YOU","HIS","HER","WAS","THIS","FROM","WITH","HAVE"}
            for t in found:
                if t not in skip:
                    counts[t] = counts.get(t, 0) + 1

        # Return top tickers mentioned 3+ times
        trending = sorted([t for t, c in counts.items() if c >= 3], key=lambda x: -counts[x])
        if trending:
            logger.info(f"[DISCORD TRENDING] {trending[:5]}")
        return trending[:10]

    except Exception as exc:
        logger.debug(f"[DISCORD TRENDING]: {exc}")
        return []


async def send_discord(text: str) -> bool:
    """
    Send a message to the configured Discord channel.
    Discord uses markdown — converts basic HTML tags from Telegram format.
    Returns True on success.
    """
    if not _enabled():
        return False

    # Convert Telegram HTML to Discord markdown
    content = (
        text
        .replace("<b>", "**").replace("</b>", "**")
        .replace("<i>", "*").replace("</i>", "*")
        .replace("<code>", "`").replace("</code>", "`")
        .replace("━━━━━━━━━━━━━━━━", "─────────────────")
        # Strip remaining HTML tags
    )
    # Remove any remaining HTML tags
    import re
    content = re.sub(r"<[^>]+>", "", content)
    # Discord max message length: 2000 chars
    content = content[:2000]

    url = f"{_DISCORD_API}/channels/{DISCORD_CHANNEL_ID}/messages"
    headers = {
        "Authorization": f"Bot {DISCORD_BOT_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"content": content}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status in (200, 201):
                    logger.debug(f"[DISCORD] Message sent successfully")
                    return True
                else:
                    body = await resp.text()
                    logger.warning(f"[DISCORD] HTTP {resp.status}: {body[:200]}")
                    return False
    except Exception as exc:
        logger.warning(f"[DISCORD] Send failed: {exc}")
        return False
