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
DISCORD_CHANNEL_ID:   str = os.getenv("DISCORD_CHANNEL_ID",    "")    # channel to SEND alerts
DISCORD_GUILD_ID:     str = os.getenv("DISCORD_GUILD_ID",       "882265638784090182")  # SKIL server ID

_DISCORD_API = "https://discord.com/api/v10"

# Cache for community sentiment
_community_sentiment: dict[str, tuple[float, float]] = {}  # ticker → (score, timestamp)
_COMMUNITY_TTL = 1800  # 30 min
_all_messages_cache: tuple[list, float] = ([], 0.0)  # (messages, timestamp)


def _enabled() -> bool:
    return bool(DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID)


async def _fetch_all_server_messages(limit_per_channel: int = 30) -> list[dict]:
    """
    Fetch recent messages from ALL text channels in the SKIL server.
    Cached 30 minutes to avoid rate limits.
    """
    import time
    global _all_messages_cache
    now = time.time()
    if _all_messages_cache[0] and now - _all_messages_cache[1] < _COMMUNITY_TTL:
        return _all_messages_cache[0]

    if not DISCORD_BOT_TOKEN or not DISCORD_GUILD_ID:
        return []

    all_messages = []
    headers = {"Authorization": f"Bot {DISCORD_BOT_TOKEN}"}

    try:
        async with aiohttp.ClientSession() as session:
            # Step 1: Get all channels in the server
            async with session.get(
                f"{_DISCORD_API}/guilds/{DISCORD_GUILD_ID}/channels",
                headers=headers, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"[DISCORD] Failed to list channels: {resp.status}")
                    return []
                channels = await resp.json()

            # Step 2: Read text channels (type 0 = GUILD_TEXT)
            text_channels = [c for c in channels if c.get("type") == 0]
            logger.info(f"[DISCORD] Reading {len(text_channels)} channels from SKIL server")

            for channel in text_channels[:15]:  # max 15 channels to avoid rate limits
                ch_id = channel["id"]
                try:
                    async with session.get(
                        f"{_DISCORD_API}/channels/{ch_id}/messages?limit={limit_per_channel}",
                        headers=headers, timeout=aiohttp.ClientTimeout(total=8)
                    ) as r:
                        if r.status == 200:
                            msgs = await r.json()
                            for m in msgs:
                                m["_channel_name"] = channel.get("name", "")
                            all_messages.extend(msgs)
                except Exception:
                    pass  # skip inaccessible channels

    except Exception as exc:
        logger.warning(f"[DISCORD] Server scan failed: {exc}")
        return []

    _all_messages_cache = (all_messages, now)
    logger.info(f"[DISCORD] Collected {len(all_messages)} messages from SKIL server")
    return all_messages


async def fetch_community_sentiment(ticker: str) -> float | None:
    """
    Scan ALL SKIL server channels for mentions of the ticker.
    Returns sentiment score 1-10 or None if not mentioned.
    """
    import time
    now = time.time()
    cached = _community_sentiment.get(ticker.upper())
    if cached and now - cached[1] < _COMMUNITY_TTL:
        return cached[0]

    messages = await _fetch_all_server_messages()
    if not messages:
        return None

    t = ticker.upper()
    _BULLISH = {"קנה","קנייה","עולה","חיובי","buy","bull","long","שורי","מומנטום","פריצה","ירוק","חזק","אחלה"}
    _BEARISH = {"מכור","מכירה","יורד","שלילי","sell","bear","short","דובי","נפילה","אדום","סכנה","חלש","זהירות"}

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

    net = bull - bear
    score = float(max(1, min(10, 5 + net)))
    _community_sentiment[t] = (score, now)
    logger.info(f"[SKIL SENTIMENT] {t}: {mentions} mentions across server, bull={bull}, bear={bear} → {score}/10")
    return score


async def get_trending_tickers() -> list[str]:
    """
    Scan ALL SKIL server channels and find most-mentioned stock tickers.
    Returns tickers ordered by mention frequency.
    """
    import re
    messages = await _fetch_all_server_messages()
    if not messages:
        return []

    counts: dict[str, int] = {}
    skip = {"THE","AND","FOR","ARE","YOU","HIS","HER","WAS","THIS","FROM","WITH","HAVE",
            "NOT","BUT","ALL","CAN","HAS","ITS","NEW","ONE","TWO","GET","USE","SAY",
            "VIP","COM","CEO","ETF","IPO","ATH","ATL","RSI"}

    for msg in messages:
        content = msg.get("content", "")
        found = re.findall(r"\b[A-Z]{2,5}\b", content)
        for t in found:
            if t not in skip:
                counts[t] = counts.get(t, 0) + 1

    trending = sorted([t for t, c in counts.items() if c >= 3], key=lambda x: -counts[x])
    if trending:
        logger.info(f"[SKIL TRENDING] Top tickers: {trending[:5]}")
    return trending[:10]


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
