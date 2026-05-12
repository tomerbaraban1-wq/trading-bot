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

DISCORD_BOT_TOKEN:  str = os.getenv("DISCORD_BOT_TOKEN",  "")
DISCORD_CHANNEL_ID: str = os.getenv("DISCORD_CHANNEL_ID", "")

_DISCORD_API = "https://discord.com/api/v10"


def _enabled() -> bool:
    return bool(DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID)


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
