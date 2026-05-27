"""
Telegram Polling Loop
======================

Alternative to webhook for LOCAL development.
When RENDER_EXTERNAL_URL is not set (local machine), this module
polls Telegram's getUpdates API every 2 seconds and feeds messages
directly into handle_telegram_update() — exactly the same handler
that the webhook uses.

Usage (auto-started by main.py when no RENDER_EXTERNAL_URL):
    from telegram_polling import start_polling_if_local

Or standalone:
    python telegram_polling.py
"""

import asyncio
import logging
import os
import time

import aiohttp

logger = logging.getLogger(__name__)

_last_update_id = 0
_running = False


async def _fetch_updates(session: aiohttp.ClientSession, token: str, offset: int) -> list:
    """Call getUpdates and return list of update objects."""
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    params = {
        "offset": offset,
        "timeout": 20,       # long-poll: wait up to 20s for new messages
        "allowed_updates": ["message", "callback_query"],
    }
    try:
        async with session.get(
            url, params=params,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("result", [])
            else:
                body = await resp.text()
                logger.warning(f"[POLLING] getUpdates returned {resp.status}: {body[:100]}")
                return []
    except asyncio.TimeoutError:
        return []   # normal — long-poll timeout
    except Exception as e:
        logger.warning(f"[POLLING] getUpdates error: {e}")
        return []


async def polling_loop():
    """
    Main polling loop. Runs forever.
    Feeds every incoming update into handle_telegram_update().
    """
    global _last_update_id, _running

    from config import settings
    token = settings.TELEGRAM_BOT_TOKEN

    if not token:
        logger.warning("[POLLING] No TELEGRAM_BOT_TOKEN — polling disabled")
        return

    # Import the same handler the webhook uses
    try:
        from telegram_chat import handle_telegram_update
    except ImportError as e:
        logger.error(f"[POLLING] Cannot import handler: {e}")
        return

    logger.info("[POLLING] Started — listening for Telegram messages...")
    _running = True

    async with aiohttp.ClientSession() as session:
        while _running:
            try:
                updates = await _fetch_updates(session, token, _last_update_id)

                for update in updates:
                    update_id = update.get("update_id", 0)
                    _last_update_id = update_id + 1  # acknowledge this update

                    try:
                        await handle_telegram_update(update)
                        logger.debug(f"[POLLING] Processed update {update_id}")
                    except Exception as e:
                        logger.error(f"[POLLING] Handler error for update {update_id}: {e}")

            except asyncio.CancelledError:
                logger.info("[POLLING] Stopped")
                _running = False
                return
            except Exception as e:
                logger.error(f"[POLLING] Loop error: {e}")
                await asyncio.sleep(5)


def is_local_mode() -> bool:
    """Return True if running locally (no RENDER_EXTERNAL_URL set)."""
    return not os.getenv("RENDER_EXTERNAL_URL", "").strip()


async def start_polling_if_local():
    """
    Start polling loop if no external URL is configured.
    Call this from main.py startup.
    On Render: does nothing (webhook handles messages).
    Locally: starts polling so user messages are processed.
    """
    if is_local_mode():
        logger.info("[POLLING] Local mode detected — starting polling (no RENDER_EXTERNAL_URL)")
        asyncio.create_task(polling_loop())
    else:
        logger.info("[POLLING] Cloud mode — using webhook (RENDER_EXTERNAL_URL is set)")


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    print("=" * 50)
    print("Telegram Polling Mode")
    print("Bot will receive messages directly (no webhook needed)")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    print()

    asyncio.run(polling_loop())
