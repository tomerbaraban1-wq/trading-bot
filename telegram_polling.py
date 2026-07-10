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


# Track consecutive 409 conflicts (webhook + polling both active)
_consecutive_409 = 0
_MAX_409_BEFORE_PAUSE = 5    # after 5 conflicts, auto-disable webhook
_pause_until_ts = 0.0          # epoch seconds — pause polling until this time

# Throttle the (otherwise very noisy) 409 warnings. A second instance elsewhere
# (e.g. a cloud/Render copy) may keep re-setting a webhook every few minutes; left
# unthrottled this floods the log with a line every couple of seconds. Summarise
# at most once per _LOG_409_EVERY seconds.
_last_409_log_ts = 0.0
_suppressed_409_count = 0
_LOG_409_EVERY = 300           # seconds


async def _delete_webhook(token: str) -> bool:
    """Force-delete an existing webhook so polling can take over.
    Required when localtunnel set a webhook but we run locally with polling."""
    url = f"https://api.telegram.org/bot{token}/deleteWebhook"
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(url, json={"drop_pending_updates": False},
                              timeout=aiohttp.ClientTimeout(total=10)) as r:
                if r.status == 200:
                    logger.debug("[POLLING] Auto-deleted stale webhook (409 → polling takeover)")
                    return True
    except Exception as e:
        logger.warning(f"[POLLING] deleteWebhook failed: {e}")
    return False


async def _webhook_suppressor(token: str) -> None:
    """Belt-and-suspenders for local mode: periodically delete any webhook that a
    second instance (e.g. a cloud/Render copy) registers, narrowing the window in
    which it can intercept incoming messages. The reactive 409 handler in
    _fetch_updates already covers the common case; this just tightens it."""
    while _running:
        await asyncio.sleep(30)
        try:
            await _delete_webhook(token)
        except Exception:
            pass


async def _fetch_updates(session: aiohttp.ClientSession, token: str, offset: int) -> list:
    """Call getUpdates and return list of update objects.
    Auto-recovers from 409 conflicts by deleting the conflicting webhook."""
    global _consecutive_409, _pause_until_ts, _last_409_log_ts, _suppressed_409_count
    import time as _t

    # If we're in cool-down, skip the call entirely
    if _t.time() < _pause_until_ts:
        await asyncio.sleep(2)
        return []

    url = f"https://api.telegram.org/bot{token}/getUpdates"
    params = {
        "offset": offset,
        "timeout": 2,        # long-poll: 2s — picks up messages within 2 seconds
        "allowed_updates": ["message", "callback_query"],
    }
    try:
        async with session.get(
            url, params=params,
            timeout=aiohttp.ClientTimeout(total=8)   # 2s poll + 6s buffer for network
        ) as resp:
            if resp.status == 200:
                _consecutive_409 = 0   # reset on success
                data = await resp.json()
                return data.get("result", [])

            body = await resp.text()

            # Special handling for 409 (webhook conflict)
            if resp.status == 409:
                _consecutive_409 += 1
                # AGGRESSIVE takeover: delete the conflicting webhook on the FIRST
                # 409. A second instance (e.g. a Render deploy) keeps re-registering
                # a webhook that steals commands from local polling — reclaim fast.
                # The warning is throttled so a chronic remote conflict does not
                # flood the log with a line every couple of seconds.
                _now = _t.time()
                if _now - _last_409_log_ts >= _LOG_409_EVERY:
                    _extra = f" (+{_suppressed_409_count} more since)" if _suppressed_409_count else ""
                    logger.warning(
                        "[POLLING] Telegram 409 conflict%s — another instance (likely a "
                        "cloud/Render copy) keeps setting a webhook. Reclaiming polling. "
                        "Permanent fix: stop the second instance.", _extra
                    )
                    _last_409_log_ts = _now
                    _suppressed_409_count = 0
                else:
                    _suppressed_409_count += 1
                if await _delete_webhook(token):
                    _consecutive_409 = 0
                else:
                    # Couldn't delete — short cool-down, then retry
                    _pause_until_ts = _t.time() + 30
                    logger.warning("[POLLING] deleteWebhook failed — retrying in 30s")
                await asyncio.sleep(1)   # brief pause, then reclaim
            else:
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

    # Belt-and-suspenders: proactively clear any webhook a second instance sets,
    # so incoming messages reach local polling with minimal delay (the reactive
    # 409 handler already covers the common case).
    asyncio.create_task(_webhook_suppressor(token))

    async with aiohttp.ClientSession() as session:
        while _running:
            try:
                updates = await _fetch_updates(session, token, _last_update_id)

                for update in updates:
                    update_id = update.get("update_id", 0)
                    _last_update_id = update_id + 1  # acknowledge this update

                    try:
                        # Hard timeout per update: a single hung handler (stuck
                        # network call inside a command) must never freeze the
                        # whole polling loop — that outage looks like "the bot
                        # ignores me" while everything else keeps running.
                        await asyncio.wait_for(handle_telegram_update(update), timeout=180)
                        logger.debug(f"[POLLING] Processed update {update_id}")
                    except asyncio.TimeoutError:
                        logger.error(f"[POLLING] Handler TIMED OUT (180s) for update {update_id} — skipped, polling continues")
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
