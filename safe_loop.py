"""
Safe Loop Wrapper — אחיד exception handling עבור כל הלולאות הבלתי סופיות
כל לולאה while True צריכה להיות בחוט זה
"""
import asyncio
import logging
import time
from typing import Callable, Optional
from functools import wraps

logger = logging.getLogger(__name__)


async def safe_loop(
    loop_fn: Callable,
    loop_name: str,
    interval: float = 5.0,
    max_consecutive_errors: int = 10,
    telegram_alert_threshold: int = 5,
):
    """
    Safe wrapper for infinite loops — handles crashes gracefully

    Usage:
        async def my_loop():
            while True:
                # your code
                await asyncio.sleep(5)

        # Instead of: asyncio.create_task(my_loop())
        # Use:       await safe_loop(my_loop, "my_loop_name", interval=5)

    Args:
        loop_fn: async function that contains while True loop
        loop_name: descriptive name for logging
        interval: sleep interval (used for detection of loop responsiveness)
        max_consecutive_errors: how many consecutive errors before giving up
        telegram_alert_threshold: alert Telegram after this many consecutive errors
    """
    error_count = 0
    last_success = time.time()

    while True:
        try:
            # Run the loop function (should contain while True)
            await loop_fn()

            # If we exit normally, reset error count
            error_count = 0
            last_success = time.time()

        except asyncio.CancelledError:
            logger.info(f"[{loop_name}] Cancelled (graceful shutdown)")
            break

        except Exception as e:
            error_count += 1
            uptime_seconds = time.time() - last_success

            logger.error(
                f"[{loop_name}] CRASH #{error_count}/{max_consecutive_errors} "
                f"after {uptime_seconds:.0f}s: {type(e).__name__}: {str(e)[:100]}"
            )

            # Check if we should give up
            if error_count >= max_consecutive_errors:
                logger.critical(
                    f"[{loop_name}] GIVING UP after {max_consecutive_errors} consecutive crashes"
                )

                # Send Telegram alert
                if error_count >= telegram_alert_threshold:
                    try:
                        import os
                        from config import settings
                        import requests

                        token = settings.TELEGRAM_BOT_TOKEN
                        chat = settings.TELEGRAM_CHAT_ID
                        if token and chat:
                            requests.post(
                                f"https://api.telegram.org/bot{token}/sendMessage",
                                json={
                                    "chat_id": chat,
                                    "text": (
                                        f"🔴 <b>LOOP CRASH</b>\n"
                                        f"<code>{loop_name}</code> failed {error_count}x\n"
                                        f"Last error: <code>{type(e).__name__}</code>\n"
                                        f"Watchdog will restart bot soon"
                                    ),
                                    "parse_mode": "HTML",
                                },
                                timeout=5,
                            )
                    except Exception as alert_err:
                        logger.warning(f"Failed to send alert: {alert_err}")

                # Exit the loop (let watchdog handle restart)
                raise

            # Send alert on repeated errors
            if error_count == telegram_alert_threshold:
                try:
                    import os
                    from config import settings
                    import requests

                    token = settings.TELEGRAM_BOT_TOKEN
                    chat = settings.TELEGRAM_CHAT_ID
                    if token and chat:
                        requests.post(
                            f"https://api.telegram.org/bot{token}/sendMessage",
                            json={
                                "chat_id": chat,
                                "text": (
                                    f"⚠️ <b>LOOP ISSUE</b>\n"
                                    f"<code>{loop_name}</code> crashed {error_count}x\n"
                                    f"Will restart after {max_consecutive_errors} failures"
                                ),
                                "parse_mode": "HTML",
                            },
                            timeout=5,
                        )
                except Exception:
                    pass

            # Wait before retry (exponential backoff)
            wait_time = min(2 ** (error_count - 1), 60)
            logger.warning(f"[{loop_name}] Retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)


def safe_loop_decorator(
    loop_name: Optional[str] = None,
    interval: float = 5.0,
    max_errors: int = 10,
):
    """
    Decorator version of safe_loop

    Usage:
        @safe_loop_decorator("my_loop_name")
        async def my_loop():
            while True:
                # your code
                await asyncio.sleep(5)

        # Create as: asyncio.create_task(my_loop())
    """

    def decorator(func):
        @wraps(func)
        async def wrapper():
            name = loop_name or func.__name__
            await safe_loop(func, name, interval=interval, max_consecutive_errors=max_errors)

        return wrapper

    return decorator
