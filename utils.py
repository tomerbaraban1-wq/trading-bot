import asyncio
import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


def retry_sync(func: Callable, *args: Any, max_retries: int = 2, base_delay: float = 1.0) -> Any:
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return func(*args)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Retry {attempt + 1}/{max_retries} for {func.__name__} after {delay}s: {e}")
                time.sleep(delay)
    raise last_error


async def retry_async(func: Callable, *args: Any, max_retries: int = 3, base_delay: float = 1.0) -> Any:
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return await func(*args)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Retry {attempt + 1}/{max_retries} for {func.__name__} after {delay}s: {e}")
                await asyncio.sleep(delay)
    raise last_error
