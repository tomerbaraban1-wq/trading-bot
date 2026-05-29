"""
API Resilience v2 — Timeout + Retry decorator for all external APIs
כל API call תקבל automatic retry with exponential backoff
"""
import asyncio
import logging
import functools
import time
from typing import Callable, Any, Optional, TypeVar, Union
from collections.abc import Coroutine

logger = logging.getLogger(__name__)

T = TypeVar('T')


class APIError(Exception):
    """Custom error for API failures"""
    pass


def with_retry(
    max_retries: int = 3,
    backoff_base: float = 0.5,
    backoff_max: float = 30.0,
    timeout: Optional[float] = 10.0,
    fallback: Any = None,
):
    """
    Decorator for async functions — adds timeout + retry with exponential backoff

    Usage:
        @with_retry(max_retries=3, timeout=15)
        async def call_api():
            ...

    Args:
        max_retries: max retry attempts
        backoff_base: initial wait time (0.5s)
        backoff_max: max wait between retries (30s)
        timeout: timeout for single attempt (None = no timeout)
        fallback: return value if all retries fail (None = raise exception)
    """

    def decorator(func: Callable[..., Coroutine]) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_error = None
            wait_time = backoff_base

            for attempt in range(max_retries + 1):
                try:
                    # Apply timeout if specified
                    if timeout:
                        result = await asyncio.wait_for(
                            func(*args, **kwargs),
                            timeout=timeout
                        )
                    else:
                        result = await func(*args, **kwargs)

                    # Success on first try
                    if attempt > 0:
                        logger.info(f"✅ {func.__name__} succeeded on attempt {attempt + 1}")

                    return result

                except asyncio.TimeoutError as e:
                    last_error = e
                    error_type = "TIMEOUT"

                except (
                    asyncio.CancelledError,
                    KeyboardInterrupt,
                    SystemExit,
                ) as e:
                    # Don't retry these
                    raise

                except Exception as e:
                    last_error = e
                    error_type = type(e).__name__

                # This attempt failed
                if attempt < max_retries:
                    logger.warning(
                        f"⚠️  {func.__name__} failed ({error_type}): {str(e)[:100]} "
                        f"— retry in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(wait_time)

                    # Exponential backoff with jitter
                    wait_time = min(wait_time * 2 + (wait_time * 0.1), backoff_max)
                else:
                    # All retries exhausted
                    logger.error(
                        f"❌ {func.__name__} failed after {max_retries + 1} attempts: {str(last_error)[:200]}"
                    )

            # All retries failed
            if fallback is not None:
                logger.warning(f"   Using fallback value for {func.__name__}")
                return fallback
            else:
                raise APIError(f"{func.__name__} failed after {max_retries + 1} attempts: {last_error}")

        return wrapper

    return decorator


def sync_with_retry(
    max_retries: int = 3,
    backoff_base: float = 0.5,
    backoff_max: float = 30.0,
    timeout: Optional[float] = None,
    fallback: Any = None,
):
    """
    Decorator for synchronous (non-async) functions

    Note: timeout is NOT enforced for sync functions (use asyncio for that)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_error = None
            wait_time = backoff_base

            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)

                    if attempt > 0:
                        logger.info(f"✅ {func.__name__} succeeded on attempt {attempt + 1}")

                    return result

                except (KeyboardInterrupt, SystemExit) as e:
                    raise

                except Exception as e:
                    last_error = e

                # This attempt failed
                if attempt < max_retries:
                    logger.warning(
                        f"⚠️  {func.__name__} failed: {str(e)[:100]} "
                        f"— retry in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    wait_time = min(wait_time * 2 + (wait_time * 0.1), backoff_max)
                else:
                    logger.error(
                        f"❌ {func.__name__} failed after {max_retries + 1} attempts"
                    )

            if fallback is not None:
                logger.warning(f"   Using fallback for {func.__name__}")
                return fallback
            else:
                raise APIError(f"{func.__name__} failed after {max_retries + 1} attempts: {last_error}")

        return wrapper

    return decorator


# Preset configurations for common APIs
ALPACA_RETRY = with_retry(max_retries=3, timeout=15, backoff_base=0.5)
GROQ_RETRY = with_retry(max_retries=2, timeout=20, backoff_base=0.5)  # Slower LLM
TELEGRAM_RETRY = with_retry(max_retries=2, timeout=10, backoff_base=0.3)
YFINANCE_RETRY = with_retry(max_retries=3, timeout=10, backoff_base=0.5)
BROKER_RETRY = with_retry(max_retries=3, timeout=10, backoff_base=1.0)
NEWS_RETRY = with_retry(max_retries=2, timeout=8, backoff_base=0.5)


# Helper function for testing
async def test_resilience():
    """Test the retry mechanism"""
    import random

    call_count = 0

    @with_retry(max_retries=2, timeout=5)
    async def flaky_api():
        nonlocal call_count
        call_count += 1

        if call_count < 3:
            logger.info(f"Simulating failure #{call_count}")
            await asyncio.sleep(0.1)
            raise Exception(f"Simulated failure #{call_count}")

        logger.info("Success!")
        return {"result": "ok"}

    try:
        result = await flaky_api()
        logger.info(f"Final result: {result}")
    except APIError as e:
        logger.error(f"Failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_resilience())
