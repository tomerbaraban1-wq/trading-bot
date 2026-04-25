import json
import time
import threading
import logging
from openai import OpenAI
from config import settings
from models import SentimentResult
from news_service import get_headlines

logger = logging.getLogger(__name__)

_client = None
_sentiment_cache: dict = {}
_cache_lock = threading.Lock()  # guards _sentiment_cache across threads
CACHE_TTL = 300  # 5 minutes


# ──────────────────────────────────────────────────────────────────────────────
# Keyword-based sentiment fallback (used when Groq is unavailable)
# ──────────────────────────────────────────────────────────────────────────────
_BULLISH_KEYWORDS = {
    "beat", "beats", "surge", "surges", "soar", "soars", "rally", "rallies",
    "upgrade", "upgraded", "outperform", "buy rating", "record high",
    "strong earnings", "exceed", "exceeds", "exceeded", "growth", "profit",
    "profits", "bullish", "breakthrough", "win", "wins", "won", "approval",
    "approved", "expand", "expansion", "raise guidance", "raised guidance",
    "boost", "boosts", "all-time high", "milestone", "partnership", "deal",
}
_BEARISH_KEYWORDS = {
    "miss", "misses", "missed", "plunge", "plunges", "crash", "crashes",
    "downgrade", "downgraded", "underperform", "sell rating", "lawsuit",
    "fraud", "investigation", "probe", "fine", "fined", "penalty",
    "bankruptcy", "loss", "losses", "decline", "declines", "drop", "drops",
    "fall", "falls", "tumble", "tumbles", "bearish", "warning", "warns",
    "cut guidance", "lowered guidance", "layoff", "layoffs", "scandal",
    "recall", "recalled", "sec", "doj", "antitrust", "weak",
}


def _keyword_sentiment(headlines: list[str]) -> tuple[int, str]:
    """Naive keyword-based sentiment scorer (1-10).

    Used when the Groq API is unavailable so the bot still uses the news
    instead of defaulting to a fully neutral 5.
    """
    if not headlines:
        return 5, "no headlines"

    text = " ".join(headlines).lower()
    bull = sum(1 for kw in _BULLISH_KEYWORDS if kw in text)
    bear = sum(1 for kw in _BEARISH_KEYWORDS if kw in text)

    if bull == 0 and bear == 0:
        return 5, "no sentiment keywords matched"

    # Net = bull - bear, scaled into 1..10 around neutral 5
    net = bull - bear
    score = 5 + max(-4, min(4, net))
    return int(score), f"bull_kw={bull}, bear_kw={bear}, net={net:+d}"


def _get_client() -> OpenAI:
    global _client
    if _client is None and settings.GROQ_API_KEY:
        _client = OpenAI(
            api_key=settings.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
    return _client


def score_sentiment(ticker: str) -> SentimentResult:
    """
    Score news sentiment for a ticker (1-10).
    MANDATORY before any buy order.
    """
    # Check cache
    now = time.time()
    with _cache_lock:
        cached = _sentiment_cache.get(ticker)
    if cached is not None and now - cached.timestamp < CACHE_TTL:
        return cached

    # Fetch headlines
    headlines = get_headlines(ticker, limit=5)

    if not headlines:
        # No news = neutral (score 5)
        result = SentimentResult(
            ticker=ticker,
            score=5,
            headlines=[],
            reasoning="No recent news found - defaulting to neutral",
            timestamp=now,
        )
        with _cache_lock:
            _sentiment_cache[ticker] = result
        return result

    # Score with LLM
    client = _get_client()
    if not client:
        # No API key configured — fall back to keyword-based sentiment so the
        # bot still ANALYSES the news (not just defaults to neutral).
        score, reasoning = _keyword_sentiment(headlines)
        logger.info(
            f"[SENTIMENT] {ticker}: GROQ unavailable — keyword fallback score={score}/10 ({reasoning})"
        )
        result = SentimentResult(
            ticker=ticker,
            score=score,
            headlines=headlines,
            reasoning=f"[keyword fallback] {reasoning}",
            timestamp=now,
        )
        with _cache_lock:
            _sentiment_cache[ticker] = result
        return result

    headlines_text = "\n".join(f"- {h}" for h in headlines)

    system_prompt = (
        "You are a financial sentiment analyst. Given news headlines about a stock, "
        "rate the overall market sentiment on a scale of 1-10:\n"
        "1-2: Critically bearish (lawsuits, fraud, bankruptcy, terrible earnings)\n"
        "3-4: Bearish (missed earnings, downgrades, sector decline)\n"
        "5-6: Neutral (mixed signals, no strong direction)\n"
        "7-8: Bullish (upgrades, good earnings, positive outlook)\n"
        "9-10: Very bullish (breakthrough news, major contracts, explosive growth)\n\n"
        'Respond with ONLY a JSON object: {"score": N, "reasoning": "brief explanation"}'
    )

    user_prompt = f"Stock: {ticker}\n\nRecent headlines:\n{headlines_text}"

    try:
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=200,
            temperature=0.3,
        )
        raw = response.choices[0].message.content.strip()

        # Parse JSON response
        # Handle potential markdown wrapping
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        data = json.loads(raw)
        score = max(1, min(10, int(data.get("score", 5))))
        reasoning = data.get("reasoning", "No reasoning provided")

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(f"Failed to parse sentiment response for {ticker}: {e}")
        score = 5
        reasoning = "Failed to parse LLM response - defaulting to neutral"
    except Exception as e:
        # Don't let a transient LLM/network error block trading — fall back
        # to keyword-based sentiment so the bot still uses the actual news.
        logger.error(f"Sentiment scoring failed for {ticker}: {e} — using keyword fallback")
        score, reasoning = _keyword_sentiment(headlines)
        reasoning = f"[keyword fallback after LLM error] {reasoning}"

    result = SentimentResult(
        ticker=ticker,
        score=score,
        headlines=headlines,
        reasoning=reasoning,
        timestamp=now,
    )
    with _cache_lock:
        _sentiment_cache[ticker] = result

    logger.info(f"Sentiment for {ticker}: {score}/10 - {reasoning}")
    return result


def check_emergency_sentiment(ticker: str) -> bool:
    """
    Check if sentiment is critically bearish for an open position.
    Returns True if emergency exit should be triggered.
    """
    try:
        # Force fresh check (bypass cache)
        with _cache_lock:
            _sentiment_cache.pop(ticker, None)
        result = score_sentiment(ticker)
        if result.score <= settings.SENTIMENT_EMERGENCY_SCORE:
            logger.warning(
                f"EMERGENCY SENTIMENT for {ticker}: {result.score}/10 - {result.reasoning}"
            )
            return True
        return False
    except Exception as e:
        logger.error(f"Emergency sentiment check failed for {ticker}: {e}")
        return False  # Don't exit on API failure
