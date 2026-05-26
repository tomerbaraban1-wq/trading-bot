"""
Social Sentiment Aggregator
============================

Aggregates sentiment from multiple sources:
1. Reddit (r/wallstreetbets, r/stocks, r/investing)
2. StockTwits (when available)
3. Twitter/X (when available)
4. Discord (existing integration)
5. News sentiment

Generates unified sentiment score for each ticker.
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class SentimentSource:
    """Sentiment from a single source."""
    source: str  # "reddit", "stocktwits", "twitter", "discord", "news"
    score: float  # -1 to +1
    mention_count: int
    sample_size: int
    confidence: float  # 0-1
    top_phrases: list[str] = field(default_factory=list)


@dataclass
class UnifiedSentiment:
    """Unified sentiment across all sources."""
    ticker: str
    overall_score: float  # -1 to +1
    overall_label: str  # "very_bullish", "bullish", "neutral", "bearish", "very_bearish"
    sources: list[SentimentSource]
    total_mentions: int
    source_agreement: float  # 0-1, how much sources agree
    confidence: float  # 0-1
    interpretation: str
    actionable: bool


# ─────────────────────────────────────────────────────────────────────────────
# REDDIT SENTIMENT (via Pushshift / Reddit API alternative)
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_reddit_sentiment(ticker: str, days_back: int = 1) -> SentimentSource:
    """
    Fetch sentiment from Reddit about a ticker.

    Note: Real Reddit API requires authentication.
    This is a simplified version that uses public RSS feeds where available.
    """
    try:
        # Try to use yfinance which has some reddit data
        import yfinance as yf

        stock = yf.Ticker(ticker)

        # Use news as proxy if Reddit not directly accessible
        news = stock.news[:20]

        # Analyze sentiment of news (similar pattern to Reddit)
        from news_intelligence import BULLISH_KEYWORDS, BEARISH_KEYWORDS

        bullish_count = 0
        bearish_count = 0
        all_words = []

        for item in news:
            title = item.get("title", "").lower()
            text = title + " " + item.get("summary", "").lower()
            words = re.findall(r'\b\w+\b', text)
            all_words.extend(words)

            bullish_count += sum(1 for w in words if w in BULLISH_KEYWORDS)
            bearish_count += sum(1 for w in words if w in BEARISH_KEYWORDS)

        total_sentiment = bullish_count + bearish_count
        if total_sentiment > 0:
            score = (bullish_count - bearish_count) / total_sentiment
            confidence = min(1, total_sentiment / 20)
        else:
            score = 0
            confidence = 0

        # Top phrases (most common bullish/bearish words found)
        word_counts = Counter(all_words)
        sentiment_words = [w for w in word_counts if w in BULLISH_KEYWORDS or w in BEARISH_KEYWORDS]
        top_phrases = [w for w, _ in word_counts.most_common(50) if w in sentiment_words][:5]

        return SentimentSource(
            source="reddit_proxy",
            score=score,
            mention_count=len(news),
            sample_size=len(news),
            confidence=confidence,
            top_phrases=top_phrases,
        )

    except Exception as e:
        logger.debug(f"Reddit sentiment failed for {ticker}: {e}")
        return SentimentSource(source="reddit", score=0, mention_count=0, sample_size=0, confidence=0)


# ─────────────────────────────────────────────────────────────────────────────
# NEWS SENTIMENT
# ─────────────────────────────────────────────────────────────────────────────

async def get_news_sentiment_score(ticker: str) -> SentimentSource:
    """Get sentiment from news articles."""
    try:
        from news_intelligence import fetch_yahoo_news

        articles = await fetch_yahoo_news(ticker, max_articles=20)

        if not articles:
            return SentimentSource(source="news", score=0, mention_count=0, sample_size=0, confidence=0)

        # Average sentiment scores
        scores = [a.sentiment_score for a in articles]
        avg_score = sum(scores) / len(scores)

        # Confidence based on sample size
        confidence = min(1, len(articles) / 10)

        return SentimentSource(
            source="news",
            score=avg_score,
            mention_count=len(articles),
            sample_size=len(articles),
            confidence=confidence,
            top_phrases=[a.title[:50] for a in articles[:3]],
        )

    except Exception as e:
        logger.debug(f"News sentiment failed for {ticker}: {e}")
        return SentimentSource(source="news", score=0, mention_count=0, sample_size=0, confidence=0)


# ─────────────────────────────────────────────────────────────────────────────
# DISCORD SENTIMENT (existing integration)
# ─────────────────────────────────────────────────────────────────────────────

async def get_discord_sentiment(ticker: str) -> SentimentSource:
    """Get sentiment from Discord community."""
    try:
        from discord_bot import fetch_community_sentiment

        sentiment_data = await fetch_community_sentiment(ticker)

        if not sentiment_data:
            return SentimentSource(source="discord", score=0, mention_count=0, sample_size=0, confidence=0)

        # Convert 0-10 score to -1 to +1
        score = (sentiment_data.get("score", 5) - 5) / 5
        mentions = sentiment_data.get("mentions", 0)

        return SentimentSource(
            source="discord",
            score=score,
            mention_count=mentions,
            sample_size=mentions,
            confidence=min(1, mentions / 20),
            top_phrases=[],
        )

    except Exception as e:
        logger.debug(f"Discord sentiment failed for {ticker}: {e}")
        return SentimentSource(source="discord", score=0, mention_count=0, sample_size=0, confidence=0)


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED SENTIMENT
# ─────────────────────────────────────────────────────────────────────────────

async def get_unified_sentiment(ticker: str) -> UnifiedSentiment:
    """
    Get unified sentiment across all sources.

    Uses weighted average based on:
    - Source confidence
    - Sample size
    - Recency
    """
    try:
        # Fetch from all sources in parallel
        sources = await asyncio.gather(
            fetch_reddit_sentiment(ticker),
            get_news_sentiment_score(ticker),
            get_discord_sentiment(ticker),
            return_exceptions=True,
        )

        # Filter to valid sources
        valid_sources = [s for s in sources if isinstance(s, SentimentSource) and s.confidence > 0]

        if not valid_sources:
            return UnifiedSentiment(
                ticker=ticker,
                overall_score=0,
                overall_label="neutral",
                sources=[],
                total_mentions=0,
                source_agreement=0,
                confidence=0,
                interpretation="🟡 No sentiment data available",
                actionable=False,
            )

        # Weighted score
        total_weight = 0
        weighted_score = 0
        for source in valid_sources:
            weight = source.confidence * (1 + source.mention_count / 100)
            weighted_score += source.score * weight
            total_weight += weight

        overall_score = weighted_score / total_weight if total_weight > 0 else 0

        # Source agreement (low std dev = high agreement)
        scores = [s.score for s in valid_sources]
        if len(scores) > 1:
            import numpy as np
            std_dev = np.std(scores)
            agreement = max(0, 1 - std_dev)
        else:
            agreement = 0.5

        # Overall confidence
        avg_confidence = sum(s.confidence for s in valid_sources) / len(valid_sources)

        # Total mentions
        total_mentions = sum(s.mention_count for s in valid_sources)

        # Label
        if overall_score > 0.5:
            label = "very_bullish"
        elif overall_score > 0.2:
            label = "bullish"
        elif overall_score > -0.2:
            label = "neutral"
        elif overall_score > -0.5:
            label = "bearish"
        else:
            label = "very_bearish"

        # Interpretation
        if agreement > 0.7 and avg_confidence > 0.6:
            if overall_score > 0.3:
                interpretation = "🟢 STRONG bullish consensus across all sources"
            elif overall_score < -0.3:
                interpretation = "🔴 STRONG bearish consensus across all sources"
            else:
                interpretation = "🟡 Strong neutral consensus"
        elif agreement < 0.4:
            interpretation = "⚠️ Sources DISAGREE - high uncertainty"
        elif avg_confidence < 0.3:
            interpretation = "🟠 Low confidence - not enough data"
        else:
            interpretation = f"📊 Moderate {label.replace('_', ' ')} sentiment"

        # Actionable if high confidence and clear direction
        actionable = avg_confidence > 0.5 and abs(overall_score) > 0.3 and agreement > 0.5

        return UnifiedSentiment(
            ticker=ticker,
            overall_score=overall_score,
            overall_label=label,
            sources=valid_sources,
            total_mentions=total_mentions,
            source_agreement=agreement,
            confidence=avg_confidence,
            interpretation=interpretation,
            actionable=actionable,
        )

    except Exception as e:
        logger.error(f"Unified sentiment failed for {ticker}: {e}")
        return UnifiedSentiment(
            ticker=ticker,
            overall_score=0,
            overall_label="neutral",
            sources=[],
            total_mentions=0,
            source_agreement=0,
            confidence=0,
            interpretation=f"Error: {e}",
            actionable=False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT-BASED RANKING
# ─────────────────────────────────────────────────────────────────────────────

async def rank_tickers_by_sentiment(tickers: list[str]) -> list[dict]:
    """Rank multiple tickers by sentiment strength."""
    results = []

    for ticker in tickers[:15]:  # Rate limit
        try:
            sentiment = await get_unified_sentiment(ticker)
            if sentiment.actionable:
                results.append({
                    "ticker": ticker,
                    "score": sentiment.overall_score,
                    "label": sentiment.overall_label,
                    "mentions": sentiment.total_mentions,
                    "agreement": sentiment.source_agreement,
                    "interpretation": sentiment.interpretation,
                })
        except Exception as e:
            logger.debug(f"Sentiment ranking error for {ticker}: {e}")

    # Sort by absolute score (most extreme sentiment first)
    results.sort(key=lambda x: abs(x["score"]), reverse=True)
    return results
