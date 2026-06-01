"""
News Intelligence Module
========================

Real-time news analysis for trading decisions.

Features:
1. News aggregation from multiple sources
2. Sentiment analysis (bullish/bearish/neutral)
3. Impact scoring (how much news might move price)
4. Topic classification (earnings, M&A, regulation, etc)
5. Catalyst detection (events that drive price)
6. Pre-market news scanner
7. Breaking news alerts
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
class NewsArticle:
    """A single news article."""
    title: str
    summary: str
    source: str
    url: str
    published_at: str
    tickers_mentioned: list[str] = field(default_factory=list)
    sentiment: str = "neutral"  # bullish, bearish, neutral
    sentiment_score: float = 0  # -1 to +1
    impact_score: float = 0  # 0-10
    topics: list[str] = field(default_factory=list)
    is_breaking: bool = False
    is_catalyst: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT KEYWORDS
# ─────────────────────────────────────────────────────────────────────────────

BULLISH_KEYWORDS = {
    "beats", "surges", "rallies", "soars", "jumps", "climbs", "rises", "gains",
    "outperforms", "exceeds", "strong", "record", "high", "growth", "expansion",
    "breakthrough", "innovation", "upgrade", "raises", "boost", "positive", "win",
    "profit", "earnings beat", "buy rating", "outperform", "bullish"
}

BEARISH_KEYWORDS = {
    "misses", "falls", "drops", "plunges", "tumbles", "slides", "declines",
    "underperforms", "warns", "weak", "low", "decline", "contraction",
    "lawsuit", "investigation", "downgrade", "cuts", "negative", "loss",
    "earnings miss", "sell rating", "underperform", "bearish", "concern"
}

HIGH_IMPACT_TOPICS = {
    "earnings": ["earnings", "quarterly", "Q1", "Q2", "Q3", "Q4", "revenue", "EPS"],
    "merger": ["merger", "acquisition", "M&A", "takeover", "buyout"],
    "regulation": ["regulation", "SEC", "FDA", "approval", "investigation", "fine"],
    "leadership": ["CEO", "CFO", "executive", "resignation", "appointment"],
    "product": ["launch", "product", "patent", "lawsuit"],
    "macro": ["fed", "rates", "inflation", "GDP", "unemployment"],
}


# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_sentiment(text: str) -> tuple[str, float]:
    """
    Analyze sentiment of news text.

    Returns: (sentiment_label, score)
    where score is -1 (bearish) to +1 (bullish)
    """
    if not text:
        return "neutral", 0

    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)

    bullish_count = sum(1 for w in words if w in BULLISH_KEYWORDS)
    bearish_count = sum(1 for w in words if w in BEARISH_KEYWORDS)

    total = bullish_count + bearish_count
    if total == 0:
        return "neutral", 0

    score = (bullish_count - bearish_count) / max(total, 1)

    # Normalize to -1 to 1
    score = max(-1, min(1, score))

    if score > 0.3:
        sentiment = "bullish"
    elif score < -0.3:
        sentiment = "bearish"
    else:
        sentiment = "neutral"

    return sentiment, score


def classify_topics(text: str) -> list[str]:
    """Classify article topics."""
    text_lower = text.lower()
    topics = []

    for topic, keywords in HIGH_IMPACT_TOPICS.items():
        if any(kw.lower() in text_lower for kw in keywords):
            topics.append(topic)

    return topics


def calculate_impact_score(article: NewsArticle) -> float:
    """
    Calculate how much this news might move price.

    Higher scores = bigger expected move.
    """
    score = 0

    # Strong sentiment
    score += abs(article.sentiment_score) * 3

    # High-impact topics
    high_impact = {"earnings", "merger", "regulation", "leadership"}
    matching_topics = set(article.topics) & high_impact
    score += len(matching_topics) * 2

    # Breaking news
    if article.is_breaking:
        score += 3

    # Multiple tickers mentioned (broad market impact)
    if len(article.tickers_mentioned) > 3:
        score += 1

    return min(10, score)


# ─────────────────────────────────────────────────────────────────────────────
# TICKER EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_tickers(text: str, known_tickers: Optional[set] = None) -> list[str]:
    """
    Extract ticker symbols from text.

    Looks for:
    - $TICKER format
    - All-caps 1-5 char words that match known tickers
    """
    if not text:
        return []

    tickers = set()

    # $TICKER format
    dollar_pattern = re.findall(r'\$([A-Z]{1,5})\b', text)
    tickers.update(dollar_pattern)

    # All-caps words (filter against known tickers if provided)
    caps_pattern = re.findall(r'\b([A-Z]{2,5})\b', text)
    if known_tickers:
        tickers.update(t for t in caps_pattern if t in known_tickers)
    else:
        # Common stop words to exclude
        stopwords = {"USA", "CEO", "CFO", "GDP", "API", "SEC", "FDA", "NYSE", "ETF",
                    "IPO", "IRS", "FBI", "USD", "EUR", "AI", "ML", "VR"}
        tickers.update(t for t in caps_pattern if t not in stopwords)

    return sorted(list(tickers))


# ─────────────────────────────────────────────────────────────────────────────
# NEWS FETCHING
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_yahoo_news(ticker: str, max_articles: int = 10) -> list[NewsArticle]:
    """
    Fetch news from Yahoo Finance for a specific ticker.
    """
    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        news_items = stock.news[:max_articles]

        articles = []
        for item in news_items:
            try:
                title = item.get("title", "")
                summary = item.get("summary", "") or title

                # Analyze sentiment
                sentiment, score = analyze_sentiment(f"{title}. {summary}")

                # Extract topics
                topics = classify_topics(f"{title}. {summary}")

                # Extract tickers
                tickers = extract_tickers(f"{title}. {summary}")
                if ticker not in tickers:
                    tickers.append(ticker)

                # Check if breaking (within last hour)
                pub_time = item.get("providerPublishTime", 0)
                is_breaking = False
                if pub_time:
                    age_hours = (datetime.now(timezone.utc).timestamp() - pub_time) / 3600
                    is_breaking = age_hours < 1

                article = NewsArticle(
                    title=title,
                    summary=summary,
                    source=item.get("publisher", "Yahoo"),
                    url=item.get("link", ""),
                    published_at=datetime.fromtimestamp(pub_time, timezone.utc).isoformat() if pub_time else "",
                    tickers_mentioned=tickers,
                    sentiment=sentiment,
                    sentiment_score=score,
                    topics=topics,
                    is_breaking=is_breaking,
                )
                article.impact_score = calculate_impact_score(article)
                article.is_catalyst = article.impact_score >= 5

                articles.append(article)

            except Exception as e:
                logger.debug(f"Failed to parse news item: {e}")

        return articles

    except Exception as e:
        logger.error(f"Yahoo news fetch failed for {ticker}: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# NEWS AGGREGATION FOR PORTFOLIO
# ─────────────────────────────────────────────────────────────────────────────

async def get_portfolio_news() -> dict:
    """
    Get aggregated news for all positions in portfolio.

    Returns prioritized news by impact.
    """
    try:
        import broker

        positions = await asyncio.to_thread(broker.get_positions)
        if not positions:
            return {"articles": [], "summary": "No positions"}

        all_articles = []
        for p in positions[:10]:  # Limit to 10 positions
            articles = await fetch_yahoo_news(p.get("ticker"), max_articles=5)
            all_articles.extend(articles)

        # Deduplicate by title
        seen_titles = set()
        unique_articles = []
        for art in all_articles:
            if art.title not in seen_titles:
                seen_titles.add(art.title)
                unique_articles.append(art)

        # Sort by impact score
        unique_articles.sort(key=lambda a: a.impact_score, reverse=True)

        # Count sentiment
        sentiments = Counter(a.sentiment for a in unique_articles)

        # Find catalysts
        catalysts = [a for a in unique_articles if a.is_catalyst]
        breaking = [a for a in unique_articles if a.is_breaking]

        return {
            "total_articles": len(unique_articles),
            "sentiment_breakdown": dict(sentiments),
            "top_articles": [
                {
                    "title": a.title,
                    "source": a.source,
                    "sentiment": a.sentiment,
                    "impact": a.impact_score,
                    "tickers": a.tickers_mentioned,
                    "topics": a.topics,
                    "is_breaking": a.is_breaking,
                }
                for a in unique_articles[:10]
            ],
            "catalysts": [
                {
                    "title": a.title,
                    "tickers": a.tickers_mentioned,
                    "sentiment": a.sentiment,
                    "impact": a.impact_score,
                }
                for a in catalysts[:5]
            ],
            "breaking_news": [
                {
                    "title": a.title,
                    "tickers": a.tickers_mentioned,
                    "sentiment": a.sentiment,
                }
                for a in breaking[:5]
            ],
        }

    except Exception as e:
        logger.error(f"Portfolio news fetch failed: {e}")
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# CATALYST DETECTION
# ─────────────────────────────────────────────────────────────────────────────

async def detect_catalysts(ticker: str) -> dict:
    """
    Detect news catalysts that might drive price movement.

    Returns alerts for high-impact news.
    """
    try:
        articles = await fetch_yahoo_news(ticker, max_articles=20)

        # Filter for catalysts
        catalysts = [a for a in articles if a.is_catalyst]
        breaking = [a for a in articles if a.is_breaking]

        # Aggregate sentiment from recent news (last 24h)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        recent_articles = []
        for a in articles:
            try:
                pub = datetime.fromisoformat(a.published_at.replace("Z", "+00:00"))
                if pub > cutoff:
                    recent_articles.append(a)
            except Exception:
                pass

        avg_sentiment = (sum(a.sentiment_score for a in recent_articles) / len(recent_articles)) if recent_articles else 0

        return {
            "ticker": ticker,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "has_catalysts": len(catalysts) > 0,
            "has_breaking_news": len(breaking) > 0,
            "catalysts_count": len(catalysts),
            "breaking_count": len(breaking),
            "avg_sentiment_24h": avg_sentiment,
            "sentiment_label": (
                "🟢 Bullish news flow" if avg_sentiment > 0.3 else
                "🔴 Bearish news flow" if avg_sentiment < -0.3 else
                "🟡 Mixed/Neutral news"
            ),
            "top_catalysts": [
                {
                    "title": a.title,
                    "sentiment": a.sentiment,
                    "impact": a.impact_score,
                    "topics": a.topics,
                }
                for a in catalysts[:3]
            ],
            "recommendation": (
                "⚠️ Hot news - monitor closely" if catalysts and breaking else
                "📰 Active news flow" if catalysts else
                "🟢 Quiet news environment"
            ),
        }

    except Exception as e:
        logger.error(f"Catalyst detection failed for {ticker}: {e}")
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# PRE-MARKET NEWS SCANNER
# ─────────────────────────────────────────────────────────────────────────────

async def scan_pre_market_news(watchlist: list[str]) -> dict:
    """
    Scan watchlist for pre-market news catalysts.

    Returns tickers with significant overnight news.
    """
    catalysts_found = []

    for ticker in watchlist[:20]:  # Limit to 20 to avoid rate limits
        try:
            result = await detect_catalysts(ticker)
            if result.get("has_catalysts") or result.get("has_breaking_news"):
                catalysts_found.append({
                    "ticker": ticker,
                    "sentiment": result.get("sentiment_label"),
                    "catalysts_count": result.get("catalysts_count", 0),
                    "breaking_count": result.get("breaking_count", 0),
                    "recommendation": result.get("recommendation"),
                })
        except Exception:
            continue

    # Sort by sentiment strength
    catalysts_found.sort(
        key=lambda x: x.get("catalysts_count", 0) + x.get("breaking_count", 0),
        reverse=True
    )

    return {
        "scanned": len(watchlist),
        "catalysts_found": len(catalysts_found),
        "top_movers": catalysts_found[:10],
    }
