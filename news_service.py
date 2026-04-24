import re
import time
import logging
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
from config import settings

logger = logging.getLogger(__name__)

RSS_FEEDS = [
    ("MarketWatch",   "https://feeds.marketwatch.com/marketwatch/topstories/"),
    ("CNBC",          "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
    ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
    ("Reuters",       "https://feeds.reuters.com/reuters/businessNews"),
    ("AP Business",   "https://rsshub.app/apnews/topics/business-news"),
    ("Seeking Alpha", "https://seekingalpha.com/market_currents.xml"),
    ("Investopedia",  "https://www.investopedia.com/feedbuilder/feed/getfeed?feedName=rss_headline"),
    ("Google News",   "https://news.google.com/rss/search?q=stock+market&hl=en-US&gl=US&ceid=US:en"),
]

_news_cache: dict = {}
_cache_time: dict = {}


def get_headlines(ticker: str, limit: int = 5) -> list[str]:
    """Get top N news headlines for a ticker from RSS feeds."""
    cache_key = ticker.upper()
    now = time.time()
    if cache_key in _news_cache and (now - _cache_time.get(cache_key, 0)) < settings.NEWS_CACHE_TTL:
        return _news_cache[cache_key][:limit]

    all_headlines = []
    search_terms = [ticker.upper(), ticker.lower()]

    for source_name, feed_url in RSS_FEEDS:
        try:
            resp = requests.get(
                feed_url,
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0 TradingBot/1.0"},
            )
            if resp.status_code != 200:
                continue
            items = _parse_rss(resp.content, source_name)
            for item in items:
                text = (item["headline"] + " " + item.get("summary", "")).lower()
                if any(term.lower() in text for term in search_terms):
                    all_headlines.append(item["headline"])
        except requests.exceptions.RequestException:
            continue

    # Deduplication
    seen = set()
    unique = []
    for h in all_headlines:
        key = h.strip().lower()[:50]
        if key not in seen:
            seen.add(key)
            unique.append(h)

    _news_cache[cache_key] = unique
    _cache_time[cache_key] = now

    return unique[:limit]


def get_general_headlines(limit: int = 10) -> list[str]:
    """Get general market headlines (no ticker filter)."""
    cache_key = "__GENERAL__"
    now = time.time()
    if cache_key in _news_cache and (now - _cache_time.get(cache_key, 0)) < settings.NEWS_CACHE_TTL:
        return _news_cache[cache_key][:limit]

    all_headlines = []
    for source_name, feed_url in RSS_FEEDS:
        try:
            resp = requests.get(
                feed_url,
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0 TradingBot/1.0"},
            )
            if resp.status_code != 200:
                continue
            items = _parse_rss(resp.content, source_name)
            all_headlines.extend([item["headline"] for item in items[:5]])
        except requests.exceptions.RequestException:
            continue

    # Dedup
    seen = set()
    unique = []
    for h in all_headlines:
        key = h.strip().lower()[:50]
        if key not in seen:
            seen.add(key)
            unique.append(h)

    _news_cache[cache_key] = unique
    _cache_time[cache_key] = now
    return unique[:limit]


def _parse_rss(content: bytes, source_name: str) -> list[dict]:
    items = []
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return items

    for item_el in root.iter("item"):
        title = _get_text(item_el, "title") or ""
        summary = _get_text(item_el, "description") or ""
        summary = _strip_html(summary)
        if len(summary) > 300:
            summary = summary[:300] + "..."

        items.append({
            "headline": title,
            "summary": summary if summary else title,
            "source": source_name,
        })
    return items


def _get_text(element, tag: str) -> str:
    el = element.find(tag)
    if el is not None and el.text:
        return el.text.strip()
    return ""


def _strip_html(text: str) -> str:
    clean = re.sub(r"<[^>]+>", "", text)
    clean = clean.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    clean = clean.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    return clean.strip()
