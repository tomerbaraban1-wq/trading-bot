import re
import time
import threading
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
_cache_lock = threading.Lock()   # guards _news_cache / _cache_time against concurrent threads
_NEWS_CACHE_MAX = 100   # max entries — evict oldest to prevent memory growth


def _cache_get(key: str) -> list[str] | None:
    """Thread-safe cache read. Returns None if missing or stale."""
    with _cache_lock:
        if key in _news_cache and (time.time() - _cache_time.get(key, 0)) < settings.NEWS_CACHE_TTL:
            return list(_news_cache[key])
    return None


def _cache_set(key: str, headlines: list[str]) -> None:
    """Thread-safe cache write with LRU eviction."""
    with _cache_lock:
        if len(_news_cache) >= _NEWS_CACHE_MAX:
            try:
                oldest = min(_cache_time.items(), key=lambda x: x[1])
                del _news_cache[oldest[0]]
                del _cache_time[oldest[0]]
            except (ValueError, KeyError):
                pass
        _news_cache[key] = headlines
        _cache_time[key] = time.time()


def get_headlines(ticker: str, limit: int = 5) -> list[str]:
    """Get top N news headlines for a ticker from RSS feeds."""
    cache_key = ticker.upper()
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached[:limit]

    all_headlines = []
    # Use word-boundary regex so ticker "ON" doesn't match every headline
    # containing the word "on", "F" doesn't match "of", etc.
    pattern = re.compile(rf"\b{re.escape(ticker)}\b", re.IGNORECASE)

    for source_name, feed_url in RSS_FEEDS:
        try:
            resp = requests.get(
                feed_url,
                timeout=5,   # reduced from 10s — 8 feeds × 5s = 40s max
                headers={"User-Agent": "Mozilla/5.0 TradingBot/1.0"},
            )
            if resp.status_code != 200:
                continue
            items = _parse_rss(resp.content, source_name)
            for item in items:
                text = item["headline"] + " " + item.get("summary", "")
                if pattern.search(text):
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

    _cache_set(cache_key, unique)
    return unique[:limit]


def get_general_headlines(limit: int = 10) -> list[str]:
    """Get general market headlines (no ticker filter)."""
    cache_key = "__GENERAL__"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached[:limit]

    all_headlines = []
    for source_name, feed_url in RSS_FEEDS:
        try:
            resp = requests.get(
                feed_url,
                timeout=5,   # reduced from 10s
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

    _cache_set(cache_key, unique)
    return unique[:limit]


def _parse_rss(content: bytes, source_name: str) -> list[dict]:
    """Parse RSS 2.0 and Atom feed formats."""
    items = []
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return items

    # RSS 2.0: <item> elements with <title> + <description>
    for item_el in root.iter("item"):
        title = _get_text(item_el, "title") or ""
        summary = _get_text(item_el, "description") or ""
        summary = _strip_html(summary)
        if len(summary) > 300:
            summary = summary[:300] + "..."
        if title:
            items.append({
                "headline": title,
                "summary": summary if summary else title,
                "source": source_name,
            })

    # Atom: <entry> elements with <title> + <summary> (Reuters, Seeking Alpha, etc.)
    if not items:
        for entry_el in root.iter("{http://www.w3.org/2005/Atom}entry"):
            title = _get_text(entry_el, "{http://www.w3.org/2005/Atom}title") or ""
            summary = _get_text(entry_el, "{http://www.w3.org/2005/Atom}summary") or ""
            summary = _strip_html(summary)
            if len(summary) > 300:
                summary = summary[:300] + "..."
            if title:
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
