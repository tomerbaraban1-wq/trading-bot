"""
Hebrew Translation Service
===========================

Automatic English-to-Hebrew translation for Telegram messages.

Features:
1. Google Translate API (free unofficial endpoint)
2. Smart caching (LRU cache + persistent DB cache)
3. Skip detection (tickers, numbers, code, emojis)
4. Batch translation for efficiency
5. Fallback to multiple providers
6. Preserve HTML formatting
7. Auto-detect language to skip already-Hebrew text
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from typing import Optional
from functools import lru_cache

import aiohttp

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Enable translation by default
TRANSLATION_ENABLED = os.getenv("TRANSLATION_ENABLED", "true").lower() == "true"
TARGET_LANGUAGE = "iw"  # Hebrew (also accepted: 'he')
SOURCE_LANGUAGE = "en"

# Cache settings
CACHE_TTL_SECONDS = 86400 * 30  # 30 days
MAX_CACHE_ENTRIES = 10000

# ─────────────────────────────────────────────────────────────────────────────
# SMART DETECTION - What NOT to translate
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that should NOT be translated
PRESERVE_PATTERNS = [
    re.compile(r'\$[A-Z]{1,5}(-[A-Z])?'),       # Stock tickers ($AAPL, $TSLA)
    re.compile(r'\b[A-Z]{2,5}\b'),               # All-caps tickers/acronyms (AAPL, NASDAQ)
    re.compile(r'\b\d+(\.\d+)?[%$]?\b'),         # Numbers, percentages, dollar amounts
    re.compile(r'<[^>]+>'),                      # HTML tags
    re.compile(r'<code>.*?</code>', re.DOTALL),  # Code blocks
    re.compile(r'<pre>.*?</pre>', re.DOTALL),    # Pre blocks
    re.compile(r'https?://\S+'),                 # URLs
    re.compile(r'/\w+'),                         # Commands
    re.compile(r'@\w+'),                         # Mentions
    re.compile(r'#\w+'),                         # Hashtags
    re.compile(r'━+'),                           # Separators
]


def is_hebrew(text: str) -> bool:
    """Check if text already contains Hebrew characters."""
    hebrew_chars = sum(1 for c in text if '֐' <= c <= '׿')
    total_alpha = sum(1 for c in text if c.isalpha())

    if total_alpha == 0:
        return False

    # If more than 30% Hebrew, consider it Hebrew
    return (hebrew_chars / total_alpha) > 0.3


def needs_translation(text: str) -> bool:
    """
    Determine if text needs translation.

    Skip if:
    - Already Hebrew
    - Only emojis/symbols
    - Only numbers/tickers
    - Empty
    """
    if not text or not text.strip():
        return False

    # Skip if already Hebrew
    if is_hebrew(text):
        return False

    # Check if there's actual English text to translate
    # Remove all preserve patterns and check what's left
    cleaned = text
    for pattern in PRESERVE_PATTERNS:
        cleaned = pattern.sub('', cleaned)

    # Remove emojis
    cleaned = re.sub(r'[\U0001F300-\U0001F9FF]', '', cleaned)
    cleaned = re.sub(r'[☀-➿]', '', cleaned)

    # Check if there are English words left
    english_words = re.findall(r'[a-zA-Z]{2,}', cleaned)
    return len(english_words) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# CACHING SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

class TranslationCache:
    """LRU + persistent cache for translations."""

    def __init__(self):
        self.memory_cache: dict = {}
        self.access_times: dict = {}
        self._db_initialized = False

    def _ensure_db(self):
        """Initialize translation cache table."""
        if self._db_initialized:
            return
        try:
            import database
            conn = database.get_connection()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS translation_cache (
                    text_hash TEXT PRIMARY KEY,
                    source_text TEXT,
                    translated_text TEXT,
                    target_lang TEXT,
                    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    hit_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_translation_cached_at
                ON translation_cache(cached_at)
            """)
            conn.commit()
            self._db_initialized = True
        except Exception as e:
            logger.debug(f"Translation cache DB init failed: {e}")

    def _hash(self, text: str, target_lang: str) -> str:
        """Create cache key from text + lang."""
        h = hashlib.sha256(f"{text}|{target_lang}".encode()).hexdigest()
        return h[:32]

    def get(self, text: str, target_lang: str = TARGET_LANGUAGE) -> Optional[str]:
        """Get translation from cache."""
        key = self._hash(text, target_lang)

        # Check memory cache first
        if key in self.memory_cache:
            self.access_times[key] = time.time()
            return self.memory_cache[key]

        # Check DB cache
        try:
            self._ensure_db()
            import database
            conn = database.get_connection()
            row = conn.execute("""
                SELECT translated_text FROM translation_cache
                WHERE text_hash = ? AND target_lang = ?
            """, (key, target_lang)).fetchone()

            if row:
                translated = row[0]
                # Load into memory cache
                self.memory_cache[key] = translated
                self.access_times[key] = time.time()

                # Increment hit count
                conn.execute("""
                    UPDATE translation_cache
                    SET hit_count = hit_count + 1
                    WHERE text_hash = ?
                """, (key,))
                conn.commit()

                return translated
        except Exception as e:
            logger.debug(f"Cache get failed: {e}")

        return None

    def set(self, text: str, translation: str, target_lang: str = TARGET_LANGUAGE) -> None:
        """Store translation in cache."""
        key = self._hash(text, target_lang)

        # Add to memory cache
        self.memory_cache[key] = translation
        self.access_times[key] = time.time()

        # Evict if too large
        if len(self.memory_cache) > MAX_CACHE_ENTRIES:
            # Remove oldest 20%
            oldest = sorted(self.access_times.items(), key=lambda x: x[1])[:MAX_CACHE_ENTRIES // 5]
            for k, _ in oldest:
                self.memory_cache.pop(k, None)
                self.access_times.pop(k, None)

        # Persist to DB
        try:
            self._ensure_db()
            import database
            conn = database.get_connection()
            conn.execute("""
                INSERT OR REPLACE INTO translation_cache
                (text_hash, source_text, translated_text, target_lang)
                VALUES (?, ?, ?, ?)
            """, (key, text[:500], translation, target_lang))
            conn.commit()
        except Exception as e:
            logger.debug(f"Cache set failed: {e}")

    def get_stats(self) -> dict:
        """Get cache statistics."""
        try:
            self._ensure_db()
            import database
            conn = database.get_connection()
            row = conn.execute("""
                SELECT COUNT(*) as total, SUM(hit_count) as total_hits
                FROM translation_cache
            """).fetchone()
            return {
                "memory_entries": len(self.memory_cache),
                "db_entries": row[0] if row else 0,
                "total_hits": row[1] if row else 0,
            }
        except:
            return {"memory_entries": len(self.memory_cache), "db_entries": 0, "total_hits": 0}


_cache = TranslationCache()


# ─────────────────────────────────────────────────────────────────────────────
# TRANSLATION PROVIDERS
# ─────────────────────────────────────────────────────────────────────────────

async def _translate_google_free(text: str, target_lang: str = TARGET_LANGUAGE) -> Optional[str]:
    """
    Use Google Translate's free unofficial endpoint.
    Endpoint: translate.googleapis.com/translate_a/single

    This is the same API the Google Translate widget uses.
    """
    if not text or len(text) > 5000:
        return None

    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            "client": "gtx",
            "sl": SOURCE_LANGUAGE,
            "tl": target_lang,
            "dt": "t",
            "q": text,
        }

        timeout = aiohttp.ClientTimeout(total=8)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    return None

                data = await response.json()
                # Response: [[["translated text", "original", ...]], ...]
                if data and isinstance(data, list) and len(data) > 0:
                    translations = data[0]
                    # Combine all segments
                    result = "".join(seg[0] for seg in translations if seg and seg[0])
                    return result

        return None

    except asyncio.TimeoutError:
        logger.debug("Google translate timeout")
        return None
    except Exception as e:
        logger.debug(f"Google translate failed: {e}")
        return None


async def _translate_mymemory(text: str, target_lang: str = TARGET_LANGUAGE) -> Optional[str]:
    """
    Fallback: MyMemory free translation API.
    1000 requests/day per IP without auth.
    """
    if not text or len(text) > 500:
        return None

    try:
        # MyMemory uses 'he' for Hebrew, not 'iw'
        target = "he" if target_lang == "iw" else target_lang

        url = "https://api.mymemory.translated.net/get"
        params = {
            "q": text,
            "langpair": f"{SOURCE_LANGUAGE}|{target}",
        }

        timeout = aiohttp.ClientTimeout(total=8)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    return None

                data = await response.json()
                if data and "responseData" in data:
                    return data["responseData"].get("translatedText")

        return None

    except Exception as e:
        logger.debug(f"MyMemory translate failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CORE TRANSLATION
# ─────────────────────────────────────────────────────────────────────────────

async def translate_to_hebrew(text: str, target_lang: str = TARGET_LANGUAGE) -> str:
    """
    Translate English text to Hebrew.

    Smart logic:
    1. Check if translation needed (skip Hebrew/emoji-only)
    2. Check cache first
    3. Try Google Translate (free endpoint)
    4. Fallback to MyMemory
    5. Cache result
    6. Return original on failure

    Preserves:
    - HTML tags (<b>, <i>, etc.)
    - Tickers ($AAPL, AAPL)
    - Numbers and percentages
    - URLs
    - Commands (/help)
    - Emojis
    """
    if not TRANSLATION_ENABLED:
        return text

    if not text or not needs_translation(text):
        return text

    # Check cache
    cached = _cache.get(text, target_lang)
    if cached:
        return cached

    # Try Google Translate first
    translated = await _translate_google_free(text, target_lang)

    # Fallback to MyMemory
    if not translated:
        translated = await _translate_mymemory(text, target_lang)

    # Cache and return
    if translated:
        _cache.set(text, translated, target_lang)
        return translated

    # If translation fails, return original
    return text


# ─────────────────────────────────────────────────────────────────────────────
# HTML-AWARE TRANSLATION
# ─────────────────────────────────────────────────────────────────────────────

async def translate_preserving_html(html_text: str) -> str:
    """
    Translate HTML-formatted text while preserving tags.

    Strategy:
    1. Extract text segments between HTML tags
    2. Translate each segment
    3. Reconstruct with tags
    """
    if not TRANSLATION_ENABLED:
        return html_text

    if not needs_translation(html_text):
        return html_text

    try:
        # Find all HTML tags and text segments
        # Split by tags but keep them
        parts = re.split(r'(<[^>]+>)', html_text)

        translated_parts = []
        for part in parts:
            if part.startswith('<') and part.endswith('>'):
                # HTML tag - keep as-is
                translated_parts.append(part)
            elif part.strip() and needs_translation(part):
                # Text content - translate
                translated = await translate_to_hebrew(part)
                translated_parts.append(translated)
            else:
                translated_parts.append(part)

        return "".join(translated_parts)

    except Exception as e:
        logger.debug(f"HTML translation failed: {e}")
        return html_text


# ─────────────────────────────────────────────────────────────────────────────
# LINE-BY-LINE TRANSLATION
# ─────────────────────────────────────────────────────────────────────────────

async def translate_message(message: str) -> str:
    """
    Smart message translation for Telegram messages.

    Translates each line separately to:
    1. Preserve formatting (separators, line breaks)
    2. Skip lines that don't need translation
    3. Handle mixed-language content
    """
    if not TRANSLATION_ENABLED:
        return message

    if not message:
        return message

    # Quick check: skip if already mostly Hebrew
    if is_hebrew(message):
        return message

    lines = message.split('\n')
    translated_lines = []

    for line in lines:
        # Empty lines
        if not line.strip():
            translated_lines.append(line)
            continue

        # Separators (━━━ lines)
        if all(c in '━─-_═══│' for c in line.strip()):
            translated_lines.append(line)
            continue

        # Check if line needs translation
        if not needs_translation(line):
            translated_lines.append(line)
            continue

        # Translate the line with HTML preservation
        translated = await translate_preserving_html(line)
        translated_lines.append(translated)

    return '\n'.join(translated_lines)


# ─────────────────────────────────────────────────────────────────────────────
# BATCH TRANSLATION
# ─────────────────────────────────────────────────────────────────────────────

async def translate_batch(texts: list[str]) -> list[str]:
    """Translate multiple texts efficiently using parallel calls."""
    if not TRANSLATION_ENABLED:
        return texts

    if not texts:
        return []

    # Filter texts that need translation
    tasks = []
    indices = []
    for i, text in enumerate(texts):
        if text and needs_translation(text):
            tasks.append(translate_to_hebrew(text))
            indices.append(i)

    if not tasks:
        return texts

    # Translate in parallel
    translations = await asyncio.gather(*tasks, return_exceptions=True)

    # Reconstruct list
    result = list(texts)
    for idx, translation in zip(indices, translations):
        if isinstance(translation, str):
            result[idx] = translation

    return result


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def disable_translation() -> None:
    """Disable translation globally."""
    global TRANSLATION_ENABLED
    TRANSLATION_ENABLED = False
    logger.info("Translation disabled")


def enable_translation() -> None:
    """Enable translation globally."""
    global TRANSLATION_ENABLED
    TRANSLATION_ENABLED = True
    logger.info("Translation enabled")


def get_translation_stats() -> dict:
    """Get translation system statistics."""
    return {
        "enabled": TRANSLATION_ENABLED,
        "target_language": TARGET_LANGUAGE,
        "cache_stats": _cache.get_stats(),
        "providers": ["Google Translate (free)", "MyMemory (fallback)"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# FINANCIAL TERMS DICTIONARY (Pre-translated for accuracy)
# ─────────────────────────────────────────────────────────────────────────────

# Pre-translated financial/trading terms for accuracy + speed
# These get applied BEFORE Google Translate for terms that have specific meanings
FINANCIAL_TERMS = {
    # Trade actions
    "BUY": "קנייה",
    "SELL": "מכירה",
    "HOLD": "החזק",
    "STRONG BUY": "קנייה חזקה",
    "STRONG SELL": "מכירה חזקה",
    "Entry": "כניסה",
    "Exit": "יציאה",
    "Stop Loss": "סטופ-לוס",
    "Take Profit": "טייק-פרופיט",
    "Position": "פוזיציה",
    "Trade": "עסקה",
    "Order": "הוראה",

    # Performance
    "Win rate": "אחוז הצלחה",
    "Profit": "רווח",
    "Loss": "הפסד",
    "Gain": "רווח",
    "P&L": "רווח/הפסד",
    "Return": "תשואה",
    "Drawdown": "ירידה מהשיא",
    "Sharpe Ratio": "יחס Sharpe",
    "Sortino Ratio": "יחס Sortino",
    "Risk Score": "ציון סיכון",
    "Confidence": "ביטחון",
    "Volatility": "תנודתיות",

    # Market terms
    "Bullish": "שורי (עליה)",
    "Bearish": "דובי (ירידה)",
    "Uptrend": "מגמת עליה",
    "Downtrend": "מגמת ירידה",
    "Sideways": "טווח אופקי",
    "Overbought": "קנייתי-יתר",
    "Oversold": "מכירתי-יתר",
    "Breakout": "פריצה",
    "Support": "תמיכה",
    "Resistance": "התנגדות",
    "Volume": "נפח",
    "Momentum": "מומנטום",

    # Analysis
    "Performance Report": "דוח ביצועים",
    "Risk Analysis": "ניתוח סיכון",
    "Market Forecast": "תחזית שוק",
    "Daily Summary": "סיכום יומי",
    "Weekly Report": "דוח שבועי",
    "Top Performers": "המובילים",
    "Top Winners": "הזוכים",
    "Top Losers": "המפסידים",
    "Best Trade": "העסקה הטובה",
    "Worst Trade": "העסקה הגרועה",
    "Insights": "תובנות",
    "Recommendations": "המלצות",

    # AI/Strategy
    "AI Decision": "החלטת AI",
    "Strategy": "אסטרטגיה",
    "Pattern": "תבנית",
    "Signal": "איתות",
    "Indicator": "אינדיקטור",
    "Backtest": "בקטסט",
    "Optimization": "אופטימיזציה",
    "Confluence": "הצטלבות",

    # Time
    "Today": "היום",
    "Yesterday": "אתמול",
    "Week": "שבוע",
    "Month": "חודש",
    "Year": "שנה",
    "Hour": "שעה",
    "Minute": "דקה",
    "Day": "יום",
    "Days": "ימים",

    # Status
    "Active": "פעיל",
    "Closed": "סגור",
    "Pending": "ממתין",
    "Cancelled": "בוטל",
    "Filled": "התבצע",
    "Open": "פתוח",
    "Critical": "קריטי",
    "Warning": "אזהרה",
    "Info": "מידע",
    "Success": "הצלחה",
    "Failed": "נכשל",
    "Error": "שגיאה",

    # Portfolio
    "Portfolio": "תיק",
    "Account": "חשבון",
    "Balance": "יתרה",
    "Cash": "מזומן",
    "Equity": "הון",
    "Diversification": "פיזור",
    "Allocation": "הקצאה",
    "Rebalance": "איזון מחדש",
    "Hedge": "גידור",

    # Common phrases
    "currently": "כרגע",
    "available": "זמין",
    "Loading": "טוען",
    "Failed to": "נכשל",
    "successfully": "בהצלחה",
    "Please wait": "נא להמתין",
    "Try again": "נסה שוב",
    "No data": "אין נתונים",
}


def apply_financial_glossary(text: str) -> str:
    """
    Apply pre-translated financial terms.
    Faster than API calls and more accurate for trading terminology.
    """
    if not text:
        return text

    # Sort by length (longest first) to avoid partial matches
    sorted_terms = sorted(FINANCIAL_TERMS.items(), key=lambda x: len(x[0]), reverse=True)

    result = text
    for english, hebrew in sorted_terms:
        # Case-sensitive match for exact phrases
        # Use word boundaries to avoid partial replacements
        pattern = r'\b' + re.escape(english) + r'\b'
        result = re.sub(pattern, hebrew, result, flags=re.IGNORECASE if english.islower() else 0)

    return result


async def translate_message_smart(message: str) -> str:
    """
    Smart translation with financial glossary applied first.

    Process:
    1. Apply financial glossary (fast, accurate for known terms)
    2. Translate remaining English via API
    3. Cache result
    """
    if not TRANSLATION_ENABLED or not message:
        return message

    if is_hebrew(message):
        return message

    # Step 1: Apply financial glossary first
    text = apply_financial_glossary(message)

    # Step 2: Check if any English remains
    if not needs_translation(text):
        return text

    # Step 3: Translate remaining English text
    return await translate_message(text)
