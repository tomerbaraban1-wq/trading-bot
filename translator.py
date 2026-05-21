"""
מודול תרגום לעברית עם 2 שכבות הגנה:
1. Groq LLM (מהיר, איכותי, אבל יכול להיכשל)
2. Google Translate חינמי (יציב, איכות סבירה, fallback)

הפונקציה הראשית: translate_to_hebrew(text) — מחזירה תמיד עברית
"""
import logging
import json
import urllib.parse
import urllib.request
import os

logger = logging.getLogger(__name__)


def _is_already_hebrew(text: str) -> bool:
    """Heuristic: text contains enough Hebrew to be considered translated.
    Threshold 20% — permissive enough to allow stock names mixed in
    (e.g. "אפל AAPL עלתה 5%" = ~25% Hebrew chars).
    """
    if not text:
        return True
    heb_chars = sum(1 for c in text if "֐" <= c <= "׿")
    # Count only letters, not digits/punctuation, in denominator
    letter_count = sum(1 for c in text if c.isalpha() or "֐" <= c <= "׿")
    if letter_count == 0:
        return True   # nothing to translate
    return heb_chars >= letter_count * 0.20


def _google_translate(text: str, source: str = "en", target: str = "he") -> str | None:
    """
    Free Google Translate endpoint (used by Chrome extension).
    No API key needed. Returns translated text or None on failure.

    Note: Google's free endpoint rejects very long queries (~5000 chars).
    For longer text, the caller should split into chunks.
    """
    if not text or not text.strip():
        return text
    # Hard cap to prevent URL-length rejection (5000 chars is Google's limit)
    MAX_LEN = 4500
    if len(text) > MAX_LEN:
        text = text[:MAX_LEN]
    try:
        url = (
            "https://translate.googleapis.com/translate_a/single"
            f"?client=gtx&sl={source}&tl={target}&dt=t&q={urllib.parse.quote(text)}"
        )
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        })
        # Use REQUESTS_CA_BUNDLE if available
        cert = os.getenv("REQUESTS_CA_BUNDLE")
        import ssl
        ctx = ssl.create_default_context(cafile=cert) if cert and os.path.exists(cert) else None
        with urllib.request.urlopen(req, timeout=8, context=ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if not data or not data[0]:
                return None
            # data[0] is list of [translated, original, ...]
            parts = [seg[0] for seg in data[0] if seg and seg[0]]
            return "".join(parts).strip()
    except Exception as e:
        logger.debug(f"[TRANSLATE] Google Translate failed: {e}")
        return None


def _groq_translate(text: str) -> str | None:
    """Use Groq LLM to translate to Hebrew. Returns None on failure."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        cli = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        resp = cli.chat.completions.create(
            model=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
            messages=[{"role": "user", "content":
                f"תרגם לעברית קצרה (שמור שמות מניות באנגלית כמו AAPL, MSFT): {text}\n"
                f"החזר רק את התרגום, בלי הסבר."}],
            max_tokens=200, temperature=0.2, timeout=20,
        )
        result = resp.choices[0].message.content.strip()
        # Strip common prefixes the LLM might add
        for prefix in ("תרגום:", "תרגום -", "התרגום:"):
            if result.startswith(prefix):
                result = result[len(prefix):].strip()
        return result if _is_already_hebrew(result) else None
    except Exception as e:
        logger.debug(f"[TRANSLATE] Groq failed: {e}")
        return None


def translate_to_hebrew(text: str) -> str:
    """
    מתרגם טקסט לעברית. תמיד מחזיר משהו (גם אם התרגום נכשל — מחזיר מקור).

    סדר ניסיונות:
    1. אם הטקסט כבר בעברית — מחזיר כמו שהוא
    2. Groq LLM (איכותי)
    3. Google Translate (יציב, חינמי)
    4. אם הכל נכשל — מחזיר את הטקסט המקורי באנגלית
    """
    if not text or not text.strip():
        return text or ""

    # Step 1: skip if already Hebrew
    if _is_already_hebrew(text):
        return text

    # Step 2: try Groq first (faster, higher quality)
    groq_result = _groq_translate(text)
    if groq_result and _is_already_hebrew(groq_result):
        return groq_result

    # Step 3: fallback to Google Translate
    google_result = _google_translate(text)
    if google_result and _is_already_hebrew(google_result):
        return google_result

    # Step 4: nothing worked — return original
    return text


def translate_headlines(headlines: list[str]) -> list[str]:
    """Translate a list of headlines to Hebrew. Always returns same length.

    Strategy:
    1. Skip if all already Hebrew.
    2. Try Google batch translation (one HTTP call — fast).
    3. Fallback: translate each individually with Google ONLY (no Groq —
       too slow for N headlines, would exceed asyncio timeout).
    """
    if not headlines:
        return []
    # Skip if all already Hebrew
    if all(_is_already_hebrew(h) for h in headlines):
        return headlines
    # Try to translate as a batch with Google (fast — one HTTP call)
    try:
        batch_text = "\n@@@\n".join(headlines)
        translated = _google_translate(batch_text)
        if translated and "@@@" in translated:
            parts = [p.strip() for p in translated.split("@@@")]
            if len(parts) == len(headlines):
                return [p if _is_already_hebrew(p) else headlines[i]
                        for i, p in enumerate(parts)]
    except Exception:
        pass
    # Fallback: translate each individually with Google ONLY (skip Groq — too slow)
    result = []
    for h in headlines:
        if _is_already_hebrew(h):
            result.append(h)
            continue
        try:
            tr = _google_translate(h)
            result.append(tr if tr and _is_already_hebrew(tr) else h)
        except Exception:
            result.append(h)
    return result
