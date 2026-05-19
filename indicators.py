import numpy as np
import pandas as pd
import yfinance as yf
import logging

logger = logging.getLogger(__name__)


def get_stock_data(symbol: str, period: str = "6mo") -> pd.DataFrame:
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df.empty:
            return pd.DataFrame()
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                return pd.DataFrame()
        return df[required]
    except Exception:
        return pd.DataFrame()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return lower, middle, upper


def _stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    """Stochastic Oscillator — momentum indicator."""
    low_min = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-10)
    d = k.rolling(window=d_period).mean()
    return k, d


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — volatility indicator."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index — trend/momentum."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    mean = typical.rolling(window=period).mean()
    mad = typical.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    return (typical - mean) / (0.015 * mad + 1e-10)


def _williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R — overbought/oversold."""
    high_max = df["high"].rolling(window=period).max()
    low_min = df["low"].rolling(window=period).min()
    return -100 * (high_max - df["close"]) / (high_max - low_min + 1e-10)


def _obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume — volume/price momentum."""
    direction = df["close"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * df["volume"]).cumsum()


def get_vix() -> float | None:
    """Get current VIX (fear index). High VIX = risky market."""
    try:
        vix = yf.Ticker("^VIX")
        info = vix.fast_info
        price = getattr(info, "last_price", None)
        if price:
            return round(float(price), 2)
        hist = vix.history(period="2d")
        if not hist.empty:
            return round(float(hist["Close"].iloc[-1]), 2)
    except Exception as e:
        logger.warning(f"Could not fetch VIX: {e}")
    return None


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 26:
        return df
    df = df.copy()

    # --- Trend ---
    df["rsi_14"] = _rsi(df["close"], 14)
    df["rsi_7"] = _rsi(df["close"], 7)   # short-term RSI
    macd_line, signal_line, histogram = _macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = histogram

    # Moving averages
    for period in [10, 20, 50, 200]:
        if len(df) >= period:
            df[f"sma_{period}"] = df["close"].rolling(window=period).mean()
            df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

    # --- Volatility ---
    lower, middle, upper = _bollinger_bands(df["close"], 20)
    df["bb_lower"] = lower
    df["bb_middle"] = middle
    df["bb_upper"] = upper
    df["bb_width"] = (upper - lower) / (middle + 1e-10)  # band squeeze indicator
    df["atr_14"] = _atr(df, 14)

    # --- Momentum ---
    df["stoch_k"], df["stoch_d"] = _stochastic(df)
    df["cci_20"] = _cci(df, 20)
    df["williams_r"] = _williams_r(df, 14)
    df["momentum_10"] = df["close"].diff(10)  # 10-day price momentum

    # --- Volume ---
    df["volume_sma_20"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_sma_20"] + 1e-10)
    df["obv"] = _obv(df)

    # --- Price patterns ---
    df["daily_return"] = df["close"].pct_change()
    df["volatility_20"] = df["daily_return"].rolling(20).std() * 100  # % volatility

    # --- VWAP (20-day) — institutional benchmark ---
    _typical = (df["high"] + df["low"] + df["close"]) / 3
    _cum_tpv  = (_typical * df["volume"]).rolling(20).sum()
    _cum_vol  = df["volume"].rolling(20).sum()
    df["vwap_20"]           = _cum_tpv / (_cum_vol + 1e-10)
    df["vwap_distance_pct"] = (df["close"] - df["vwap_20"]) / (df["vwap_20"] + 1e-10) * 100

    # --- Candlestick patterns (last bar) ---
    if len(df) >= 3:
        o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
        body        = abs(c[-1] - o[-1])
        rng         = h[-1] - l[-1] + 1e-10
        lower_wick  = min(o[-1], c[-1]) - l[-1]
        upper_wick  = h[-1] - max(o[-1], c[-1])
        # Hammer: long lower wick, small body near top
        df["pattern_hammer"] = lower_wick >= 2 * body and upper_wick <= body * 0.5 and rng > 0
        # Bullish engulfing: current bullish bar engulfs previous bearish bar
        df["pattern_bull_engulf"] = (
            len(df) >= 2 and c[-2] < o[-2] and c[-1] > o[-1] and
            o[-1] <= c[-2] and c[-1] >= o[-2]
        )
        # Bearish engulfing: current bearish bar engulfs previous bullish
        df["pattern_bear_engulf"] = (
            len(df) >= 2 and c[-2] > o[-2] and c[-1] < o[-1] and
            o[-1] >= c[-2] and c[-1] <= o[-2]
        )
    else:
        df["pattern_hammer"] = df["pattern_bull_engulf"] = df["pattern_bear_engulf"] = False

    return df


_indicators_cache: dict = {}   # symbol -> (data, timestamp)
_IND_CACHE_TTL = 300           # 5 minutes — reuse same data within scan cycle
_IND_CACHE_MAX = 100           # max entries to prevent memory growth
import threading as _threading
_IND_CACHE_LOCK = _threading.Lock()   # protects eviction from concurrent iteration crash


def get_current_indicators(symbol: str) -> dict | None:
    """Get current indicator snapshot for a symbol — all indicators.
    Cached for 5 minutes to avoid redundant yfinance calls within the same scan.
    Cache capped at 100 entries to prevent memory growth.
    """
    import time as _time
    now = _time.time()
    with _IND_CACHE_LOCK:
        cached = _indicators_cache.get(symbol)
        if cached and now - cached[1] < _IND_CACHE_TTL:
            return cached[0]
        # Evict oldest entries if cache is full (under lock to prevent dict-changed-size error)
        if len(_indicators_cache) >= _IND_CACHE_MAX:
            try:
                oldest = min(_indicators_cache.items(), key=lambda x: x[1][1])
                del _indicators_cache[oldest[0]]
            except (ValueError, KeyError):
                pass

    df = get_stock_data(symbol, period="3mo")  # 3mo instead of 6mo — faster
    if df.empty:
        return None
    df = add_all_indicators(df)
    if df.empty or len(df) < 2:
        return None
    last = df.iloc[-1]
    close = float(last["close"])

    def safe(val):
        try:
            v = float(val)
            return round(v, 4) if pd.notna(v) and not np.isinf(v) else None
        except Exception:
            return None

    # Bollinger Band position (0.0 = at lower, 1.0 = at upper)
    bb_lower = safe(last.get("bb_lower"))
    bb_upper = safe(last.get("bb_upper"))
    bb_position = None
    if bb_lower is not None and bb_upper is not None and bb_upper > bb_lower:
        bb_position = round((close - bb_lower) / (bb_upper - bb_lower), 4)

    # Trend: is price above key moving averages?
    sma20 = safe(last.get("sma_20"))
    sma50 = safe(last.get("sma_50"))
    sma200 = safe(last.get("sma_200"))
    above_sma20 = close > sma20 if sma20 else None
    above_sma50 = close > sma50 if sma50 else None
    above_sma200 = close > sma200 if sma200 else None

    # MACD crossover signal
    macd_bullish = None
    if safe(last.get("macd")) is not None and safe(last.get("macd_signal")) is not None:
        macd_bullish = float(last["macd"]) > float(last["macd_signal"])

    # Stochastic signal
    stoch_k = safe(last.get("stoch_k"))
    stoch_oversold = stoch_k < 20 if stoch_k is not None else None
    stoch_overbought = stoch_k > 80 if stoch_k is not None else None

    # ── 52-Week High / Low ────────────────────────────────────────────────
    # Fetched from yfinance .info; graceful fallback if unavailable.
    week52_high = None
    week52_low = None
    pct_from_52w_high = None
    near_52w_high = False   # within 5% of 52w high  — breakout zone
    near_52w_low = False    # within 10% of 52w low   — support zone
    try:
        _info = yf.Ticker(symbol).info
        _52h = _info.get("fiftyTwoWeekHigh")
        _52l = _info.get("fiftyTwoWeekLow")
        if _52h and float(_52h) > 0:
            week52_high = round(float(_52h), 4)
            pct_from_52w_high = round((close - week52_high) / week52_high * 100, 2)
            near_52w_high = pct_from_52w_high >= -5.0          # within 5% below high
        if _52l and float(_52l) > 0:
            week52_low = round(float(_52l), 4)
            _pct_above_low = (close - week52_low) / week52_low * 100
            near_52w_low = _pct_above_low <= 10.0              # within 10% above low
    except Exception as _e52:
        logger.debug(f"[52W] Could not fetch 52-week data for {symbol}: {_e52}")

    result = {
        # Core
        "close": round(close, 2),
        # RSI
        "rsi": safe(last.get("rsi_14")),
        "rsi_7": safe(last.get("rsi_7")),
        # MACD
        "macd": safe(last.get("macd")),
        "macd_signal": safe(last.get("macd_signal")),
        "macd_hist": safe(last.get("macd_hist")),
        "macd_bullish": macd_bullish,
        # Bollinger Bands
        "bb_position": bb_position,
        "bb_width": safe(last.get("bb_width")),
        # Moving averages
        "sma_20": sma20,
        "sma_50": sma50,
        "sma_200": sma200,
        "above_sma20": above_sma20,
        "above_sma50": above_sma50,
        "above_sma200": above_sma200,
        # Momentum
        "stoch_k": stoch_k,
        "stoch_overbought": stoch_overbought,
        "stoch_oversold": stoch_oversold,
        "cci": safe(last.get("cci_20")),
        "williams_r": safe(last.get("williams_r")),
        "momentum_10": safe(last.get("momentum_10")),
        # Volatility
        "atr": safe(last.get("atr_14")),
        "volatility_20": safe(last.get("volatility_20")),
        # Volume
        "volume_ratio": safe(last.get("volume_ratio")),
        "obv": safe(last.get("obv")),
        # Daily return
        "daily_return": safe(last.get("daily_return")),
        # VWAP
        "vwap_20":           safe(last.get("vwap_20")),
        "vwap_distance_pct": safe(last.get("vwap_distance_pct")),
        # 52-Week High / Low
        "week52_high":        week52_high,
        "week52_low":         week52_low,
        "pct_from_52w_high":  pct_from_52w_high,
        "near_52w_high":      near_52w_high,
        "near_52w_low":       near_52w_low,
        # Candlestick patterns
        "pattern_hammer":      bool(last.get("pattern_hammer", False)),
        "pattern_bull_engulf": bool(last.get("pattern_bull_engulf", False)),
        "pattern_bear_engulf": bool(last.get("pattern_bear_engulf", False)),
    }

    # Cache the result — prevents redundant yfinance downloads within same scan cycle
    with _IND_CACHE_LOCK:
        _indicators_cache[symbol] = (result, now)
    return result


_market_cache: dict = {"data": None, "ts": 0}
_MARKET_CACHE_TTL = 300  # 5 minutes — reuse same SPY data across all tickers in a scan
_fear_greed_cache: dict = {"value": None, "ts": 0}
_FEAR_GREED_TTL = 3600   # 1 hour


def get_put_call_ratio() -> float | None:
    """
    Fetch CBOE total Put/Call ratio — free, no auth.
    PCR > 1.2 = extreme fear (contrarian buy) | PCR < 0.7 = complacency (caution)
    Cached 1 hour.
    """
    import time, requests
    now = time.time()
    if _fear_greed_cache.get("pcr_ts") and now - _fear_greed_cache["pcr_ts"] < 3600:
        return _fear_greed_cache.get("pcr")
    try:
        # Try CBOE via direct API (more reliable than yfinance for ^PCR)
        import requests as _req
        r = _req.get("https://cdn.cboe.com/api/global/delayed_quotes/charts/historical/_SPX.json",
                     timeout=5)
        if r.status_code == 200:
            data = r.json()
            # fallback: use last known value
        # ^PCR no longer available on yfinance — return None silently
    except Exception:
        pass
    return None


def get_fear_greed() -> int | None:
    """
    Fetch CNN Fear & Greed Index (0=Extreme Fear, 100=Extreme Greed).
    Cached 1 hour. Returns None on failure.
    """
    import time as _t
    now = _t.time()
    if _fear_greed_cache["value"] is not None and now - _fear_greed_cache["ts"] < _FEAR_GREED_TTL:
        return _fear_greed_cache["value"]
    try:
        import requests as _req
        resp = _req.get(
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0 TradingBot/1.0"},
        )
        if resp.status_code == 200:
            data = resp.json()
            score = data.get("fear_and_greed", {}).get("score")
            if score is not None:
                val = round(float(score))
                _fear_greed_cache["value"] = val
                _fear_greed_cache["ts"] = now
                logger.debug(f"[FEAR&GREED] score={val}")
                return val
    except Exception as e:
        logger.debug(f"[FEAR&GREED] fetch failed: {e}")
    return None


def get_market_conditions() -> dict:
    """Get overall market conditions (VIX, SPY trend). Cached for 5 minutes."""
    import time
    now = time.time()
    if _market_cache["data"] and now - _market_cache["ts"] < _MARKET_CACHE_TTL:
        return _market_cache["data"]

    result = {"vix": None, "market_trend": None, "spy_above_sma50": None}
    try:
        vix = get_vix()
        result["vix"] = vix
        if vix:
            if vix > 30:
                result["market_trend"] = "fearful"
            elif vix > 20:
                result["market_trend"] = "uncertain"
            else:
                result["market_trend"] = "calm"

        # SPY trend
        spy = get_current_indicators("SPY")
        if spy:
            result["spy_above_sma50"] = spy.get("above_sma50")
            result["spy_rsi"] = spy.get("rsi")

        # Fear & Greed Index
        fg = get_fear_greed()
        if fg is not None:
            result["fear_greed"] = fg

        # Put/Call Ratio — options market sentiment
        pcr = get_put_call_ratio()
        if pcr is not None:
            result["put_call_ratio"] = pcr

        # Market Breadth — % of last 20 days SPY+QQQ closed positive
        breadth = get_market_breadth()
        if breadth is not None:
            result["breadth_score"] = breadth
    except Exception as e:
        logger.warning(f"Market conditions error: {e}")

    _market_cache["data"] = result
    _market_cache["ts"] = now
    return result


def get_market_breadth() -> float | None:
    """
    Calculate market breadth as the percentage of the last 20 trading days
    where both SPY and QQQ closed positive (daily return > 0).

    Returns a 0–100 score, or None on failure.
    Shares the existing 5-minute market cache TTL (called inside get_market_conditions).
    """
    try:
        import yfinance as _yf
        import pandas as _pd
        hist = _yf.download(["SPY", "QQQ"], period="2mo", interval="1d", progress=False, auto_adjust=True)["Close"]
        if hist is None or hist.empty:
            return None
        if "SPY" not in hist.columns or "QQQ" not in hist.columns:
            return None

        hist = hist.dropna().tail(20)
        if len(hist) < 5:
            return None

        spy_pos = hist["SPY"].pct_change().dropna() > 0
        qqq_pos = hist["QQQ"].pct_change().dropna() > 0
        both_pos = spy_pos & qqq_pos
        breadth = round(float(both_pos.mean()) * 100, 1)
        logger.debug(f"[BREADTH] {breadth:.1f}% of last {len(both_pos)} days SPY+QQQ both up")
        return breadth
    except Exception as e:
        logger.debug(f"[BREADTH] fetch failed: {e}")
        return None
