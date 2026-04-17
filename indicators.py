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
    rs = avg_gain / avg_loss
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

    return df


def get_current_indicators(symbol: str) -> dict | None:
    """Get current indicator snapshot for a symbol — all indicators."""
    df = get_stock_data(symbol, period="6mo")
    if df.empty:
        return None
    df = add_all_indicators(df)
    if df.empty or len(df) < 2:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2]
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

    return {
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
    }


def get_market_conditions() -> dict:
    """Get overall market conditions (VIX, SPY trend)."""
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
    except Exception as e:
        logger.warning(f"Market conditions error: {e}")
    return result
