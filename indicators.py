import numpy as np
import pandas as pd
import yfinance as yf


def get_stock_data(symbol: str, period: str = "3mo") -> pd.DataFrame:
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


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 20:
        return df
    df = df.copy()
    df["rsi_14"] = _rsi(df["close"], 14)
    macd_line, signal_line, histogram = _macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = histogram
    lower, middle, upper = _bollinger_bands(df["close"], 20)
    df["bb_lower"] = lower
    df["bb_middle"] = middle
    df["bb_upper"] = upper
    for period in [20, 50]:
        df[f"sma_{period}"] = df["close"].rolling(window=period).mean()
    df["volume_sma_20"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
    return df


def get_current_indicators(symbol: str) -> dict | None:
    """Get current indicator snapshot for a symbol."""
    df = get_stock_data(symbol, period="3mo")
    if df.empty:
        return None
    df = add_all_indicators(df)
    if df.empty:
        return None
    last = df.iloc[-1]
    close = last["close"]
    bb_lower = last.get("bb_lower", 0)
    bb_upper = last.get("bb_upper", 0)

    if pd.notna(bb_lower) and pd.notna(bb_upper) and bb_upper > bb_lower:
        if close <= bb_lower:
            bb_pos = "lower"
        elif close >= bb_upper:
            bb_pos = "upper"
        else:
            bb_pos = "middle"
    else:
        bb_pos = "unknown"

    return {
        "rsi": round(float(last.get("rsi_14", 0)), 2) if pd.notna(last.get("rsi_14")) else None,
        "macd": round(float(last.get("macd", 0)), 4) if pd.notna(last.get("macd")) else None,
        "macd_signal": round(float(last.get("macd_signal", 0)), 4) if pd.notna(last.get("macd_signal")) else None,
        "bb_position": bb_pos,
        "volume_ratio": round(float(last.get("volume_ratio", 0)), 2) if pd.notna(last.get("volume_ratio")) else None,
        "close": round(float(close), 2),
    }
