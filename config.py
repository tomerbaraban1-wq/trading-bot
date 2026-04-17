import os
import sys
from pathlib import Path
from dotenv import load_dotenv

_PROJECT_DIR = Path(__file__).parent
load_dotenv(_PROJECT_DIR / ".env")


class Settings:
    # Active broker selection
    ACTIVE_BROKER: str = os.getenv("ACTIVE_BROKER", "alpaca_paper")

    # Alpaca Broker
    ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
    ALPACA_BASE_URL: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    # Interactive Brokers (IBKR)
    IBKR_HOST: str = os.getenv("IBKR_HOST", "127.0.0.1")
    IBKR_PORT: int = int(os.getenv("IBKR_PORT", "7497"))

    # Oanda
    OANDA_API_KEY: str = os.getenv("OANDA_API_KEY", "")
    OANDA_ACCOUNT_ID: str = os.getenv("OANDA_ACCOUNT_ID", "")

    # Tradier
    TRADIER_TOKEN: str = os.getenv("TRADIER_TOKEN", "")
    TRADIER_ACCOUNT: str = os.getenv("TRADIER_ACCOUNT", "")

    # Tastytrade
    TASTYTRADE_USERNAME: str = os.getenv("TASTYTRADE_USERNAME", "")
    TASTYTRADE_PASSWORD: str = os.getenv("TASTYTRADE_PASSWORD", "")
    TASTYTRADE_ACCOUNT: str = os.getenv("TASTYTRADE_ACCOUNT", "")

    # Schwab
    SCHWAB_API_KEY: str = os.getenv("SCHWAB_API_KEY", "")
    SCHWAB_SECRET: str = os.getenv("SCHWAB_SECRET", "")
    SCHWAB_ACCOUNT: str = os.getenv("SCHWAB_ACCOUNT", "")
    SCHWAB_CALLBACK_URL: str = os.getenv("SCHWAB_CALLBACK_URL", "https://127.0.0.1")

    # Binance
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_SECRET: str = os.getenv("BINANCE_SECRET", "")

    # Kraken
    KRAKEN_API_KEY: str = os.getenv("KRAKEN_API_KEY", "")
    KRAKEN_SECRET: str = os.getenv("KRAKEN_SECRET", "")

    # Coinbase
    COINBASE_API_KEY: str = os.getenv("COINBASE_API_KEY", "")
    COINBASE_SECRET: str = os.getenv("COINBASE_SECRET", "")

    # Robinhood
    ROBINHOOD_EMAIL: str = os.getenv("ROBINHOOD_EMAIL", "")
    ROBINHOOD_PASSWORD: str = os.getenv("ROBINHOOD_PASSWORD", "")

    # Webull
    WEBULL_EMAIL: str = os.getenv("WEBULL_EMAIL", "")
    WEBULL_PASSWORD: str = os.getenv("WEBULL_PASSWORD", "")
    WEBULL_DEVICE_ID: str = os.getenv("WEBULL_DEVICE_ID", "")
    WEBULL_TRADING_PIN: str = os.getenv("WEBULL_TRADING_PIN", "")

    # Bybit
    BYBIT_API_KEY: str = os.getenv("BYBIT_API_KEY", "")
    BYBIT_SECRET: str = os.getenv("BYBIT_SECRET", "")

    # OKX
    OKX_API_KEY: str = os.getenv("OKX_API_KEY", "")
    OKX_SECRET: str = os.getenv("OKX_SECRET", "")
    OKX_PASSPHRASE: str = os.getenv("OKX_PASSPHRASE", "")

    # KuCoin
    KUCOIN_API_KEY: str = os.getenv("KUCOIN_API_KEY", "")
    KUCOIN_SECRET: str = os.getenv("KUCOIN_SECRET", "")
    KUCOIN_PASSPHRASE: str = os.getenv("KUCOIN_PASSPHRASE", "")

    # Gemini
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_SECRET: str = os.getenv("GEMINI_SECRET", "")

    # TradeStation
    TRADESTATION_API_KEY: str = os.getenv("TRADESTATION_API_KEY", "")
    TRADESTATION_SECRET: str = os.getenv("TRADESTATION_SECRET", "")
    TRADESTATION_ACCOUNT: str = os.getenv("TRADESTATION_ACCOUNT", "")
    TRADESTATION_REFRESH_TOKEN: str = os.getenv("TRADESTATION_REFRESH_TOKEN", "")

    # Groq LLM
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    # Budget
    MAX_BUDGET: float = float(os.getenv("MAX_BUDGET", "10000"))
    MAX_POSITION_PCT: float = float(os.getenv("MAX_POSITION_PCT", "20"))
    TRAILING_STOP_PCT: float = float(os.getenv("TRAILING_STOP_PCT", "2.0"))
    STOP_LOSS_PCT: float = 5.0   # sell if drops 5%
    TAKE_PROFIT_PCT: float = 10.0  # sell if gains 10%

    # Tax
    TAX_RATE: float = float(os.getenv("TAX_RATE", "0.25"))

    # Webhook
    WEBHOOK_SECRET: str = os.getenv("WEBHOOK_SECRET", "")

    # Database
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", str(_PROJECT_DIR / "data" / "trading.db"))

    # Sentiment
    SENTIMENT_MIN_SCORE: int = int(os.getenv("SENTIMENT_MIN_SCORE", "4"))
    SENTIMENT_EMERGENCY_SCORE: int = int(os.getenv("SENTIMENT_EMERGENCY_SCORE", "2"))
    NEWS_CACHE_TTL: int = int(os.getenv("NEWS_CACHE_TTL", "120"))

    # Heartbeat
    HEARTBEAT_INTERVAL_MINUTES: int = int(os.getenv("HEARTBEAT_INTERVAL_MINUTES", "5"))

    # Telegram
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    def validate(self):
        if self.ACTIVE_BROKER == "tv_paper":
            return  # No API keys needed for paper trading

        missing = []
        if not self.ALPACA_API_KEY:
            missing.append("ALPACA_API_KEY")
        if not self.ALPACA_SECRET_KEY:
            missing.append("ALPACA_SECRET_KEY")
        if not self.GROQ_API_KEY:
            missing.append("GROQ_API_KEY")
        if not self.WEBHOOK_SECRET:
            missing.append("WEBHOOK_SECRET")

        if missing:
            print("--- ERROR: Missing required .env variables ---")
            for var in missing:
                print(f"   {var}=...")
            sys.exit(1)

        if self.MAX_BUDGET <= 0:
            print("--- ERROR: MAX_BUDGET must be positive ---")
            sys.exit(1)


settings = Settings()
