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

    # Budget & Position Sizing (calibrated from live trade data)
    MAX_BUDGET: float = float(os.getenv("MAX_BUDGET", "10000"))
    MAX_POSITION_PCT: float = float(os.getenv("MAX_POSITION_PCT", "15"))   # max 15% per trade
    TRAILING_STOP_PCT: float = float(os.getenv("TRAILING_STOP_PCT", "2.0"))
    STOP_LOSS_PCT: float = float(os.getenv("STOP_LOSS_PCT", "3.5"))
    TAKE_PROFIT_PCT: float = float(os.getenv("TAKE_PROFIT_PCT", "15.0"))
    MAX_OPEN_POSITIONS: int = int(os.getenv("MAX_OPEN_POSITIONS", "4"))  # reduced 6→4: focus on best

    # Time management (calibrated: >48h = 27% WR, 2-12h = 50% WR)
    MAX_HOLD_HOURS: float = float(os.getenv("MAX_HOLD_HOURS", "24.0"))      # exit after 24h max
    MIN_HOLD_MINUTES: int = int(os.getenv("MIN_HOLD_MINUTES", "20"))        # never sell in first 20 min

    # Entry quality filters (calibrated from 33% → target 55% win rate)
    MIN_BUY_SCORE: int = int(os.getenv("MIN_BUY_SCORE", "65"))              # raised 51→65
    MIN_VOLUME_RATIO: float = float(os.getenv("MIN_VOLUME_RATIO", "0.75"))  # raised 0.5→0.75
    REQUIRE_ABOVE_SMA50: bool = os.getenv("REQUIRE_ABOVE_SMA50", "true").lower() in ("true", "1", "yes")
    MAX_DAILY_LOSSES: int = int(os.getenv("MAX_DAILY_LOSSES", "3"))

    # Profit protection (lesson: let profits go to stop instead of riding to zero)
    BREAKEVEN_TRIGGER_PCT: float = float(os.getenv("BREAKEVEN_TRIGGER_PCT", "0.5"))
    PROFIT_PROTECT_ENABLED: bool = os.getenv("PROFIT_PROTECT_ENABLED", "true").lower() in ("true","1","yes")
    PROFIT_PROTECT_PEAK_PCT: float = float(os.getenv("PROFIT_PROTECT_PEAK_PCT", "1.5"))
    PROFIT_PROTECT_FLOOR_PCT: float = float(os.getenv("PROFIT_PROTECT_FLOOR_PCT", "0.2"))

    # Drawdown control (professional trading rules)
    MAX_DAILY_LOSS_PCT: float = float(os.getenv("MAX_DAILY_LOSS_PCT", "2.0"))   # stop after 2% daily loss
    MAX_WEEKLY_LOSS_PCT: float = float(os.getenv("MAX_WEEKLY_LOSS_PCT", "5.0")) # stop after 5% weekly
    MAX_CONSECUTIVE_LOSSES: int = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "3")) # pause after 3 losses

    # Tax
    TAX_RATE: float = float(os.getenv("TAX_RATE", "0.25"))

    # Webhook
    WEBHOOK_SECRET: str = os.getenv("WEBHOOK_SECRET", "")

    # Database
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", str(_PROJECT_DIR / "data" / "trading.db"))
    HARDENED_DURABILITY: bool = os.getenv("HARDENED_DURABILITY", "true").lower() in ("true", "1", "yes")
    ENABLE_WAL_CHECKPOINT: bool = os.getenv("ENABLE_WAL_CHECKPOINT", "true").lower() in ("true", "1", "yes")

    # Sentiment
    SENTIMENT_MIN_SCORE: int = int(os.getenv("SENTIMENT_MIN_SCORE", "5"))  # balanced: 5/10 (was 6, too strict)
    SENTIMENT_EMERGENCY_SCORE: int = int(os.getenv("SENTIMENT_EMERGENCY_SCORE", "2"))
    NEWS_CACHE_TTL: int = int(os.getenv("NEWS_CACHE_TTL", "300"))  # 5 min default (was 2 min)

    # Heartbeat
    HEARTBEAT_INTERVAL_MINUTES: int = int(os.getenv("HEARTBEAT_INTERVAL_MINUTES", "5"))

    # Telegram
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # Discord
    DISCORD_BOT_TOKEN: str = os.getenv("DISCORD_BOT_TOKEN", "")
    DISCORD_CHANNEL_ID: str = os.getenv("DISCORD_CHANNEL_ID", "")
    DISCORD_GUILD_ID: str = os.getenv("DISCORD_GUILD_ID", "882265638784090182")  # SKIL server

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

        # Secret strength check
        if self.WEBHOOK_SECRET and len(self.WEBHOOK_SECRET) < 16:
            print("--- WARNING: WEBHOOK_SECRET is weak (< 16 chars) — use a longer secret ---")

        # Warn if using default/example secrets
        _weak_secrets = {"secret", "password", "123456", "tradebot", "webhook", "test"}
        if self.WEBHOOK_SECRET and self.WEBHOOK_SECRET.lower() in _weak_secrets:
            print("--- WARNING: WEBHOOK_SECRET is too common — change it in .env ---")


settings = Settings()
