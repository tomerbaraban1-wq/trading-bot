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
    MAX_POSITION_PCT: float = float(os.getenv("MAX_POSITION_PCT", "18"))   # 15→18: bigger positions for high-conviction trades
    # ── PROFIT OPTIMIZATION (תוקן: רווחים גדולים יותר) ─────────────────
    TRAILING_STOP_PCT: float = float(os.getenv("TRAILING_STOP_PCT", "3.0"))    # 2.0→3.0: יותר אוויר לרווחים לרוץ
    STOP_LOSS_PCT: float = float(os.getenv("STOP_LOSS_PCT", "3.5"))            # סטופ-לוס קשיח, ללא שינוי
    TAKE_PROFIT_PCT: float = float(os.getenv("TAKE_PROFIT_PCT", "10.0"))       # 15→10: יעד ריאלי יותר (רוב המניות לא מגיעות ל-15%)
    MAX_OPEN_POSITIONS: int = int(os.getenv("MAX_OPEN_POSITIONS", "20"))

    # ── Tiered profit-taking — לוקח רווחים בשלבים ─────────────────────────
    PARTIAL_PROFIT_1_PCT: float = float(os.getenv("PARTIAL_PROFIT_1_PCT", "3.0"))    # +3% → מכור 30%
    PARTIAL_PROFIT_1_SIZE: float = float(os.getenv("PARTIAL_PROFIT_1_SIZE", "0.30"))
    PARTIAL_PROFIT_2_PCT: float = float(os.getenv("PARTIAL_PROFIT_2_PCT", "6.0"))    # +6% → מכור 40% נוסף
    PARTIAL_PROFIT_2_SIZE: float = float(os.getenv("PARTIAL_PROFIT_2_SIZE", "0.40"))

    # Time management (calibrated: >48h = 27% WR, 2-12h = 50% WR)
    # שינוי: 24→36 שעות מקסימום (לפעמים מניות צריכות זמן להתאושש)
    MAX_HOLD_HOURS: float = float(os.getenv("MAX_HOLD_HOURS", "36.0"))
    # MIN_HOLD: 20→10 דק' — יציאה מהירה ממפסידים → פינוי תקציב לעסקאות חדשות
    MIN_HOLD_MINUTES: int = int(os.getenv("MIN_HOLD_MINUTES", "10"))

    # ── Entry quality filters — מאוזן בין כניסה לאיכות ──────────────────
    MIN_BUY_SCORE: int = int(os.getenv("MIN_BUY_SCORE", "55"))              # 60→55: bot scoring stocks at 51-58 due to overbought market
    MAX_BB_POSITION: float = float(os.getenv("MAX_BB_POSITION", "0.97"))    # 0.92→0.97: market BB at 89-107%, only block extreme tops
    MIN_VOLUME_RATIO: float = float(os.getenv("MIN_VOLUME_RATIO", "0.50"))  # 0.75→0.50: market data unreliable (yfinance), allow lower
    REQUIRE_ABOVE_SMA50: bool = os.getenv("REQUIRE_ABOVE_SMA50", "true").lower() in ("true", "1", "yes")
    MAX_DAILY_LOSSES: int = int(os.getenv("MAX_DAILY_LOSSES", "3"))

    # Profit protection — מתי להגן על רווחים שנוצרו
    # שינוי: 0.5→2.0 (לא להגן ב-0.5% — זה יוצא במהירות)
    BREAKEVEN_TRIGGER_PCT: float = float(os.getenv("BREAKEVEN_TRIGGER_PCT", "2.0"))
    PROFIT_PROTECT_ENABLED: bool = os.getenv("PROFIT_PROTECT_ENABLED", "true").lower() in ("true","1","yes")
    PROFIT_PROTECT_PEAK_PCT: float = float(os.getenv("PROFIT_PROTECT_PEAK_PCT", "2.5"))    # 1.5→2.5
    PROFIT_PROTECT_FLOOR_PCT: float = float(os.getenv("PROFIT_PROTECT_FLOOR_PCT", "1.0"))  # 0.2→1.0: שמירת 1% רווח

    # Drawdown control (professional trading rules)
    MAX_DAILY_LOSS_PCT: float = float(os.getenv("MAX_DAILY_LOSS_PCT", "2.0"))   # stop after 2% daily loss
    MAX_WEEKLY_LOSS_PCT: float = float(os.getenv("MAX_WEEKLY_LOSS_PCT", "5.0")) # stop after 5% weekly
    MAX_CONSECUTIVE_LOSSES: int = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "3")) # pause after 3 losses

    # ══════════════════════════════════════════════════════════════════════════
    # INSTITUTIONAL-GRADE RISK MANAGEMENT (Hedge Fund Style)
    # ══════════════════════════════════════════════════════════════════════════

    # 1. Portfolio Heat: sum of all open risk should not exceed X% of equity
    # Industry standard: 6-10%. We use 8% for moderate aggression.
    # If 8 positions each with 1% risk = 8% total heat → block new entries
    MAX_PORTFOLIO_HEAT_PCT: float = float(os.getenv("MAX_PORTFOLIO_HEAT_PCT", "8.0"))

    # 2. Time-of-day filter: avoid first/last N minutes of session
    # First 15 min: highest volatility, often false moves (algos absorb open)
    # Last 15 min: closing volatility, can have market-on-close imbalances
    AVOID_FIRST_MINUTES: int = int(os.getenv("AVOID_FIRST_MINUTES", "10"))   # 15→10: less restrictive
    AVOID_LAST_MINUTES: int = int(os.getenv("AVOID_LAST_MINUTES", "5"))      # 15→5: was blocking too much trading time

    # 3. Anti-overtrading: max losses in short window → cool down
    MAX_LOSSES_PER_HOUR: int = int(os.getenv("MAX_LOSSES_PER_HOUR", "3"))
    COOLDOWN_AFTER_LOSSES_MIN: int = int(os.getenv("COOLDOWN_AFTER_LOSSES_MIN", "30"))

    # 4. Volatility-adjusted stop (ATR multiplier)
    # Wider stops on volatile stocks, tighter on stable ones
    ATR_STOP_MULTIPLIER: float = float(os.getenv("ATR_STOP_MULTIPLIER", "2.0"))   # 2× ATR = professional standard
    ATR_TP_MULTIPLIER: float = float(os.getenv("ATR_TP_MULTIPLIER", "3.5"))       # 3.5× ATR for target

    # 5. Weekend exit — close positions Friday before close (avoid weekend gap risk)
    EXIT_BEFORE_WEEKEND: bool = os.getenv("EXIT_BEFORE_WEEKEND", "false").lower() in ("true", "1", "yes")

    # 6. Pre-FOMC / pre-CPI defensive mode (don't add risk before major events)
    # Set via env var on event days
    PRE_EVENT_DEFENSIVE: bool = os.getenv("PRE_EVENT_DEFENSIVE", "false").lower() in ("true", "1", "yes")

    # ══════════════════════════════════════════════════════════════════════════
    # QUANT HEDGE FUND FEATURES (Renaissance / DE Shaw style)
    # ══════════════════════════════════════════════════════════════════════════

    # 7. Sharpe ratio guard — if rolling Sharpe drops, reduce sizing
    # Sharpe < 0.5  → cut risk by 50%
    # Sharpe < 0    → cut risk by 75% (strategy losing edge)
    # Sharpe > 1.5  → can scale up by 25%
    SHARPE_GUARD_ENABLED: bool = os.getenv("SHARPE_GUARD_ENABLED", "true").lower() in ("true", "1", "yes")
    SHARPE_LOW_THRESHOLD: float = float(os.getenv("SHARPE_LOW_THRESHOLD", "0.5"))
    SHARPE_HIGH_THRESHOLD: float = float(os.getenv("SHARPE_HIGH_THRESHOLD", "1.5"))

    # 8. Max acceptable slippage — skip trades with bad fills
    # Industry standard: 0.1% for liquid stocks, 0.3% for less liquid
    MAX_SLIPPAGE_PCT: float = float(os.getenv("MAX_SLIPPAGE_PCT", "0.5"))

    # 9. Smart Order Routing — use LIMIT instead of MARKET orders by default
    # Reduces slippage but may not fill in fast-moving markets
    USE_LIMIT_ORDERS: bool = os.getenv("USE_LIMIT_ORDERS", "true").lower() in ("true", "1", "yes")
    LIMIT_ORDER_OFFSET_PCT: float = float(os.getenv("LIMIT_ORDER_OFFSET_PCT", "0.1"))  # 0.1% above bid for buy

    # 10. Adaptive performance — recent streak adjusts risk
    # 5 wins in a row → +20% risk (riding the wave)
    # 3 losses in a row → -30% risk (preserve capital)
    STREAK_BOOST_WINS: int = int(os.getenv("STREAK_BOOST_WINS", "5"))
    STREAK_BOOST_PCT: float = float(os.getenv("STREAK_BOOST_PCT", "20"))
    STREAK_PROTECT_LOSSES: int = int(os.getenv("STREAK_PROTECT_LOSSES", "3"))
    STREAK_PROTECT_PCT: float = float(os.getenv("STREAK_PROTECT_PCT", "30"))

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
