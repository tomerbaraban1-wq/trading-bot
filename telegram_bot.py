"""
Telegram Notifications
Sends alerts when the bot buys, sells, or detects important events.
"""
import logging
import aiohttp
from config import settings

logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN: str = getattr(settings, "TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = getattr(settings, "TELEGRAM_CHAT_ID", "")


def _enabled() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


async def send_message(text: str):
    """Send a Telegram message asynchronously."""
    if not _enabled():
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    logger.warning(f"Telegram send failed: {resp.status}")
    except Exception as e:
        logger.warning(f"Telegram error: {e}")


async def notify_buy(ticker: str, qty: int, price: float, score: float, sentiment: int):
    await send_message(
        f"🟢 <b>BUY</b> {ticker}\n"
        f"📦 {qty} shares @ ${price:.2f}\n"
        f"🎯 Score: {score}/100 | Sentiment: {sentiment}/10"
    )


async def notify_sell(ticker: str, price: float, pnl_gross: float, reason: str):
    emoji = "💰" if pnl_gross >= 0 else "🔴"
    await send_message(
        f"{emoji} <b>SELL</b> {ticker}\n"
        f"💵 Exit @ ${price:.2f} | PnL: ${pnl_gross:+.2f}\n"
        f"📌 Reason: {reason}"
    )


async def notify_emergency(ticker: str, reason: str):
    await send_message(
        f"🚨 <b>EMERGENCY EXIT</b> {ticker}\n"
        f"⚠️ {reason}"
    )


async def notify_weekly_report(report_html: str):
    """Send the weekly performance report (pre-formatted HTML)."""
    await send_message(report_html)


async def notify_daily_summary(total_trades: int, wins: int, losses: int,
                                total_pnl: float, open_positions: int, equity: float):
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    pnl_emoji = "📈" if total_pnl >= 0 else "📉"
    await send_message(
        f"📊 <b>Daily Summary</b>\n"
        f"🔄 Trades today: {total_trades} (W:{wins} / L:{losses})\n"
        f"🎯 Win rate: {win_rate:.0f}%\n"
        f"{pnl_emoji} PnL today: ${total_pnl:+.2f}\n"
        f"📂 Open positions: {open_positions}\n"
        f"💼 Equity: ${equity:,.2f}"
    )
