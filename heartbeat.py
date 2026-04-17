import asyncio
import logging
from config import settings
import broker
import budget
import database
from sentiment import check_emergency_sentiment
from trade_logger import log_trade_close

logger = logging.getLogger(__name__)


async def heartbeat_loop():
    """Background task: log heartbeat every N minutes."""
    while True:
        try:
            await asyncio.sleep(settings.HEARTBEAT_INTERVAL_MINUTES * 60)
            status = budget.get_budget_status()
            open_trades = database.get_open_trades()

            database.save_heartbeat(
                open_positions=len(open_trades),
                budget_used_pct=status.get("budget_used_pct", 0),
                total_equity=status.get("positions_value", 0) + status.get("cash_available", 0),
                notes=f"Open: {[t['ticker'] for t in open_trades]}" if open_trades else "No open positions",
            )

            logger.info(
                f"HEARTBEAT: {len(open_trades)} positions | "
                f"Budget: {status.get('budget_used_pct', 0):.1f}% used | "
                f"Equity: ${status.get('positions_value', 0) + status.get('cash_available', 0):,.2f}"
            )

            # Cleanup old heartbeats
            database.cleanup_old_heartbeats(days=7)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")


async def sentiment_monitor():
    """Background task: re-check sentiment for open positions every 15 minutes."""
    while True:
        try:
            await asyncio.sleep(15 * 60)  # 15 minutes

            open_trades = database.get_open_trades()
            if not open_trades:
                continue

            for trade in open_trades:
                ticker = trade["ticker"]
                if trade["action"] != "buy":
                    continue

                logger.info(f"Sentiment monitor: checking {ticker}...")
                is_emergency = check_emergency_sentiment(ticker)

                if is_emergency:
                    logger.warning(f"EMERGENCY: Sentiment critically bearish for {ticker}! Executing exit...")
                    await _emergency_exit(trade)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Sentiment monitor error: {e}")


async def stop_loss_monitor():
    """Background task: check open trades for stop loss / take profit every 60 seconds."""
    while True:
        try:
            await asyncio.sleep(60)

            open_trades = database.get_open_trades()
            if not open_trades:
                continue

            for trade in open_trades:
                if trade["action"] != "buy":
                    continue

                ticker = trade["ticker"]
                try:
                    position = broker.get_position(ticker)
                    if not position:
                        continue

                    plpc = float(position.get("unrealized_plpc", 0)) * 100

                    if plpc <= -settings.STOP_LOSS_PCT:
                        logger.warning(f"STOP LOSS: selling {ticker} (P&L: {plpc:.2f}%)")
                        order = broker.submit_sell(ticker)
                        # Use actual fill price from broker, not the pre-sell snapshot
                        exit_price = float(order.get("price") or position.get("current_price", trade["entry_price"]))
                        pnl_gross = (exit_price - trade["entry_price"]) * trade["qty"]
                        from tax_tracker import process_trade_close
                        tax_result = process_trade_close(trade["id"], pnl_gross)
                        pnl_net = pnl_gross - tax_result["tax_amount"]
                        log_trade_close(
                            trade["id"], exit_price, pnl_gross, pnl_net,
                            tax_result["tax_amount"], 0.0, "stop_loss",
                        )

                    elif plpc >= settings.TAKE_PROFIT_PCT:
                        logger.info(f"TAKE PROFIT: selling {ticker} (P&L: {plpc:.2f}%)")
                        order = broker.submit_sell(ticker)
                        exit_price = float(order.get("price") or position.get("current_price", trade["entry_price"]))
                        pnl_gross = (exit_price - trade["entry_price"]) * trade["qty"]
                        from tax_tracker import process_trade_close
                        tax_result = process_trade_close(trade["id"], pnl_gross)
                        pnl_net = pnl_gross - tax_result["tax_amount"]
                        log_trade_close(
                            trade["id"], exit_price, pnl_gross, pnl_net,
                            tax_result["tax_amount"], 0.0, "take_profit",
                        )

                except Exception as e:
                    logger.error(f"Stop loss monitor error for {ticker}: {e}")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Stop loss monitor error: {e}")


async def _emergency_exit(trade: dict):
    """Execute an emergency exit for a trade."""
    ticker = trade["ticker"]
    try:
        position = broker.get_position(ticker)
        if not position:
            logger.warning(f"Emergency exit: no broker position for {ticker}")
            return

        order = broker.submit_sell(ticker)
        exit_price = position["current_price"]
        pnl_gross = (exit_price - trade["entry_price"]) * trade["qty"]

        from tax_tracker import process_trade_close
        tax_result = process_trade_close(trade["id"], pnl_gross)
        pnl_net = pnl_gross - tax_result["tax_amount"]

        log_trade_close(
            trade["id"], exit_price, pnl_gross, pnl_net,
            tax_result["tax_amount"], 0.0, "emergency_exit",
        )

        logger.warning(
            f"EMERGENCY EXIT COMPLETE: {ticker} | PnL=${pnl_gross:+.2f} | "
            f"Reason: Critically bearish sentiment"
        )
    except Exception as e:
        logger.error(f"Emergency exit FAILED for {ticker}: {e}")
