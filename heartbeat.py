import asyncio
import logging
import aiohttp
from config import settings
import broker
import budget
import database
from sentiment import check_emergency_sentiment
from trade_logger import log_trade_close

logger = logging.getLogger(__name__)


async def keep_alive_loop():
    """Ping ourselves every 10 minutes so Render free tier never sleeps."""
    await asyncio.sleep(30)
    while True:
        try:
            port = getattr(settings, "PORT", 8000)
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://localhost:{port}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    logger.debug(f"Keep-alive ping: {resp.status}")
        except Exception:
            pass
        await asyncio.sleep(10 * 60)  # every 10 minutes


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


async def auto_invest_loop():
    """Background task: scan and buy every 5 minutes using full composite scoring."""
    await asyncio.sleep(60)  # wait 1 min after startup
    while True:
        try:
            logger.info("AUTO-INVEST: Starting scheduled scan with composite scoring...")
            from scanner import WATCHLIST
            from sentiment import score_sentiment
            from scoring import get_composite_score
            from budget import get_budget_status, check_can_buy
            from trade_logger import log_trade
            import asyncio as _asyncio

            status = get_budget_status()
            remaining = float(status.get("cash_available", 0))

            if remaining < 10:
                logger.info(f"AUTO-INVEST: Not enough cash (${remaining:.2f}), skipping")
            else:
                # Step 1: Score tickers in PARALLEL for speed
                candidates = [
                    t for t in WATCHLIST[:10]
                    if not database.get_open_trade_by_ticker(t)
                ]

                async def score_ticker(ticker):
                    try:
                        sentiment = await _asyncio.to_thread(score_sentiment, ticker)
                        composite = await _asyncio.to_thread(
                            get_composite_score, ticker, sentiment.score
                        )
                        logger.info(
                            f"AUTO-INVEST: {ticker} → {composite['composite_score']}/100 "
                            f"({'✅ BUY' if composite['should_buy'] else '❌ SKIP'})"
                        )
                        if composite["should_buy"]:
                            return (ticker, composite["composite_score"], sentiment)
                    except Exception as e:
                        logger.warning(f"AUTO-INVEST: score error for {ticker}: {e}")
                    return None

                results = await _asyncio.gather(*[score_ticker(t) for t in candidates])
                scored = [r for r in results if r is not None]

                # Step 2: Sort by score — buy highest scoring first
                scored.sort(key=lambda x: x[1], reverse=True)
                bought = 0

                for ticker, comp_score, sentiment in scored:
                    try:
                        price = await _asyncio.to_thread(broker.get_price, ticker)
                        if not price or price <= 0:
                            continue
                        can_buy, qty, reason = check_can_buy(ticker, price, remaining)
                        if not can_buy or qty <= 0:
                            logger.info(f"AUTO-INVEST: {ticker} budget skip: {reason}")
                            continue
                        order = await _asyncio.to_thread(broker.submit_buy, ticker, qty)
                        actual_price = float(order.get("price") or price)
                        spent = actual_price * qty
                        remaining -= spent
                        bought += 1
                        log_trade(ticker, "buy", qty, actual_price,
                                  sentiment_score=sentiment.score,
                                  sentiment_reasoning=sentiment.reasoning)
                        logger.info(
                            f"AUTO-INVEST: ✅ Bought {qty}x {ticker} @ ${actual_price:.2f} "
                            f"(score={comp_score}/100)"
                        )
                        if remaining < 10:
                            break
                    except Exception as e:
                        logger.error(f"AUTO-INVEST: Error buying {ticker}: {e}")

                logger.info(f"AUTO-INVEST: Done. Bought {bought} stocks. Cash left: ${remaining:.2f}")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"AUTO-INVEST loop error: {e}")

        await asyncio.sleep(5 * 60)  # run every 5 minutes


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
