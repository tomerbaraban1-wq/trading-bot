import asyncio
import logging
import aiohttp
from config import settings
import broker
import budget
import database
from sentiment import check_emergency_sentiment
from trade_logger import log_trade_close
from telegram_bot import notify_buy, notify_sell, notify_emergency, notify_daily_summary
from circuit_breaker import check_circuit_breaker, record_trade_result

logger = logging.getLogger(__name__)

# Smart sell throttle: ticker -> last_check_timestamp (check max every 5 minutes)
_smart_sell_last_check: dict = {}


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
            status = await asyncio.to_thread(budget.get_budget_status)
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
                try:
                    is_emergency = await asyncio.wait_for(
                        asyncio.to_thread(check_emergency_sentiment, ticker), timeout=45
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Sentiment monitor: {ticker} timed out, skipping")
                    continue

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
                    position = await asyncio.wait_for(
                        asyncio.to_thread(broker.get_position, ticker), timeout=15
                    )
                    if not position:
                        continue

                    plpc = float(position.get("unrealized_plpc", 0)) * 100

                    if plpc <= -settings.STOP_LOSS_PCT:
                        logger.warning(f"STOP LOSS: selling {ticker} (P&L: {plpc:.2f}%)")
                        order = await asyncio.wait_for(
                            asyncio.to_thread(broker.submit_sell, ticker), timeout=15
                        )
                        exit_price = float(order.get("price") or position.get("current_price", trade["entry_price"]))
                        pnl_gross = (exit_price - trade["entry_price"]) * trade["qty"]
                        from tax_tracker import process_trade_close
                        tax_result = process_trade_close(trade["id"], pnl_gross)
                        pnl_net = pnl_gross - tax_result["tax_amount"]
                        log_trade_close(
                            trade["id"], exit_price, pnl_gross, pnl_net,
                            tax_result["tax_amount"], 0.0, "stop_loss",
                        )
                        record_trade_result(pnl_gross)
                        await notify_sell(ticker, exit_price, pnl_gross, f"Stop Loss ({plpc:.1f}%)")

                    elif plpc >= settings.TAKE_PROFIT_PCT:
                        logger.info(f"TAKE PROFIT: selling {ticker} (P&L: {plpc:.2f}%)")
                        order = await asyncio.wait_for(
                            asyncio.to_thread(broker.submit_sell, ticker), timeout=15
                        )
                        exit_price = float(order.get("price") or position.get("current_price", trade["entry_price"]))
                        pnl_gross = (exit_price - trade["entry_price"]) * trade["qty"]
                        from tax_tracker import process_trade_close
                        tax_result = process_trade_close(trade["id"], pnl_gross)
                        pnl_net = pnl_gross - tax_result["tax_amount"]
                        log_trade_close(
                            trade["id"], exit_price, pnl_gross, pnl_net,
                            tax_result["tax_amount"], 0.0, "take_profit",
                        )
                        record_trade_result(pnl_gross)
                        await notify_sell(ticker, exit_price, pnl_gross, f"Take Profit ({plpc:.1f}%)")

                    else:
                        # Smart sell: exit if composite score drops too low (max once per 5 min)
                        import time as _time
                        last = _smart_sell_last_check.get(ticker, 0)
                        if _time.time() - last >= 300:
                            try:
                                _smart_sell_last_check[ticker] = _time.time()
                                from scoring import get_composite_score
                                score_result = await asyncio.to_thread(get_composite_score, ticker, 5)
                                comp = score_result["composite_score"]
                                if comp < 30:
                                    logger.warning(f"SMART SELL: {ticker} composite score={comp}/100 — exiting weak position")
                                    order = await asyncio.wait_for(
                                        asyncio.to_thread(broker.submit_sell, ticker), timeout=15
                                    )
                                    exit_price = float(order.get("price") or position.get("current_price", trade["entry_price"]))
                                    pnl_gross = (exit_price - trade["entry_price"]) * trade["qty"]
                                    from tax_tracker import process_trade_close
                                    tax_result = process_trade_close(trade["id"], pnl_gross)
                                    pnl_net = pnl_gross - tax_result["tax_amount"]
                                    log_trade_close(
                                        trade["id"], exit_price, pnl_gross, pnl_net,
                                        tax_result["tax_amount"], 0.0, "smart_sell",
                                    )
                                    record_trade_result(pnl_gross)
                                    await notify_sell(ticker, exit_price, pnl_gross, f"Smart Sell (score={comp}/100)")
                            except Exception as se:
                                logger.warning(f"Smart sell check error for {ticker}: {se}")

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
            import random
            from scanner import WATCHLIST
            from sentiment import score_sentiment
            from scoring import get_composite_score
            from budget import get_budget_status, check_can_buy
            import asyncio as _asyncio

            # Only trade during market hours (wrapped — may call broker API)
            if not await asyncio.to_thread(broker.is_market_open):
                logger.info("AUTO-INVEST: Market is closed, skipping scan")
                await asyncio.sleep(5 * 60)
                continue

            # Circuit breaker check — stop if daily loss limit exceeded
            ok, cb_reason = check_circuit_breaker()
            if not ok:
                logger.warning(f"AUTO-INVEST: {cb_reason} — skipping scan")
                await asyncio.sleep(5 * 60)
                continue

            logger.info("AUTO-INVEST: Starting scheduled scan with composite scoring...")

            status = await _asyncio.to_thread(get_budget_status)
            remaining = float(status.get("cash_available", 0))

            if remaining < 10:
                logger.info(f"AUTO-INVEST: Not enough cash (${remaining:.2f}), skipping")
            else:
                # Step 1: Shuffle watchlist for diversification — different stocks each scan
                shuffled = WATCHLIST.copy()
                random.shuffle(shuffled)
                candidates = [
                    t for t in shuffled[:6]
                    if not database.get_open_trade_by_ticker(t)
                ]

                bought = 0

                # Sequential scan — one ticker at a time to avoid memory/thread issues
                for ticker in candidates:
                    if remaining < 10:
                        break
                    try:
                        # Score with timeout protection
                        sentiment = await _asyncio.wait_for(
                            _asyncio.to_thread(score_sentiment, ticker), timeout=30
                        )
                        composite = await _asyncio.wait_for(
                            _asyncio.to_thread(get_composite_score, ticker, sentiment.score), timeout=30
                        )
                        score = composite["composite_score"]
                        logger.info(
                            f"AUTO-INVEST: {ticker} → {score}/100 "
                            f"({'✅ BUY' if composite['should_buy'] else '❌ SKIP'})"
                        )
                        if not composite["should_buy"]:
                            continue

                        price = await _asyncio.wait_for(
                            _asyncio.to_thread(broker.get_price, ticker), timeout=15
                        )
                        if not price or price <= 0:
                            continue

                        can_buy, qty, reason = check_can_buy(price)
                        if not can_buy or qty <= 0:
                            logger.info(f"AUTO-INVEST: {ticker} budget skip: {reason}")
                            continue

                        order = await _asyncio.wait_for(
                            _asyncio.to_thread(broker.submit_buy, ticker, qty), timeout=15
                        )
                        actual_price = float(order.get("price") or price)
                        spent = actual_price * qty
                        remaining -= spent
                        bought += 1

                        from database import save_trade
                        save_trade({
                            "ticker": ticker, "action": "buy", "qty": qty,
                            "entry_price": actual_price, "trailing_stop_pct": None,
                            "rsi": None, "macd": None, "macd_signal": None,
                            "bb_position": None, "volume_ratio": None,
                            "sentiment_score": sentiment.score,
                            "sentiment_reasoning": sentiment.reasoning,
                        })
                        logger.info(
                            f"AUTO-INVEST: ✅ Bought {qty}x {ticker} @ ${actual_price:.2f} "
                            f"(score={score}/100)"
                        )
                        await notify_buy(ticker, qty, actual_price, score, sentiment.score)

                    except _asyncio.TimeoutError:
                        logger.warning(f"AUTO-INVEST: {ticker} timed out, skipping")
                    except Exception as e:
                        logger.error(f"AUTO-INVEST: Error on {ticker}: {e}")

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
        position = await asyncio.wait_for(
            asyncio.to_thread(broker.get_position, ticker), timeout=15
        )
        if not position:
            logger.warning(f"Emergency exit: no broker position for {ticker}")
            return

        order = await asyncio.wait_for(
            asyncio.to_thread(broker.submit_sell, ticker), timeout=15
        )
        exit_price = float(order.get("price") or position.get("current_price", trade["entry_price"]))
        pnl_gross = (exit_price - trade["entry_price"]) * trade["qty"]

        from tax_tracker import process_trade_close
        tax_result = process_trade_close(trade["id"], pnl_gross)
        pnl_net = pnl_gross - tax_result["tax_amount"]

        log_trade_close(
            trade["id"], exit_price, pnl_gross, pnl_net,
            tax_result["tax_amount"], 0.0, "emergency_exit",
        )
        record_trade_result(pnl_gross)
        await notify_emergency(ticker, f"Critically bearish sentiment | PnL=${pnl_gross:+.2f}")
        logger.warning(
            f"EMERGENCY EXIT COMPLETE: {ticker} | PnL=${pnl_gross:+.2f} | "
            f"Reason: Critically bearish sentiment"
        )
    except Exception as e:
        logger.error(f"Emergency exit FAILED for {ticker}: {e}")


async def daily_summary_loop():
    """Background task: send daily summary to Telegram at market close (~4pm ET)."""
    import datetime
    while True:
        try:
            now = datetime.datetime.utcnow()
            # Market closes at ~20:00 UTC (4pm ET / 23:00 Israel time)
            target = now.replace(hour=20, minute=5, second=0, microsecond=0)
            if now >= target:
                target += datetime.timedelta(days=1)
            wait_seconds = (target - now).total_seconds()
            await asyncio.sleep(wait_seconds)

            # Build summary from today's closed trades
            today = datetime.datetime.utcnow().date()
            all_trades = database.get_trade_history(limit=200)
            today_trades = [
                t for t in all_trades
                if t.get("exit_time") and t["exit_time"][:10] == str(today)
            ]
            wins = [t for t in today_trades if (t.get("pnl_gross") or 0) > 0]
            losses = [t for t in today_trades if (t.get("pnl_gross") or 0) <= 0]
            total_pnl = sum(t.get("pnl_gross") or 0 for t in today_trades)

            open_trades = database.get_open_trades()
            status = await asyncio.to_thread(budget.get_budget_status)
            equity = status.get("positions_value", 0) + status.get("cash_available", 0)

            await notify_daily_summary(
                total_trades=len(today_trades),
                wins=len(wins),
                losses=len(losses),
                total_pnl=total_pnl,
                open_positions=len(open_trades),
                equity=equity,
            )
            logger.info(f"Daily summary sent: {len(today_trades)} trades, PnL=${total_pnl:+.2f}")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Daily summary error: {e}")
            await asyncio.sleep(3600)  # retry in 1 hour on error
