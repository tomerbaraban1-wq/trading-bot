import asyncio
import logging
import aiohttp
from config import settings
import broker
import budget
import database
from sentiment import check_emergency_sentiment
from trade_logger import log_trade_open, log_trade_close
from telegram_bot import notify_buy, notify_sell, notify_emergency, notify_daily_summary, notify_weekly_report
from circuit_breaker import check_circuit_breaker, record_trade_result
from slippage import limit_buy_price, limit_sell_price, estimate as slippage_estimate

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
                        cur_price = float(position.get("current_price", trade["entry_price"]))
                        lim_sell = limit_sell_price(cur_price)
                        order = await asyncio.wait_for(
                            asyncio.to_thread(broker.submit_sell, ticker), timeout=15
                        )
                        exit_price = float(order.get("price") or lim_sell)
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
                        cur_price = float(position.get("current_price", trade["entry_price"]))
                        lim_sell = limit_sell_price(cur_price)
                        order = await asyncio.wait_for(
                            asyncio.to_thread(broker.submit_sell, ticker), timeout=15
                        )
                        exit_price = float(order.get("price") or lim_sell)
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

            # Trading hours / liquidity / FOMC blackout guard
            from trading_hours import is_ok_to_trade
            hours_ok, hours_reason = is_ok_to_trade()
            if not hours_ok:
                logger.info(f"AUTO-INVEST: {hours_reason} — skipping scan")
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

                        # Sanity check — price plausibility + velocity + data completeness
                        from sanity_check import run_all as sanity_run
                        sane, sane_reason = await _asyncio.wait_for(
                            _asyncio.to_thread(sanity_run, ticker, price, None), timeout=20
                        )
                        if not sane:
                            logger.warning(f"AUTO-INVEST: {ticker} SANITY FAIL — {sane_reason}")
                            continue

                        # Risk-based position sizing (replaces naive "available/price")
                        from budget import compute_position_size
                        qty, sizing_meta = await _asyncio.to_thread(compute_position_size, price)
                        if qty <= 0:
                            logger.info(f"AUTO-INVEST: {ticker} sizing=0 → skip ({sizing_meta})")
                            continue

                        # ATR-based dynamic limit price (replaces fixed 0.1% offset)
                        lim_price = await _asyncio.to_thread(limit_buy_price, price, ticker)
                        slip = await _asyncio.to_thread(slippage_estimate, price, qty, "buy", ticker)

                        from utils import retry_sync
                        order = await _asyncio.wait_for(
                            _asyncio.to_thread(retry_sync, broker.submit_buy, ticker, qty, lim_price, max_retries=2), timeout=30
                        )
                        actual_price = float(order.get("price") or lim_price)
                        spent        = actual_price * qty
                        remaining   -= spent
                        bought      += 1

                        from models import WebhookPayload, TradeAction
                        fake_payload = WebhookPayload(
                            secret=settings.WEBHOOK_SECRET,
                            ticker=ticker, action=TradeAction.BUY, price=actual_price,
                        )
                        log_trade_open(fake_payload, sentiment, order, qty, sizing_meta, slip)
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


async def weekly_report_loop():
    """Background task: compute & send weekly performance report every Sunday at 20:10 UTC."""
    import datetime
    while True:
        try:
            now = datetime.datetime.utcnow()

            # Target: next Sunday at 20:10 UTC
            days_until_sunday = (6 - now.weekday()) % 7   # weekday(): Mon=0 … Sun=6
            if days_until_sunday == 0:
                # Today is Sunday — check if 20:10 has already passed
                target = now.replace(hour=20, minute=10, second=0, microsecond=0)
                if now >= target:
                    days_until_sunday = 7   # next Sunday
            if days_until_sunday > 0:
                target = (now + datetime.timedelta(days=days_until_sunday)).replace(
                    hour=20, minute=10, second=0, microsecond=0
                )

            wait_seconds = (target - now).total_seconds()
            logger.info(f"Weekly report scheduled in {wait_seconds/3600:.1f}h (Sunday 20:10 UTC)")
            await asyncio.sleep(wait_seconds)

            # Compute 4-week report
            from performance import compute as perf_compute, export_csv, format_telegram
            report = await asyncio.to_thread(perf_compute, 4)
            html   = format_telegram(report)

            # Export CSV
            try:
                csv_path = await asyncio.to_thread(export_csv, report)
                logger.info(f"Weekly CSV saved: {csv_path}")
            except Exception as csv_err:
                logger.warning(f"Weekly CSV export failed: {csv_err}")

            await notify_weekly_report(html)
            logger.info(f"Weekly report sent: {report.total_trades} trades | "
                        f"Sharpe={report.sharpe_ratio} | DD={report.max_drawdown_pct:.2f}%")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Weekly report error: {e}")
            await asyncio.sleep(3600)
