import asyncio
import logging
import aiohttp
from config import settings
import broker
import budget
import database
from sentiment import check_emergency_sentiment
from trade_logger import log_trade_open, log_trade_close
from telegram_bot import (
    notify_buy, notify_sell, notify_emergency,
    notify_daily_summary, notify_weekly_report,
    notify_error, notify_circuit_breaker_tripped,
)
from circuit_breaker import check_circuit_breaker, record_trade_result, get_status as cb_status
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
            asyncio.ensure_future(notify_error("loop_error", "", f"sentiment_monitor: {e}"))


async def _close_position(
    trade:      dict,
    cur_price:  float,
    status:     str,
    label:      str,
) -> bool:
    """
    Execute a sell order, log the close, update circuit breaker, notify Telegram.
    Returns True on success, False on broker failure.
    Called by stop_loss_monitor for every exit type.
    """
    ticker   = trade["ticker"]
    lim_sell = limit_sell_price(cur_price)
    try:
        order = await asyncio.wait_for(
            asyncio.to_thread(broker.submit_sell, ticker), timeout=15
        )
    except Exception as sell_err:
        asyncio.ensure_future(notify_error("stop_loss_fail", ticker, str(sell_err)))
        return False

    exit_price = float(order.get("price") or lim_sell)
    pnl_gross  = (exit_price - trade["entry_price"]) * trade["qty"]

    from tax_tracker import process_trade_close
    tax_result = process_trade_close(trade["id"], pnl_gross)
    pnl_net    = pnl_gross - tax_result["tax_amount"]

    log_trade_close(
        trade["id"], exit_price, pnl_gross, pnl_net,
        tax_result["tax_amount"], 0.0, status,
    )

    was_ok, _ = check_circuit_breaker()
    record_trade_result(pnl_gross)
    is_ok, _  = check_circuit_breaker()
    if not is_ok and was_ok:
        st = cb_status()
        asyncio.ensure_future(notify_circuit_breaker_tripped(
            st["daily_pnl"], st["max_daily_loss"], st["trip_reason"]
        ))

    await notify_sell(ticker, exit_price, pnl_gross, label)
    return True


async def stop_loss_monitor():
    """
    Background task: check open trades every 60 seconds.

    Exit hierarchy (checked in order):
      1. ATR Trailing Stop  — price fell through the dynamic floor
      2. Take Profit        — price rose above the fixed ceiling (TAKE_PROFIT_PCT)
      3. Smart Sell         — composite score collapsed (< 30/100)

    ATR trailing stop replaces the old fixed STOP_LOSS_PCT.
    The stop trails upward with the high watermark, locking in gains,
    while giving each asset room proportional to its own volatility.
    """
    from atr_stop import compute_initial_stop, update_trailing_stop, should_exit_confirmed
    import os as _os
    MAX_HOLD_HOURS: float = float(_os.getenv("MAX_HOLD_HOURS", "48.0"))

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

                    cur_price = float(position.get("current_price", trade["entry_price"]))
                    plpc      = float(position.get("unrealized_plpc", 0)) * 100

                    # ── 1. ATR Trailing Stop ──────────────────────────────────
                    atr_stop = trade.get("atr_stop_price")
                    high_wm  = trade.get("high_watermark") or trade["entry_price"]

                    # Initialise on first encounter (new trade or legacy trade)
                    if atr_stop is None:
                        atr_stop, stop_meta = await asyncio.to_thread(
                            compute_initial_stop, ticker, trade["entry_price"]
                        )
                        high_wm = trade["entry_price"]
                        await asyncio.to_thread(
                            database.update_trade_stop, trade["id"], atr_stop, high_wm
                        )
                        logger.info(
                            f"[ATR STOP] {ticker}: initialised stop=${atr_stop:.2f} "
                            f"({stop_meta['stop_pct']:.2f}% from entry)"
                        )

                    # Trail the stop upward as price rises
                    new_stop, new_wm, raised = await asyncio.to_thread(
                        update_trailing_stop,
                        ticker, cur_price, atr_stop, high_wm, trade["entry_price"]
                    )
                    if raised or new_wm != high_wm:
                        await asyncio.to_thread(
                            database.update_trade_stop, trade["id"], new_stop, new_wm
                        )
                        if raised:
                            logger.info(
                                f"[ATR STOP] {ticker}: stop raised "
                                f"${atr_stop:.2f} → ${new_stop:.2f} "
                                f"(price=${cur_price:.2f} | wm=${new_wm:.2f})"
                            )
                        atr_stop = new_stop
                        high_wm  = new_wm

                    # ── 1a. Time-Based Exit — free capital after MAX_HOLD_HOURS ─────
                    from datetime import datetime, timezone as _tz
                    entry_ts = trade.get("entry_time")
                    if entry_ts:
                        try:
                            entry_dt = datetime.strptime(
                                str(entry_ts)[:19], "%Y-%m-%d %H:%M:%S"
                            ).replace(tzinfo=_tz.utc)
                            hours_held = (
                                datetime.now(_tz.utc) - entry_dt
                            ).total_seconds() / 3600
                            if hours_held >= MAX_HOLD_HOURS:
                                logger.info(
                                    f"[TIME EXIT] {ticker}: held {hours_held:.1f}h "
                                    f"≥ {MAX_HOLD_HOURS}h — closing"
                                )
                                await _close_position(
                                    trade, cur_price, "time_exit",
                                    f"Time-based exit ({hours_held:.1f}h held, "
                                    f"limit={MAX_HOLD_HOURS}h)",
                                )
                                continue
                        except Exception as te:
                            logger.debug(f"[TIME EXIT] {ticker}: parse error: {te}")

                    # ── 1b. ATR Trailing Stop (flash-crash confirmed) ─────────
                    flash_exit, flash_reason = await asyncio.to_thread(
                        should_exit_confirmed, ticker, cur_price, atr_stop
                    )
                    if flash_exit:
                        logger.warning(
                            f"[ATR STOP] {ticker}: CONFIRMED EXIT "
                            f"price=${cur_price:.2f} stop=${atr_stop:.2f} "
                            f"(P&L: {plpc:.2f}%) — {flash_reason}"
                        )
                        await _close_position(
                            trade, cur_price, "stop_loss",
                            f"ATR Trailing Stop (stop=${atr_stop:.2f} | {plpc:.1f}%)"
                        )
                        continue  # trade closed — skip other checks
                    elif cur_price <= atr_stop:
                        # Price below stop but NOT confirmed by closed candle
                        logger.info(
                            f"[FLASH GUARD] {ticker}: stop not yet confirmed — holding. "
                            f"{flash_reason}"
                        )

                    # ── 2. Take Profit (fixed ceiling) ────────────────────────
                    if plpc >= settings.TAKE_PROFIT_PCT:
                        logger.info(
                            f"[TAKE PROFIT] {ticker}: {plpc:.2f}% "
                            f"≥ {settings.TAKE_PROFIT_PCT}%"
                        )
                        await _close_position(
                            trade, cur_price, "take_profit",
                            f"Take Profit ({plpc:.1f}%)"
                        )
                        continue

                    # ── 3. Smart Sell (score collapse, max once per 5 min) ────
                    import time as _time
                    last = _smart_sell_last_check.get(ticker, 0)
                    if _time.time() - last >= 300:
                        _smart_sell_last_check[ticker] = _time.time()
                        try:
                            from scoring import get_composite_score
                            score_result = await asyncio.to_thread(
                                get_composite_score, ticker, 5
                            )
                            comp = score_result["composite_score"]
                            if comp < 30:
                                logger.warning(
                                    f"[SMART SELL] {ticker}: score={comp}/100 — exiting"
                                )
                                await _close_position(
                                    trade, cur_price, "smart_sell",
                                    f"Smart Sell (score={comp}/100)"
                                )
                        except Exception as se:
                            logger.warning(f"Smart sell check error for {ticker}: {se}")
                            asyncio.ensure_future(
                                notify_error("stop_loss_fail", ticker, f"Smart sell: {se}")
                            )

                except Exception as e:
                    logger.error(f"Stop loss monitor error for {ticker}: {e}")
                    asyncio.ensure_future(notify_error("stop_loss_fail", ticker, str(e)))

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Stop loss monitor error: {e}")
            asyncio.ensure_future(notify_error("loop_error", "", f"stop_loss_monitor: {e}"))


async def auto_invest_loop():
    """Background task: scan and buy every 5 minutes using full composite scoring."""
    await asyncio.sleep(60)  # wait 1 min after startup
    while True:
        try:
            import random
            import shadow as _shadow
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
                        _vol_ratio: float | None = None   # set after volume check
                        logger.info(
                            f"AUTO-INVEST: {ticker} → {score}/100 "
                            f"({'✅ BUY' if composite['should_buy'] else '❌ SKIP'})"
                        )
                        if not composite["should_buy"]:
                            asyncio.ensure_future(_asyncio.to_thread(
                                _shadow.evaluate, ticker, price, score, sentiment.score,
                                None, "score",
                                f"composite_score={score:.0f} below threshold", "auto_invest",
                            ))
                            continue

                        price = await _asyncio.wait_for(
                            _asyncio.to_thread(broker.get_price, ticker), timeout=15
                        )
                        if not price or price <= 0:
                            continue

                        # Market regime filter — skip in ranging/choppy markets
                        from market_regime import get_regime as _get_regime
                        try:
                            _regime, _adx, _regime_details = await _asyncio.wait_for(
                                _asyncio.to_thread(_get_regime, ticker), timeout=20
                            )
                            if _regime == "ranging":
                                logger.info(
                                    f"AUTO-INVEST: {ticker} skipped — ranging market "
                                    f"(ADX={_adx:.1f} < {_regime_details.get('threshold', 25)})"
                                )
                                asyncio.ensure_future(_asyncio.to_thread(
                                    _shadow.evaluate, ticker, price, score, sentiment.score,
                                    _vol_ratio, "market_regime",
                                    f"ranging market ADX={_adx:.1f}", "auto_invest",
                                ))
                                continue
                        except _asyncio.TimeoutError:
                            logger.warning(f"[ADX] {ticker} regime check timed out — proceeding (fail-open)")

                        # Sanity check — price plausibility + velocity + data completeness
                        from sanity_check import run_all as sanity_run
                        sane, sane_reason = await _asyncio.wait_for(
                            _asyncio.to_thread(sanity_run, ticker, price, None), timeout=20
                        )
                        if not sane:
                            logger.warning(f"AUTO-INVEST: {ticker} SANITY FAIL — {sane_reason}")
                            asyncio.ensure_future(_asyncio.to_thread(
                                _shadow.evaluate, ticker, price, score, sentiment.score,
                                None, "sanity", sane_reason, "auto_invest",
                            ))
                            continue

                        # Volume confirmation — skip low-volume signals
                        from volume_confirm import check as vol_check
                        try:
                            vol_passed, vol_reason, vol_details = await _asyncio.wait_for(
                                _asyncio.to_thread(vol_check, ticker), timeout=15
                            )
                            _vol_ratio = vol_details.get("ratio")
                            if not vol_passed:
                                logger.info(f"AUTO-INVEST: {ticker} volume skip — {vol_reason}")
                                asyncio.ensure_future(_asyncio.to_thread(
                                    _shadow.evaluate, ticker, price, score, sentiment.score,
                                    _vol_ratio, "volume", vol_reason, "auto_invest",
                                ))
                                continue
                        except _asyncio.TimeoutError:
                            logger.warning(f"[VOLUME] {ticker} check timed out — proceeding (fail-open)")

                        # Correlation filter — skip if too correlated with open positions
                        from correlation import check as corr_check
                        try:
                            corr_blocked, corr_reason, corr_details = await _asyncio.wait_for(
                                _asyncio.to_thread(corr_check, ticker), timeout=25
                            )
                            if corr_blocked:
                                logger.info(
                                    f"AUTO-INVEST: {ticker} skipped — {corr_reason} "
                                    f"(max_corr={corr_details.get('max_correlation', '?')})"
                                )
                                asyncio.ensure_future(_asyncio.to_thread(
                                    _shadow.evaluate, ticker, price, score, sentiment.score,
                                    _vol_ratio, "correlation", corr_reason, "auto_invest",
                                ))
                                continue
                        except _asyncio.TimeoutError:
                            logger.warning(f"[CORR] {ticker} check timed out — proceeding (fail-open)")

                        # Risk-based position sizing (replaces naive "available/price")
                        from budget import compute_position_size
                        qty, sizing_meta = await _asyncio.to_thread(compute_position_size, price)
                        if qty <= 0:
                            logger.info(f"AUTO-INVEST: {ticker} sizing=0 → skip ({sizing_meta})")
                            asyncio.ensure_future(_asyncio.to_thread(
                                _shadow.evaluate, ticker, price, score, sentiment.score,
                                _vol_ratio, "budget", f"sizing=0 at ${price:.2f}", "auto_invest",
                            ))
                            continue

                        # Slippage estimate (for metadata/audit — iceberg manages actual limit internally)
                        slip = await _asyncio.to_thread(slippage_estimate, price, qty, "buy", ticker)

                        # Execute — iceberg splits large orders automatically
                        from iceberg import iceberg_buy
                        order = await iceberg_buy(ticker, qty, price)
                        actual_price = float(order.get("price") or price)
                        spent        = actual_price * qty
                        remaining   -= spent
                        bought      += 1

                        from models import WebhookPayload, TradeAction
                        fake_payload = WebhookPayload(
                            secret=settings.WEBHOOK_SECRET,
                            ticker=ticker, action=TradeAction.BUY, price=actual_price,
                        )
                        trade_id = log_trade_open(fake_payload, sentiment, order, qty, sizing_meta, slip)

                        # Set ATR trailing stop immediately after fill
                        try:
                            from atr_stop import compute_initial_stop
                            atr_stop_price, stop_meta = await _asyncio.to_thread(
                                compute_initial_stop, ticker, actual_price
                            )
                            await _asyncio.to_thread(
                                database.update_trade_stop, trade_id, atr_stop_price, actual_price
                            )
                            logger.info(
                                f"[ATR STOP] {ticker}: stop set @ ${atr_stop_price:.2f} "
                                f"({stop_meta['stop_pct']:.2f}% from entry)"
                            )
                        except Exception as stop_err:
                            logger.warning(f"[ATR STOP] {ticker}: failed to set stop: {stop_err}")

                        # Shadow: live also traded — record agreement
                        asyncio.ensure_future(_asyncio.to_thread(
                            _shadow.evaluate, ticker, actual_price, score, sentiment.score,
                            _vol_ratio, None, "", "auto_invest",
                        ))
                        await notify_buy(ticker, qty, actual_price, score, sentiment.score)

                    except _asyncio.TimeoutError:
                        logger.warning(f"AUTO-INVEST: {ticker} timed out, skipping")
                        asyncio.ensure_future(notify_error("api_timeout", ticker, "Auto-invest order timed out"))
                    except Exception as e:
                        logger.error(f"AUTO-INVEST: Error on {ticker}: {e}")
                        asyncio.ensure_future(notify_error("order_failed", ticker, str(e)))

                logger.info(f"AUTO-INVEST: Done. Bought {bought} stocks. Cash left: ${remaining:.2f}")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"AUTO-INVEST loop error: {e}")
            asyncio.ensure_future(notify_error("loop_error", "", f"auto_invest_loop: {e}"))

        await asyncio.sleep(5 * 60)  # run every 5 minutes


async def shadow_monitor_loop():
    """
    Background task: tick all open shadow paper positions every 5 minutes.
    Applies ATR trailing stop and take-profit ceiling — mirrors live stop_loss_monitor
    but operates on the shadow_trades table only (no real orders ever submitted).
    """
    await asyncio.sleep(90)   # staggered start so it doesn't compete with startup I/O
    while True:
        try:
            import shadow as _shadow
            await asyncio.to_thread(_shadow.tick_open_positions)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Shadow monitor error: {e}")
        await asyncio.sleep(5 * 60)


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
            wins      = [t for t in today_trades if (t.get("pnl_gross") or 0) > 0]
            losses    = [t for t in today_trades if (t.get("pnl_gross") or 0) <= 0]
            total_pnl = sum(t.get("pnl_gross") or 0 for t in today_trades)
            total_tax = sum(t.get("tax_reserved") or 0 for t in today_trades)
            total_net = sum(t.get("pnl_net") or 0 for t in today_trades)

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
                tax_reserved=total_tax,
                realized_pnl_net=total_net,
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
