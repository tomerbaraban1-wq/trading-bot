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
    send_message,
)
from circuit_breaker import check_circuit_breaker, record_trade_result, get_status as cb_status
from slippage import limit_buy_price, limit_sell_price, estimate as slippage_estimate, record as slippage_record

logger = logging.getLogger(__name__)

# Smart sell throttle: ticker -> last_check_timestamp (check max every 5 minutes)
_smart_sell_last_check: dict = {}
_smart_sell_lock = asyncio.Lock()  # Protect race condition on dict access

# Track background tasks to prevent fire-and-forget errors
_background_tasks = set()

# Smart sell: track consecutive low-score cycles per ticker (confirmation buffer)
_smart_sell_low_count: dict[str, int] = {}

def _create_background_task(coro):
    """Create a background task and track it to prevent garbage collection."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


async def keep_alive_loop():
    """
    Ping our own /health endpoint every 10 minutes to prevent Render free-tier spin-down.

    Priority:
      1. RENDER_EXTERNAL_URL  — set automatically by Render for every web service
      2. SELF_PING_URL        — manual override in .env (useful for custom domains)
      3. localhost fallback   — last resort (won't prevent Render spin-down, but keeps
                                local/Docker deployments healthy)
    """
    import os as _os
    render_external = _os.getenv("RENDER_EXTERNAL_URL", "").rstrip("/")
    self_ping       = _os.getenv("SELF_PING_URL", "").rstrip("/")

    if render_external:
        base_url = render_external
        logger.info(f"Keep-alive: will ping {base_url}/health (Render external URL)")
    elif self_ping:
        base_url = self_ping
        logger.info(f"Keep-alive: will ping {base_url}/health (SELF_PING_URL override)")
    else:
        port     = getattr(settings, "PORT", 8000)
        base_url = f"http://localhost:{port}"
        logger.info(f"Keep-alive: will ping {base_url}/health (localhost fallback — "
                    "set RENDER_EXTERNAL_URL to prevent Render spin-down)")

    await asyncio.sleep(60)   # wait 60 s after startup before first ping

    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{base_url}/health",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    logger.debug(f"Keep-alive ping → {base_url}/health: {resp.status}")
        except Exception as exc:
            logger.debug(f"Keep-alive ping failed (harmless): {exc}")
        await asyncio.sleep(10 * 60)   # every 10 minutes


async def heartbeat_loop():
    """Background task: log heartbeat every N minutes."""
    while True:
        try:
            await asyncio.sleep(settings.HEARTBEAT_INTERVAL_MINUTES * 60)

            # Timeout guards prevent heartbeat from hanging indefinitely on slow I/O
            try:
                status = await asyncio.wait_for(
                    asyncio.to_thread(budget.get_budget_status), timeout=20
                )
            except asyncio.TimeoutError:
                logger.warning("HEARTBEAT: budget.get_budget_status timed out — using empty status")
                status = {}

            try:
                open_trades = await asyncio.wait_for(
                    asyncio.to_thread(database.get_open_trades), timeout=10
                )
            except asyncio.TimeoutError:
                logger.warning("HEARTBEAT: database.get_open_trades timed out — skipping this cycle")
                continue

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

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")


async def heartbeat_cleanup_loop():
    """Background task: cleanup old data every 6 hours to prevent unbounded DB growth."""
    await asyncio.sleep(60)  # Initial delay
    while True:
        try:
            await asyncio.sleep(6 * 60 * 60)  # Run every 6 hours
            await asyncio.to_thread(database.cleanup_old_heartbeats, days=7)
            await asyncio.to_thread(database.cleanup_old_data, days=30)
            logger.debug("Cleaned up old records (heartbeats, shadow, slippage, learning)")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Heartbeat cleanup error: {e}")


async def sentiment_monitor():
    """Background task: re-check sentiment for open positions every 15 minutes."""
    while True:
        try:
            await asyncio.sleep(15 * 60)  # 15 minutes

            open_trades = await asyncio.to_thread(database.get_open_trades)
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
            _create_background_task(notify_error("loop_error", "", f"sentiment_monitor: {e}"))


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
        # Pass cur_price to avoid redundant yfinance fetch inside submit_sell
        order = await asyncio.wait_for(
            asyncio.to_thread(broker.submit_sell, ticker, None, cur_price), timeout=20
        )
    except asyncio.TimeoutError:
        logger.warning(f"[SELL] {ticker}: timed out, retrying without price hint...")
        try:
            order = await asyncio.wait_for(
                asyncio.to_thread(broker.submit_sell, ticker), timeout=30
            )
        except Exception as retry_err:
            _create_background_task(notify_error("stop_loss_fail", ticker, f"timeout+retry: {retry_err}"))
            return False
    except Exception as sell_err:
        _create_background_task(notify_error("stop_loss_fail", ticker, str(sell_err)))
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
        _create_background_task(notify_circuit_breaker_tripped(
            st["daily_pnl"], st["max_daily_loss"], st["trip_reason"]
        ))

    await notify_sell(ticker, exit_price, pnl_gross, label)

    # Cleanup per-ticker state so re-entry into the same ticker works correctly
    _smart_sell_last_check.pop(ticker, None)   # allow immediate smart-sell check on re-entry
    _position_alert_sent.pop(ticker, None)     # allow fresh movement alerts on re-entry

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
    # Stale-trade detection: a trade must be reported as missing this many
    # consecutive times before we auto-close it (defends against transient
    # broker API hiccups returning None spuriously).
    STALE_THRESHOLD: int = int(_os.getenv("STALE_TRADE_THRESHOLD", "3"))
    _stale_counter: dict[int, int] = {}

    while True:
        try:
            await asyncio.sleep(60)

            open_trades = await asyncio.to_thread(database.get_open_trades)
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
                        # Broker has no position but DB says "open" — possibly
                        # a stale record after a restart, or possibly a transient
                        # API glitch. Require N consecutive misses before closing.
                        miss = _stale_counter.get(trade["id"], 0) + 1
                        _stale_counter[trade["id"]] = miss
                        if miss < STALE_THRESHOLD:
                            logger.warning(
                                f"[STOP LOSS] {ticker}: broker returned no position "
                                f"({miss}/{STALE_THRESHOLD}) — will auto-close if it persists"
                            )
                            continue
                        logger.warning(
                            f"[STOP LOSS] {ticker}: DB has open trade #{trade['id']} "
                            f"but broker shows no position for {STALE_THRESHOLD} checks "
                            f"— auto-closing stale record"
                        )
                        log_trade_close(
                            trade["id"],
                            trade["entry_price"],  # exit at entry = no P&L
                            0.0, 0.0, 0.0, 0.0,
                            "stale_restart",
                        )
                        _stale_counter.pop(trade["id"], None)
                        # Cleanup per-ticker state for re-entry
                        _smart_sell_last_check.pop(ticker, None)
                        _position_alert_sent.pop(ticker, None)
                        continue
                    # Position found — reset stale counter
                    _stale_counter.pop(trade["id"], None)

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
                            # Notify Telegram when stop is raised significantly (≥0.5%)
                            if atr_stop > 0 and (new_stop - atr_stop) / atr_stop >= 0.005:
                                _entry   = trade["entry_price"]
                                _qty     = trade["qty"]
                                _pnl_now = (cur_price - _entry) * _qty
                                _pnl_pct = (cur_price - _entry) / _entry * 100
                                _stop_dist = (cur_price - new_stop) / cur_price * 100
                                # Estimate take profit (~3× stop distance)
                                try:
                                    _stop_dist_abs = _entry - new_stop
                                    _tp_price = round(_entry + _stop_dist_abs * 3, 2)
                                except Exception:
                                    _tp_price = 0
                                _pnl_label = "🟢 רווח" if _pnl_now >= 0 else "🔴 הפסד"
                                try:
                                    from telegram_chat import _fmt_price as _fp
                                    _ep = _fp(_entry)
                                    _cp = _fp(cur_price)
                                    _sp = _fp(new_stop)
                                    _tp = _fp(_tp_price) if _tp_price else "N/A"
                                    _pp = _fp(abs(_pnl_now))
                                except Exception:
                                    _ep = f"${_entry:.2f}"
                                    _cp = f"${cur_price:.2f}"
                                    _sp = f"${new_stop:.2f}"
                                    _tp = f"${_tp_price:.2f}" if _tp_price else "N/A"
                                    _pp = f"${abs(_pnl_now):.2f}"
                                _create_background_task(send_message(
                                    f"🛡️ <b>הגנה הועלתה!</b>  ✨\n"
                                    f"━━━━━━━━━━━━━━━━\n"
                                    f"🏦  <b>{ticker}</b>  ·  {_qty} מניות\n\n"
                                    f"💵  קנייה:         {_ep}\n"
                                    f"🎯  יעד רווח:    {_tp}\n"
                                    f"🛡️  הגנה (Stop): <b>{_sp}</b>\n\n"
                                    f"📍  עכשיו:  {_cp}  <b>({_pnl_pct:+.1f}%)</b>\n"
                                    f"{'💚' if _pnl_now >= 0 else '❤️'}  {'רווח' if _pnl_now >= 0 else 'הפסד'}:  <b>{_pp}</b>"
                                ))
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
                                    f"יציאה לפי זמן ({hours_held:.1f} שעות, מגבלה={MAX_HOLD_HOURS}ש')",
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
                            f"עצירה נגררת (עצירה=${atr_stop:.2f} | {plpc:.1f}%)"
                        )
                        continue  # trade closed — skip other checks
                    elif cur_price <= atr_stop:
                        # Price below stop but NOT confirmed by closed candle
                        logger.info(
                            f"[FLASH GUARD] {ticker}: stop not yet confirmed — holding. "
                            f"{flash_reason}"
                        )

                    # ── 2. Take Profit — ATR-based (4× ATR) or fixed ceiling ──
                    # ATR-based TP is achievable for low-vol stocks (MO ~2%, V ~4%)
                    # Fixed TP is the safety cap for high-vol stocks
                    try:
                        from atr_stop import _fetch_atr
                        _atr_val = await asyncio.to_thread(_fetch_atr, ticker, trade["entry_price"])
                        _atr_tp_pct = min(
                            settings.TAKE_PROFIT_PCT,                    # cap at fixed TP
                            max(3.0, (_atr_val / trade["entry_price"]) * 100 * 4)  # 4×ATR, min 3%
                        )
                    except Exception:
                        _atr_tp_pct = settings.TAKE_PROFIT_PCT

                    # Scale-out exit plan (sell in thirds):
                    #   Stage 1 — sell 50% at 50% of ATR TP
                    #   Stage 2 — sell 25% (of original) at 80% of ATR TP
                    #   Stage 3 — sell remaining 25% at full ATR TP
                    _stage1_pct = max(2.0, _atr_tp_pct * 0.5)   # 50% of full TP
                    _stage2_pct = max(2.5, _atr_tp_pct * 0.8)   # 80% of full TP
                    # Stage 1 is considered done when the high_watermark already reached
                    # the stage-1 level (price was above it at some point and partial was taken)
                    _stage1_done = bool(trade.get("high_watermark") and
                                        trade.get("high_watermark", 0) >= trade["entry_price"] * (1 + _stage1_pct / 100 + 0.001))
                    # Stage 2 is considered done when watermark reached stage-2 level
                    _stage2_done = bool(trade.get("high_watermark") and
                                        trade.get("high_watermark", 0) >= trade["entry_price"] * (1 + _stage2_pct / 100 + 0.001))

                    if not _stage1_done and plpc >= _stage1_pct:
                        # Stage 1: sell 50% of the full position
                        _half_qty = round(trade["qty"] * 0.5, 6)
                        if _half_qty > 0:
                            try:
                                _half_order = await asyncio.wait_for(
                                    asyncio.to_thread(broker.submit_sell, ticker, _half_qty, cur_price),
                                    timeout=30
                                )
                                _half_pnl = (cur_price - trade["entry_price"]) * _half_qty
                                logger.info(f"[PARTIAL TP S1] {ticker}: sold 50% ({_half_qty} shares) "
                                            f"@ ${cur_price:.2f} (+{plpc:.1f}%) | PnL=${_half_pnl:+.2f}")
                                await send_message(
                                    f"💰 <b>רווח חלקי שלב 1 — {ticker}</b>\n"
                                    f"מכרתי 50% מהפוזיציה\n"
                                    f"📊 {_half_qty} מניות @ ${cur_price:.2f} (+{plpc:.1f}%)\n"
                                    f"💵 רווח ממומש: <b>${_half_pnl:+.2f}</b>\n"
                                    f"השאר ממשיך לרוץ עם עצירה נגררת ✅"
                                )
                            except Exception as _pe:
                                logger.warning(f"[PARTIAL TP S1] {ticker}: half-sell failed: {_pe}")

                    elif _stage1_done and not _stage2_done and plpc >= _stage2_pct:
                        # Stage 2: sell another 25% of original position (= 50% of current remaining)
                        # After Stage 1, remaining qty ≈ 50% of original, so we sell half of that
                        _quarter_qty = round(trade["qty"] * 0.5, 6)   # 50% of what's left
                        if _quarter_qty > 0:
                            try:
                                _s2_order = await asyncio.wait_for(
                                    asyncio.to_thread(broker.submit_sell, ticker, _quarter_qty, cur_price),
                                    timeout=30
                                )
                                _s2_pnl = (cur_price - trade["entry_price"]) * _quarter_qty
                                logger.info(f"[PARTIAL TP S2] {ticker}: sold 25% ({_quarter_qty} shares) "
                                            f"@ ${cur_price:.2f} (+{plpc:.1f}%) | PnL=${_s2_pnl:+.2f}")
                                await send_message(
                                    f"💰 <b>רווח חלקי שלב 2 — {ticker}</b>\n"
                                    f"מכרתי עוד 25% מהפוזיציה המקורית\n"
                                    f"📊 {_quarter_qty} מניות @ ${cur_price:.2f} (+{plpc:.1f}%)\n"
                                    f"💵 רווח ממומש: <b>${_s2_pnl:+.2f}</b>\n"
                                    f"25% האחרון ממשיך לרוץ עד יעד מלא ✅"
                                )
                            except Exception as _pe:
                                logger.warning(f"[PARTIAL TP S2] {ticker}: quarter-sell failed: {_pe}")

                    elif plpc >= _atr_tp_pct:
                        # Stage 3 (full TP): sell remaining position
                        logger.info(
                            f"[TAKE PROFIT] {ticker}: {plpc:.2f}% ≥ {_atr_tp_pct:.1f}% "
                            f"(ATR-based full TP)"
                        )
                        await _close_position(
                            trade, cur_price, "take_profit",
                            f"רווח יעד ({plpc:.1f}% ≥ {_atr_tp_pct:.1f}%)"
                        )
                        continue

                    # ── 3. Smart Sell (score collapse, max once per 5 min) ────
                    import time as _time
                    # Hold lock ONLY to read/update timestamp — release before expensive I/O
                    async with _smart_sell_lock:
                        last = _smart_sell_last_check.get(ticker, 0)
                        if _time.time() - last < 300:
                            continue  # Skip check if run too recently
                        _smart_sell_last_check[ticker] = _time.time()
                    # Lock released — now do expensive scoring outside the lock
                    try:
                        from scoring import get_composite_score
                        score_result = await asyncio.to_thread(
                            get_composite_score, ticker, 5
                        )
                        comp = score_result.get("composite_score")
                        if comp is None:
                            raise ValueError("composite_score missing from result")

                        # Profit-adjusted threshold: tighter exit when sitting on gains
                        smart_threshold = 35
                        if plpc >= 8.0:
                            smart_threshold = 45   # lock in large gains aggressively
                        elif plpc >= 4.0:
                            smart_threshold = 40   # moderate protection of profits

                        if comp < smart_threshold:
                            # Require 2 consecutive low scores before selling (noise filter)
                            _smart_sell_low_count[ticker] = _smart_sell_low_count.get(ticker, 0) + 1
                            if _smart_sell_low_count[ticker] >= 2:
                                logger.warning(
                                    f"[SMART SELL] {ticker}: score={comp}/100 "
                                    f"(threshold={smart_threshold}, confirmed) — exiting"
                                )
                                _smart_sell_low_count.pop(ticker, None)
                                await _close_position(
                                    trade, cur_price, "smart_sell",
                                    f"מכירה חכמה (ציון={comp}/100)"
                                )
                            else:
                                logger.info(
                                    f"[SMART SELL] {ticker}: score={comp} < {smart_threshold} "
                                    f"— waiting for confirmation (1/2)"
                                )
                        else:
                            _smart_sell_low_count.pop(ticker, None)   # reset on recovery
                    except Exception as se:
                        logger.warning(f"Smart sell check error for {ticker}: {se}")
                        _create_background_task(
                                notify_error("stop_loss_fail", ticker, f"Smart sell: {se}")
                            )

                except Exception as e:
                    logger.error(f"Stop loss monitor error for {ticker}: {e}")
                    _create_background_task(notify_error("stop_loss_fail", ticker, str(e)))

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Stop loss monitor error: {e}")
            _create_background_task(notify_error("loop_error", "", f"stop_loss_monitor: {e}"))


async def auto_invest_loop():
    """Background task: scan and buy every 5 minutes using full composite scoring."""
    await asyncio.sleep(60)  # wait 1 min after startup
    while True:
        try:
            import os as _os
            import random
            import shadow as _shadow
            from scanner import get_watchlist as _get_watchlist
            WATCHLIST = _get_watchlist()
            from sentiment import score_sentiment
            from scoring import get_composite_score
            from budget import get_budget_status, check_can_buy
            import asyncio as _asyncio

            # /pause command check — user can pause buying via Telegram
            if _os.getenv("BOT_PAUSED", "").lower() == "true":
                logger.info("AUTO-INVEST: Bot paused by user — skipping scan")
                await asyncio.sleep(5 * 60)
                continue

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

            # SPY trend guard — skip if overall market is in downtrend
            try:
                from indicators import get_market_conditions
                _mkt = await asyncio.to_thread(get_market_conditions)
                if _mkt.get("spy_above_sma50") is False:
                    logger.info("AUTO-INVEST: SPY below SMA50 (downtrend) — skipping buys")
                    await asyncio.sleep(5 * 60)
                    continue
                _vix = _mkt.get("vix")
                if _vix and _vix > 30:
                    logger.info(f"AUTO-INVEST: VIX={_vix:.1f} extreme fear — skipping buys")
                    await asyncio.sleep(5 * 60)
                    continue
            except Exception:
                pass   # fail-open: proceed if market data unavailable

            # High-impact economic event guard (CPI / NFP) — check BEFORE
            # is_ok_to_trade() so we can send a Telegram notification on these days.
            from trading_hours import is_ok_to_trade, is_high_impact_day
            _econ_impact, _econ_event = is_high_impact_day()
            if _econ_impact:
                logger.warning(f"AUTO-INVEST: High-impact day — {_econ_event} — skipping buys")
                _create_background_task(
                    send_message(
                        f"📅 <b>אירוע כלכלי משמעותי היום</b>\n"
                        f"⛔ <b>{_econ_event}</b>\n"
                        f"קניות חסומות היום למניעת תנודתיות לפני הפרסום.\n"
                        f"פוזיציות קיימות ומכירות אינן מושפעות."
                    )
                )
                await asyncio.sleep(5 * 60)
                continue

            # Trading hours / liquidity / FOMC blackout guard
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
                # Max open positions (configurable via MAX_OPEN_POSITIONS env var)
                MAX_OPEN_POSITIONS = settings.MAX_OPEN_POSITIONS
                open_count = len(database.get_open_trades())
                if open_count >= MAX_OPEN_POSITIONS:
                    # Auto-rebalance: check if any open position scores very low
                    # If so, sell it to make room for a better opportunity
                    _rebalance_threshold = int(_os.getenv("REBALANCE_EXIT_SCORE", "30"))
                    _rebalanced = False
                    try:
                        _open = database.get_open_trades()
                        for _t in _open:
                            _tsym = _t["ticker"]
                            _tscore = await _asyncio.wait_for(
                                _asyncio.to_thread(get_composite_score, _tsym, 5), timeout=20
                            )
                            if _tscore.get("composite_score", 100) < _rebalance_threshold:
                                _pos = await _asyncio.to_thread(broker.get_position, _tsym)
                                if _pos:
                                    _cprice = float(_pos.get("current_price", _t["entry_price"]))
                                    logger.info(f"AUTO-REBALANCE: selling weak {_tsym} "
                                                f"(score={_tscore['composite_score']}) to make room")
                                    await _close_position(_t, _cprice, "smart_sell",
                                                          f"איזון מחדש (ציון={_tscore['composite_score']})")
                                    _rebalanced = True
                                    break
                    except Exception as _re:
                        logger.debug(f"Auto-rebalance check failed: {_re}")

                    if not _rebalanced:
                        logger.info(f"AUTO-INVEST: max positions reached ({open_count}/{MAX_OPEN_POSITIONS}), skipping scan")
                        await _asyncio.sleep(5 * 60)
                        continue

                # Sector rotation: prioritize stocks from leading sectors
                try:
                    from sector_rotation import prioritize_by_sector, get_leading_sectors
                    _sectors = await _asyncio.to_thread(get_leading_sectors)
                    if _sectors:
                        _top_sector = _sectors[0]
                        logger.info(f"[SECTOR] Leading: {_top_sector['name']} ({_top_sector['return_pct']:+.1f}%)")
                    shuffled = await _asyncio.to_thread(prioritize_by_sector, shuffled)
                except Exception:
                    pass  # fail-open: proceed with random order

                # Smart scan: put high-momentum stocks first, then random rotation
                SCAN_PER_CYCLE = int(_os.getenv("SCAN_PER_CYCLE", "10"))
                # Momentum pre-filter: quick 1-day % change check (cheap, no full scoring)
                _momentum_tickers = []
                _normal_tickers = []
                try:
                    import yfinance as _yf
                    _sample = [t for t in shuffled[:30] if not database.get_open_trade_by_ticker(t)]
                    if _sample:
                        _prices = _yf.download(_sample, period="2d", progress=False, auto_adjust=True)
                        if not _prices.empty and "Close" in _prices.columns:
                            _close = _prices["Close"]
                            for _t in _sample:
                                try:
                                    _chg = float((_close[_t].iloc[-1] - _close[_t].iloc[-2]) / _close[_t].iloc[-2] * 100)
                                    if _chg > 0.5:   # positive momentum
                                        _momentum_tickers.append(_t)
                                    else:
                                        _normal_tickers.append(_t)
                                except Exception:
                                    _normal_tickers.append(_t)
                except Exception:
                    _normal_tickers = [t for t in shuffled[:SCAN_PER_CYCLE] if not database.get_open_trade_by_ticker(t)]

                # Momentum stocks first, then fill with normal rotation
                _prioritized = _momentum_tickers + _normal_tickers
                candidates = _prioritized[:SCAN_PER_CYCLE]

                bought = 0

                # Sequential scan — one ticker at a time to avoid memory/thread issues
                for ticker in candidates:
                    if remaining < 10:
                        break
                    try:
                        # Earnings blackout check — skip if earnings report is within 3 days
                        try:
                            from earnings import check_earnings_risk
                            earn_risky, earn_reason, earn_days = await _asyncio.wait_for(
                                _asyncio.to_thread(check_earnings_risk, ticker), timeout=10
                            )
                            if earn_risky:
                                logger.info(f"AUTO-INVEST: {ticker} EARNINGS BLACKOUT — {earn_reason}")
                                continue
                        except Exception:
                            pass  # fail-open: proceed if earnings check fails

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
                        # Learning check — skip if ticker matches known loss patterns
                        try:
                            from learning import should_override_buy as _sob
                            from indicators import get_current_indicators as _gci
                            _ind = _gci(ticker) or {}
                            _block, _block_reason = _sob(ticker, _ind)
                            if _block:
                                logger.info(f"AUTO-INVEST: {ticker} blocked by learning — {_block_reason}")
                                continue
                        except Exception:
                            pass  # fail-open: proceed if learning check fails

                        if not composite["should_buy"]:
                            # price not yet fetched at this point — pass 0.0 as placeholder
                            _create_background_task(_asyncio.to_thread(
                                _shadow.evaluate, ticker, 0.0, score, sentiment.score,
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
                                _create_background_task(_asyncio.to_thread(
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
                            _create_background_task(_asyncio.to_thread(
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
                                _create_background_task(_asyncio.to_thread(
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
                                _create_background_task(_asyncio.to_thread(
                                    _shadow.evaluate, ticker, price, score, sentiment.score,
                                    _vol_ratio, "correlation", corr_reason, "auto_invest",
                                ))
                                continue
                        except _asyncio.TimeoutError:
                            logger.warning(f"[CORR] {ticker} check timed out — proceeding (fail-open)")

                        # Risk-based position sizing (replaces naive "available/price")
                        # Pass composite score so high-conviction signals get larger positions
                        from budget import compute_position_size
                        qty, sizing_meta = await _asyncio.to_thread(compute_position_size, price, score)
                        if qty <= 0:
                            logger.info(f"AUTO-INVEST: {ticker} sizing=0 → skip ({sizing_meta})")
                            _create_background_task(_asyncio.to_thread(
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
                        filled_qty   = float(order.get("filled_qty", qty))  # use actual fill
                        spent        = actual_price * filled_qty
                        remaining   -= spent
                        bought      += 1

                        # Record actual slippage (signal price vs fill price)
                        _create_background_task(_asyncio.to_thread(
                            slippage_record, price, actual_price, filled_qty, "buy", ticker
                        ))

                        from models import WebhookPayload, TradeAction
                        fake_payload = WebhookPayload(
                            secret=settings.WEBHOOK_SECRET,
                            ticker=ticker, action=TradeAction.BUY, price=actual_price,
                        )
                        trade_id = log_trade_open(fake_payload, sentiment, order, filled_qty, sizing_meta, slip)

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
                        _create_background_task(_asyncio.to_thread(
                            _shadow.evaluate, ticker, actual_price, score, sentiment.score,
                            _vol_ratio, None, "", "auto_invest",
                        ))
                        await notify_buy(ticker, filled_qty, actual_price, score, sentiment.score)

                    except _asyncio.TimeoutError:
                        logger.warning(f"AUTO-INVEST: {ticker} timed out, skipping")
                        _create_background_task(notify_error("api_timeout", ticker, "Auto-invest order timed out"))
                    except Exception as e:
                        logger.error(f"AUTO-INVEST: Error on {ticker}: {e}")
                        _create_background_task(notify_error("order_failed", ticker, str(e)))

                logger.info(f"AUTO-INVEST: Done. Bought {bought} stocks. Cash left: ${remaining:.2f}")

                # Send scan summary to Telegram (only if something notable happened)
                if bought > 0 or len(candidates) > 0:
                    try:
                        _scan_lines = [f"🔍 <b>סיכום סריקה</b>"]
                        if bought > 0:
                            _scan_lines.append(f"✅ נקנו {bought} מניות")
                        else:
                            _scan_lines.append(f"⏭️ לא נמצאו הזדמנויות ({len(candidates)} מניות נסרקו)")
                        _scan_lines.append(f"💵 מזומן: ${remaining:.2f}")
                        _create_background_task(send_message("\n".join(_scan_lines)))
                    except Exception:
                        pass

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"AUTO-INVEST loop error: {e}")
            _create_background_task(notify_error("loop_error", "", f"auto_invest_loop: {e}"))

        await asyncio.sleep(5 * 60)  # run every 5 minutes


async def morning_briefing_loop():
    """
    Every trading day, send a briefing 30 minutes before market open.
    Uses Alpaca's clock API to get the exact next open time — handles
    NYSE DST transitions, early closes, and holidays automatically.
    """
    import datetime as _dt
    await asyncio.sleep(10)   # short initial delay (was 60s — caused missed briefings on restart)
    _briefing_sent_date = None   # track so we only send once per day
    while True:
        try:
            # Ask broker when the market next opens
            clock = await asyncio.wait_for(
                asyncio.to_thread(broker.get_clock), timeout=10
            )
            if not clock:
                await asyncio.sleep(5 * 60)
                continue

            next_open = clock.get("next_open")   # ISO datetime string
            is_open   = clock.get("is_open", False)

            if not next_open:
                await asyncio.sleep(5 * 60)
                continue

            # Parse next open time
            if isinstance(next_open, str):
                next_open_dt = _dt.datetime.fromisoformat(next_open.replace("Z", "+00:00"))
            else:
                next_open_dt = next_open

            now_utc = _dt.datetime.now(_dt.timezone.utc)
            briefing_time = next_open_dt - _dt.timedelta(minutes=30)
            today_str = now_utc.strftime("%Y-%m-%d")

            # Already sent today — skip
            if _briefing_sent_date == today_str:
                await asyncio.sleep(5 * 60)
                continue

            # Catch-up window: send if we are within 25 minutes PAST briefing_time
            # (handles restarts during the briefing window or just after market open)
            catch_up_deadline = briefing_time + _dt.timedelta(minutes=25)
            if briefing_time <= now_utc <= catch_up_deadline:
                pass  # in the send window — fall through to send
            elif now_utc < briefing_time:
                # Too early — wait
                wait_sec = (briefing_time - now_utc).total_seconds()
                await asyncio.sleep(min(wait_sec, 5 * 60))
                continue
            else:
                # Past the catch-up window — skip until tomorrow
                await asyncio.sleep(5 * 60)
                continue

            from news_service import get_general_headlines
            from scanner import get_watchlist as _get_wl
            WATCHLIST = _get_wl()
            from sentiment import score_sentiment
            from scoring import get_composite_score

            headlines = await asyncio.to_thread(get_general_headlines, 5)

            # Translate headlines to Hebrew using Groq LLM
            if headlines and settings.GROQ_API_KEY:
                try:
                    from openai import OpenAI as _OAI
                    _cli = _OAI(api_key=settings.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
                    _raw = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))
                    _prompt = f"תרגם את הכותרות הבאות לעברית קצרה (עד 10 מילים כל אחת). החזר רק את הכותרות המתורגמות, ממוספרות:\n{_raw}"
                    _resp = await asyncio.to_thread(
                        lambda: _cli.chat.completions.create(
                            model=settings.LLM_MODEL,
                            messages=[{"role": "user", "content": _prompt}],
                            max_tokens=300, temperature=0.3,
                        )
                    )
                    _lines = _resp.choices[0].message.content.strip().split("\n")
                    _translated = [l.split(". ", 1)[-1].strip() for l in _lines if l.strip()]
                    headlines = _translated[:5] if _translated else headlines
                except Exception as _te:
                    logger.debug(f"[BRIEFING] Translation failed: {_te}")

            news_text = "\n".join(f"• {h}" for h in headlines) if headlines else "אין חדשות זמינות"

            # Score top 3 candidates quickly
            top = []
            for ticker in WATCHLIST[:8]:
                try:
                    sent = await asyncio.wait_for(
                        asyncio.to_thread(score_sentiment, ticker), timeout=20
                    )
                    comp = await asyncio.wait_for(
                        asyncio.to_thread(get_composite_score, ticker, sent.score), timeout=20
                    )
                    top.append((ticker, comp["composite_score"], sent.score))
                except Exception:
                    continue
            top.sort(key=lambda x: x[1], reverse=True)

            candidates = ""
            for ticker, score, sent in top[:3]:
                candidates += f"\n📊 <b>{ticker}</b> — ציון {score:.0f}/100 | סנטימנט {sent}/10"

            await send_message(
                f"☀️ <b>תדרוך בוקר — שוק נפתח בעוד 30 דקות</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📰 <b>חדשות מובילות:</b>\n{news_text}\n\n"
                f"🎯 <b>מועמדים מובילים היום:</b>{candidates if candidates else chr(10) + 'טרם חושב'}"
            )
            _briefing_sent_date = today_str
            logger.info("Morning briefing sent")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Morning briefing error: {e}")
            await asyncio.sleep(3600)


_position_alert_sent: dict[str, float] = {}   # ticker → last alert pct
_price_alerts_fired: set = set()              # "TICKER:PRICE" already fired


async def price_alert_loop():
    """
    Check user-defined price alerts every 2 minutes.
    Alerts set via /alert TICKER PRICE command stored in USER_ALERTS env var.
    """
    await asyncio.sleep(120)
    while True:
        try:
            import os as _os
            alerts_str = _os.getenv("USER_ALERTS", "")
            if alerts_str:
                alerts = [a.strip() for a in alerts_str.split(",") if ":" in a.strip()]
                for alert in alerts:
                    try:
                        ticker, target_str = alert.split(":", 1)
                        target = float(target_str)
                        key = f"{ticker}:{target}"
                        if key in _price_alerts_fired:
                            continue
                        cur = await asyncio.wait_for(
                            asyncio.to_thread(broker.get_price, ticker), timeout=8
                        )
                        if cur and abs(cur - target) / target < 0.01:  # within 1%
                            _price_alerts_fired.add(key)
                            direction = "📈 עלה" if cur >= target else "📉 ירד"
                            await send_message(
                                f"🔔 <b>התראת מחיר — {ticker}</b>\n"
                                f"━━━━━━━━━━━━━━━━\n"
                                f"🎯 יעד: ${target:.2f}\n"
                                f"💵 מחיר עכשיו: ${cur:.2f}\n"
                                f"{direction} ליעד! ✅"
                            )
                    except Exception:
                        continue
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"Price alert loop error: {e}")
        await asyncio.sleep(2 * 60)


async def position_alert_loop():
    """
    Check open positions every 2 minutes. If any position moves ±5% from entry,
    send an immediate Telegram alert. Rate-limited to once per position per 2%.
    """
    await asyncio.sleep(180)
    while True:
        try:
            open_trades = await asyncio.to_thread(database.get_open_trades)
            for trade in open_trades:
                if trade["action"] != "buy":
                    continue
                ticker = trade["ticker"]
                entry  = trade["entry_price"]
                try:
                    pos = await asyncio.wait_for(
                        asyncio.to_thread(broker.get_position, ticker), timeout=10
                    )
                    if not pos:
                        continue
                    cur   = float(pos.get("current_price", entry))
                    pct   = (cur - entry) / entry * 100
                    unreal = float(pos.get("unrealized_pl", (cur - entry) * trade["qty"]))

                    last = _position_alert_sent.get(ticker, 0.0)
                    # Alert if moved ±5% and hasn't been alerted within 2% of this level
                    if abs(pct) >= 5.0 and abs(pct - last) >= 2.0:
                        _position_alert_sent[ticker] = pct
                        emoji = "🚀" if pct > 0 else "⚠️"
                        await send_message(
                            f"{emoji} <b>התראת תנועה — {ticker}</b>\n"
                            f"📊 שינוי: <b>{pct:+.2f}%</b> מכניסה\n"
                            f"💵 מחיר: ${cur:.2f}  (כניסה: ${entry:.2f})\n"
                            f"💰 רווח/הפסד לא ממומש: <b>${unreal:+.2f}</b>"
                        )

                    # Stop-loss proximity alert — warn when within 1.5% of stop
                    atr_stop = trade.get("atr_stop_price")
                    _stop_key = f"stop_{ticker}"
                    if atr_stop and atr_stop > 0:
                        dist_pct = (cur - atr_stop) / cur * 100
                        if dist_pct <= 1.5 and _position_alert_sent.get(_stop_key) != round(dist_pct, 0):
                            _position_alert_sent[_stop_key] = round(dist_pct, 0)
                            await send_message(
                                f"🚨 <b>התראת Stop Loss קרוב — {ticker}</b>\n"
                                f"💵 מחיר עכשיו: ${cur:.2f}\n"
                                f"🛡️ Stop Loss: ${atr_stop:.2f}\n"
                                f"⚡ מרחק: <b>{dist_pct:.1f}%</b> בלבד!"
                            )
                except Exception:
                    continue
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Position alert error: {e}")
        await asyncio.sleep(2 * 60)


async def eod_sweep_loop():
    """
    End-of-day sweep (3:45 PM ET): sell 'dead money' positions.
    A position is dead money if: flat (-1% to +1%) AND composite score < 55.
    Frees capital for better opportunities tomorrow.
    """
    import datetime as _dt
    _utc = _dt.timezone.utc
    _last_sweep_date = None
    await asyncio.sleep(120)
    while True:
        try:
            now_utc = _dt.datetime.now(_utc)
            try:
                from trading_hours import _now_et
                now_et = _now_et()
                is_sweep_time = (now_et.hour == 15 and now_et.minute >= 45)
            except Exception:
                is_sweep_time = False

            today = now_utc.date()
            if is_sweep_time and _last_sweep_date != today:
                _last_sweep_date = today
                logger.info("[EOD SWEEP] Running end-of-day dead-money check...")

                open_trades = await asyncio.to_thread(database.get_open_trades)
                swept = 0
                for trade in open_trades:
                    if trade["action"] != "buy":
                        continue
                    ticker = trade["ticker"]
                    try:
                        pos = await asyncio.wait_for(
                            asyncio.to_thread(broker.get_position, ticker), timeout=10
                        )
                        if not pos:
                            continue
                        plpc = float(pos.get("unrealized_plpc", 0)) * 100
                        cur  = float(pos.get("current_price", trade["entry_price"]))

                        # Dead money: flat AND score deteriorating
                        if -1.0 <= plpc <= 1.0:
                            from scoring import get_composite_score
                            score_r = await asyncio.wait_for(
                                asyncio.to_thread(get_composite_score, ticker, 5), timeout=20
                            )
                            if score_r.get("composite_score", 100) < 55:
                                logger.info(f"[EOD SWEEP] {ticker}: flat ({plpc:.1f}%) + weak score — selling")
                                await _close_position(trade, cur, "smart_sell",
                                                      f"ניקוי סוף יום: שטוח {plpc:.1f}%, ציון={score_r['composite_score']:.0f}")
                                swept += 1
                    except Exception:
                        continue

                if swept:
                    await send_message(
                        f"🌙 <b>סיכום סוף יום</b>\n"
                        f"מכרתי {swept} פוזיציות 'כסף מת' — שטוחות + ציון נמוך\n"
                        f"✅ ניפינו הון לחר מחר"
                    )

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"EOD sweep error: {e}")
        await asyncio.sleep(60)


async def news_refresh_loop():
    """
    Pre-fetch news for all watchlist stocks every 60 seconds — always running,
    not just during market hours. This ensures the cache is warm before open,
    and that emergency sentiment checks work on weekends / after-hours too.
    """
    await asyncio.sleep(90)   # staggered start
    while True:
        try:
            from scanner import get_watchlist as _gwl
            WATCHLIST = _gwl()
            from news_service import get_headlines, get_general_headlines
            # Refresh general market headlines
            await asyncio.to_thread(get_general_headlines, 10)
            # Refresh per-ticker headlines for open positions + watchlist
            open_trades = await asyncio.to_thread(database.get_open_trades)
            # Only refresh news for open positions (not watchlist) to save memory
            tickers = list({t["ticker"] for t in open_trades})[:5]
            for ticker in tickers:
                try:
                    await asyncio.to_thread(get_headlines, ticker, 5)
                except Exception:
                    pass
            logger.debug(f"News cache refreshed for {len(tickers)} tickers")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"News refresh error (non-critical): {e}")
        await asyncio.sleep(300)   # refresh every 5 minutes (was 60s) — reduces memory pressure


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

        # Cleanup per-ticker state so re-entry works correctly (same as _close_position)
        _smart_sell_last_check.pop(ticker, None)
        _position_alert_sent.pop(ticker, None)

        await notify_emergency(ticker, f"סנטימנט שלילי קיצוני | רווח/הפסד=${pnl_gross:+.2f}")
        logger.warning(
            f"EMERGENCY EXIT COMPLETE: {ticker} | PnL=${pnl_gross:+.2f} | "
            f"סיבה: סנטימנט שלילי קיצוני"
        )
    except Exception as e:
        logger.error(f"Emergency exit FAILED for {ticker}: {e}")


async def daily_summary_loop():
    """Background task: send daily summary to Telegram at market close (~4pm ET).
    Accounts for EDT (UTC-4, summer) and EST (UTC-5, winter) automatically.
    """
    import datetime
    _utc = datetime.timezone.utc
    while True:
        try:
            now = datetime.datetime.now(_utc)
            # Determine if US is on EDT (Mar-Nov) or EST (Nov-Mar)
            # Market closes at 20:00 UTC in EDT, 21:00 UTC in EST
            is_edt = 3 <= now.month <= 10
            close_hour = 20 if is_edt else 21
            target = now.replace(hour=close_hour, minute=5, second=0, microsecond=0)
            if now >= target:
                target += datetime.timedelta(days=1)

            # Check every minute instead of sleeping for hours
            while datetime.datetime.now(_utc) < target:
                await asyncio.sleep(60)

            # Skip weekends — no trading, nothing to summarise
            if datetime.datetime.now(_utc).weekday() >= 5:  # 5=Sat, 6=Sun
                logger.debug("Daily summary: skipping weekend")
                await asyncio.sleep(60)
                continue

            # Build summary from today's trades
            today = datetime.datetime.now(_utc).date()
            all_trades = database.get_trade_history(limit=200)

            # Closed today (sells)
            closed_today = [
                t for t in all_trades
                if t.get("exit_time") and t["exit_time"][:10] == str(today)
            ]
            # Opened today (buys)
            opened_today = [
                t for t in all_trades
                if t.get("entry_time") and t["entry_time"][:10] == str(today)
            ]

            wins      = [t for t in closed_today if (t.get("pnl_gross") or 0) > 0]
            losses    = [t for t in closed_today if (t.get("pnl_gross") or 0) <= 0]
            total_pnl = sum(t.get("pnl_gross") or 0 for t in closed_today)
            total_tax = sum(t.get("tax_reserved") or 0 for t in closed_today)
            total_net = sum(t.get("pnl_net") or 0 for t in closed_today)

            open_trades = await asyncio.to_thread(database.get_open_trades)
            status = await asyncio.to_thread(budget.get_budget_status)
            equity = status.get("positions_value", 0) + status.get("cash_available", 0)

            await notify_daily_summary(
                total_trades=len(closed_today),
                wins=len(wins),
                losses=len(losses),
                total_pnl=total_pnl,
                open_positions=len(open_trades),
                equity=equity,
                tax_reserved=total_tax,
                realized_pnl_net=total_net,
                buys_today=len(opened_today),
            )
            logger.info(
                f"Daily summary sent: buys={len(opened_today)}, "
                f"sells={len(closed_today)}, PnL=${total_pnl:+.2f}"
            )

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Daily summary error: {e}")
            await asyncio.sleep(3600)  # retry in 1 hour on error


async def portfolio_update_loop():
    """
    Send a live portfolio snapshot to Telegram every hour during market hours.
    Shows every open position with current price, unrealized P&L and % change.
    """
    await asyncio.sleep(120)   # wait 2 min after startup before first send
    while True:
        try:
            # Use broker API for market hours — handles DST + holidays automatically
            market_open = await asyncio.wait_for(
                asyncio.to_thread(broker.is_market_open), timeout=10
            )
            if not market_open:
                await asyncio.sleep(15 * 60)   # check again in 15 min
                continue

            open_trades = await asyncio.to_thread(database.get_open_trades)

            if not open_trades:
                # Don't spam "no positions" every hour — skip silently
                pass
            else:
                lines = ["📂 <b>תיק עכשיו</b>\n━━━━━━━━━━━━━━━━"]
                total_unrealized = 0.0

                for trade in open_trades:
                    ticker = trade["ticker"]
                    qty    = trade["qty"]
                    entry  = trade["entry_price"]
                    try:
                        pos = await asyncio.wait_for(
                            asyncio.to_thread(broker.get_position, ticker), timeout=10
                        )
                        cur_price    = float(pos.get("current_price", entry))
                        unrealized   = float(pos.get("unrealized_pl", (cur_price - entry) * qty))
                        unrealized_pct = float(pos.get("unrealized_plpc", 0)) * 100
                    except Exception:
                        cur_price      = entry
                        unrealized     = 0.0
                        unrealized_pct = 0.0

                    total_unrealized += unrealized
                    emoji = "📈" if unrealized >= 0 else "📉"
                    lines.append(
                        f"\n{emoji} <b>{ticker}</b>\n"
                        f"   כמות: {qty} מניות\n"
                        f"   כניסה: ${entry:.2f}  →  עכשיו: ${cur_price:.2f}\n"
                        f"   רווח/הפסד: <b>${unrealized:+.2f}</b> ({unrealized_pct:+.2f}%)"
                    )

                total_emoji = "📈" if total_unrealized >= 0 else "📉"
                lines.append(f"\n━━━━━━━━━━━━━━━━\n{total_emoji} סה״כ לא ממומש: <b>${total_unrealized:+.2f}</b>")
                await send_message("\n".join(lines))

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Portfolio update error: {e}")

        await asyncio.sleep(60 * 60)   # שלח כל שעה


async def backtest_learning_loop():
    """Weekly: run historical backtest and update MIN_BUY_SCORE automatically."""
    import datetime as _dt
    _utc = _dt.timezone.utc
    await asyncio.sleep(300)  # 5 min after startup
    while True:
        try:
            now = _dt.datetime.now(_utc)
            days_to_sun = (6 - now.weekday()) % 7 or 7
            target = (now + _dt.timedelta(days=days_to_sun)).replace(
                hour=19, minute=0, second=0, microsecond=0)
            while _dt.datetime.now(_utc) < target:
                await asyncio.sleep(300)

            logger.info("[BACKTEST] Starting weekly historical learning...")
            from backtest_learner import run_backtest, apply_insights
            from scanner import get_watchlist as _gwl
            result = await asyncio.to_thread(run_backtest, _gwl()[:25])
            update = apply_insights()
            await send_message(
                f"🎓 <b>למידה מהיסטוריה</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📊 {result.tickers_analyzed} מניות | {result.total_signals} סיגנלים\n"
                f"✅ אחוז הצלחה היסטורי: <b>{result.overall_win_rate:.1f}%</b> | תשואה: {result.avg_return:+.2f}%\n"
                f"🎯 ציון מומלץ: <b>{result.optimal_min_score}</b>\n"
                + (f"🔄 עודכן: {update['old_score']} → <b>{update['new_score']}</b>" if update.get('applied') else "✅ ציון נוכחי אופטימלי")
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Backtest learning error: {e}")
            await asyncio.sleep(3600)


async def weekly_report_loop():
    """Background task: compute & send weekly performance report every Sunday at 20:10 UTC."""
    import datetime
    _utc = datetime.timezone.utc
    while True:
        try:
            now = datetime.datetime.now(_utc)

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

            # Check every minute instead of sleeping for days
            while datetime.datetime.now(_utc) < target:
                await asyncio.sleep(60)

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
