import asyncio
import logging
import aiohttp
import os
import datetime as _dt
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
    notify_trending_tickers, notify_daily_goal_progress,
    notify_sentiment_alert, notify_market_summary,
    notify_risk_metrics,
)
from discord_bot import (
    send_discord_embed, send_discord_trade_open,
    send_discord_trade_close, send_discord_emergency,
    send_discord_circuit_breaker, send_discord_daily_summary,
    fetch_community_sentiment, get_trending_tickers,
)
from circuit_breaker import check_circuit_breaker, record_trade_result, get_status as cb_status
from slippage import limit_buy_price, limit_sell_price, estimate as slippage_estimate, record as slippage_record
from continuous_learner import run_continuous_learning_cycle, get_learning_summary

logger = logging.getLogger(__name__)

# Smart sell throttle: ticker -> last_check_timestamp (check max every 5 minutes)
_smart_sell_last_check: dict = {}
_smart_sell_lock: asyncio.Lock | None = None  # initialized in start_loops() after event loop starts

# Track background tasks to prevent fire-and-forget errors
_background_tasks = set()

# Smart sell: track consecutive low-score cycles per ticker (confirmation buffer)
_smart_sell_low_count: dict[str, int] = {}

# Stop-raise alert deduplication: ticker → last alert pct
_position_alert_sent: dict[str, float] = {}

# Momentum exit: track last 3 prices per ticker to detect declining trend
_price_history: dict[str, list] = {}   # ticker → [price1, price2, price3] (oldest→newest)
# Price-target alerts already fired: "TICKER:PRICE" strings
_price_alerts_fired: set = set()
# In-memory guard: partial-sell stage already executed this cycle
# Prevents double-sell if watermark DB write fails
_partial_sell_done: set[str] = set()  # "trade_id:stage" strings

def _create_background_task(coro):
    """Create a background task and track it to prevent garbage collection."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


def _is_quiet() -> bool:
    """Return True when QUIET_MODE is enabled — suppress non-critical Telegram alerts."""
    import os as _os
    return _os.getenv("QUIET_MODE", "").lower() == "true"


async def telegram_context_warmup_loop():
    """Pre-build Telegram context every 3 min so replies are instant.
    Forces cache invalidation so the next user message gets fresh data immediately.
    """
    await asyncio.sleep(20)   # start fast after boot
    while True:
        try:
            tc = __import__('telegram_chat')
            # Force cache expiry then rebuild so next message hits warm cache
            tc._context_cache = (0.0, {})
            await asyncio.wait_for(
                asyncio.to_thread(tc._build_context),
                timeout=25,
            )
            logger.debug("[CHAT] Context pre-warmed")
        except asyncio.TimeoutError:
            logger.warning("[CHAT] Context warmup timeout — skipping cycle")
        except Exception:
            pass
        await asyncio.sleep(180)   # every 3 minutes (cache TTL is 5 min)


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
        logger.info(f"Keep-alive: will ping {base_url}/ping every 14 min (Render external URL)")
    elif self_ping:
        base_url = self_ping
        logger.info(f"Keep-alive: will ping {base_url}/ping every 14 min (SELF_PING_URL override)")
    else:
        port     = getattr(settings, "PORT", 8000)
        base_url = f"http://localhost:{port}"
        logger.info(f"Keep-alive: will ping {base_url}/ping (localhost — "
                    "set RENDER_EXTERNAL_URL to prevent Render spin-down)")

    await asyncio.sleep(60)   # wait 60 s after startup before first ping

    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{base_url}/ping",   # /ping is lighter than /health (no DB calls)
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    logger.debug(f"Keep-alive ping → {base_url}/ping: {resp.status}")
        except Exception as exc:
            logger.debug(f"Keep-alive ping failed (harmless): {exc}")
        await asyncio.sleep(8 * 60)   # every 8 min — keeps Render awake (spins down after 15 min)


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
            await asyncio.wait_for(asyncio.to_thread(database.cleanup_old_heartbeats, days=7), timeout=30)
            await asyncio.wait_for(asyncio.to_thread(database.cleanup_old_data, days=30), timeout=30)
            logger.debug("Cleaned up old records (heartbeats, shadow, slippage, learning)")

            # ── In-memory cache cleanup — prevent unbounded growth ────────
            # Keep only tickers with open positions
            try:
                _open_now = await asyncio.to_thread(database.get_open_trades)
                _open_tickers = {t["ticker"] for t in (_open_now or [])}
                # _price_history: remove closed tickers
                for _tk in list(_price_history.keys()):
                    if _tk not in _open_tickers:
                        _price_history.pop(_tk, None)
                # _smart_sell_low_count: remove closed tickers
                for _tk in list(_smart_sell_low_count.keys()):
                    if _tk not in _open_tickers:
                        _smart_sell_low_count.pop(_tk, None)
                # _smart_sell_last_check: remove closed tickers
                for _tk in list(_smart_sell_last_check.keys()):
                    if _tk not in _open_tickers:
                        _smart_sell_last_check.pop(_tk, None)
            except Exception:
                pass

            # ── Translation cache cleanup ─────────────────────────────────
            try:
                from translation_service import _cache as _trans_cache
                deleted = await asyncio.to_thread(_trans_cache.cleanup_expired)
                if deleted > 0:
                    logger.debug(f"Translation cache: deleted {deleted} expired entries")
            except Exception as e:
                logger.debug(f"Translation cache cleanup failed: {e}")

            # _price_alerts_fired: cap at 500 entries (old alerts don't matter)
            if len(_price_alerts_fired) > 500:
                old_count = len(_price_alerts_fired)
                _price_alerts_fired.clear()
                logger.debug(f"Cleared _price_alerts_fired ({old_count} entries)")

            # _position_alert_sent: remove near_tp_* keys for closed trades
            closed_ids = set()
            try:
                closed = await asyncio.to_thread(database.get_closed_trades_ids) if hasattr(database, 'get_closed_trades_ids') else []
                closed_ids = {str(t) for t in (closed or [])}
            except Exception:
                pass
            stale_keys = [k for k in list(_position_alert_sent.keys())
                          if k.startswith("near_tp_") and k.split("_")[-1] in closed_ids]
            for k in stale_keys:
                _position_alert_sent.pop(k, None)
            if len(_position_alert_sent) > 200:
                # safety valve — clear all if too many accumulate
                _position_alert_sent.clear()
                logger.debug("Cleared _position_alert_sent (overflow)")

            # _partial_sell_done: cap at 1000 entries
            if len(_partial_sell_done) > 1000:
                _partial_sell_done.clear()
                logger.debug("Cleared _partial_sell_done (overflow)")

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

                # Also fetch community sentiment score and alert if significant
                try:
                    sentiment_score = await fetch_community_sentiment(ticker)
                    if sentiment_score is not None:
                        if sentiment_score >= 7:
                            _create_background_task(notify_sentiment_alert(
                                ticker=ticker,
                                sentiment_score=sentiment_score,
                                direction="bullish"
                            ))
                        elif sentiment_score <= 3:
                            _create_background_task(notify_sentiment_alert(
                                ticker=ticker,
                                sentiment_score=sentiment_score,
                                direction="bearish"
                            ))
                except Exception as sentiment_err:
                    logger.debug(f"Community sentiment fetch for {ticker} failed: {sentiment_err}")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Sentiment monitor error: {e}")
            _create_background_task(notify_error("loop_error", "", f"שגיאה ב-sentiment_monitor"))


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
    lim_sell = limit_sell_price(cur_price, ticker)

    # ── Pre-action notification — tell user BEFORE executing ─────────────────
    _entry   = float(trade.get("entry_price") or cur_price)
    _qty     = float(trade.get("qty") or 0)
    _pnl_est = (cur_price - _entry) * _qty
    _pct_est = (cur_price - _entry) / _entry * 100 if _entry else 0
    _action_labels = {
        "stop_loss":   "🛑 סטופ לוס הופעל",
        "take_profit": "🎯 יעד רווח הושג",
        "smart_sell":  "🧠 מכירה חכמה (ציון נפל)",
        "news_exit":   "📰 יציאה בגלל חדשות שליליות",
        "time_exit":   "⏱ יציאה לפי זמן",
        "eod_sweep":   "🌙 ניקוי סוף יום",
    }
    _action_label = _action_labels.get(status, f"📌 {status}")
    _pnl_icon = "💚" if _pnl_est >= 0 else "❤️"
    try:
        from telegram_chat import _fmt_price as _fp_pre
        _cp_str = _fp_pre(cur_price)
        _ep_str = _fp_pre(_entry)
    except Exception:
        _cp_str = f"${cur_price:.2f}"
        _ep_str = f"${_entry:.2f}"
    _create_background_task(send_message(
        f"⚡ <b>הבוט עומד למכור — {ticker}</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"📌  {_action_label}\n"
        f"💵  מחיר עכשיו:  {_cp_str}\n"
        f"📍  מחיר קנייה:  {_ep_str}\n"
        f"{_pnl_icon}  רווח/הפסד משוער: <b>${_pnl_est:+.2f}</b>  ({_pct_est:+.1f}%)\n"
        f"🔢  כמות: {_qty} מניות"
    ))

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
            _create_background_task(notify_error("stop_loss_fail", ticker, f"timeout + retry נכשל"))
            return False
    except TypeError as sig_err:
        # Signature mismatch (e.g. broker wrapper missing price param) — retry without price
        logger.warning(f"[SELL] {ticker}: signature error ({sig_err}), retrying without price...")
        try:
            order = await asyncio.wait_for(
                asyncio.to_thread(broker.submit_sell, ticker), timeout=30
            )
        except Exception as retry_err:
            _create_background_task(notify_error("stop_loss_fail", ticker, f"חתימה שגויה + retry נכשל"))
            return False
    except Exception as sell_err:
        _create_background_task(notify_error("stop_loss_fail", ticker, f"שגיאת מכירה"))
        return False

    exit_price  = float(order.get("price") or lim_sell)
    # Use cur_price as fallback (same as pre-notification) — NOT exit_price which makes PnL=0
    _entry      = float(trade.get("entry_price") or cur_price or exit_price)
    if _entry <= 0:
        logger.error(f"[CLOSE] {ticker}: entry_price invalid — using exit_price (PnL will be 0)")
        _entry = exit_price
    pnl_gross   = (exit_price - _entry) * float(trade.get("qty") or 0)

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

    # ── Professional Drawdown Control ─────────────────────────────────
    try:
        from drawdown_control import record_trade_loss, record_trade_win, get_status as dd_status
        if pnl_gross < 0:
            # Calculate loss as % of budget
            from config import settings as _cfg_dd
            loss_pct = abs(pnl_gross) / _cfg_dd.MAX_BUDGET * 100
            dd_result = record_trade_loss(loss_pct)
            if dd_result.get("triggered"):
                _create_background_task(send_message(
                    f"🚨 <b>Drawdown Control מופעל</b>\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"❌ הסיבה: {dd_result.get('mode')}\n"
                    f"📉 הפסד יומי: {dd_result.get('daily_loss_pct', 0):.1f}%\n"
                    f"🔢 הפסדים רצופים: {dd_result.get('consecutive_losses', 0)}\n"
                    f"⏸️ עוצר קניות {dd_result.get('pause_hours', 0):.0f} שעות!"
                ))
        else:
            record_trade_win()
    except Exception as _dd_err:
        logger.debug(f"Drawdown control update failed: {_dd_err}")

    # ── Recovery Protocol: react to losses ────────────────────────────
    if pnl_gross < 0:
        try:
            from recovery_protocol import get_current_mode, tighten_stops_after_loss, send_recovery_status
            _rmode = get_current_mode()
            if _rmode in ("caution", "recovery"):
                _create_background_task(tighten_stops_after_loss())
                _create_background_task(send_recovery_status())
        except Exception as _rp_err:
            logger.debug(f"Recovery protocol reaction failed: {_rp_err}")

    # Fire-and-forget — Telegram failure must NOT prevent position cleanup or
    # cause _close_position to return False (trade is already closed in broker+DB)
    _create_background_task(notify_sell(ticker, exit_price, pnl_gross, label))

    # Cleanup per-ticker state so re-entry into the same ticker works correctly
    _smart_sell_last_check.pop(ticker, None)
    _position_alert_sent.pop(ticker, None)
    _smart_sell_low_count.pop(ticker, None)  # reset confirmation counter — prevents 1-shot trigger on re-entry
    # Clean up partial-sell guards to prevent memory leak over many trades
    _trade_id = trade.get("id")
    if _trade_id:
        _partial_sell_done.discard(f"{_trade_id}:s1")
        _partial_sell_done.discard(f"{_trade_id}:s2")

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
    MAX_HOLD_HOURS: float = float(_os.getenv("MAX_HOLD_HOURS", "24.0"))  # reduced from 48 → 24h (27% WR after 48h)
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

                    # ── PRE-CHECK NOTIFICATION — לפני בדיקת הפוזיציה ─────────────
                    # שולח פעם ב-10 דקות לכל פוזיציה
                    try:
                        import time as _t2
                        _check_key = f"pre_check_{trade['id']}"
                        if not hasattr(send_message, '_pre_check_last'):
                            send_message._pre_check_last = {}
                        if _t2.time() - send_message._pre_check_last.get(_check_key, 0) > 600:
                            send_message._pre_check_last[_check_key] = _t2.time()
                            _tv2 = f'https://www.tradingview.com/chart/?symbol={ticker}'
                            _atr_now = trade.get("atr_stop_price")
                            _stop_dist = f" | 🛑 {((cur_price-float(_atr_now))/cur_price*100):.1f}% לסטופ" if _atr_now else ""
                            _pl_now = float(position.get("unrealized_pl", 0))
                            _em = "🟢" if plpc >= 0 else "🔴"
                            _create_background_task(send_message(
                                f"🔎 <b>בודק פוזיציה — <a href=\"{_tv2}\">{ticker}</a></b>\n"
                                f"  {_em} מחיר: ${cur_price:.2f}  |  {plpc:+.1f}%  |  ${_pl_now:+.2f}{_stop_dist}"
                            ))
                    except Exception:
                        pass

                    # ── Live position status — דיווח על מצב פוזיציה פעם ב-15 דקות ──
                    try:
                        from action_log import notify_action as _na
                        _pos_key = f"pos_status_{trade['id']}"
                        import time as _t
                        _last_pos_report = getattr(_na, '_pos_last', {})
                        if _t.time() - _last_pos_report.get(_pos_key, 0) > 900:  # 15 min
                            if not hasattr(_na, '_pos_last'):
                                _na._pos_last = {}
                            _na._pos_last[_pos_key] = _t.time()
                            _tv = f'https://www.tradingview.com/chart/?symbol={ticker}'
                            _pl = float(position.get("unrealized_pl", 0))
                            _em = "🟢" if _pl >= 0 else "🔴"
                            _atr_s = trade.get("atr_stop_price")
                            _stop_line = f"\n🛑 Stop: ${_atr_s:.2f} ({((cur_price-_atr_s)/cur_price*100):.1f}% מרחק)" if _atr_s else ""
                            asyncio.create_task(send_message(
                                f"{_em} <b><a href=\"{_tv}\">{ticker}</a></b>  {plpc:+.1f}%  ${_pl:+.2f}{_stop_line}"
                            ))
                    except Exception:
                        pass

                    # ── 1. ATR Trailing Stop ──────────────────────────────────
                    atr_stop = trade.get("atr_stop_price")
                    high_wm  = trade.get("high_watermark") or trade["entry_price"]

                    # Initialise on first encounter (new trade or legacy trade)
                    if atr_stop is None:
                        try:
                            atr_stop, stop_meta = await asyncio.wait_for(
                                asyncio.to_thread(
                                    compute_initial_stop, ticker, trade["entry_price"]
                                ),
                                timeout=25,
                            )
                        except asyncio.TimeoutError:
                            logger.warning(
                                f"[ATR STOP] {ticker}: compute_initial_stop timed out — "
                                f"using fallback (entry × 0.96)"
                            )
                            atr_stop = trade["entry_price"] * 0.96
                            stop_meta = {"stop_pct": 4.0, "fallback": True}
                        high_wm = trade["entry_price"]
                        await asyncio.to_thread(
                            database.update_trade_stop, trade["id"], atr_stop, high_wm
                        )
                        logger.info(
                            f"[ATR STOP] {ticker}: initialised stop=${atr_stop:.2f} "
                            f"({stop_meta['stop_pct']:.2f}% from entry)"
                        )

                    # ── Pre-loss guardian: "אם לא אמכור — אהיה בהפסד" ──────────────
                    # If price is declining AND approaching entry from above → sell now
                    # This catches the "slowly bleeding" case before stop-loss fires.
                    try:
                        _ph2 = _price_history.get(ticker, [])
                        if len(_ph2) >= 2 and 0 < plpc < 1.0:
                            # In thin profit zone AND declining
                            _trending_down = _ph2[-1] < _ph2[-2]
                            # How far from stop loss? (% remaining buffer)
                            _stop_buffer = ((cur_price - atr_stop) / cur_price * 100) if atr_stop else 99
                            # If only 0.5% away from stop and declining → sell NOW
                            _preempt_key = f"preempt_{trade['id']}"
                            if _trending_down and _stop_buffer < 0.5 and not _position_alert_sent.get(_preempt_key):
                                _position_alert_sent[_preempt_key] = True
                                logger.info(
                                    f"[PRE-LOSS] {ticker}: plpc={plpc:+.1f}% declining, "
                                    f"stop buffer={_stop_buffer:.2f}% — selling before loss"
                                )
                                await _close_position(
                                    trade, cur_price, "smart_sell",
                                    f"מניעת הפסד: מחיר יורד לסטופ ({_stop_buffer:.2f}% מהסטופ)"
                                )
                                _create_background_task(send_message(
                                    f"⚡ <b>מכרתי לפני הפסד — {ticker}</b>\n"
                                    f"━━━━━━━━━━━━━━━━\n"
                                    f"📉 המחיר ירד ל-{plpc:+.1f}%\n"
                                    f"🛑 הסטופ נמצא {_stop_buffer:.2f}% מתחת\n"
                                    f"✅ יצאתי עם {plpc:+.2f}% — לא חיכיתי להפסד!"
                                ))
                                continue
                    except Exception:
                        pass

                    # ── Break-even lock: once price > entry + 0.5%, floor stop at entry ──
                    # Tightened: was 1.0%, now 0.5% — lock in sooner
                    _entry_price = trade["entry_price"]
                    _breakeven_trigger = float(_os.getenv("BREAKEVEN_TRIGGER_PCT", "0.5"))
                    if (atr_stop is not None
                            and atr_stop < _entry_price
                            and plpc >= _breakeven_trigger):
                        _be_stop = round(_entry_price * 1.002, 4)  # 0.2% above entry (covers slippage)
                        if _be_stop > atr_stop:
                            atr_stop = _be_stop
                            await asyncio.to_thread(
                                database.update_trade_stop, trade["id"], atr_stop, high_wm
                            )
                            logger.info(
                                f"[BE LOCK] {ticker}: stop locked to entry "
                                f"${_be_stop:.2f} (+{plpc:.1f}% | trigger={_breakeven_trigger}%)"
                            )

                    # ── Profit-fade exit: "רווח קטן > הפסד" ─────────────────────────
                    # If the position once reached profit_protect_peak% and has since
                    # faded back toward entry, sell now and keep the small gain.
                    # Better to book +0.3% than ride it back to -3%.
                    _profit_protect_enabled = _os.getenv("PROFIT_PROTECT_ENABLED", "true").lower() == "true"
                    if _profit_protect_enabled:
                        try:
                            _peak_pct = ((high_wm - _entry_price) / _entry_price * 100) if _entry_price > 0 else 0
                            _fade_threshold  = float(_os.getenv("PROFIT_PROTECT_PEAK_PCT",  "1.5"))  # had this much profit
                            _exit_floor_pct  = float(_os.getenv("PROFIT_PROTECT_FLOOR_PCT", "0.2"))  # exit at this profit
                            _pp_key = f"pp_{trade['id']}"
                            if (
                                _peak_pct >= _fade_threshold      # was in profit
                                and 0 <= plpc <= _exit_floor_pct  # profit is fading toward entry
                                and not _position_alert_sent.get(_pp_key)
                            ):
                                _position_alert_sent[_pp_key] = True
                                logger.info(
                                    f"[PROFIT-FADE] {ticker}: was +{_peak_pct:.1f}% → now +{plpc:.1f}% — selling to protect gain"
                                )
                                await _close_position(
                                    trade, cur_price, "smart_sell",
                                    f"הרווח נשחק ({_peak_pct:.1f}%→{plpc:.1f}%) — נועל רווח קטן"
                                )
                                _create_background_task(send_message(
                                    f"🔒 <b>נעלתי רווח קטן — {ticker}</b>\n"
                                    f"━━━━━━━━━━━━━━━━\n"
                                    f"📈 שיא: <b>+{_peak_pct:.1f}%</b>\n"
                                    f"💵 מכרתי ב: <b>+{plpc:.1f}%</b>\n"
                                    f"✅ עדיף רווח קטן מאשר להמתין ולהפסיד!\n"
                                    f"💡 הפסד הנמנע: ~{settings.STOP_LOSS_PCT:.1f}%"
                                ))
                                continue
                        except Exception as _pp_err:
                            logger.debug(f"[PROFIT-FADE] {ticker}: check error: {_pp_err}")

                    # ── Declining-momentum exit: sell when trending down while in profit ──
                    # If position has been in profit AND price is now declining for 2 checks,
                    # sell before it crosses below entry.
                    try:
                        _dm_enabled = _os.getenv("DECLINE_MOMENTUM_EXIT", "true").lower() == "true"
                        _peak_pct2 = ((high_wm - _entry_price) / _entry_price * 100) if _entry_price > 0 else 0
                        if _dm_enabled and _peak_pct2 >= 1.0 and 0 < plpc < 1.0:
                            # Was in profit, now falling — check price history for decline
                            _ph = _price_history.get(ticker, [])
                            if len(_ph) >= 3:
                                # All last 3 prices declining and still positive
                                _declining = all(_ph[i] > _ph[i+1] for i in range(len(_ph)-1))
                                _dm_key = f"dm_{trade['id']}"
                                if _declining and not _position_alert_sent.get(_dm_key):
                                    _position_alert_sent[_dm_key] = True
                                    logger.info(
                                        f"[DECLINE EXIT] {ticker}: declining from +{_peak_pct2:.1f}% → now +{plpc:.1f}% — exiting"
                                    )
                                    await _close_position(
                                        trade, cur_price, "smart_sell",
                                        f"ירידת מומנטום מהשיא (peak={_peak_pct2:.1f}%, now={plpc:.1f}%)"
                                    )
                                    continue
                    except Exception:
                        pass

                    # Trail the stop upward as price rises
                    try:
                        new_stop, new_wm, raised = await asyncio.wait_for(
                            asyncio.to_thread(
                                update_trailing_stop,
                                ticker, cur_price, atr_stop, high_wm, trade["entry_price"]
                            ),
                            timeout=20,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"[ATR STOP] {ticker}: update_trailing_stop timed out — keeping current"
                        )
                        new_stop, new_wm, raised = atr_stop, high_wm, False
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
                            # Stop-raise alerts — ENABLED (user wants all actions reported)
                            if not _is_quiet() and atr_stop > 0 and (new_stop - atr_stop) / atr_stop >= 0.005:
                                _entry   = trade["entry_price"]
                                _qty     = trade["qty"]
                                _pnl_now = (cur_price - _entry) * _qty
                                _pnl_pct = (cur_price - _entry) / _entry * 100
                                _stop_dist = (cur_price - new_stop) / cur_price * 100
                                # Estimate take profit (~3× stop distance)
                                try:
                                    _stop_dist_abs = abs(new_stop - _entry)  # always positive (handles break-even lock where stop > entry)
                                    _tp_price = round(_entry + _stop_dist_abs * 3, 2)
                                except Exception:
                                    _tp_price = 0
                                _pnl_label = "🟢 רווח" if _pnl_now >= 0 else "🔴 הפסד"
                                try:
                                    from telegram_chat import _fmt_price as _fp
                                    _ep = _fp(_entry)
                                    _cp = _fp(cur_price)
                                    _sp = _fp(new_stop)
                                    _tp = _fp(_tp_price) if _tp_price else "—"
                                    _pp = _fp(abs(_pnl_now))
                                except Exception:
                                    _ep = f"${_entry:.2f}"
                                    _cp = f"${cur_price:.2f}"
                                    _sp = f"${new_stop:.2f}"
                                    _tp = f"${_tp_price:.2f}" if _tp_price else "—"
                                    _pp = f"${abs(_pnl_now):.2f}"
                                # Is the new stop above or below entry? → profit or loss if triggered
                                _stop_in_profit = new_stop > _entry
                                _stop_label = "סטופ לוס ברווח 💚" if _stop_in_profit else "סטופ לוס בהפסד ❤️"
                                _create_background_task(send_message(
                                    f"🛡️ <b>סטופ לוס הועלה</b>  ✨\n"
                                    f"━━━━━━━━━━━━━━━━\n"
                                    f"💹  <b>{ticker}</b>  ·  {_qty} מניות\n\n"
                                    f"📌  מחיר קנייה:      {_ep}\n"
                                    f"🎯  יעד רווח:   {_tp}\n"
                                    f"🛑  {_stop_label}:  <b>{_sp}</b>\n\n"
                                    f"📍  מחיר עכשיו:   {_cp}  <b>({_pnl_pct:+.1f}%)</b>\n"
                                    f"{'💚' if _pnl_now >= 0 else '❤️'}  {'רווח' if _pnl_now >= 0 else 'הפסד'} כרגע:  <b>{_pp}</b>"
                                ))
                        atr_stop = new_stop
                        high_wm  = new_wm

                    # ── 1d. Stop Approaching Alert — alert when <20% from stop ──────
                    try:
                        if atr_stop and atr_stop > 0 and cur_price > atr_stop:
                            _stop_dist_pct = (cur_price - atr_stop) / cur_price * 100
                            if _stop_dist_pct < 1.5:  # within 1.5% of stop
                                from telegram_bot import notify_stop_approaching as _nsa
                                await asyncio.shield(asyncio.create_task(
                                    _nsa(ticker, cur_price, atr_stop,
                                         trade["entry_price"], _stop_dist_pct)
                                ))
                    except Exception:
                        pass

                    # ── 1c. Partial Take Profits ─────────────────────────────────────
                    # Sell 25% at +5%, +10%, +18% — lock in gains while letting winners run
                    _partial_enabled = _os.getenv("PARTIAL_EXITS_ENABLED", "true").lower() == "true"
                    if _partial_enabled and plpc >= 5.0:
                        try:
                            from partial_exit_engine import execute_partial_exit as _partial_exit
                            _partial_done = await asyncio.wait_for(
                                _partial_exit(trade, cur_price, plpc),
                                timeout=15,
                            )
                            if _partial_done:
                                # Re-fetch trade with updated qty
                                _fresh_trades = await asyncio.to_thread(database.get_open_trades)
                                for _ft in (_fresh_trades or []):
                                    if _ft["id"] == trade["id"]:
                                        trade = _ft
                                        break
                        except asyncio.TimeoutError:
                            logger.debug(f"[PARTIAL EXIT] {ticker}: timed out")
                        except Exception as _pe:
                            logger.debug(f"[PARTIAL EXIT] {ticker}: {_pe}")

                    # ── 1b. Smart Trailing — tighten stop at profit milestones ──────
                    # Accelerates trail at +3%, +5%, +8%, +12%, +20%
                    try:
                        from pro_exit_system import calculate_smart_trailing_stop as _smart_trail
                        from atr_stop import get_atr as _get_atr
                        _atr_val = 0.0
                        try:
                            _atr_val = float(await asyncio.wait_for(
                                asyncio.to_thread(_get_atr, ticker), timeout=5
                            ) or 0)
                        except Exception:
                            pass
                        _trail_result = _smart_trail(
                            entry_price=trade["entry_price"],
                            current_price=cur_price,
                            original_stop=atr_stop,
                            atr=_atr_val,
                            high_water_mark=high_wm,
                        )
                        if _trail_result["tightened"] and _trail_result["new_stop"] > atr_stop:
                            _tighter_stop = _trail_result["new_stop"]
                            await asyncio.to_thread(
                                database.update_trade_stop, trade["id"], _tighter_stop, high_wm
                            )
                            logger.info(
                                f"[SMART TRAIL] {ticker}: stop tightened "
                                f"${atr_stop:.2f} → ${_tighter_stop:.2f} "
                                f"({_trail_result['reason']})"
                            )
                            atr_stop = _tighter_stop
                    except Exception as _st_err:
                        logger.debug(f"[SMART TRAIL] {ticker}: {_st_err}")

                    # ── 1a. Minimum hold guard — never sell within MIN_HOLD_MINUTES ──
                    # Prevents immediate sell-after-buy caused by brief score dips or
                    # ATR stop calculated before price stabilises after fill.
                    _MIN_HOLD_MIN = int(_os.getenv("MIN_HOLD_MINUTES", "10"))
                    try:
                        from datetime import datetime, timezone as _tz0
                        _et0 = trade.get("entry_time")
                        if _et0:
                            _ed0 = datetime.strptime(str(_et0)[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=_tz0.utc)
                            _held_min = (datetime.now(_tz0.utc) - _ed0).total_seconds() / 60
                            if _held_min < _MIN_HOLD_MIN:
                                # Too new — skip all sell checks this cycle
                                logger.debug(f"[MIN HOLD] {ticker}: held {_held_min:.1f}m < {_MIN_HOLD_MIN}m — skip sell checks")
                                continue
                    except Exception:
                        pass

                    # ── 1b. Time-Based Exit — free capital after MAX_HOLD_HOURS ─────
                    # PROFIT BOOST: Let strong winners (>10%) run beyond MAX_HOLD_HOURS.
                    # ATR trailing stop will protect them. Cuts losses early but keeps winners.
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
                            # Profit-aware time exit: extend hold time if trade is profitable
                            #   - <0%  profit:        exit at MAX_HOLD_HOURS  (cut losers fast)
                            #   - 0-5% profit:        exit at MAX_HOLD_HOURS + 24h
                            #   - 5-10% profit:       exit at MAX_HOLD_HOURS + 48h
                            #   - 10%+ profit:        no time limit — trailing stop only
                            if plpc >= 10.0:
                                effective_max_hold = float('inf')   # let strong winners run
                            elif plpc >= 5.0:
                                effective_max_hold = MAX_HOLD_HOURS + 48
                            elif plpc >= 0:
                                effective_max_hold = MAX_HOLD_HOURS + 24
                            else:
                                effective_max_hold = MAX_HOLD_HOURS   # cut losers fast
                            if hours_held >= effective_max_hold:
                                logger.info(
                                    f"[TIME EXIT] {ticker}: held {hours_held:.1f}h "
                                    f"≥ {effective_max_hold:.0f}h (PnL={plpc:+.1f}%) — closing"
                                )
                                await _close_position(
                                    trade, cur_price, "time_exit",
                                    f"יציאה לפי זמן ({hours_held:.1f} שעות, רווח {plpc:+.1f}%)",
                                )
                                continue
                        except Exception as te:
                            logger.debug(f"[TIME EXIT] {ticker}: parse error: {te}")

                    # ── 1b. ATR Trailing Stop (flash-crash confirmed) ─────────
                    try:
                        flash_exit, flash_reason = await asyncio.wait_for(
                            asyncio.to_thread(
                                should_exit_confirmed, ticker, cur_price, atr_stop
                            ),
                            timeout=15,
                        )
                    except asyncio.TimeoutError:
                        flash_exit, flash_reason = False, "timeout — holding position"
                        logger.warning(f"[FLASH] {ticker}: should_exit_confirmed timed out")
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

                    # ── 1b0. Pre-Earnings Protection — close profitable positions before earnings ──
                    # מגן רווחים — אם יש דוח רווחים בעוד 1-2 ימים ויש רווח 1.5%+
                    # מוכר את החצי הראשון לפני התנודתיות הצפויה
                    try:
                        from earnings import check_earnings_risk as _cer
                        _ern_risky, _ern_reason, _ern_days = await asyncio.wait_for(
                            asyncio.to_thread(_cer, ticker), timeout=10
                        )
                        if _ern_risky and _ern_days is not None and _ern_days <= 2 and plpc >= 1.5:
                            # We have profit + earnings imminent → lock 50% of profit
                            _pre_e_key = f"pre_earn_{trade['id']}"
                            if not _position_alert_sent.get(_pre_e_key):
                                _position_alert_sent[_pre_e_key] = True
                                _half_qty = round(trade["qty"] * 0.5, 6)
                                try:
                                    await asyncio.wait_for(
                                        asyncio.to_thread(broker.submit_sell, ticker, _half_qty, cur_price),
                                        timeout=30,
                                    )
                                    _new_qty = round(trade["qty"] - _half_qty, 6)
                                    await asyncio.to_thread(database.update_trade_qty, trade["id"], _new_qty)
                                    trade = dict(trade); trade["qty"] = _new_qty
                                    _create_background_task(send_message(
                                        f"📑 <b>מגן לפני דוח — {ticker}</b>\n"
                                        f"━━━━━━━━━━━━━━━━\n"
                                        f"⚠️ דוח רווחים בעוד {_ern_days} ימים\n"
                                        f"💰 רווח נוכחי: {plpc:+.1f}%\n"
                                        f"🔢 מוכר 50% ({_half_qty} מניות) לנעילת רווח\n"
                                        f"📌 חצי שני נשאר עם stop בטוח"
                                    ))
                                    logger.info(f"[PRE-EARN] {ticker}: sold 50% before earnings in {_ern_days}d")
                                except Exception as _ee:
                                    logger.warning(f"[PRE-EARN] {ticker}: protection sale failed: {_ee}")
                    except Exception:
                        pass

                    # ── 1b1. Stagnant Position Exit — sell if flat/losing for 12h ──
                    # מניה שלא זזה (-0.5% ל-+1%) 12 שעות = הון מבוזבז + סיכון
                    try:
                        from datetime import datetime as _dt_sg, timezone as _tz_sg
                        _entry_ts = trade.get("entry_time")
                        if _entry_ts and -1.5 <= plpc <= 1.0:
                            _entry_dt = _dt_sg.strptime(str(_entry_ts)[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=_tz_sg.utc)
                            _hours_held = (_dt_sg.now(_tz_sg.utc) - _entry_dt).total_seconds() / 3600
                            # Negative/flat: exit after 12h. Small profit: exit after 18h.
                            _stagnant_hours = 12 if plpc < 0 else 18
                            if _hours_held >= _stagnant_hours:
                                logger.info(
                                    f"[STAGNANT] {ticker}: {_hours_held:.0f}h held, plpc={plpc:+.1f}% — selling stale position"
                                )
                                await _close_position(
                                    trade, cur_price, "time_exit",
                                    f"מניה תקועה {_hours_held:.0f}ש' ({plpc:+.1f}%) — משחרר הון"
                                )
                                _create_background_task(send_message(
                                    f"⏰ <b>מכרתי מניה תקועה — {ticker}</b>\n"
                                    f"━━━━━━━━━━━━━━━━\n"
                                    f"⏱️ החזקתי {_hours_held:.0f} שעות\n"
                                    f"💵 P&L: {plpc:+.1f}%\n"
                                    f"✅ שיחררתי הון לעסקה טובה יותר"
                                ))
                                continue
                    except Exception:
                        pass

                    # ── 1b2. Profit Milestone Alerts — celebrate winning positions! ──
                    # שולח התראה ב-+2%, +5%, +10%, +15% רווח (פעם אחת לכל יעד)
                    try:
                        for _milestone, _emoji, _msg in [
                            (2.0,  "🎯", "רווח נעול! Stage 1 מתקרב"),
                            (5.0,  "🚀", "רווח יפה — לוקח בקרוב חלק שני"),
                            (10.0, "💎", "רווח דו-ספרתי!"),
                            (15.0, "🏆", "רווח מצוין — מתקרבים ליעד מלא!"),
                        ]:
                            _ms_key = f"ms_{trade['id']}_{int(_milestone)}"
                            if plpc >= _milestone and not _position_alert_sent.get(_ms_key):
                                _position_alert_sent[_ms_key] = True
                                try:
                                    from telegram_chat import _fmt_price as _fpm
                                    _create_background_task(send_message(
                                        f"{_emoji} <b>{ticker} ב-+{plpc:.1f}%!</b>\n"
                                        f"━━━━━━━━━━━━━━━━\n"
                                        f"💵 מחיר עכשיו: {_fpm(cur_price)}\n"
                                        f"📌 נכנסתי ב: {_fpm(trade['entry_price'])}\n"
                                        f"💡 {_msg}"
                                    ))
                                except Exception:
                                    pass
                                break   # only one milestone per cycle
                    except Exception:
                        pass

                    # ── 1c. Near-TP Alert — warn when within 2% of take profit ─
                    try:
                        from atr_stop import _fetch_atr as _atr_near
                        _atr_near_val = await asyncio.wait_for(
                            asyncio.to_thread(_atr_near, ticker, trade["entry_price"]),
                            timeout=15,
                        )
                        _near_tp_pct  = min(
                            settings.TAKE_PROFIT_PCT,
                            max(4.0, (_atr_near_val / trade["entry_price"]) * 100 * 6)
                        )
                        _gap_to_tp = _near_tp_pct - plpc   # how far from TP
                        _near_key  = f"near_tp_{trade['id']}"
                        if 0 < _gap_to_tp <= 2.0 and not _position_alert_sent.get(_near_key):
                            _position_alert_sent[_near_key] = True
                            try:
                                from telegram_chat import _fmt_price as _fpp
                                _tp_price = round(trade["entry_price"] * (1 + _near_tp_pct/100), 2)
                                _create_background_task(send_message(
                                    f"🎯 <b>קרוב ליעד הרווח — {ticker}</b>\n"
                                    f"━━━━━━━━━━━━━━━━\n"
                                    f"📍 מחיר עכשיו: {_fpp(cur_price)} (רווח {plpc:+.1f}%)\n"
                                    f"🎯 יעד למכירה: {_fpp(_tp_price)} ({_near_tp_pct:.1f}%)\n"
                                    f"⏳ עוד {_gap_to_tp:.1f}% וזה ימכר ברווח!"
                                ))
                            except Exception:
                                pass
                    except Exception:
                        pass

                    # ── 2. Take Profit — ATR-based (6× ATR) or fixed ceiling ──
                    # ATR-based TP is achievable for low-vol stocks (MO ~2%, V ~4%)
                    # Fixed TP is the safety cap for high-vol stocks
                    try:
                        from atr_stop import _fetch_atr
                        _atr_val = await asyncio.wait_for(
                            asyncio.to_thread(_fetch_atr, ticker, trade["entry_price"]),
                            timeout=15,
                        )
                        _atr_tp_pct = min(
                            settings.TAKE_PROFIT_PCT,                    # cap at fixed TP
                            max(4.0, (_atr_val / trade["entry_price"]) * 100 * 6)  # 6×ATR — let winners run
                        )
                    except Exception:
                        _atr_tp_pct = settings.TAKE_PROFIT_PCT

                    # ── 2a. Progressive profit-zone tightening — multi-tier stop tightening ──
                    # +5%  → 1.2× ATR (lock most gains)
                    # +10% → 0.8× ATR (very tight)
                    # +15% → 0.5× ATR (extremely tight — almost guaranteed exit on next dip)
                    if plpc >= 5.0:
                        try:
                            from atr_stop import _fetch_atr as _atr_fn
                            import os as _os2
                            # Progressive multiplier — tighter as profits grow
                            if plpc >= 15.0:
                                _tight_mult = 0.5
                            elif plpc >= 10.0:
                                _tight_mult = 0.8
                            else:
                                _tight_mult = float(_os2.getenv("ATR_TIGHT_MULTIPLIER", "1.2"))
                            _atr2 = await asyncio.wait_for(
                                asyncio.to_thread(_atr_fn, ticker, trade["entry_price"]),
                                timeout=15,
                            )
                            _tight_stop = round(cur_price - _atr2 * _tight_mult, 4)
                            if _tight_stop > atr_stop:
                                atr_stop = _tight_stop
                                await asyncio.to_thread(
                                    database.update_trade_stop, trade["id"], atr_stop, high_wm
                                )
                                logger.info(
                                    f"[TIGHT TRAIL] {ticker}: progressive lock — stop=${atr_stop:.2f} "
                                    f"({_tight_mult}×ATR @ +{plpc:.1f}%)"
                                )
                        except Exception:
                            pass  # fail-open

                    # Scale-out exit plan (sell in thirds):
                    #   Stage 1 — sell 50% at 50% of ATR TP
                    #   Stage 2 — sell 25% (of original) at 80% of ATR TP
                    #   Stage 3 — sell remaining 25% at full ATR TP
                    # MAX WIN-RATE MODE: lock in at +1.5%, second lock at +4%
                    _stage1_pct = max(1.5, _atr_tp_pct * 0.15)  # 15% of full TP (≈1.5-2%)
                    _stage2_pct = max(3.0, _atr_tp_pct * 0.35)  # 35% of full TP (≈4-5%)
                    # Stage 1 is considered done when the high_watermark already reached
                    # the stage-1 level (price was above it at some point and partial was taken)
                    _stage1_done = bool(trade.get("high_watermark") and
                                        trade.get("high_watermark", 0) >= trade["entry_price"] * (1 + _stage1_pct / 100 + 0.001))
                    # Stage 2 is considered done only if Stage 1 is also done
                    # (prevents Stage 2 being True while Stage 1 is False, skipping Stage 1 entirely)
                    _stage2_done = _stage1_done and bool(trade.get("high_watermark") and
                                        trade.get("high_watermark", 0) >= trade["entry_price"] * (1 + _stage2_pct / 100 + 0.001))

                    _s1_guard_key = f"{trade['id']}:s1"
                    _s2_guard_key = f"{trade['id']}:s2"
                    if not _stage1_done and plpc >= _stage1_pct and _s1_guard_key not in _partial_sell_done:
                        # MAX-WIN MODE: Stage 1 sells 50% — lock in profit aggressively
                        _orig_qty = trade["qty"]
                        _half_qty = round(_orig_qty * 0.5, 6)
                        if _half_qty > 0:
                            _create_background_task(send_message(
                                f"💰 <b>נעלתי רווח — {ticker}</b>\n"
                                f"━━━━━━━━━━━━━━━━\n"
                                f"🎯 רווח: {plpc:+.1f}%\n"
                                f"💵 מחיר: ${cur_price:.2f}\n"
                                f"🔢 מכרתי: {_half_qty} מניות (50% מהפוזיציה)\n"
                                f"📌 חצי שני נשאר — הסטופ עולה לכניסה (אפס סיכון)"
                            ))
                            try:
                                _half_order = await asyncio.wait_for(
                                    asyncio.to_thread(broker.submit_sell, ticker, _half_qty, cur_price),
                                    timeout=30
                                )
                                _half_pnl = (cur_price - trade["entry_price"]) * _half_qty
                                # ── CRITICAL FIX: update DB so next cycle doesn't re-sell ──
                                _new_qty = round(_orig_qty - _half_qty, 6)
                                await asyncio.to_thread(
                                    database.update_trade_qty, trade["id"], _new_qty
                                )
                                trade = dict(trade)   # make mutable copy
                                trade["qty"] = _new_qty   # update in-memory too
                                # Tax + P&L tracking for partial close
                                try:
                                    from tax_tracker import process_trade_close as _ptc
                                    _ptc(trade["id"], _half_pnl)
                                except Exception:
                                    pass
                                record_trade_result(_half_pnl)   # update circuit breaker
                                # MAX-WIN MODE: After Stage 1, move stop AGGRESSIVELY to LOCK IN PROFIT
                                # Stop goes to entry+0.5% — second half can ONLY win (not lose)
                                _stop_at_winning = round(trade["entry_price"] * 1.005, 4)  # +0.5% above entry
                                atr_stop = max(atr_stop, _stop_at_winning)   # always raise, never lower
                                _s1_wm_pct   = max(_stage1_pct + 3.0, 10.0)
                                _s1_wm_mark  = round(trade["entry_price"] * (1 + _s1_wm_pct / 100), 4)
                                _s1_wm_final = max(cur_price, _s1_wm_mark)
                                try:
                                    await asyncio.to_thread(
                                        database.update_trade_stop, trade["id"], atr_stop, _s1_wm_final
                                    )
                                    high_wm = _s1_wm_final
                                except Exception as _wm_err:
                                    logger.critical(f"[PARTIAL TP S1] {ticker}: WATERMARK UPDATE FAILED: {_wm_err} — NOT marking done to prevent double-sell!")
                                    # Do NOT add to _partial_sell_done — let next cycle retry the sell
                                    # but skip adding guard so Stage1 can be retried safely
                                else:
                                    _partial_sell_done.add(_s1_guard_key)  # guard only on success
                                logger.info(f"[PARTIAL TP S1] {ticker}: sold 33% ({_half_qty} shares) "
                                            f"@ ${cur_price:.2f} (+{plpc:.1f}%) | PnL=${_half_pnl:+.2f} | remaining={_new_qty}")
                                _create_background_task(send_message(
                                    f"💰 <b>נעלתי רווח קטן — {ticker}</b>\n"
                                    f"━━━━━━━━━━━━━━━━\n"
                                    f"📊 מכרתי שליש מהפוזיציה ({_half_qty} מניות) ב-${cur_price:.2f}\n"
                                    f"📈 רווח: {plpc:+.1f}%\n"
                                    f"💚 הרווח על המכירה: <b>${_half_pnl:+.2f}</b>\n"
                                    f"📌 שני שלישים נשארים לרווח גדול יותר"
                                ))
                            except Exception as _pe:
                                logger.warning(f"[PARTIAL TP S1] {ticker}: half-sell failed: {_pe}")
                                # Do NOT continue — let ATR/TP/smart-sell still run this cycle
                            else:
                                # Only skip Smart Sell if the guard was successfully set
                                # (i.e. watermark write succeeded). If watermark failed,
                                # don't skip — Smart Sell can still run as a safety exit.
                                if _s1_guard_key in _partial_sell_done:
                                    continue  # sell + watermark OK — skip Smart Sell

                    elif _stage1_done and not _stage2_done and plpc >= _stage2_pct and _s2_guard_key not in _partial_sell_done:
                        # Stage 2: sell half of remaining (≈ 33% of original — symmetric thirds with Stage 1)
                        _orig_qty    = trade["qty"]   # already updated after Stage 1 (now ~67% of original)
                        _quarter_qty = round(_orig_qty * 0.5, 6)   # 50% of remaining ≈ 33% of original
                        if _quarter_qty > 0:
                            _create_background_task(send_message(
                                f"⚡ <b>נועל עוד רווח — {ticker}</b>\n"
                                f"━━━━━━━━━━━━━━━━\n"
                                f"📊 מוכר חצי ממה שנשאר ({_quarter_qty} מניות)\n"
                                f"📈 רווח: {plpc:+.1f}% | מחיר: ${cur_price:.2f}"
                            ))
                            try:
                                _s2_order = await asyncio.wait_for(
                                    asyncio.to_thread(broker.submit_sell, ticker, _quarter_qty, cur_price),
                                    timeout=30
                                )
                                _s2_pnl = (cur_price - trade["entry_price"]) * _quarter_qty
                                # ── CRITICAL FIX: update DB quantity ──
                                _new_qty = round(_orig_qty - _quarter_qty, 6)
                                await asyncio.to_thread(
                                    database.update_trade_qty, trade["id"], _new_qty
                                )
                                trade = dict(trade)
                                trade["qty"] = _new_qty
                                try:
                                    from tax_tracker import process_trade_close as _ptc2
                                    _ptc2(trade["id"], _s2_pnl)
                                except Exception:
                                    pass
                                record_trade_result(_s2_pnl)
                                # Mark Stage 2 done to prevent double-sell next cycle
                                _s2_wm_mark = round(trade["entry_price"] * (1 + _stage2_pct / 100 + 0.003), 4)
                                _s2_wm_final = max(cur_price, _s2_wm_mark)
                                try:
                                    await asyncio.to_thread(
                                        database.update_trade_stop, trade["id"], atr_stop, _s2_wm_final
                                    )
                                    high_wm = _s2_wm_final
                                except Exception as _wm2_err:
                                    logger.critical(f"[PARTIAL TP S2] {ticker}: WATERMARK UPDATE FAILED: {_wm2_err} — NOT marking done")
                                else:
                                    _partial_sell_done.add(_s2_guard_key)  # guard only on success
                                logger.info(f"[PARTIAL TP S2] {ticker}: sold 25% ({_quarter_qty} shares) "
                                            f"@ ${cur_price:.2f} (+{plpc:.1f}%) | PnL=${_s2_pnl:+.2f} | remaining={_new_qty}")
                                _create_background_task(send_message(
                                    f"💰 <b>נועל עוד רווח — {ticker}</b>\n"
                                    f"━━━━━━━━━━━━━━━━\n"
                                    f"🎯 רווח נוכחי: {plpc:+.1f}%\n"
                                    f"💵 מחיר עכשיו: ${cur_price:.2f}\n"
                                    f"🔢 מוכר עכשיו: {_quarter_qty} מניות (חצי ממה שנשאר)\n"
                                    f"💚 רווח על המכירה הזאת: <b>${_s2_pnl:+.2f}</b>\n"
                                    f"📌 הנותר ממשיך לעלות ליעד מלא"
                                ))
                            except Exception as _pe:
                                logger.warning(f"[PARTIAL TP S2] {ticker}: quarter-sell failed: {_pe}")
                                # Do NOT continue — let ATR stop / take-profit / smart-sell run this cycle
                            else:
                                continue  # sell succeeded — skip Smart Sell this cycle

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

                    # ── 2b. Momentum Exit — sell if declining 3 checks in a row while in profit ──
                    try:
                        _ph = _price_history.setdefault(ticker, [])
                        _ph.append(cur_price)
                        if len(_ph) > 3:
                            _ph.pop(0)  # keep only last 3 prices

                        if (len(_ph) == 3
                                and _ph[0] > _ph[1] > _ph[2]   # 3 consecutive lower prices
                                and plpc > 0.5):                 # still in profit (> 0.5%)
                            _drop_from_peak = (_ph[0] - _ph[2]) / _ph[0] * 100
                            logger.info(
                                f"[MOMENTUM EXIT] {ticker}: 3 consecutive lower prices "
                                f"({_ph[0]:.2f}→{_ph[1]:.2f}→{_ph[2]:.2f}) | "
                                f"still +{plpc:.1f}% profit — selling to lock in gains"
                            )
                            _price_history.pop(ticker, None)
                            await _close_position(
                                trade, cur_price, "momentum_exit",
                                f"ירידה רצופה 3 בדיקות | נעילת רווח {plpc:+.1f}%"
                            )
                            continue
                    except Exception:
                        pass

                    # ── 3. Smart Sell (score collapse, max once per 5 min) ────
                    import time as _time
                    global _smart_sell_lock
                    if _smart_sell_lock is None:  # ensure lock exists (first cycle)
                        _smart_sell_lock = asyncio.Lock()
                    # Hold lock ONLY to read/update timestamp — release before expensive I/O
                    # Check interval: faster when near break-even or in loss
                    _near_entry = abs(plpc) < 1.0   # within ±1% of entry
                    _check_interval = 90 if _near_entry else 240  # 90s near entry, 4 min otherwise
                    async with _smart_sell_lock:
                        last = _smart_sell_last_check.get(ticker, 0)
                        if _time.time() - last < _check_interval:
                            continue  # Skip if checked recently
                        _smart_sell_last_check[ticker] = _time.time()
                    # Lock released — do scoring outside the lock
                    try:
                        from scoring import get_composite_score
                        score_result = await asyncio.to_thread(
                            get_composite_score, ticker, 5
                        )
                        comp = score_result.get("composite_score")
                        if comp is None:
                            raise ValueError("composite_score missing from result")

                        # ── Adaptive threshold: exit sooner when score falls ──
                        # When in profit: protect gains aggressively
                        # When at/near entry: exit before it becomes a loss
                        # When in loss: exit fast (stop already handles big loss)
                        if plpc >= 8.0:
                            smart_threshold = 50   # was 45 — tighter
                        elif plpc >= 4.0:
                            smart_threshold = 45   # was 40
                        elif plpc >= 1.0:
                            smart_threshold = 42   # new: small profit — protect it
                        elif plpc >= -0.5:
                            smart_threshold = 48   # near breakeven: exit fast on bad score
                        else:
                            smart_threshold = 38   # already losing: quick exit if score bad

                        # ── No-confirmation for near-entry positions ──────────
                        # If near entry (±1%) and score bad → sell immediately, no wait
                        if comp < smart_threshold:
                            if _near_entry:
                                # Near entry + bad score = sell NOW (1 check)
                                logger.warning(
                                    f"[SMART SELL] {ticker}: score={comp} < {smart_threshold} "
                                    f"near entry ({plpc:+.1f}%) — immediate exit"
                                )
                                _smart_sell_low_count.pop(ticker, None)
                                await _close_position(
                                    trade, cur_price, "smart_sell",
                                    f"ציון נפל ({comp}/100) ליד כניסה — מניעת הפסד"
                                )
                                _create_background_task(send_message(
                                    f"🧠 <b>מכרתי לפני הפסד — {ticker}</b>\n"
                                    f"━━━━━━━━━━━━━━━━\n"
                                    f"📊 ציון נפל ל-{comp}/100\n"
                                    f"💵 P&L: {plpc:+.1f}%\n"
                                    f"✅ יצאתי בזמן לפני שהפך להפסד!"
                                ))
                            else:
                                # Not near entry: wait for 2 confirmations (noise filter)
                                _smart_sell_low_count[ticker] = _smart_sell_low_count.get(ticker, 0) + 1
                                if _smart_sell_low_count[ticker] >= 2:
                                    logger.warning(
                                        f"[SMART SELL] {ticker}: score={comp}/100 "
                                        f"(threshold={smart_threshold}, confirmed) — exiting"
                                    )
                                    _smart_sell_low_count.pop(ticker, None)
                                    await _close_position(
                                        trade, cur_price, "smart_sell",
                                        f"מכירה חכמה מאושרת (ציון={comp}/100)"
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
                                notify_error("stop_loss_fail", ticker, f"מכירה חכמה נכשלה")
                            )

                except Exception as e:
                    logger.error(f"Stop loss monitor error for {ticker}: {e}")
                    _create_background_task(notify_error("stop_loss_fail", ticker, f"שגיאה"))

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Stop loss monitor error: {e}")
            _create_background_task(notify_error("loop_error", "", f"שגיאה ב-stop_loss_monitor"))
            await asyncio.sleep(60)  # prevent CPU spin-loop on repeated DB/broker failure


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

            # ── Recovery Protocol check ────────────────────────────────────
            try:
                from recovery_protocol import get_current_mode, should_skip_buy_in_recovery
                _recovery_mode = get_current_mode()
                if should_skip_buy_in_recovery(_recovery_mode):
                    logger.info(f"AUTO-INVEST: Recovery mode={_recovery_mode} — skipping buys")
                    await asyncio.sleep(5 * 60)
                    continue
            except Exception as _rp_err:
                _recovery_mode = "normal"
                logger.debug(f"Recovery protocol check failed: {_rp_err}")

            # Only trade during market hours — timeout prevents loop hang on broker outage
            try:
                mkt_open = await asyncio.wait_for(
                    asyncio.to_thread(broker.is_market_open), timeout=15
                )
            except asyncio.TimeoutError:
                logger.warning("AUTO-INVEST: is_market_open timed out — skipping scan")
                await asyncio.sleep(5 * 60)
                continue
            if not mkt_open:
                logger.info("AUTO-INVEST: Market is closed, skipping scan")
                await asyncio.sleep(5 * 60)
                continue

            # Circuit breaker check — stop if daily loss limit exceeded
            ok, cb_reason = check_circuit_breaker()
            if not ok:
                logger.warning(f"AUTO-INVEST: {cb_reason} — skipping scan")
                await asyncio.sleep(5 * 60)
                continue

            # ── Time-of-day filter: avoid first/last 30 min (wide spreads, volatility) ──
            try:
                from trading_hours import _now_et  # DST-aware (zoneinfo)
                _et_now   = _now_et()
                _et_hour  = _et_now.hour
                _et_min   = _et_now.minute
                _minutes_since_open  = (_et_hour - 9) * 60 + _et_min - 30   # since 9:30 ET
                _minutes_before_close = (16 * 60) - (_et_hour * 60 + _et_min)  # to 4:00 ET
                # OPTIMIZED: block only first 15 min (volatile spread) + last 15 min (EOD)
                if _minutes_since_open < 15:
                    logger.info(f"AUTO-INVEST: First 15 min ({_et_hour:02d}:{_et_min:02d} ET) — skipping (wide spreads)")
                    await asyncio.sleep(5 * 60)
                    continue
                if _minutes_before_close < 15:
                    logger.info(f"AUTO-INVEST: Last 15 min before close — skipping (EOD volatility)")
                    await asyncio.sleep(5 * 60)
                    continue
            except Exception:
                pass

            # SPY trend guard — skip if overall market is in downtrend
            try:
                from indicators import get_market_conditions
                _mkt = await asyncio.wait_for(
                    asyncio.to_thread(get_market_conditions), timeout=20
                )
                if _mkt.get("spy_above_sma50") is False:
                    logger.info("AUTO-INVEST: SPY below SMA50 (downtrend) — skipping buys")
                    await asyncio.sleep(5 * 60)
                    continue
                _vix = _mkt.get("vix")
                # MAX-WIN MODE: was 30 (extreme fear), now 22 (mild fear) — only calm markets
                if _vix and _vix > 22:
                    logger.info(f"AUTO-INVEST: VIX={_vix:.1f} too high for max-win mode (max 22) — skipping buys")
                    await asyncio.sleep(5 * 60)
                    continue
            except asyncio.TimeoutError:
                logger.warning("AUTO-INVEST: market conditions timeout — proceeding fail-open")
            except Exception as _mkt_e:
                logger.debug(f"AUTO-INVEST: market conditions error ({type(_mkt_e).__name__}) — proceeding fail-open")

            # ── Event Memory: record today + read scenario signal (with timeouts) ──
            try:
                from event_memory import auto_record_today, get_scenario_signal, record_market_scenario
                await asyncio.wait_for(asyncio.to_thread(auto_record_today), timeout=20)
                await asyncio.wait_for(asyncio.to_thread(record_market_scenario), timeout=20)

                _sig, _sig_reason = await asyncio.wait_for(
                    asyncio.to_thread(get_scenario_signal), timeout=20
                )
                if _sig in ("bearish", "caution"):
                    logger.warning(f"[SCENARIO MEMORY] {_sig.upper()}: {_sig_reason}")
                elif _sig == "bullish":
                    logger.info(f"[SCENARIO MEMORY] BULLISH: {_sig_reason}")
            except asyncio.TimeoutError:
                logger.warning("[SCENARIO MEMORY] timed out — skipping")
            except Exception as _se:
                logger.debug(f"[SCENARIO MEMORY] {type(_se).__name__}")

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
                # ── PRE-WAIT NOTIFICATION ────────────────────────────────────
                try:
                    from live_reporter import send_scan_report as _sr3, send_hourly_pulse
                    await send_hourly_pulse()   # עדיין שולח pulse שעתי גם כשהשוק סגור
                    await _sr3(0, None, 0, hours_reason, 0, 0)
                except Exception:
                    pass
                await asyncio.sleep(5 * 60)
                continue

            # ── Event Memory signal: warn if similar events caused market drops ──
            try:
                from event_memory import get_event_signal
                from trading_hours import _FOMC_DATES, _ECONOMIC_DATES
                import datetime as _dt_ev
                _today_ev = _dt_ev.date.today()
                _ev_type  = None
                if _today_ev in _FOMC_DATES:
                    _ev_type = "FOMC"
                elif _today_ev in _ECONOMIC_DATES:
                    _ev_label = _ECONOMIC_DATES[_today_ev]
                    _ev_type  = "CPI" if "CPI" in _ev_label else ("NFP" if "NFP" in _ev_label else None)

                if _ev_type:
                    _caution, _ev_reason = get_event_signal(_ev_type)
                    if _caution == "high":
                        logger.warning(f"[EVENT MEMORY] {_ev_type} — HIGH caution: {_ev_reason}")
                        _create_background_task(send_message(
                            f"⚠️ <b>זיכרון אירועים: {_ev_type}</b>\n"
                            f"📉 בעבר השוק ירד ברוב הפעמים בצאת {_ev_type}:\n"
                            f"<i>{_ev_reason}</i>\n"
                            f"הבוט ימשיך לסרוק אבל יהיה זהיר יותר."
                        ))
                    elif _caution == "positive":
                        logger.info(f"[EVENT MEMORY] {_ev_type} — POSITIVE: {_ev_reason}")
            except Exception:
                pass

            logger.info("AUTO-INVEST: Starting scheduled scan with composite scoring...")

            # ── Live Reporter — שעתי ──────────────────────────────────────────
            try:
                from live_reporter import send_hourly_pulse, check_and_report_market_change
                await check_and_report_market_change()
                await send_hourly_pulse()
            except Exception:
                pass

            try:
                status = await _asyncio.wait_for(
                    _asyncio.to_thread(get_budget_status), timeout=20
                )
            except _asyncio.TimeoutError:
                logger.warning("AUTO-INVEST: budget status timed out — skipping cycle")
                await asyncio.sleep(60)
                continue
            remaining = float(status.get("cash_available", 0))

            # ── תנאי עצירה: רק חוסר מזומן — לא מגבלת מספר פוזיציות ──────────────
            # הבוט ממשיך לקנות כל עוד יש כסף בתקציב
            _min_pos_size = float(_os.getenv("MIN_POSITION_NOTIONAL", "200"))   # מינימום $200 לפוזיציה
            if remaining < _min_pos_size:
                logger.info(f"AUTO-INVEST: Not enough cash (${remaining:.2f} < ${_min_pos_size:.0f} min), skipping")
                try:
                    from live_reporter import send_scan_report as _sr2
                    await _sr2(0, None, 0, f"אין מזומן — כל התקציב מושקע! (${remaining:.0f} נותר)", 0, remaining)
                except Exception:
                    pass
            else:
                # Step 1: Shuffle watchlist for diversification — different stocks each scan
                shuffled = WATCHLIST.copy()
                random.shuffle(shuffled)
                # מספר הפוזיציות המקסימלי — ברירת מחדל 20 (כמה שהתקציב מאפשר)
                MAX_OPEN_POSITIONS = settings.MAX_OPEN_POSITIONS
                open_count = len(database.get_open_trades())
                if open_count >= MAX_OPEN_POSITIONS:
                    # Max positions reached — try rebalance or skip
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
                                _pos = await _asyncio.wait_for(
                                    _asyncio.to_thread(broker.get_position, _tsym), timeout=15
                                )
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

                # ── Advanced Momentum Pre-Filter (momentum_filter.py) ─────
                # Rank stocks by 5-day return + volume trend + SMA position
                # before expensive full scoring. Focus scan on best setups.
                try:
                    from momentum_filter import get_top_momentum_tickers
                    _cands_for_momentum = [
                        t for t in shuffled[:50]
                        if not database.get_open_trade_by_ticker(t)
                    ]
                    if len(_cands_for_momentum) >= 5:
                        _top_momentum = await _asyncio.wait_for(
                            get_top_momentum_tickers(_cands_for_momentum, top_n=20),
                            timeout=60,
                        )
                        # Add remaining at the end
                        _remaining = [t for t in shuffled if t not in _top_momentum]
                        shuffled = _top_momentum + _remaining
                        logger.info(
                            f"[MOMENTUM FILTER] Top tickers: {_top_momentum[:5]}"
                        )
                except Exception as _mf_err:
                    logger.debug(f"Momentum filter skipped: {_mf_err}")

                # Momentum pre-filter: quick 1-day % change check (cheap, no full scoring)
                _momentum_tickers = []
                _normal_tickers = []
                try:
                    import yfinance as _yf
                    _sample = [t for t in shuffled[:30] if not database.get_open_trade_by_ticker(t)]
                    if _sample:
                        # Wrap yfinance with timeout — prevents hangs
                        _prices = await _asyncio.wait_for(
                            _asyncio.to_thread(
                                _yf.download, _sample, period="2d",
                                progress=False, auto_adjust=True
                            ),
                            timeout=20
                        )
                        # yf.download returns MultiIndex columns for multiple tickers
                        # Check top-level with get_level_values to avoid always-False bug
                        _has_close = (
                            "Close" in _prices.columns.get_level_values(0)
                            if hasattr(_prices.columns, "get_level_values")
                            else "Close" in _prices.columns
                        )
                        if not _prices.empty and _has_close:
                            _close = _prices["Close"]
                            for _t in _sample:
                                try:
                                    _chg = float((_close[_t].iloc[-1] - _close[_t].iloc[-2]) / _close[_t].iloc[-2] * 100)
                                    if _chg > 1.5:        # strong momentum (top priority)
                                        _momentum_tickers.insert(0, _t)
                                    elif _chg > 0.5:      # mild momentum
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
                _bought_list: list[dict] = []   # collect all buys for one combined Telegram message

                # ── PRE-SCAN NOTIFICATION — לפני הסריקה ──────────────────────────
                try:
                    from datetime import datetime, timezone, timedelta as _td
                    _now_il = datetime.now(timezone.utc) + _td(hours=3)
                    _create_background_task(send_message(
                        f"🔍 <b>מתחיל סריקה</b> | {_now_il.strftime('%H:%M')}\n"
                        f"📊 בודק {len(candidates)} מניות\n"
                        f"💵 מזומן זמין: ${remaining:,.0f}\n"
                        f"⏳ <i>מחפש הזדמנויות...</i>"
                    ))
                except Exception:
                    pass

                # ── Action Log — התחלת סריקה ────────────────────────────────────
                try:
                    from action_log import start_scan as _start_scan
                    _start_scan(len(candidates), remaining)
                except Exception:
                    pass

                # ── SPY Gate — לא קונים כשהשוק יורד ────────────────────────────
                # אם SPY ירד >1% היום → לא קונים כלום
                try:
                    import yfinance as _yf_spy
                    _spy_hist = await _asyncio.wait_for(
                        _asyncio.to_thread(
                            lambda: _yf_spy.Ticker("SPY").history(period="2d", interval="1d")
                        ), timeout=10
                    )
                    if _spy_hist is not None and len(_spy_hist) >= 2:
                        _spy_chg = float(
                            (_spy_hist["Close"].iloc[-1] - _spy_hist["Close"].iloc[-2])
                            / _spy_hist["Close"].iloc[-2] * 100
                        )
                        if _spy_chg < -1.0:
                            logger.warning(
                                f"AUTO-INVEST: SPY down {_spy_chg:.2f}% today — "
                                "skipping all buys (market gate)"
                            )
                            _create_background_task(send_message(
                                f"🛡️ <b>SPY Gate פעיל</b>\n"
                                f"📉 S&P500 ירד <b>{_spy_chg:.1f}%</b> היום\n"
                                f"⛔ הבוט לא יקנה — ממתין לשוק טוב יותר\n"
                                f"✅ פוזיציות קיימות מוגנות עם Stop Loss"
                            ))
                            await _asyncio.sleep(5 * 60)
                            continue
                        elif _spy_chg < -0.5:
                            logger.info(f"AUTO-INVEST: SPY {_spy_chg:.2f}% — caution mode (smaller positions)")
                except Exception:
                    pass  # fail-open

                # ── PRE-FETCH: הורד נתוני OHLCV לכל המועמדים בקריאה אחת ──────
                # yfinance batch download is ~5x faster than N individual calls.
                # This populates the yfinance_cache so _score_candidate() never
                # waits for a download — it reads from in-memory cache.
                try:
                    from yfinance_cache import prefetch_batch as _prefetch
                    _t_prefetch_start = _time_module.time()
                    await _asyncio.wait_for(
                        _asyncio.to_thread(_prefetch, candidates, "3mo"),
                        timeout=30,
                    )
                    logger.info(
                        f"AUTO-INVEST: prefetch {len(candidates)} tickers in "
                        f"{_time_module.time()-_t_prefetch_start:.1f}s"
                    )
                except Exception as _pf_err:
                    logger.debug(f"AUTO-INVEST: prefetch failed (non-critical): {_pf_err}")

                # ── שלב 1: ציון מקבילי לכל המועמדים (PARALLEL) ──────────────
                # סורקים את כולם בו-זמנית → הרבה יותר מהיר
                _scored_candidates: list[tuple[str, float, object, object]] = []  # (ticker, score, composite, sentiment)

                async def _score_candidate(ticker: str):
                    """Score a single candidate — runs in parallel."""
                    try:
                        from action_log import log_ticker

                        # Earnings blackout check
                        try:
                            from earnings import check_earnings_risk
                            earn_risky, earn_reason, earn_days = await _asyncio.wait_for(
                                _asyncio.to_thread(check_earnings_risk, ticker), timeout=10
                            )
                            if earn_risky:
                                logger.info(f"AUTO-INVEST: {ticker} EARNINGS BLACKOUT — {earn_reason}")
                                log_ticker(ticker, 0, False, f"דוח בעוד {earn_days}d — מסוכן")
                                return None
                        except Exception:
                            pass

                        sentiment = await _asyncio.wait_for(
                            _asyncio.to_thread(score_sentiment, ticker), timeout=60
                        )
                        composite = await _asyncio.wait_for(
                            _asyncio.to_thread(get_composite_score, ticker, sentiment.score), timeout=60
                        )
                        score = composite["composite_score"]
                        logger.info(f"AUTO-INVEST: {ticker} → {score}/100 ({'✅ BUY' if composite['should_buy'] else '❌ SKIP'})")

                        if not composite["should_buy"]:
                            reason = composite.get("hard_block_reason") or f"ציון {score:.0f} < {composite.get('min_score', 70)}"
                            log_ticker(ticker, score, False, reason)
                            return None

                        # ── PROFIT BOOST: Skip overbought entries (RSI too high) ─────
                        # Data shows: late entries = small or no upside, but full downside risk
                        _ind_chk = composite.get("indicators", {})
                        _rsi_chk = float(_ind_chk.get("rsi") or 50.0)
                        _max_rsi = float(_os.getenv("MAX_RSI_FOR_ENTRY", "72"))
                        if _rsi_chk > _max_rsi:
                            logger.info(f"AUTO-INVEST: {ticker} RSI {_rsi_chk:.0f} > {_max_rsi:.0f} — overbought, skipping")
                            log_ticker(ticker, score, False, f"RSI={_rsi_chk:.0f} — overbought")
                            return None

                        # AI Score Enhancement
                        try:
                            from score_enhancer import enhance_score as _enhance_score
                            _ind = composite.get("indicators", {})
                            _enhancement = await _asyncio.wait_for(
                                _enhance_score(
                                    ticker=ticker,
                                    base_score=score,
                                    rsi=_ind.get("rsi", 50.0),
                                    macd=_ind.get("macd", 0.0),
                                    volume_ratio=_ind.get("volume_ratio", 1.0),
                                    sentiment_score=sentiment.score if hasattr(sentiment, "score") else 5.0,
                                ),
                                timeout=45,
                            )
                            if _enhancement.get("skip_trade"):
                                logger.info(f"AUTO-INVEST: {ticker} AI skip — {_enhancement['skip_reason']}")
                                log_ticker(ticker, score, False, f"AI: {_enhancement['skip_reason']}")
                                return None
                            score = _enhancement.get("enhanced_score", score)
                        except Exception:
                            pass

                        # Buffett Quality Filter
                        _buffett_score = 50
                        try:
                            from buffett_analysis import get_buffett_analysis as _bff
                            _buf = await _asyncio.wait_for(
                                _asyncio.to_thread(_bff, ticker), timeout=15
                            )
                            _buffett_score = _buf.get("score", 50)
                            if _buffett_score < 50:
                                logger.info(f"AUTO-INVEST: {ticker} Buffett={_buffett_score:.0f} below quality bar — skip")
                                log_ticker(ticker, score, False, f"Buffett {_buffett_score:.0f}/100 — איכות נמוכה")
                                return None
                        except Exception:
                            pass

                        # News boost
                        _news_boost = 0
                        if hasattr(sentiment, 'score') and sentiment.score >= 9:
                            _news_boost = 8
                        elif hasattr(sentiment, 'score') and sentiment.score >= 8:
                            _news_boost = 5
                        _effective_score = score + _news_boost
                        if _effective_score < 60:
                            log_ticker(ticker, score, False, f"ציון {_effective_score:.0f} < 60")
                            return None

                        # Intraday momentum — don't buy falling knife
                        try:
                            _intra = await _asyncio.wait_for(
                                _asyncio.to_thread(broker.get_price, ticker), timeout=10
                            )
                            if _intra:
                                _intra_ind = composite.get("indicators", {})
                                _prev_close = _intra_ind.get("prev_close") or _intra_ind.get("close")
                                if _prev_close and _prev_close > 0:
                                    _day_chg = (_intra - _prev_close) / _prev_close * 100
                                    if _day_chg < -1.5:
                                        logger.info(f"AUTO-INVEST: {ticker} down {_day_chg:.1f}% — knife falling, skip")
                                        log_ticker(ticker, score, False, f"יורד {_day_chg:.1f}% היום — סכין נופל")
                                        return None
                        except Exception:
                            pass

                        # Pro Entry gate
                        try:
                            from pro_entry_system import pro_entry_gate as _pro_gate
                            _pro_result = await _asyncio.wait_for(_pro_gate(ticker, score), timeout=30)
                            if not _pro_result.get("should_enter", True):
                                _skip_r = _pro_result.get('skip_reason', f"דרגה {_pro_result.get('grade', '?')}")
                                logger.info(f"AUTO-INVEST: {ticker} PRO BLOCKED — {_skip_r}")
                                log_ticker(ticker, score, False, f"Pro Gate: {_skip_r}")
                                return None
                            score = max(0, min(100, score + _pro_result.get("score_adjustment", 0)))
                        except Exception:
                            pass

                        # Pre-buy checklist
                        try:
                            from pre_buy_checklist import run_pre_buy_checklist
                            _ind3 = composite.get("indicators", {})
                            _checklist = await _asyncio.wait_for(
                                run_pre_buy_checklist(
                                    ticker=ticker, score=score,
                                    rsi=_ind3.get("rsi", 50),
                                    volume_ratio=_ind3.get("volume_ratio", 1.0),
                                    above_sma50=_ind3.get("above_sma50", True),
                                    above_sma200=_ind3.get("above_sma200", True),
                                    open_positions_count=len(database.get_open_trades() or []),
                                ), timeout=20,
                            )
                            if not _checklist.get("pass"):
                                _failed_c = ", ".join(_checklist.get("failed_checks", []))[:40]
                                log_ticker(ticker, score, False, f"Checklist: {_failed_c}")
                                return None
                            score = min(100, score + _checklist.get("confidence_boost", 0))
                        except Exception:
                            pass

                        # Learning block check
                        try:
                            from learning import should_override_buy as _sob
                            from indicators import get_current_indicators as _gci
                            _ind4 = await _asyncio.wait_for(_asyncio.to_thread(_gci, ticker), timeout=15) or {}
                            _block, _block_reason = _sob(ticker, _ind4)
                            if _block:
                                logger.info(f"AUTO-INVEST: {ticker} blocked by learning — {_block_reason}")
                                log_ticker(ticker, score, False, f"למידה: {_block_reason[:40]}")
                                return None
                        except Exception:
                            pass

                        # ✅ עבר את כל הפילטרים
                        _buffett_str = f"Buffett {_buffett_score:.0f}" if _buffett_score != 50 else ""
                        log_ticker(ticker, score, True, extra=_buffett_str)

                        _combined = score * 0.6 + _buffett_score * 0.4
                        return (ticker, _combined, composite, sentiment, _buffett_score)

                    except Exception as _err:
                        logger.debug(f"AUTO-INVEST: {ticker} parallel score failed — {type(_err).__name__}")
                        return None

                # ── סריקה מקבילית — כל המועמדים בו-זמנית ──────────────────────
                _parallel_results = await _asyncio.gather(
                    *[_score_candidate(t) for t in candidates],
                    return_exceptions=False
                )
                _scored_candidates = [r for r in _parallel_results if r is not None]

                # (keep old sequential loop as dead code marker — replaced by parallel above)
                for ticker in []:
                    if remaining < 10:
                        break
                    try:
                        # Earnings blackout check
                        try:
                            from earnings import check_earnings_risk
                            earn_risky, earn_reason, earn_days = await _asyncio.wait_for(
                                _asyncio.to_thread(check_earnings_risk, ticker), timeout=10
                            )
                            if earn_risky:
                                logger.info(f"AUTO-INVEST: {ticker} EARNINGS BLACKOUT — {earn_reason}")
                                continue
                        except Exception:
                            pass

                        sentiment = await _asyncio.wait_for(
                            _asyncio.to_thread(score_sentiment, ticker), timeout=60
                        )
                        composite = await _asyncio.wait_for(
                            _asyncio.to_thread(get_composite_score, ticker, sentiment.score), timeout=60
                        )
                        score = composite["composite_score"]
                        logger.info(f"AUTO-INVEST: {ticker} → {score}/100 ({'✅ BUY' if composite['should_buy'] else '❌ SKIP'})")

                        if not composite["should_buy"]:
                            continue

                        # ── AI Score Enhancement — ML + Patterns + MTF + News ──────────
                        try:
                            from score_enhancer import enhance_score as _enhance_score
                            _ind = composite.get("indicators", {})
                            _enhancement = await _asyncio.wait_for(
                                _enhance_score(
                                    ticker=ticker,
                                    base_score=score,
                                    rsi=_ind.get("rsi", 50.0),
                                    macd=_ind.get("macd", 0.0),
                                    volume_ratio=_ind.get("volume_ratio", 1.0),
                                    sentiment_score=sentiment.score if hasattr(sentiment, "score") else 5.0,
                                ),
                                timeout=45,   # generous timeout for parallel AI calls
                            )

                            # Skip if AI signals say avoid
                            if _enhancement.get("skip_trade"):
                                logger.info(
                                    f"AUTO-INVEST: {ticker} AI skip — {_enhancement['skip_reason']}"
                                )
                                continue

                            # Use enhanced score
                            _enhanced = _enhancement.get("enhanced_score", score)
                            _adj = _enhancement.get("adjustment", 0)
                            if _adj != 0:
                                logger.info(
                                    f"AUTO-INVEST: {ticker} AI enhancement: "
                                    f"{score:.1f} → {_enhanced:.1f} ({_adj:+.1f} pts)"
                                )
                                # Notify on big adjustments
                                if abs(_adj) >= 5:
                                    try:
                                        from telegram_bot import notify_score_enhancement
                                        _create_background_task(
                                            notify_score_enhancement(
                                                ticker=ticker,
                                                original_score=score,
                                                enhanced_score=_enhanced,
                                                adjustment=_adj,
                                                skip_trade=False,
                                                skip_reason="",
                                                signals=_enhancement.get("signals", {}),
                                            )
                                        )
                                    except Exception:
                                        pass
                            score = _enhanced

                        except (_asyncio.TimeoutError, Exception) as _enh_err:
                            # Fail-open: use original score if enhancement fails
                            logger.debug(f"AUTO-INVEST: {ticker} enhancement skipped ({type(_enh_err).__name__})")

                        # ── Buffett Quality Filter — MAX-WIN MODE: only quality companies ──
                        # קונה רק חברות עם Buffett >= 50 (איכות בינונית+)
                        _buffett_score = None
                        try:
                            from buffett_analysis import get_buffett_analysis as _bff
                            _buf = await _asyncio.wait_for(
                                _asyncio.to_thread(_bff, ticker), timeout=15
                            )
                            _buffett_score = _buf.get("score", 50)
                            # STRICTER: was 30, now 50 — require above-average quality
                            if _buffett_score < 50:
                                logger.info(f"AUTO-INVEST: {ticker} Buffett={_buffett_score:.0f} below quality bar (50) — skip")
                                continue
                        except Exception:
                            _buffett_score = 50   # neutral if Buffett analysis fails

                        # ── Technical Score Floor — MAX-WIN MODE: require very strong score ──
                        # EXCEPTION: very positive news (sentiment 8+) → score boost of 5 allowed
                        _news_boost = 0
                        if hasattr(sentiment, 'score') and sentiment.score >= 8:
                            _news_boost = 5
                            logger.info(f"AUTO-INVEST: {ticker} positive news ({sentiment.score}/10) → +5 boost")
                        elif hasattr(sentiment, 'score') and sentiment.score >= 9:
                            _news_boost = 8
                            logger.info(f"AUTO-INVEST: {ticker} extremely positive news ({sentiment.score}/10) → +8 boost")
                        _effective_score = score + _news_boost
                        if _effective_score < 60:
                            logger.info(f"AUTO-INVEST: {ticker} score={score:.0f}+news{_news_boost} below max-win threshold (60) — skip")
                            continue

                        # ── Intraday momentum filter — relaxed (allow flat/mild dip) ──
                        # Buying into FALLING knife has poor WR, but flat/mild dip is OK
                        try:
                            _intra = await _asyncio.wait_for(
                                _asyncio.to_thread(broker.get_price, ticker), timeout=10
                            )
                            if _intra:
                                _intra_ind = composite.get("indicators", {})
                                _prev_close = _intra_ind.get("prev_close") or _intra_ind.get("close")
                                if _prev_close and _prev_close > 0:
                                    _day_chg = (_intra - _prev_close) / _prev_close * 100
                                    if _day_chg < -1.5:    # was -0.5, now -1.5 (more permissive)
                                        logger.info(f"AUTO-INVEST: {ticker} down {_day_chg:.1f}% today — skip (knife falling)")
                                        continue
                        except Exception:
                            pass  # fail-open — don't block on price fetch error

                        # ── Professional Entry Analysis (pro_entry_system) ────
                        try:
                            from pro_entry_system import pro_entry_gate as _pro_gate
                            _pro_result = await _asyncio.wait_for(
                                _pro_gate(ticker, score), timeout=30
                            )
                            if not _pro_result.get("should_enter", True):
                                _skip_reason = _pro_result.get("skip_reason", "Pro gate")
                                logger.info(f"AUTO-INVEST: {ticker} PRO BLOCKED — {_skip_reason}")
                                continue
                            # Apply score adjustment from professional analysis
                            _pro_adj = _pro_result.get("score_adjustment", 0)
                            if abs(_pro_adj) > 0:
                                score = max(0, min(100, score + _pro_adj))
                                logger.info(f"AUTO-INVEST: {ticker} Pro grade={_pro_result.get('grade')} adj={_pro_adj:+.0f} → {score:.0f}")
                        except (_asyncio.TimeoutError, Exception) as _pro_err:
                            logger.debug(f"Pro gate skipped: {type(_pro_err).__name__}")

                        # ── Drawdown Control ───────────────────────────────────
                        try:
                            from drawdown_control import get_drawdown_mode, get_size_multiplier
                            _dd_mode = get_drawdown_mode()
                            if _dd_mode == "PAUSE":
                                logger.info(f"AUTO-INVEST: Drawdown PAUSE active — no new buys")
                                break  # Stop scanning entirely when in drawdown pause
                        except Exception:
                            pass

                        # ── Pre-Buy Checklist — 7 final quality gates ─────────
                        try:
                            from pre_buy_checklist import run_pre_buy_checklist
                            _ind3 = composite.get("indicators", {})
                            _open_count = len(database.get_open_trades() or [])
                            _checklist = await _asyncio.wait_for(
                                run_pre_buy_checklist(
                                    ticker=ticker,
                                    score=score,
                                    rsi=_ind3.get("rsi", 50),
                                    volume_ratio=_ind3.get("volume_ratio", 1.0),
                                    above_sma50=_ind3.get("above_sma50", True),
                                    above_sma200=_ind3.get("above_sma200", True),
                                    open_positions_count=_open_count,
                                ),
                                timeout=20,
                            )
                            if not _checklist.get("pass"):
                                _failed = ", ".join(_checklist.get("failed_checks", []))[:100]
                                logger.info(f"AUTO-INVEST: {ticker} BLOCKED by checklist — {_failed}")
                                continue
                            # Apply confidence boost to score
                            _boost = _checklist.get("confidence_boost", 0)
                            if _boost > 0:
                                score = min(100, score + _boost)
                                logger.info(f"AUTO-INVEST: {ticker} checklist boost +{_boost} → score={score:.1f}")
                        except (_asyncio.TimeoutError, Exception) as _cl_err:
                            logger.debug(f"AUTO-INVEST: {ticker} checklist skipped ({type(_cl_err).__name__})")

                        # Learning check — wrap with timeout to prevent yfinance hangs
                        try:
                            from learning import should_override_buy as _sob
                            from indicators import get_current_indicators as _gci
                            _ind = await _asyncio.wait_for(
                                _asyncio.to_thread(_gci, ticker), timeout=15
                            ) or {}
                            _block, _block_reason = _sob(ticker, _ind)
                            if _block:
                                logger.info(f"AUTO-INVEST: {ticker} blocked by learning — {_block_reason}")
                                continue
                        except _asyncio.TimeoutError:
                            logger.warning(f"AUTO-INVEST: {ticker} indicators timed out (15s) — skipping learning check")
                        except Exception as _le:
                            logger.debug(f"AUTO-INVEST: {ticker} learning error — {type(_le).__name__}")

                        # Combined score: 60% technical + 40% Buffett quality
                        _combined_score = score * 0.6 + (_buffett_score or 50) * 0.4
                        _scored_candidates.append((ticker, _combined_score, composite, sentiment, _buffett_score or 50))
                    except _asyncio.TimeoutError:
                        logger.warning(f"AUTO-INVEST: {ticker} scoring timed out — skipping")
                    except Exception as _se:
                        logger.debug(f"AUTO-INVEST: {ticker} scoring error — {type(_se).__name__}")

                # ממיינים לפי ציון משוקלל יורד — הטוב ביותר ראשון
                _scored_candidates.sort(key=lambda x: x[1], reverse=True)
                logger.info(f"AUTO-INVEST: {len(_scored_candidates)} candidates above threshold, best-first: "
                            + ", ".join(f"{t}=ציון{s:.0f}(באפט{b:.0f})" for t, s, _, _, b in _scored_candidates[:3]))

                # ── Action Log — שלח דוח סריקה מלא ─────────────────────────────
                try:
                    from action_log import flush_scan_report as _flush, log_event as _log_ev
                    if not _scored_candidates:
                        _log_ev("💤", "אין מניות עם ציון מספיק — ממשיך לסריקה הבאה")
                    await _flush()
                except Exception:
                    pass

                # ── שלב 2: בדיקות מלאות וקנייה — לפי סדר ציון יורד ────────────
                for ticker, score, composite, sentiment, _buffett_score in _scored_candidates:
                    if remaining < 10:
                        break
                    try:
                        _vol_ratio: float | None = None

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

                        # Sector diversification — max 2 stocks per sector
                        try:
                            from sector_guard import check_sector_concentration as _sector_check
                            _open_trades_now = database.get_open_trades() or []
                            _open_tickers_now = [_ot["ticker"] for _ot in _open_trades_now]
                            _sector_result = _sector_check(ticker, _open_tickers_now)
                            if not _sector_result["allowed"]:
                                logger.info(f"AUTO-INVEST: {ticker} SECTOR BLOCKED — {_sector_result['reason']}")
                                continue
                        except Exception:
                            pass  # fail-open

                        try:
                            from sector_rotation import get_sector_for_ticker as _sector_of

                            # Avoid defensive/commodity sectors in bull market
                            _WEAK_SECTORS_IN_BULL = {
                                "XLE",   # אנרגיה (GLD, GDX)
                                "GLD",   # זהב
                                "XLV",   # בריאות
                                "XLU",   # תשתיות
                                "XLP",   # צרכנות בסיסית (MCD)
                            }
                            # ETFs now ALLOWED (user requested both stocks + ETFs to be tradeable)
                            # Only volatile/inverse ETFs blocked (too risky for paper trading)
                            _BLOCKED_ETFS = {
                                "VXX","UVXY","SVXY",   # volatility ETFs — extreme decay
                                "SQQQ","SPXS","SDOW",  # inverse ETFs — bet against market
                                "TZA","FAZ",           # 3x inverse — too risky
                            }
                            if ticker.upper() in _BLOCKED_ETFS:
                                logger.info(f"AUTO-INVEST: {ticker} skipped — volatility/inverse ETF (too risky)")
                                continue
                        except Exception:
                            pass  # fail-open: proceed if sector check fails

                        # Smart position sizing — scale by score + market conditions
                        try:
                            from smart_position_sizing import (
                                get_consecutive_losses as _get_consec,
                                get_today_pnl_pct as _get_pnl_pct,
                            )
                            _consec = _get_consec()
                            _pnl_pct_today = _get_pnl_pct()
                            if _consec >= 2 or _pnl_pct_today < -2:
                                # Struggling day: reduce conviction for sizing
                                _conviction = max(50, _conviction - 10)
                                logger.info(
                                    f"AUTO-INVEST: {ticker} sizing reduced — "
                                    f"consec={_consec} pnl={_pnl_pct_today:.1f}%"
                                )
                        except Exception:
                            pass

                        # ── Adaptive Position Sizing — ממ adaptive_trader.py ────────
                        # נכוון גודל פוזיציה לפי win rate, drawdown, consecutive losses
                        _adaptive_risk_factor = 1.0
                        try:
                            from adaptive_trader import get_adaptive_trading_params as _gatp
                            _adapt = await _asyncio.wait_for(
                                _gatp(
                                    base_quantity=1.0,
                                    base_min_buy_score=float(_os.getenv("MIN_BUY_SCORE", "70")),
                                    base_stop_loss_pct=settings.STOP_LOSS_PCT,
                                    base_take_profit_pct=settings.TAKE_PROFIT_PCT,
                                ),
                                timeout=10,
                            )
                            _adaptive_risk_factor = _adapt.get("position_sizing", {}).get("risk_factor", 1.0)
                            # Clamp to safe range
                            _adaptive_risk_factor = max(0.3, min(1.5, _adaptive_risk_factor))
                            if abs(_adaptive_risk_factor - 1.0) > 0.1:
                                logger.info(
                                    f"AUTO-INVEST: {ticker} adaptive risk_factor={_adaptive_risk_factor:.2f} "
                                    f"(reason={_adapt.get('position_sizing', {}).get('reason', '?')})"
                                )
                        except Exception:
                            pass  # fail-open: use original sizing

                        # Risk-based position sizing — adjusted by Buffett quality AND market volatility
                        _conviction = score * _adaptive_risk_factor
                        if _buffett_score >= 80:
                            _conviction += 10   # premium quality → bigger position
                        elif _buffett_score >= 70:
                            _conviction += 5
                        # VIX defense — shrink position size in high-fear market
                        try:
                            from indicators import get_market_conditions as _gmc
                            _vix_now = (await _asyncio.wait_for(
                                _asyncio.to_thread(_gmc), timeout=8
                            )).get("vix", 0)
                            if _vix_now and _vix_now > 25:
                                _conviction -= 8   # high fear → smaller position (acts like low conviction)
                                logger.info(f"AUTO-INVEST: {ticker} VIX={_vix_now:.1f} — shrinking position")
                            elif _vix_now and _vix_now < 15:
                                _conviction += 3   # very calm → slightly larger (more confidence)
                        except Exception:
                            pass
                        from budget import compute_position_size
                        qty, sizing_meta = await _asyncio.to_thread(compute_position_size, price, _conviction)
                        if qty <= 0:
                            logger.info(f"AUTO-INVEST: {ticker} sizing=0 → skip ({sizing_meta})")
                            _create_background_task(_asyncio.to_thread(
                                _shadow.evaluate, ticker, price, score, sentiment.score,
                                _vol_ratio, "budget", f"sizing=0 at ${price:.2f}", "auto_invest",
                            ))
                            continue

                        # Slippage estimate (for metadata/audit — iceberg manages actual limit internally)
                        slip = await _asyncio.to_thread(slippage_estimate, price, qty, "buy", ticker)

                        # ── PRE-BUY NOTIFICATION — לפני הקנייה ──────────────────────
                        try:
                            _notional = round(price * qty, 2)
                            _stop_p   = round(price * (1 - settings.STOP_LOSS_PCT / 100), 2)
                            _tp_p     = round(price * (1 + settings.TAKE_PROFIT_PCT / 100), 2)
                            _score_bar = "🟩" * round(score / 10) + "⬜" * (10 - round(score / 10))
                            _tv_link = f'https://www.tradingview.com/chart/?symbol={ticker}'
                            _create_background_task(send_message(
                                f"⏳ <b>עומד לקנות — <a href=\"{_tv_link}\">{ticker}</a></b>\n"
                                f"━━━━━━━━━━━━━━━━\n"
                                f"💵 מחיר: <b>${price:.2f}</b>  |  🔢 כמות: {qty:.4f}\n"
                                f"💰 סכום: <b>${_notional:,.2f}</b>\n"
                                f"━━━━━━━━━━━━━━━━\n"
                                f"📊 ציון: {_score_bar} <b>{score:.0f}/100</b>\n"
                                f"🎯 יעד רווח: ${_tp_p:.2f} (+{settings.TAKE_PROFIT_PCT:.0f}%)\n"
                                f"🛑 Stop Loss: ${_stop_p:.2f} (-{settings.STOP_LOSS_PCT:.0f}%)\n"
                                f"⏳ <i>מבצע את הקנייה...</i>"
                            ))
                        except Exception:
                            pass

                        # Acquire per-ticker lock to prevent double-buy with simultaneous webhook
                        # Acquire per-ticker lock (same lock webhook uses) — HOLD it during buy
                        try:
                            from webhook import _get_buy_lock as _gbl
                            _ticker_lock = await _gbl(ticker)
                        except Exception:
                            _ticker_lock = None

                        async def _do_buy_locked():
                            # Final duplicate check INSIDE the lock — prevents TOCTOU race
                            if database.get_open_trade_by_ticker(ticker):
                                logger.info(f"AUTO-INVEST: {ticker} already held (race condition caught)")
                                return None
                            from iceberg import iceberg_buy as _ib
                            return await _ib(ticker, qty, price)

                        if _ticker_lock is not None:
                            if _ticker_lock.locked():
                                logger.info(f"AUTO-INVEST: {ticker} buy lock held by webhook — skipping")
                                continue
                            async with _ticker_lock:
                                order = await _asyncio.wait_for(_do_buy_locked(), timeout=120)
                        else:
                            order = await _asyncio.wait_for(_do_buy_locked(), timeout=120)

                        if order is None:
                            continue  # duplicate detected inside lock

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

                        # Collect for combined Telegram message
                        _bought_list.append({
                            "ticker": ticker, "qty": filled_qty,
                            "price": actual_price, "score": score,
                            "sentiment": sentiment.score, "trade_id": trade_id,
                            "notional": actual_price * filled_qty,
                        })

                    except _asyncio.TimeoutError:
                        # Timeout during scan is normal — just skip this ticker silently.
                        # The bot will retry in the next 5-minute cycle. No Telegram alert needed.
                        logger.warning(f"AUTO-INVEST: {ticker} timed out — skipping (will retry next cycle)")
                    except Exception as e:
                        logger.error(f"AUTO-INVEST: Error on {ticker}: {e}")
                        _create_background_task(notify_error("order_failed", ticker, f"שגיאה"))

                logger.info(f"AUTO-INVEST: Done. Bought {bought} stocks. Cash left: ${remaining:.2f}")

                # ── Send ONE combined Telegram message for all buys ───────────
                if _bought_list:
                    try:
                        from config import settings as _cfg
                        from telegram_chat import _fmt_price as _fp
                        from telegram_bot import _build_progress_bar
                        n = len(_bought_list)
                        lines = [
                            f"🛒 <b>קנינו {n} {'מניה' if n==1 else 'מניות'}!</b>",
                            "━━━━━━━━━━━━━━━━"
                        ]
                        for _b in _bought_list:
                            _p   = _b["price"]
                            _sc  = _b.get("score", 0)
                            _sl  = round(_p * (1 - _cfg.STOP_LOSS_PCT  / 100), 2)
                            _tp  = round(_p * (1 + _cfg.TAKE_PROFIT_PCT / 100), 2)
                            _qty = f"{_b['qty']:.4f}".rstrip('0').rstrip('.')
                            _notional = _p * _b["qty"]
                            _rr = _cfg.TAKE_PROFIT_PCT / _cfg.STOP_LOSS_PCT
                            _bar = _build_progress_bar(_sc, fill="🟩", empty="⬜")
                            lines.append(
                                f"  <b>{_b['ticker']}</b>  {_qty}×{_fp(_p)}  "
                                f"→TP:{_fp(_tp)} / SL:{_fp(_sl)}  R/R:1:{_rr:.1f}\n"
                                f"  {_bar} {_sc:.0f}/100"
                            )
                            if _b != _bought_list[-1]:
                                lines.append("━━━━━━━━━━━━━━━━")
                        lines.append("━━━━━━━━━━━━━━━━")
                        lines.append(f"💰 מזומן נותר: {_fp(remaining)}  |  פוזיציות: {_open_count+n}")
                        _create_background_task(send_message("\n".join(lines)))
                    except Exception as _te:
                        logger.warning(f"[NOTIFY] combined buy message failed: {_te}")

        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            logger.warning("AUTO-INVEST: scan iteration timed out — moving to next cycle")
        except Exception as e:
            logger.error(f"AUTO-INVEST loop error: {e}")
            _create_background_task(notify_error("loop_error", "", f"שגיאה ב-auto_invest_loop"))

        # Configurable scan interval — default 4 min (was 5).
        # Lower = more trade opportunities, faster path to 200 trades.
        _scan_min = int(_os.getenv("SCAN_INTERVAL_MIN", "4"))
        await asyncio.sleep(_scan_min * 60)


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
                # Too early — sleep precisely to briefing_time
                # Use 1-minute intervals when <5 min away to be accurate
                wait_sec = (briefing_time - now_utc).total_seconds()
                if wait_sec > 5 * 60:
                    await asyncio.sleep(min(wait_sec - 60, 5 * 60))  # wake up 1 min before
                else:
                    await asyncio.sleep(max(wait_sec, 1))  # sleep exactly the remaining time
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

            headlines = await asyncio.to_thread(get_general_headlines, 6)

            # Get news for open positions too
            from news_service import get_headlines as _get_ticker_news
            _open_trades = database.get_open_trades()
            pos_news_lines = []
            for _ot in _open_trades[:3]:
                try:
                    _ticker_hl = await asyncio.wait_for(
                        asyncio.to_thread(_get_ticker_news, _ot["ticker"], 2), timeout=10
                    )
                    for _h in _ticker_hl[:1]:
                        pos_news_lines.append(f"[{_ot['ticker']}] {_h}")
                except Exception:
                    pass

            all_headlines = pos_news_lines + headlines
            all_headlines = all_headlines[:7]

            # Translate headlines to Hebrew (Groq → Google Translate → English fallback)
            if all_headlines:
                try:
                    from translator import translate_headlines
                    all_headlines = await asyncio.wait_for(
                        asyncio.to_thread(translate_headlines, all_headlines),
                        timeout=20,
                    )
                except Exception as _te:
                    logger.debug(f"[BRIEFING] Translation failed: {_te}")
                    # Keep original headlines (English) — better than nothing

            news_text = "\n".join(f"• {h}" for h in all_headlines) if all_headlines else "אין חדשות זמינות כרגע"


            # Israel time & open positions context
            import datetime as _dt2
            _now_il = _dt2.datetime.now(_dt2.timezone.utc) + _dt2.timedelta(hours=3)
            _il_time = _now_il.strftime("%H:%M")
            _open_trades = database.get_open_trades()
            open_pos_text = ""
            if _open_trades:
                for _ot in _open_trades[:5]:
                    try:
                        _pos = await asyncio.wait_for(
                            asyncio.to_thread(broker.get_position, _ot["ticker"]), timeout=8
                        )
                        _pct = float(_pos.get("unrealized_plpc", 0)) * 100 if _pos else 0
                        _pl  = float(_pos.get("unrealized_pl", 0)) if _pos else 0
                        _icon = "🟢" if _pct >= 0 else "🔴"
                        _tv = f'<a href="https://www.tradingview.com/chart/?symbol={_ot["ticker"]}">{_ot["ticker"]}</a>'
                        open_pos_text += f"\n  {_icon} {_tv}  {_pct:+.1f}%  ${_pl:+.2f}"
                    except Exception:
                        _tv = f'<a href="https://www.tradingview.com/chart/?symbol={_ot["ticker"]}">{_ot["ticker"]}</a>'
                        open_pos_text += f"\n  📌 {_tv}"

            # VIX
            try:
                from indicators import get_vix as _gvix, get_fear_greed as _gfg
                _vix = _gvix()
                _fg  = _gfg()
                _vix_str = f"🌡️ VIX: {_vix:.1f}" if _vix else ""
                _fg_str  = f"  |  😨 F&G: {_fg}" if _fg else ""
                market_line = f"{_vix_str}{_fg_str}"
            except Exception:
                market_line = ""

            _is_edt = 3 <= _now_il.month <= 10
            _open_time  = "16:30" if _is_edt else "15:30"
            _close_time = "23:00" if _is_edt else "22:00"

            # ── Top 5 Buffett picks for the day ──────────────────────────
            picks_text = ""
            try:
                from scanner import get_watchlist as _gwl_p
                from buffett_analysis import get_buffett_analysis as _ba_p
                _tickers_p = _gwl_p()[:15]
                _picks = []
                for _tk in _tickers_p:
                    try:
                        _r = await asyncio.wait_for(
                            asyncio.to_thread(_ba_p, _tk),
                            timeout=12,
                        )
                        if _r.get("score", 0) >= 65:
                            _picks.append((_tk, _r.get("score", 0), _r.get("moat", "?")))
                    except Exception:
                        continue
                if _picks:
                    _picks.sort(key=lambda x: x[1], reverse=True)
                    _top5 = _picks[:5]
                    _picks_lines = []
                    for _tk, _s, _m in _top5:
                        _icon = {"strong": "💪", "medium": "🛡️", "weak": "⚠️"}.get(_m, "?")
                        _picks_lines.append(f"   {_icon} <b>{_tk}</b> — איכות {_s:.0f}/100")
                    picks_text = "\n🎯 <b>Top 5 איכותיות להיום (Buffett):</b>\n" + "\n".join(_picks_lines)
            except Exception:
                pass

            await send_message(
                f"☀️ <b>בוקר טוב! שוק נפתח בעוד 30 דקות</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"🕐  שעה: {_il_time} ישראל  |  פתיחה: {_open_time}  סגירה: {_close_time}\n"
                + (f"{market_line}\n" if market_line else "")
                + (f"\n📂 <b>פוזיציות פתוחות:</b>{open_pos_text}\n" if open_pos_text else "")
                + picks_text
                + f"\n\n📰 <b>חדשות בולטות:</b>\n{news_text}"
            )
            _briefing_sent_date = today_str

            # Send market summary with market open status
            _create_background_task(notify_market_summary(
                market_status="open",
                top_gainers=[],  # Would need real-time market data
                top_losers=[],   # Would need real-time market data
            ))

            logger.info("Morning briefing sent")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Morning briefing error: {e}")
            await asyncio.sleep(3600)


# (Moved to top of file — defined near _smart_sell_low_count)


async def earnings_monitor_loop():
    """
    בודק כל 30 דקות אם יצא דוח לאחת מהפוזיציות הפתוחות.
    פועל מיידית:
      - הפסד גדול (miss + ירידה >3%): מוכר מיידי
      - הפסד קטן (miss): מהדק stop ל-1% מתחת מחיר
      - הצלחה (beat): מרפה stop קצת + שולח התראה
      - הצלחה גדולה (beat + עלייה >5%): שולח "שקול הוספה"
    """
    await asyncio.sleep(5 * 60)  # wait 5 min after startup
    _checked: dict[str, str] = {}  # ticker → last earnings date checked

    while True:
        try:
            open_trades = await asyncio.to_thread(database.get_open_trades)
            if not open_trades:
                await asyncio.sleep(30 * 60)
                continue

            from earnings import check_post_earnings_momentum

            for trade in open_trades:
                if trade.get("action") != "buy":
                    continue
                ticker = trade["ticker"]
                try:
                    em = await asyncio.wait_for(
                        asyncio.to_thread(check_post_earnings_momentum, ticker),
                        timeout=20
                    )
                    if not em.get("post_earnings"):
                        continue

                    # Key by ACTUAL earnings date (today - days_since), not days_since itself.
                    # Otherwise the key changes every day for the same report and we re-fire.
                    import datetime as _dt_em
                    _ds = em.get("days_since") or 0
                    _earn_date = (_dt_em.date.today() - _dt_em.timedelta(days=_ds)).isoformat()
                    earn_key = f"{ticker}:{_earn_date}"
                    if _checked.get(ticker) == earn_key:
                        continue  # already handled this specific earnings report
                    _checked[ticker] = earn_key

                    days   = em.get("days_since", 0)
                    beat   = em.get("beat")
                    react  = em.get("price_reaction", 0.0)  # % on earnings day
                    surprise = em.get("momentum_score", 5)

                    entry   = trade["entry_price"]
                    cur_p   = await asyncio.wait_for(
                        asyncio.to_thread(broker.get_price, ticker), timeout=10
                    ) or entry
                    plpc    = (cur_p - entry) / entry * 100
                    atr_stop = trade.get("atr_stop_price") or (entry * 0.97)

                    beat_str = "✅ עקף תחזיות" if beat is True else ("❌ פספס תחזיות" if beat is False else "❔ לא ידוע")
                    react_str = f"{react:+.1f}%" if react else "N/A"

                    # ── Miss + ירידה גדולה → מכור מיידי ──────────────────
                    if beat is False and react < -3.0:
                        logger.warning(f"[EARNINGS] {ticker}: MISS + react={react:.1f}% — selling immediately")
                        await _close_position(
                            trade, cur_p, "earnings_miss",
                            f"דוח: פספס תחזיות | תגובה {react:.1f}%"
                        )
                        _create_background_task(send_message(
                            f"📉 <b>מכרתי — דוח רע: {ticker}</b>\n"
                            f"━━━━━━━━━━━━━━━━\n"
                            f"📊 {beat_str} | תגובה: {react_str}\n"
                            f"💵 מכרתי @ ${cur_p:.2f} ({plpc:+.1f}%)\n"
                            f"⚡ פעולה מיידית — הגנה על ההון"
                        ))
                        continue

                    # ── Miss → מהדק stop ──────────────────────────────────
                    if beat is False:
                        new_stop = round(cur_p * 0.99, 4)  # 1% stop
                        if new_stop > atr_stop:
                            await asyncio.to_thread(
                                database.update_trade_stop, trade["id"], new_stop, cur_p
                            )
                            _create_background_task(send_message(
                                f"⚠️ <b>דוח חלש — הידקתי עצירה: {ticker}</b>\n"
                                f"━━━━━━━━━━━━━━━━\n"
                                f"📊 {beat_str} | תגובה: {react_str}\n"
                                f"🛑 עצירה חדשה: ${new_stop:.2f} (1% מתחת)\n"
                                f"📍 מחיר עכשיו: ${cur_p:.2f} ({plpc:+.1f}%)"
                            ))
                        continue

                    # ── Beat גדול + עלייה → שלח התראה ──────────────────
                    if beat is True and react > 5.0:
                        _create_background_task(send_message(
                            f"🚀 <b>דוח מצוין — {ticker}</b>\n"
                            f"━━━━━━━━━━━━━━━━\n"
                            f"📊 {beat_str} | תגובה: {react_str}\n"
                            f"💵 מחיר: ${cur_p:.2f} ({plpc:+.1f}% מהכניסה)\n"
                            f"💡 שקול להוסיף לפוזיציה — מומנטום חזק!"
                        ))
                        continue

                    # ── Beat רגיל → עדכן + שלח ──────────────────────────
                    if beat is True:
                        _create_background_task(send_message(
                            f"✅ <b>דוח טוב — {ticker}</b>\n"
                            f"━━━━━━━━━━━━━━━━\n"
                            f"📊 {beat_str} | תגובה: {react_str}\n"
                            f"💵 מחיר: ${cur_p:.2f} ({plpc:+.1f}%)\n"
                            f"📌 מחזיק — ממשיך לנטר"
                        ))

                except asyncio.TimeoutError:
                    logger.debug(f"[EARNINGS] {ticker}: timeout — skipping")
                except Exception as _ee:
                    logger.debug(f"[EARNINGS] {ticker}: {_ee}")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"earnings_monitor_loop error: {e}")

        await asyncio.sleep(30 * 60)  # check every 30 min


async def adaptive_threshold_loop():
    """
    סף קנייה אדפטיבי — מתאים פרמטרים לפי ביצועים אחרונים.

    כולל auto_optimizer: מעלה/מוריד MIN_BUY_SCORE לפי win rate שבועי.
    """
    await asyncio.sleep(60 * 60)  # 1 hour after startup

    # Run auto-optimizer every 2 hours
    _last_optimizer_run = 0
    _original_score: int | None = None

    while True:
        try:
            import os as _os_at
            import datetime as _dt_at

            # ── Run auto-optimizer every 2 hours ─────────────────────
            import time as _time_ao
            if _time_ao.time() - _last_optimizer_run > 2 * 3600:
                try:
                    from auto_optimizer import run_auto_optimizer
                    await asyncio.wait_for(run_auto_optimizer(), timeout=30)
                    _last_optimizer_run = _time_ao.time()
                except Exception as _ao_err:
                    logger.debug(f"Auto optimizer failed: {_ao_err}")

            # Get most recent trade
            history = await asyncio.to_thread(database.get_trade_history, limit=10)
            recent = [t for t in history if t.get("entry_time")]
            if not recent:
                await asyncio.sleep(6 * 60 * 60)
                continue

            last_trade = recent[0]
            last_entry = last_trade.get("entry_time", "")
            try:
                last_dt = _dt_at.datetime.strptime(
                    str(last_entry)[:19], "%Y-%m-%d %H:%M:%S"
                ).replace(tzinfo=_dt_at.timezone.utc)
                days_since = (_dt_at.datetime.now(_dt_at.timezone.utc) - last_dt).days
            except Exception:
                await asyncio.sleep(6 * 60 * 60)
                continue

            current_min = int(_os_at.getenv("MIN_BUY_SCORE", "70"))

            # ── Win Rate check — raise score if performance is poor ───────────
            try:
                _week_history = await asyncio.to_thread(database.get_trade_history, limit=20)
                _week_closed  = [t for t in _week_history
                                  if t.get("pnl_gross") is not None][-10:]  # last 10 closed
                if len(_week_closed) >= 5:
                    _wins = sum(1 for t in _week_closed if float(t.get("pnl_gross", 0)) >= 0)
                    _wr   = _wins / len(_week_closed) * 100
                    if _wr < 35:
                        # Win rate below 35% → raise score bar significantly
                        new_min = min(80, current_min + 5)
                        if new_min != current_min:
                            _os_at.environ["MIN_BUY_SCORE"] = str(new_min)
                            logger.info(f"[ADAPTIVE] Win rate {_wr:.0f}% < 35% — raising MIN_BUY_SCORE: {current_min} → {new_min}")
                            _create_background_task(send_message(
                                f"📊 <b>סף ציון הוגדל</b>\n"
                                f"━━━━━━━━━━━━━━━━\n"
                                f"📉 Win Rate שבועי: <b>{_wr:.0f}%</b> (נמוך מ-35%)\n"
                                f"⬆️ סף ציון: {current_min} → <b>{new_min}</b>\n"
                                f"🎯 בוחרים רק עסקאות איכותיות יותר!"
                            ))
                    elif _wr > 60:
                        # Win rate above 60% → can be slightly more aggressive
                        new_min = max(65, current_min - 2)
                        if new_min != current_min:
                            _os_at.environ["MIN_BUY_SCORE"] = str(new_min)
                            logger.info(f"[ADAPTIVE] Win rate {_wr:.0f}% > 60% — easing MIN_BUY_SCORE: {current_min} → {new_min}")
            except Exception as _wr_err:
                logger.debug(f"[ADAPTIVE] Win rate check failed: {_wr_err}")

            if days_since >= 2 and current_min > 55:
                # No trades for 2+ days — lower threshold more aggressively
                if _original_score is None:
                    _original_score = current_min
                # Faster decay: -3/check for 2-5 days, -5/check for 5+ days
                step = 5 if days_since >= 5 else 3
                new_min = max(55, current_min - step)
                _os_at.environ["MIN_BUY_SCORE"] = str(new_min)
                logger.info(f"[ADAPTIVE] {days_since}d no trades — lowering MIN_BUY_SCORE: {current_min} → {new_min} (step={step})")
                _create_background_task(send_message(
                    f"📉 <b>הקלת סף זמנית</b>\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"⏰ {days_since} ימים בלי עסקאות\n"
                    f"⬇️ סף ציון: {current_min} → <b>{new_min}</b>\n"
                    f"💡 חוזר לסף הרגיל בעסקה הבאה"
                ))
            elif days_since < 1 and _original_score is not None:
                # Just made a trade — restore original threshold
                _os_at.environ["MIN_BUY_SCORE"] = str(_original_score)
                logger.info(f"[ADAPTIVE] new trade — restoring MIN_BUY_SCORE: {current_min} → {_original_score}")
                _original_score = None

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"adaptive_threshold_loop error: {e}")

        await asyncio.sleep(6 * 60 * 60)   # check every 6 hours


async def idle_cash_alert_loop():
    """
    התראת מזומן חופשי — אם יש $1000+ מזומן יותר מ-3 ימים, התראה.
    מציע פעולה: לחפש הזדמנויות / להעלות סף סיכון.
    רץ פעם ביום.
    """
    await asyncio.sleep(60 * 60)   # 1 hour after startup
    _last_alert_date: str | None = None

    while True:
        try:
            import datetime as _dt_idle
            today_str = _dt_idle.date.today().isoformat()
            if _last_alert_date == today_str:
                await asyncio.sleep(6 * 60 * 60)
                continue

            # Check if market opens today (skip weekends)
            if _dt_idle.datetime.now().weekday() >= 5:
                await asyncio.sleep(6 * 60 * 60)
                continue

            # Get current cash
            try:
                from budget import get_budget_status
                status = await asyncio.wait_for(
                    asyncio.to_thread(get_budget_status), timeout=15
                )
                cash = float(status.get("cash_available", 0) or 0)
                equity = float(status.get("equity", 1) or 1)
                pos_value = float(status.get("positions_value", 0) or 0)
            except Exception:
                await asyncio.sleep(60 * 60)
                continue

            # Alert if cash > 50% of equity AND > $1000 (significant idle cash)
            cash_pct = (cash / equity * 100) if equity else 0
            if cash > 1000 and cash_pct > 50:
                _last_alert_date = today_str
                _create_background_task(send_message(
                    f"💵 <b>מזומן חופשי לא מנוצל</b>\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"💰 מזומן זמין: <b>${cash:,.2f}</b> ({cash_pct:.0f}% מהתיק)\n"
                    f"💼 מושקע: ${pos_value:,.2f}\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"💡 הבוט סלקטיבי — לא מצא הרבה הזדמנויות איכותיות\n"
                    f"📋 השתמש ב-/best לראות מה הכי קרוב לסף\n"
                    f"📋 או /why TICKER לבדוק מניה ספציפית"
                ))

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"idle_cash_alert error: {e}")

        await asyncio.sleep(6 * 60 * 60)   # check every 6 hours


async def drawdown_protection_loop():
    """
    הגנת drawdown — מנטר ירידת תיק מהמקסימום ההיסטורי.
    אם התיק יורד 10%+ מהשיא ההיסטורי → עוצר קניות חדשות עד שיש התאוששות 3%.

    שונה מ-circuit_breaker היומי: זה drawdown מצטבר (יכול להיות לאורך ימים).
    """
    await asyncio.sleep(20 * 60)
    _peak_equity: float | None = None
    _paused_for_drawdown = False

    while True:
        try:
            # Get current equity
            try:
                from budget import get_budget_status as _gbs
                status = await asyncio.wait_for(
                    asyncio.to_thread(_gbs), timeout=15
                )
                current_equity = float(status.get("equity", 0) or 0)
            except Exception:
                await asyncio.sleep(30 * 60)
                continue

            if current_equity <= 0:
                await asyncio.sleep(30 * 60)
                continue

            # Track peak
            if _peak_equity is None or current_equity > _peak_equity:
                _peak_equity = current_equity
                # Reset drawdown pause if we recovered
                if _paused_for_drawdown and current_equity >= _peak_equity * 0.97:
                    import os as _os_dd
                    _os_dd.environ.pop("BOT_PAUSED", None)
                    _paused_for_drawdown = False
                    _create_background_task(send_message(
                        f"📈 <b>התאוששות!</b>\n"
                        f"━━━━━━━━━━━━━━━━\n"
                        f"💼 התיק חזר לקרבת השיא — ${current_equity:,.2f}\n"
                        f"▶️ הבוט חוזר לסחור"
                    ))

            # Compute drawdown
            drawdown_pct = ((_peak_equity - current_equity) / _peak_equity * 100) if _peak_equity else 0

            # Trigger pause at 10% drawdown
            if drawdown_pct >= 10 and not _paused_for_drawdown:
                import os as _os_dd
                _os_dd.environ["BOT_PAUSED"] = "true"
                _paused_for_drawdown = True
                logger.warning(f"[DRAWDOWN] {drawdown_pct:.1f}% drawdown — pausing bot")
                _create_background_task(send_message(
                    f"🛡️ <b>הגנת Drawdown הופעלה</b>\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"📉 התיק ירד {drawdown_pct:.1f}% מהשיא\n"
                    f"💼 שיא: ${_peak_equity:,.2f} → עכשיו: ${current_equity:,.2f}\n"
                    f"⏸️ הבוט מפסיק לקנות עד שיהיה recovery"
                ))
            # Warning at 5% drawdown
            elif drawdown_pct >= 5 and not _paused_for_drawdown:
                _warn_key = f"dd_warn_{int(drawdown_pct)}"
                if not hasattr(drawdown_protection_loop, "_warned"):
                    drawdown_protection_loop._warned = set()
                if _warn_key not in drawdown_protection_loop._warned:
                    drawdown_protection_loop._warned.add(_warn_key)
                    _create_background_task(send_message(
                        f"⚠️ <b>ירידה בתיק</b>\n"
                        f"━━━━━━━━━━━━━━━━\n"
                        f"📉 התיק ירד {drawdown_pct:.1f}% מהשיא\n"
                        f"💼 שיא: ${_peak_equity:,.2f} → עכשיו: ${current_equity:,.2f}\n"
                        f"📌 מעקב — ב-10% הבוט יעצור"
                    ))

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"drawdown_protection error: {e}")

        await asyncio.sleep(30 * 60)   # every 30 min


async def volume_surge_loop():
    """
    🔴 Volume Surge Detector — כל 15 דקות.
    מזהה מניות עם נפח 3×+ מהנורמלי ושולח התראה בטלגרם.
    """
    await asyncio.sleep(5 * 60)   # warm-up 5 minutes
    while True:
        try:
            from trading_hours import is_ok_to_trade as _iot
            _ok, _ = _iot()
            if _ok:
                from scanner import get_watchlist as _gwl
                from volume_surge import run_volume_surge_alert as _vsa
                _wl = _gwl()
                await asyncio.wait_for(_vsa(_wl), timeout=60)
        except Exception as _ve:
            logger.debug(f"volume_surge_loop: {_ve}")
        await asyncio.sleep(15 * 60)   # every 15 minutes


async def rapid_move_alert_loop():
    """
    מתריע מיידית על תזוזות חזקות בפוזיציות פתוחות.
    כל 3 דקות בודק:
    - כל פוזיציה ששינתה ±2% תוך 10 דק' → התראה מיידית
    - מסנן ספקיו ימיים — מוודא שזה תזוזה אמיתית

    שולח התראה רק פעם אחת לכל כיוון לכל פוזיציה ביום.
    """
    await asyncio.sleep(5 * 60)
    _price_snapshots: dict[str, list] = {}   # ticker → [(timestamp, price), ...]
    _alerted_today: dict[str, str] = {}   # f"{ticker}_{direction}" → date

    while True:
        try:
            mkt_open = await asyncio.wait_for(
                asyncio.to_thread(broker.is_market_open), timeout=8
            )
            if not mkt_open:
                await asyncio.sleep(10 * 60)
                continue

            import datetime as _dt_ra
            today_str = _dt_ra.date.today().isoformat()
            _alerted_today = {k: v for k, v in _alerted_today.items() if v == today_str}

            open_trades = await asyncio.to_thread(database.get_open_trades)
            if not open_trades:
                await asyncio.sleep(3 * 60)
                continue

            import time as _t_ra
            now_ts = _t_ra.time()

            for trade in open_trades:
                if trade.get("action") != "buy":
                    continue
                ticker = trade["ticker"]
                try:
                    cur_p = await asyncio.wait_for(
                        asyncio.to_thread(broker.get_price, ticker), timeout=8
                    )
                    if not cur_p:
                        continue
                    # Track price snapshots (last 10 min)
                    snaps = _price_snapshots.setdefault(ticker, [])
                    snaps.append((now_ts, cur_p))
                    # Keep only last 12 minutes
                    _price_snapshots[ticker] = [(t, p) for t, p in snaps if now_ts - t < 720]

                    if len(_price_snapshots[ticker]) < 2:
                        continue

                    # Find price ~10 min ago
                    snaps_sorted = sorted(_price_snapshots[ticker], key=lambda x: x[0])
                    old_t, old_p = snaps_sorted[0]
                    elapsed_min = (now_ts - old_t) / 60
                    if elapsed_min < 7:
                        continue

                    move_pct = (cur_p - old_p) / old_p * 100
                    abs_move = abs(move_pct)

                    # 2%+ move in 7-10 min = significant
                    if abs_move >= 2.0:
                        direction = "up" if move_pct > 0 else "down"
                        alert_key = f"{ticker}_{direction}"
                        if _alerted_today.get(alert_key) == today_str:
                            continue
                        _alerted_today[alert_key] = today_str

                        entry = trade["entry_price"]
                        plpc = (cur_p - entry) / entry * 100
                        icon = "🚀" if move_pct > 0 else "⚠️"
                        action_hint = (
                            "💡 שקול לקחת רווח" if move_pct > 0 and plpc > 0
                            else "🛑 הסטופ מגן עליך" if move_pct < 0
                            else "📌 מעקב צמוד"
                        )
                        _create_background_task(send_message(
                            f"{icon} <b>תזוזה חזקה — {ticker}</b>\n"
                            f"━━━━━━━━━━━━━━━━\n"
                            f"⚡ זזה <b>{move_pct:+.2f}%</b> ב-{elapsed_min:.0f} דקות\n"
                            f"💵 ${old_p:.2f} → ${cur_p:.2f}\n"
                            f"📈 רווח כולל: {plpc:+.1f}%\n"
                            f"{action_hint}"
                        ))
                        logger.info(f"[RAPID] {ticker} {move_pct:+.2f}% in {elapsed_min:.0f}min — alerted")
                except Exception:
                    continue

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"rapid_move_alert_loop error: {e}")

        await asyncio.sleep(3 * 60)   # check every 3 min


async def self_improvement_loop():
    """
    תיקון עצמי — הבוט מזהה דפוסים של כישלון ומתאים את ההגדרות.

    אם 5 הפסדים רצופים → מעלה MIN_BUY_SCORE זמנית ב-5 נקודות (זהיר יותר)
    אם 5 ניצחונות רצופים → מוריד MIN_BUY_SCORE ב-2 נקודות (יותר אגרסיבי)
    אם הפסד יומי >3% → מקפיא קניות ל-24 שעות

    רץ כל שעה.
    """
    await asyncio.sleep(30 * 60)
    import os as _os_si

    while True:
        try:
            # Get recent closed trades
            history = await asyncio.to_thread(database.get_trade_history, limit=10)
            closed = [t for t in history if t.get("status") in (
                "closed","stop_loss","take_profit","smart_sell","momentum_exit",
                "news_exit","earnings_miss","time_exit"
            )]
            if len(closed) < 5:
                await asyncio.sleep(60 * 60)
                continue

            # Last 5 trades win/loss
            last_5 = closed[:5]
            last_5_wins = sum(1 for t in last_5 if (t.get("pnl_gross") or 0) > 0)
            last_5_losses = sum(1 for t in last_5 if (t.get("pnl_gross") or 0) <= 0)

            current_min = int(_os_si.getenv("MIN_BUY_SCORE", "60"))

            # ── Pattern 1: 5 losses in a row → tighten ─────────────────────
            if last_5_losses == 5 and current_min < 70:
                new_min = min(70, current_min + 5)
                _os_si.environ["MIN_BUY_SCORE"] = str(new_min)
                logger.warning(f"[SELF-IMPROVE] 5 losses in a row → tightening MIN_BUY_SCORE: {current_min} → {new_min}")
                _create_background_task(send_message(
                    f"🛡️ <b>מצב הגנה אוטומטי</b>\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"💡 הבוט זיהה 5 הפסדים רצופים\n"
                    f"⬆️ סף ציון: {current_min} → <b>{new_min}</b>\n"
                    f"📌 קונה רק מניות איכותיות יותר עד שהוא חוזר לטופ"
                ))

            # ── Pattern 2: 5 wins in a row → loosen ────────────────────────
            elif last_5_wins == 5 and current_min > 55:
                new_min = max(55, current_min - 2)
                _os_si.environ["MIN_BUY_SCORE"] = str(new_min)
                logger.info(f"[SELF-IMPROVE] 5 wins in a row → loosening MIN_BUY_SCORE: {current_min} → {new_min}")
                _create_background_task(send_message(
                    f"📈 <b>הבוט בכושר טוב</b>\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"🔥 5 ניצחונות רצופים!\n"
                    f"⬇️ סף ציון: {current_min} → <b>{new_min}</b>\n"
                    f"💡 הבוט יקנה גם הזדמנויות סולידיות, לא רק מצוינות"
                ))

            # ── Pattern 3: daily loss > 3% → pause ─────────────────────────
            today_pnl = sum(
                (t.get("pnl_gross") or 0)
                for t in closed[:10]
                if t.get("exit_time", "")[:10] == __import__('datetime').date.today().isoformat()
            )
            budget = float(_os_si.getenv("MAX_BUDGET", "10000"))
            daily_loss_pct = abs(today_pnl) / budget * 100 if today_pnl < 0 else 0
            if daily_loss_pct > 3 and not _os_si.getenv("BOT_PAUSED"):
                _os_si.environ["BOT_PAUSED"] = "true"
                logger.warning(f"[SELF-IMPROVE] Daily loss {daily_loss_pct:.1f}% > 3% → PAUSING bot for 24h")
                _create_background_task(send_message(
                    f"⏸️ <b>הפסקה אוטומטית</b>\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"⚠️ הפסד יומי: {daily_loss_pct:.1f}% (יותר מ-3%)\n"
                    f"💤 הבוט עוצר קניות ל-24 שעות להתאוששות\n"
                    f"📌 שלח /resume כדי להפעיל מחדש מוקדם"
                ))

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"self_improvement_loop error: {e}")

        await asyncio.sleep(60 * 60)   # check every hour


async def daily_ai_insights_loop():
    """
    סיכום AI יומי — שולח כל יום ב-23:30 (סגירת שוק) ניתוח של:
    - מה קרה היום בעסקאות
    - מה למד הבוט (אילו דפוסים עבדו / נכשלו)
    - הצעות לשיפור למחר

    מצריך Groq API key — אם לא קיים, שולח גרסה פשוטה.
    """
    import datetime as _dt_ai
    _utc = _dt_ai.timezone.utc
    _last_sent_date = None

    while True:
        try:
            now = _dt_ai.datetime.now(_utc)
            # Target: 21:00 UTC ≈ 24:00 שעון ישראל (5pm ET, after market close)
            target = now.replace(hour=21, minute=0, second=0, microsecond=0)
            if now >= target:
                target += _dt_ai.timedelta(days=1)

            # Wait until target time
            while _dt_ai.datetime.now(_utc) < target:
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    raise

            today_str = _dt_ai.datetime.now(_utc).date().isoformat()
            if _last_sent_date == today_str:
                await asyncio.sleep(3600)
                continue

            # Skip weekends
            if _dt_ai.datetime.now(_utc).weekday() >= 5:
                await asyncio.sleep(3600)
                continue

            # ── Gather today's stats ──────────────────────────────────────
            tc = await asyncio.to_thread(database.get_total_trades_count)
            today_trades = tc.get("today", 0)
            wins = tc.get("wins", 0)
            losses = tc.get("losses", 0)
            wr = (wins / (wins + losses) * 100) if (wins + losses) else 0

            # Get today's closed trades for analysis
            history = await asyncio.to_thread(database.get_trade_history, limit=20)
            today_closed = [
                t for t in history
                if t.get("exit_time") and str(t.get("exit_time", ""))[:10] == today_str
            ]
            today_pnl = sum(t.get("pnl_gross", 0) or 0 for t in today_closed)

            # ── Build AI insights using Groq ───────────────────────────────
            insights = ""
            try:
                from openai import OpenAI as _OAI_AI
                _cli = _OAI_AI(api_key=settings.GROQ_API_KEY,
                               base_url="https://api.groq.com/openai/v1")
                _prompt = (
                    f"אתה אנליסט מסחר ברמת באפט. סקור את היום:\n"
                    f"- עסקאות שנפתחו היום: {today_trades}\n"
                    f"- סה\"כ עסקאות בהיסטוריה: {tc.get('total', 0)}\n"
                    f"- ניצחונות: {wins} | הפסדים: {losses} (WR={wr:.0f}%)\n"
                    f"- רווח היום: ${today_pnl:+.2f}\n\n"
                    f"כתוב בעברית 3 שורות:\n"
                    f"1. מה היה היום (תמצית קצרה)\n"
                    f"2. מה הבוט למד (דפוס שעבד/נכשל)\n"
                    f"3. הצעה לשיפור למחר\n"
                    f"קצר וממוקד. ללא כותרות."
                )
                resp = await asyncio.wait_for(
                    asyncio.to_thread(lambda: _cli.chat.completions.create(
                        model=settings.LLM_MODEL,
                        messages=[{"role": "user", "content": _prompt}],
                        max_tokens=300, temperature=0.5,
                    )),
                    timeout=25,
                )
                insights = resp.choices[0].message.content.strip()
            except Exception as _e:
                logger.debug(f"[AI INSIGHTS] LLM failed: {_e}")
                insights = (
                    f"היום נסחרו {today_trades} עסקאות. "
                    + (f"רווח: ${today_pnl:+.2f}." if today_pnl else "תוצאה ניטרלית.")
                    + " הבוט ממשיך ללמוד ולהשתפר."
                )

            await send_message(
                f"🤖 <b>סיכום AI יומי</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📊 עסקאות היום: {today_trades}\n"
                f"💰 רווח/הפסד יומי: ${today_pnl:+.2f}\n"
                f"🎯 אחוז הצלחה כולל: {wr:.0f}% ({wins} נצח / {losses} הפסד)\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"<b>תובנות AI:</b>\n{insights}\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"😴 מחר תהיה יום חדש — לילה טוב!"
            )
            _last_sent_date = today_str
            logger.info(f"Daily AI insights sent: trades={today_trades} pnl=${today_pnl:.2f}")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"daily_ai_insights error: {e}")
            await asyncio.sleep(3600)


async def weekend_research_loop():
    """
    לולאת מחקר לסופי שבוע — הבוט לומד עומק כשהשוק סגור.
    כל שעתיים בסופי שבוע:
    - מנתח Buffett על 5 מניות מהווטצ'ליסט
    - מאתר 'הזדמנויות שבת' (מניות איכותיות שיכולות לפתוח חזק ביום שני)
    - שולח דוח שבועי על אילו מניות שווה לבדוק
    """
    await asyncio.sleep(20 * 60)   # 20 min after startup
    _last_research_date = None

    while True:
        try:
            import datetime as _dt_we
            now = _dt_we.datetime.now(_dt_we.timezone.utc)
            weekday = now.weekday()
            today_str = now.date().isoformat()

            # Only run on Saturday/Sunday
            if weekday < 5:
                await asyncio.sleep(2 * 3600)
                continue

            # Run at most once per 2 hours
            if _last_research_date == today_str and now.hour % 4 != 0:
                await asyncio.sleep(60 * 60)
                continue

            from scanner import get_watchlist as _gwl
            from buffett_analysis import get_buffett_analysis
            tickers = _gwl()[:10]

            results = []
            for t in tickers:
                try:
                    a = await asyncio.wait_for(
                        asyncio.to_thread(get_buffett_analysis, t),
                        timeout=15,
                    )
                    if a.get("score", 0) >= 70:
                        results.append((t, a.get("score", 0), a.get("moat", "?")))
                except Exception:
                    continue

            if results:
                results.sort(key=lambda x: x[1], reverse=True)
                top3 = results[:3]
                _last_research_date = today_str
                lines = [
                    "📚 <b>מחקר סוף שבוע</b>",
                    "━━━━━━━━━━━━━━━━",
                    "🏆 <b>איכותיות לבדיקה ביום שני:</b>",
                ]
                for t, s, m in top3:
                    moat_icon = {"strong": "💪", "medium": "🛡️", "weak": "⚠️"}.get(m, "?")
                    lines.append(f"   {moat_icon} <b>{t}</b>: ציון באפט <b>{s:.0f}/100</b>")
                lines.append("━━━━━━━━━━━━━━━━")
                lines.append("💡 הבוט יסרוק אותן ראשונות בפתיחת השוק")
                _create_background_task(send_message("\n".join(lines)))
                logger.info(f"[WEEKEND] research sent: top3={[t for t,_,_ in top3]}")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"weekend_research_loop error: {e}")

        await asyncio.sleep(4 * 60 * 60)   # every 4 hours during weekend


async def smart_reentry_loop():
    """
    זיהוי re-entry: אם הבוט מכר מניה ב-stop_loss / smart_sell ב-2 ימים אחרונים,
    והיא עלתה 3%+ מאז המכירה — בודק שוב.
    אם הציון מעל 65 — שולח התראה למשתמש.
    רץ כל שעה.
    """
    await asyncio.sleep(15 * 60)
    _notified: dict[str, str] = {}   # ticker → date (avoid spam)

    while True:
        try:
            mkt_open = await asyncio.wait_for(
                asyncio.to_thread(broker.is_market_open), timeout=10
            )
            if not mkt_open:
                await asyncio.sleep(60 * 60)
                continue

            import datetime as _dt_re
            today_str = _dt_re.date.today().isoformat()
            _notified = {k: v for k, v in _notified.items() if v == today_str}

            # Get recent sells (last 2 days)
            two_days_ago = (_dt_re.datetime.utcnow() - _dt_re.timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S")
            conn = database.get_connection()
            rows = conn.execute(f"""
                SELECT ticker, exit_price, exit_time, pnl_gross
                FROM trade_log
                WHERE status IN ('stop_loss','smart_sell','time_exit','momentum_exit')
                  AND exit_time > ?
                ORDER BY exit_time DESC
                LIMIT 10
            """, (two_days_ago,)).fetchall()

            for row in rows:
                ticker = row["ticker"]
                exit_p = row["exit_price"] or 0
                if _notified.get(ticker) == today_str:
                    continue
                # Skip if already re-bought
                if database.get_open_trade_by_ticker(ticker):
                    continue
                try:
                    cur_p = await asyncio.wait_for(
                        asyncio.to_thread(broker.get_price, ticker), timeout=10
                    )
                    if not cur_p or not exit_p:
                        continue
                    # Did it bounce 3%+ since we sold?
                    bounce = (cur_p - exit_p) / exit_p * 100
                    if bounce < 3.0:
                        continue
                    # Get fresh score
                    from sentiment import score_sentiment
                    from scoring import get_composite_score
                    sent = await asyncio.wait_for(
                        asyncio.to_thread(score_sentiment, ticker), timeout=15
                    )
                    score_r = await asyncio.wait_for(
                        asyncio.to_thread(get_composite_score, ticker, sent.score), timeout=15
                    )
                    score = score_r.get("composite_score", 0)
                    if score >= 65:
                        _notified[ticker] = today_str
                        _create_background_task(send_message(
                            f"🔄 <b>חזרה ל-{ticker}?</b>\n"
                            f"━━━━━━━━━━━━━━━━\n"
                            f"📉 מכרתי ב: ${exit_p:.2f}\n"
                            f"📈 עכשיו: ${cur_p:.2f} ({bounce:+.1f}%)\n"
                            f"⭐ ציון חדש: <b>{score:.0f}/100</b>\n"
                            f"💡 המניה התאוששה — שקול חזרה לפוזיציה\n"
                            f"📋 /buffett {ticker} | /score {ticker}"
                        ))
                        logger.info(f"[REENTRY] {ticker}: bounce={bounce:.1f}% score={score:.0f} — alerted")
                except Exception:
                    continue

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"smart_reentry_loop error: {e}")

        await asyncio.sleep(60 * 60)


async def golden_opportunity_loop():
    """
    מאתר "הזדמנויות זהב" — מניות עם ציון מעל 75 + ציון באפט מעל 70.
    אם אין מזומן זמין → מציע למשתמש לסגור פוזיציה חלשה.
    רץ כל שעה בזמן שוק פתוח.
    """
    await asyncio.sleep(10 * 60)   # 10 min after startup
    _seen_today: dict[str, str] = {}   # ticker → date (avoid spam same ticker daily)

    while True:
        try:
            # Only run during market hours
            mkt_open = await asyncio.wait_for(
                asyncio.to_thread(broker.is_market_open), timeout=10
            )
            if not mkt_open:
                await asyncio.sleep(30 * 60)
                continue

            import datetime as _dt_op
            today_str = _dt_op.date.today().isoformat()

            # Reset daily tracker
            _seen_today = {k: v for k, v in _seen_today.items() if v == today_str}

            from scanner import get_watchlist as _gwl
            from scoring import get_composite_score
            from sentiment import score_sentiment

            # Sample top 15 tickers
            candidates = _gwl()[:15]
            best = None
            best_score = 0

            for ticker in candidates:
                if _seen_today.get(ticker) == today_str:
                    continue
                # Skip if already in portfolio
                if database.get_open_trade_by_ticker(ticker):
                    continue
                try:
                    sent = await asyncio.wait_for(
                        asyncio.to_thread(score_sentiment, ticker), timeout=15
                    )
                    score_r = await asyncio.wait_for(
                        asyncio.to_thread(get_composite_score, ticker, sent.score), timeout=15
                    )
                    score = score_r.get("composite_score", 0)
                    if score >= 75 and score > best_score:
                        best = (ticker, score, score_r)
                        best_score = score
                except Exception:
                    continue

            if best:
                ticker, score, _r = best
                _seen_today[ticker] = today_str

                # Get Buffett score
                buf_score = 0
                try:
                    from buffett_analysis import get_buffett_analysis
                    buf = await asyncio.wait_for(
                        asyncio.to_thread(get_buffett_analysis, ticker), timeout=15
                    )
                    buf_score = buf.get("score", 0)
                except Exception:
                    pass

                # Only notify if both technical AND fundamentals strong
                if buf_score >= 60:
                    _inline_kb = {
                        "inline_keyboard": [[
                            {"text": "🎩 ניתוח באפט", "url": f"https://t.me/share/url?url=/buffett%20{ticker}"},
                            {"text": "📊 ציון מפורט", "url": f"https://t.me/share/url?url=/score%20{ticker}"},
                        ], [
                            {"text": "📰 חדשות", "url": f"https://t.me/share/url?url=/news%20{ticker}"},
                            {"text": "💲 מחיר", "url": f"https://t.me/share/url?url=/price%20{ticker}"},
                        ]]
                    }
                    _create_background_task(send_message(
                        f"🌟 <b>הזדמנות זהב — {ticker}!</b>\n"
                        f"━━━━━━━━━━━━━━━━\n"
                        f"⭐ ציון טכני: <b>{score:.0f}/100</b> (מעולה!)\n"
                        f"🎩 ציון באפט: <b>{buf_score:.0f}/100</b>\n"
                        f"💡 הבוט מזהה הזדמנות איכותית — סורק לעומק\n"
                        f"📋 לפרטים: /buffett {ticker} | /score {ticker} | /news {ticker}"
                    ))
                    logger.info(f"[GOLDEN] {ticker} score={score:.0f} buf={buf_score:.0f} — alerted")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"golden_opportunity_loop error: {e}")

        await asyncio.sleep(60 * 60)   # check every hour


async def webhook_keeper_loop():
    """
    שומר על Telegram webhook על URL הנכון.
    אם בוט אחר משכתב — מתקן תוך 5 דקות.
    """
    import os as _os
    await asyncio.sleep(5 * 60)   # 5 min after startup
    while True:
        try:
            render_url = _os.getenv("RENDER_EXTERNAL_URL", "").rstrip("/")
            token      = settings.TELEGRAM_BOT_TOKEN
            secret     = settings.WEBHOOK_SECRET
            if not render_url or not token:
                await asyncio.sleep(15 * 60)
                continue
            expected = f"{render_url}/telegram/webhook"

            # Check current webhook
            import aiohttp
            async with aiohttp.ClientSession() as sess:
                async with sess.get(
                    f"https://api.telegram.org/bot{token}/getWebhookInfo",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    info = await resp.json()
                current = info.get("result", {}).get("url", "")

                if current != expected:
                    logger.warning(f"[WEBHOOK KEEPER] webhook drifted: '{current}' → fixing to '{expected}'")
                    async with sess.post(
                        f"https://api.telegram.org/bot{token}/setWebhook",
                        json={
                            "url": expected,
                            "drop_pending_updates": False,
                            "secret_token": secret,
                        },
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as r2:
                        result = await r2.json()
                        if result.get("ok"):
                            logger.info(f"[WEBHOOK KEEPER] webhook reset to {expected}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"[WEBHOOK KEEPER] error: {e}")

        await asyncio.sleep(5 * 60)   # check every 5 min


async def market_pulse_loop():
    """
    פעימת שוק 24/7 — הבוט תמיד פעיל ומדווח מה הוא עושה.
    כל שעתיים שולח עדכון: מה הוא רואה בשוק, מה הוא עושה.

    כשהשוק פתוח: סורק להזדמנויות + מנתח חדשות
    כשהשוק סגור: מתאמן + קורא חדשות + מכין רשימה ליום הבא
    """
    await asyncio.sleep(15 * 60)   # 15 min after startup

    while True:
        try:
            # Read market state
            is_open = await asyncio.wait_for(
                asyncio.to_thread(broker.is_market_open), timeout=10
            )

            # Read market conditions (works 24/7)
            try:
                from indicators import get_market_conditions
                mkt = await asyncio.wait_for(
                    asyncio.to_thread(get_market_conditions), timeout=20
                )
                vix = mkt.get("vix", 0)
                spy_up = mkt.get("spy_above_sma50")
                spy_rsi = mkt.get("spy_rsi", 50)
            except Exception:
                vix, spy_up, spy_rsi = 0, None, 50

            # Read open positions
            open_trades = await asyncio.to_thread(database.get_open_trades)
            n_pos = len(open_trades)

            # Read trade counts (total/today)
            try:
                trade_counts = await asyncio.to_thread(database.get_total_trades_count)
            except Exception:
                trade_counts = {"total": 0, "closed": 0, "wins": 0, "losses": 0, "today": 0}

            # Build market mood line
            if vix and vix > 0:
                if vix < 18:    mood = f"🟢 שוק רגוע (VIX={vix:.1f})"
                elif vix < 25:  mood = f"🟡 שוק זהיר (VIX={vix:.1f})"
                else:           mood = f"🔴 שוק בפחד (VIX={vix:.1f})"
            else:
                mood = "📊 אין נתוני VIX"

            trend = "📈 מגמת עליה (SPY מעל ממוצע 50)" if spy_up else (
                "📉 מגמת ירידה (SPY מתחת ממוצע 50)" if spy_up is False else "⚪ מגמה לא ברורה"
            )

            # Build "what bot is doing" line
            if is_open:
                if n_pos > 0:
                    activity = f"🔍 מנהל {n_pos} פוזיציות + מחפש הזדמנויות חדשות"
                else:
                    activity = "🔍 סורק את השוק להזדמנויות"
            else:
                activity = "🧠 השוק סגור — מתאמן על נתוני עבר + קורא חדשות"

            # Trades summary line
            wr_pct = (trade_counts["wins"] / trade_counts["closed"] * 100) if trade_counts["closed"] else 0
            trades_line = (
                f"📊 סך העסקאות שעשיתי: <b>{trade_counts['total']}</b>"
                + (f" (היום: {trade_counts['today']})" if trade_counts['today'] else "")
                + "\n"
                f"   ✅ ברווח: {trade_counts['wins']} | ❌ בהפסד: {trade_counts['losses']}"
                + (f" | אחוז הצלחה: {wr_pct:.0f}%" if trade_counts['closed'] else "")
            )

            # Send update
            _create_background_task(send_message(
                f"💓 <b>פעימת שוק</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"{mood}\n"
                f"{trend}\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"⚙️ מה אני עושה עכשיו:\n"
                f"   {activity}\n\n"
                f"{trades_line}"
                + (f"\n📂 פוזיציות פתוחות: {n_pos}" if n_pos > 0 else "")
            ))

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"market_pulse error: {e}")

        await asyncio.sleep(2 * 60 * 60)   # every 2 hours


async def price_alert_loop():
    """
    Check user-defined price alerts every 2 minutes.
    Supports:
    1. Legacy: USER_ALERTS env var (ticker:price)
    2. New: telegram_bot._price_alerts (set via /alert command)
    """
    await asyncio.sleep(120)
    while True:
        try:
            import os as _os

            # ── Check in-memory alerts (set via /alert command) ──────────
            try:
                from telegram_bot import _price_alerts, check_price_alerts
                if _price_alerts:
                    prices = {}
                    for ticker in list(_price_alerts.keys()):
                        try:
                            cur = await asyncio.wait_for(
                                asyncio.to_thread(broker.get_price, ticker), timeout=8
                            )
                            if cur:
                                prices[ticker] = cur
                        except Exception:
                            pass
                    if prices:
                        await check_price_alerts(prices)
            except Exception as _pal_err:
                logger.debug(f"[PRICE ALERT] in-memory check error: {_pal_err}")

            # ── Legacy: USER_ALERTS env var ──────────────────────────────
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
                        if cur and abs(cur - target) / target < 0.01:
                            _price_alerts_fired.add(key)
                            direction = "📈 עלה" if cur >= target else "📉 ירד"
                            await send_message(
                                f"🔔 <b>התראה!</b>  {ticker}  הגיע ליעד\n"
                                f"━━━━━━━━━━━━━━━━\n"
                                f"🎯  יעד:          ${target:.2f}\n"
                                f"📍  עכשיו:     ${cur:.2f}\n"
                                f"{direction}  ✅"
                            )
                    except Exception:
                        continue
            # ── Check user reminders (set via /remind HH:MM TEXT) ────────────
            import time as _t2
            from datetime import datetime as _dt2, timezone as _tz2, timedelta as _td2
            reminders_str = _os.getenv("USER_REMINDERS", "")
            if reminders_str:
                # Israel time
                _il_off  = 3 if 3 <= _dt2.now(_tz2.utc).month <= 10 else 2
                _now_il  = _dt2.now(_tz2.utc) + _td2(hours=_il_off)
                _hhmm    = _now_il.strftime("%H:%M")
                _reminders = [r.strip() for r in reminders_str.split(",") if "|" in r.strip()]
                _fired   = []
                _keep    = []
                for rem in _reminders:
                    try:
                        rem_time, rem_text = rem.split("|", 1)
                        if rem_time.strip() == _hhmm:
                            _fired.append(rem_text.strip())
                        else:
                            _keep.append(rem)
                    except Exception:
                        _keep.append(rem)
                if _fired:
                    _os.environ["USER_REMINDERS"] = ",".join(_keep)
                    for msg in _fired:
                        await send_message(
                            f"⏰ <b>תזכורת!</b>\n━━━━━━━━━━━━━━━━\n📌 {msg}"
                        )

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
                entry  = float(trade.get("entry_price") or 0)
                if entry <= 0:
                    continue  # skip trades with invalid entry price
                try:
                    pos = await asyncio.wait_for(
                        asyncio.to_thread(broker.get_position, ticker), timeout=10
                    )
                    if not pos:
                        continue
                    cur   = float(pos.get("current_price", entry))
                    pct   = (cur - entry) / entry * 100
                    unreal = float(pos.get("unrealized_pl", (cur - entry) * trade["qty"]))

                    # Position movement alerts disabled — too noisy.
                    # Only buy/sell/news/daily-summary reach the user now.
                    _ = unreal  # suppress unused warning
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
                        f"✅ ניפינו הון למחר"
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
            from news_service import get_headlines, get_general_headlines
            # Refresh general market headlines (timeout 30s)
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(get_general_headlines, 10), timeout=30
                )
            except asyncio.TimeoutError:
                logger.warning("[NEWS] get_general_headlines timed out — skipping")
            # Refresh for open positions (highest priority — monitored for sell signals)
            open_trades = await asyncio.to_thread(database.get_open_trades)
            open_tickers = list({t["ticker"] for t in open_trades})
            # Also pre-warm top watchlist candidates so buy-path has fresh news
            wl_sample = _gwl()[:10]
            tickers_to_refresh = list(dict.fromkeys(open_tickers + wl_sample))[:15]
            for ticker in tickers_to_refresh:
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(get_headlines, ticker, 8, True),  # bypass_cache=True
                        timeout=15,
                    )
                except (asyncio.TimeoutError, Exception):
                    pass
            logger.debug(f"News refreshed: {len(open_tickers)} positions + {len(wl_sample)} watchlist candidates")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"News refresh error (non-critical): {e}")
        await asyncio.sleep(240)   # every 4 min — cache TTL is 5 min, prevents expiry race


async def news_monitor_loop():
    """
    Real-time news monitor for open positions — runs every 10 minutes.

    Logic per position:
      score ≤ 2  → IMMEDIATE SELL  (breaking disaster: fraud/bankruptcy/SEC)
      score ≤ 3  → SELL if position already in profit, else tighten stop 50%
      score ≤ 4  → Tighten ATR stop by 30% (protect against further drop)
      score ≥ 8  → Extend TP target by +2% (great news = let it run further)
      score ≥ 9  → Loosen trailing stop slightly (give big winner more room)

    Sends Telegram alert for every news event that changes the trade plan.
    """
    await asyncio.sleep(3 * 60)   # wait 3 min after startup
    _last_checked: dict[str, float] = {}   # ticker → last check timestamp
    _NEWS_CHECK_INTERVAL = 10 * 60         # 10 minutes per ticker

    while True:
        try:
            try:
                _mkt = await asyncio.wait_for(asyncio.to_thread(broker.is_market_open), timeout=15)
            except asyncio.TimeoutError:
                _mkt = False
            # NEW: בודק חדשות גם בשוק סגור (alert-only mode — בלי מכירה אחרי שעות)
            _alert_only = not _mkt

            open_trades = await asyncio.to_thread(database.get_open_trades)
            if not open_trades:
                await asyncio.sleep(5 * 60)
                continue

            import time as _t
            try:
                from sentiment import score_sentiment_live as _sent_live
            except (ImportError, AttributeError) as _imp_err:
                logger.error(f"[NEWS MONITOR] Cannot import score_sentiment_live: {_imp_err}")
                await asyncio.sleep(60)
                continue

            for trade in open_trades:
                ticker = trade.get("ticker", "")
                if not ticker:
                    continue

                # Rate-limit: check each ticker at most once per 10 minutes
                now = _t.time()
                if now - _last_checked.get(ticker, 0) < _NEWS_CHECK_INTERVAL:
                    continue
                _last_checked[ticker] = now

                try:
                    # Get fresh sentiment — bypasses cache
                    sent = await asyncio.wait_for(
                        asyncio.to_thread(_sent_live, ticker), timeout=25
                    )
                    score    = sent.score
                    headlines = sent.headlines[:3]
                    reasoning = sent.reasoning

                    # Get current price for P&L context (timeout 10s)
                    pos = await asyncio.wait_for(
                        asyncio.to_thread(broker.get_position, ticker), timeout=10
                    )
                    if not pos:
                        continue
                    cur_price = float(pos.get("current_price", trade["entry_price"]))
                    entry     = trade["entry_price"]
                    plpc      = (cur_price - entry) / entry * 100
                    atr_stop  = trade.get("atr_stop_price") or (entry * 0.97)

                    # Translate headlines + reasoning to Hebrew
                    # Uses Groq → Google Translate → English fallback chain
                    async def _translate_news(hl: list[str], reason: str) -> tuple[list[str], str]:
                        try:
                            from translator import translate_headlines, translate_to_hebrew
                            translated_hl = await asyncio.wait_for(
                                asyncio.to_thread(translate_headlines, hl),
                                timeout=15,
                            )
                            translated_reason = await asyncio.wait_for(
                                translate_to_hebrew(reason),
                                timeout=10,
                            )
                            return translated_hl or hl, translated_reason or reason
                        except Exception:
                            return hl, reason

                    headlines_he, reasoning_he = await _translate_news(headlines, reasoning)
                    news_preview = "\n".join(f"📰 {h[:90]}" for h in headlines_he) if headlines_he else "📰 לא נמצאו חדשות"
                    reasoning = reasoning_he

                    # ── 1. CRITICAL (1-2): emergency action ────────────────────
                    if score <= 2:
                        if _alert_only:
                            # שוק סגור — רק התראה, אי אפשר למכור
                            await send_message(
                                f"⚠️ <b>חדשות קריטיות אחרי שעות — {ticker}</b>\n"
                                f"━━━━━━━━━━━━━━━━\n"
                                f"📌 ציון חדשות: <b>{score}/10</b> 🔴\n\n"
                                f"{news_preview}\n\n"
                                f"💬 <b>ניתוח AI:</b> {reasoning[:120]}\n\n"
                                f"📍 שוק סגור — אמכור מיד עם הפתיחה"
                            )
                        else:
                            logger.warning(
                                f"[NEWS SELL] {ticker}: CRITICAL sentiment={score}/10 "
                                f"— emergency exit | reason: {reasoning}"
                            )
                            await _close_position(
                                trade, cur_price, "news_exit",
                                f"חדשות קריטיות (ציון={score}/10) — {reasoning[:60]}"
                            )
                            await send_message(
                                f"🚨 <b>יציאה חירום — חדשות קריטיות!</b>\n"
                                f"━━━━━━━━━━━━━━━━\n"
                                f"📌  <b>{ticker}</b>  ·  ציון חדשות: <b>{score}/10</b> 🔴\n\n"
                                f"{news_preview}\n\n"
                                f"💬 <b>ניתוח AI:</b> {reasoning[:120]}\n\n"
                                f"⚡ מכרתי מיד — הפסד/רווח: <b>{plpc:+.1f}%</b>"
                            )
                        continue

                    # שאר הציונים (3-7, 8+) רק כשהשוק פתוח
                    if _alert_only:
                        continue

                    # ── 2. BEARISH (3): sell if in profit, tighten if in loss ─
                    if score == 3:
                        if plpc >= 0:
                            logger.warning(
                                f"[NEWS SELL] {ticker}: bearish={score}/10 + in profit "
                                f"({plpc:+.1f}%) — selling to protect gains"
                            )
                            await _close_position(
                                trade, cur_price, "news_exit",
                                f"חדשות שליליות (ציון={score}/10) + ברווח — מוגן"
                            )
                            await send_message(
                                f"📉 <b>מכירה מחדשות שליליות — {ticker}</b>\n"
                                f"━━━━━━━━━━━━━━━━\n"
                                f"📌  ציון חדשות: <b>{score}/10</b> 🔴\n\n"
                                f"{news_preview}\n\n"
                                f"💬 {reasoning[:120]}\n\n"
                                f"✅ מכרתי בזמן — רווח שמור: <b>{plpc:+.1f}%</b>"
                            )
                            continue
                        else:
                            # In loss — tighten stop by 50%
                            _dist = cur_price - atr_stop
                            _new_stop = round(atr_stop + _dist * 0.5, 4)
                            if _new_stop > atr_stop:
                                await asyncio.to_thread(
                                    database.update_trade_stop, trade["id"], _new_stop,
                                    trade.get("high_watermark", entry)
                                )
                                await send_message(
                                    f"⚠️ <b>חדשות שליליות — סטופ הוידוק — {ticker}</b>\n"
                                    f"━━━━━━━━━━━━━━━━\n"
                                    f"📌  ציון חדשות: <b>{score}/10</b> 🔴\n\n"
                                    f"{news_preview}\n\n"
                                    f"💬 {reasoning[:100]}\n\n"
                                    f"🛑  סטופ חדש: <b>${_new_stop:.2f}</b>  (הוידוק 50%)"
                                )
                            continue

                    # ── 3. MILDLY BEARISH (4): tighten stop by 30% ───────────
                    if score == 4:
                        _dist = cur_price - atr_stop
                        if _dist > 0:
                            _new_stop = round(atr_stop + _dist * 0.3, 4)
                            if _new_stop > atr_stop:
                                await asyncio.to_thread(
                                    database.update_trade_stop, trade["id"], _new_stop,
                                    trade.get("high_watermark", entry)
                                )
                                logger.info(
                                    f"[NEWS TIGHTEN] {ticker}: score={score}/10 "
                                    f"— stop tightened ${atr_stop:.2f}→${_new_stop:.2f}"
                                )
                                await send_message(
                                    f"📰 <b>חדשות מעט שליליות — {ticker}</b>\n"
                                    f"━━━━━━━━━━━━━━━━\n"
                                    f"📌  ציון חדשות: <b>{score}/10</b> 🟡\n\n"
                                    f"{news_preview}\n\n"
                                    f"🛡️  הידוק סטופ: ${atr_stop:.2f} → <b>${_new_stop:.2f}</b>"
                                )
                        continue

                    # ── 4. VERY BULLISH (8-9): loosen stop silently (no Telegram alert) ──
                    if score >= 8:
                        _dist = cur_price - atr_stop
                        _new_stop = round(atr_stop - _dist * 0.15, 4)
                        if _new_stop < atr_stop and _new_stop > entry:
                            await asyncio.to_thread(
                                database.update_trade_stop, trade["id"], _new_stop,
                                trade.get("high_watermark", entry)
                            )
                        # Bullish news alert disabled — user only gets sell/buy/critical news

                except asyncio.TimeoutError:
                    logger.debug(f"[NEWS MONITOR] {ticker}: sentiment check timed out — skip")
                except Exception as e:
                    logger.debug(f"[NEWS MONITOR] {ticker}: error — {e}")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[NEWS MONITOR] loop error: {e}")

        await asyncio.sleep(60)   # check every minute — per-ticker rate limited to 10 min


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
    # Pre-action notification
    _create_background_task(send_message(
        f"🚨 <b>הבוט עומד לצאת חירום — {ticker}</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"⚠️  סנטימנט שלילי קיצוני\n"
        f"💵  מוכר את כל הפוזיציה עכשיו"
    ))
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

            # Check every minute — CancelledError must propagate for clean shutdown
            while datetime.datetime.now(_utc) < target:
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    raise  # don't swallow cancellation inside inner loop

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

            # Send Discord daily summary
            _create_background_task(send_discord_daily_summary(
                date=str(today),
                trades_count=len(closed_today),
                wins=len(wins),
                losses=len(losses),
                daily_pnl=total_pnl,
                win_rate=(len(wins) / len(closed_today) * 100) if closed_today else 0,
            ))

            # Send trending tickers from Discord community
            try:
                trending = await get_trending_tickers()
                if trending:
                    _create_background_task(notify_trending_tickers(trending))
            except Exception as trend_err:
                logger.debug(f"Trending tickers fetch failed: {trend_err}")

            # Send risk metrics (Sharpe, drawdown, win rate)
            try:
                from performance import compute as perf_compute
                report = await asyncio.to_thread(perf_compute, 1)  # 1-day report
                _create_background_task(notify_risk_metrics(
                    sharpe_ratio=report.sharpe_ratio,
                    max_drawdown=report.max_drawdown_pct,
                    win_rate=(len(wins) / len(closed_today) * 100) if closed_today else 0,
                ))
            except Exception as risk_err:
                logger.debug(f"Risk metrics calculation failed: {risk_err}")

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


async def daily_goal_progress_loop():
    """
    Send periodic updates on daily profit goal progress every 2 hours during market hours.
    Tracks current PnL against daily target and motivates the trader.
    """
    await asyncio.sleep(10 * 60)   # wait 10 min after startup
    while True:
        try:
            # Check if market is open
            market_open = await asyncio.wait_for(
                asyncio.to_thread(broker.is_market_open), timeout=10
            )
            if not market_open:
                await asyncio.sleep(15 * 60)   # check again in 15 min
                continue

            # Get daily PnL and trade count
            today = _dt.datetime.now(_dt.timezone.utc).date()
            all_trades = await asyncio.to_thread(database.get_trade_history, 200)

            closed_today = [
                t for t in all_trades
                if t.get("exit_time") and t["exit_time"][:10] == str(today)
            ]

            current_pnl = sum(t.get("pnl_gross") or 0 for t in closed_today)
            trades_count = len(closed_today)

            # Get daily target (default: 2% of max budget)
            daily_target = float(os.getenv("DAILY_PROFIT_TARGET",
                                          str(settings.MAX_BUDGET * 0.02)))

            # Send goal progress notification
            if trades_count > 0 or current_pnl != 0:
                _create_background_task(notify_daily_goal_progress(
                    current_pnl=current_pnl,
                    daily_target=daily_target,
                    trades_count=trades_count,
                ))

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"Daily goal progress error: {e}")

        await asyncio.sleep(2 * 60 * 60)   # every 2 hours during market hours


async def market_closed_training_loop():
    """
    Chart learning loop — runs whenever the market is closed.

    Flow:
      1. On startup (after 30s) — immediately learn from the bot's own past trades
      2. Then every 4 hours when market is NOT open — re-learn + run general backtest
      3. Each cycle: analyzes chart indicators at entry, explains why charts moved,
         updates MIN_BUY_SCORE based on findings
    """
    import datetime as _dt
    await asyncio.sleep(30)   # 30s after startup — then learn immediately

    _last_own_sim_date  = None
    _last_tg_notify_ts  = 0.0   # last time we sent Telegram training update

    while True:
        is_open = False   # safe default — used at end of iteration in sleep()
        try:
            import os as _os

            # ── Always run own-trade chart analysis once per day ──────────
            today_str = _dt.datetime.utcnow().strftime("%Y-%m-%d")
            if _last_own_sim_date != today_str:
                logger.info("[TRAINING] Learning from bot's own past trade charts...")
                from backtest_learner import simulate_own_trade_history
                own_summary = await asyncio.wait_for(
                    asyncio.to_thread(simulate_own_trade_history),
                    timeout=300
                )
                _last_own_sim_date = today_str
                if own_summary.get("simulated", 0) > 0:
                    logger.info(
                        f"[TRAINING] Own-trade chart analysis: "
                        f"{own_summary['simulated']} trades | "
                        f"הצלחה={own_summary['win_rate']:.0f}% | "
                        f"תשואה={own_summary['avg_return']:+.2f}%"
                    )

            # ── אימון רץ תמיד — גם בשוק פתוח (קל יותר) וגם בסגור (מלא) ──
            is_open = await asyncio.wait_for(
                asyncio.to_thread(broker.is_market_open), timeout=10
            )

            from backtest_learner import run_backtest, apply_insights
            from scanner import get_watchlist as _gwl
            # שוק פתוח = ניתוח קצר על 10 מניות; שוק סגור = ניתוח עמוק על 20
            tickers = _gwl()[:10] if is_open else _gwl()[:20]
            mode_label = "קל (שוק פתוח)" if is_open else "מלא (שוק סגור)"
            logger.info(f"[TRAINING] Running backtest — mode={mode_label}, {len(tickers)} tickers")

            # ── הודעת "מתחיל אימון" כל 30 דקות ──────────────────────────
            import time as _t
            now_ts = _t.time()
            _send_telegram_results = now_ts - _last_tg_notify_ts >= 30 * 60
            if _send_telegram_results:
                _last_tg_notify_ts = now_ts   # update immediately to prevent re-fire on timeout
                _create_background_task(send_message(
                    f"🧠 <b>מתחיל אימון — {mode_label}</b>\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"📋 מניות לניתוח:\n"
                    + "\n".join(f"   • {t}" for t in tickers)
                    + f"\n\n⏳ מנתח היסטוריה של כל מניה..."
                ))

            # ── Run general historical backtest on watchlist ──────────────
            result = await asyncio.wait_for(
                run_backtest(tickers),
                timeout=600  # 10 min max
            )
            update = await apply_insights()

            score_line = (
                f"🔄 ציון עודכן: {update['old_score']} → <b>{update['new_score']}</b>"
                if update.get("applied")
                else f"✅ ציון נוכחי אופטימלי ({_os.getenv('MIN_BUY_SCORE', '51')})"
            )

            logger.info(
                f"[TRAINING] Done: {result.tickers_analyzed} tickers | "
                f"{result.total_signals} signals | הצלחה={result.overall_win_rate:.1f}% | "
                f"optimal_score={result.optimal_min_score}"
            )

            # ── שלח תוצאות לטלגרם אם שלחנו "מתחיל אימון" — באותו מחזור ────
            if _send_telegram_results:
                own_summary_fresh = await asyncio.wait_for(
                    asyncio.to_thread(__import__('backtest_learner').simulate_own_trade_history),
                    timeout=120
                )
                own_line = (
                    f"🔁 <b>עסקאות שלי:</b> {own_summary_fresh['simulated']} | "
                    f"הצלחה={own_summary_fresh['win_rate']:.0f}% | תשואה={own_summary_fresh['avg_return']:+.1f}%\n"
                    if own_summary_fresh.get("simulated", 0) > 0 else ""
                )
                # פירוט פר-מניה — ממוין לפי אחוז הצלחה
                ticker_stats = getattr(result, "ticker_stats", [])
                if ticker_stats:
                    stats_lines = []
                    for ts in ticker_stats:
                        wr   = ts["win_rate"]
                        ret  = ts["avg_return"]
                        icon = "✅" if wr >= 55 else ("⚠️" if wr >= 40 else "❌")
                        stats_lines.append(
                            f"{icon} <b>{ts['ticker']}</b>: "
                            f"הצלחה={wr:.0f}% | תשואה={ret:+.1f}% | {ts['signals']} הזדמנויות"
                        )
                    ticker_block = "\n".join(stats_lines)
                else:
                    ticker_block = "\n".join(f"   • {t}" for t in tickers)

                _create_background_task(send_message(
                    f"🎓 <b>אימון הושלם</b>\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"{own_line}"
                    f"📊 <b>{result.tickers_analyzed} מניות שנבדקו:</b>\n"
                    f"{ticker_block}\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"✅ אחוז הצלחה כולל: <b>{result.overall_win_rate:.1f}%</b>\n"
                    f"📈 תשואה ממוצעת: {result.avg_return:+.2f}%\n"
                    f"🎯 ציון מומלץ: <b>{result.optimal_min_score}</b>\n"
                    f"{score_line}"
                ))

        except asyncio.TimeoutError:
            logger.warning("[TRAINING] Backtest timed out — will retry in 30 min")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[TRAINING] Training error: {e}")

        # Loop: 5 min during market hours, 1 min when closed (cache prevents heavy load)
        await asyncio.sleep(300 if is_open else 60)


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
            result = await run_backtest(_gwl()[:25])
            update = await apply_insights()
            await send_message(
                f"🎓 <b>למידה מהיסטוריה</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📊 {result.tickers_analyzed} מניות | {result.total_signals} הזדמנויות שזוהו\n"
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

            # Check every minute — CancelledError must propagate for clean shutdown
            while datetime.datetime.now(_utc) < target:
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    raise

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


async def continuous_learning_loop():
    """
    Hourly learning cycle: analyze errors, sentiment correlation, and live performance.
    Sends insights to trader about what the bot is learning and how to improve.

    Runs every hour during market hours to adapt in real-time.
    """
    await asyncio.sleep(10 * 60)   # wait 10 min after startup
    while True:
        try:
            # Check if market is open
            market_open = await asyncio.wait_for(
                asyncio.to_thread(broker.is_market_open), timeout=10
            )
            if not market_open:
                await asyncio.sleep(15 * 60)   # check again in 15 min
                continue

            # Run the continuous learning cycle
            from continuous_learner import run_continuous_learning_cycle, get_learning_summary

            logger.info("[LEARNING] Running continuous learning cycle...")
            results = await asyncio.wait_for(
                run_continuous_learning_cycle(),   # ✅ fixed: async, no to_thread
                timeout=120
            )

            # Send summary to trader (hourly during market hours)
            if results.get("error_patterns") or results.get("live_performance", {}).get("recommendations"):
                summary = get_learning_summary()
                _create_background_task(send_message(summary))
                logger.info("[LEARNING] Insights sent to trader")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Continuous learning error: {e}")
            await asyncio.sleep(300)  # retry in 5 min on error

        await asyncio.sleep(60 * 60)   # run every hour


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE TRADER MONITORING LOOP
# ─────────────────────────────────────────────────────────────────────────────

async def adaptive_parameters_monitor_loop():
    """
    Monitor and send adaptive trading parameters every 4 hours.
    Helps trader understand how the bot adjusts to market conditions.
    """
    await asyncio.sleep(5 * 60)   # wait 5 min after startup
    while True:
        try:
            # Only during market hours
            market_open = await asyncio.wait_for(
                asyncio.to_thread(broker.is_market_open), timeout=10
            )
            if market_open:
                from telegram_bot import notify_adaptive_parameters
                await notify_adaptive_parameters()
                logger.info("[ADAPTIVE] Parameters sent to trader")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Adaptive parameters monitor error: {e}")

        await asyncio.sleep(4 * 60 * 60)   # run every 4 hours


# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION MONITORING LOOP
# ─────────────────────────────────────────────────────────────────────────────

async def correlation_monitor_loop():
    """
    Monitor position correlations every 2 hours.
    Warns trader if positions are becoming too correlated.
    """
    await asyncio.sleep(3 * 60)   # wait 3 min after startup
    while True:
        try:
            market_open = await asyncio.wait_for(
                asyncio.to_thread(broker.is_market_open), timeout=10
            )
            if market_open:
                from telegram_bot import notify_correlation_analysis
                await notify_correlation_analysis()
                logger.info("[CORRELATION] Analysis sent to trader")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Correlation monitor error: {e}")

        await asyncio.sleep(2 * 60 * 60)   # run every 2 hours


# ─────────────────────────────────────────────────────────────────────────────
# MARKET INTELLIGENCE LOOP
# ─────────────────────────────────────────────────────────────────────────────

async def market_intelligence_loop():
    """
    Send comprehensive market analysis every 3 hours.
    Includes volatility regime, sector rotation, market breadth.
    """
    await asyncio.sleep(7 * 60)   # wait 7 min after startup
    while True:
        try:
            market_open = await asyncio.wait_for(
                asyncio.to_thread(broker.is_market_open), timeout=10
            )
            if market_open:
                from telegram_bot import notify_market_regime_analysis
                await notify_market_regime_analysis()
                logger.info("[MARKET INTEL] Regime analysis sent to trader")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Market intelligence loop error: {e}")

        await asyncio.sleep(3 * 60 * 60)   # run every 3 hours


# ─────────────────────────────────────────────────────────────────────────────
# DETAILED TRADE ANALYTICS LOOP
# ─────────────────────────────────────────────────────────────────────────────

async def detailed_analytics_loop():
    """
    Send detailed trade analytics every 2 hours.
    Shows P&L by ticker, best/worst trades, performance comparison.
    """
    await asyncio.sleep(9 * 60)   # wait 9 min after startup
    while True:
        try:
            market_open = await asyncio.wait_for(
                asyncio.to_thread(broker.is_market_open), timeout=10
            )
            if market_open:
                from telegram_bot import (
                    notify_detailed_trade_analytics,
                    notify_performance_comparison,
                    notify_ai_trading_insights
                )

                # Rotate through different analytics
                hour = asyncio.get_event_loop().time()
                analytics_cycle = int(hour / 3600) % 3

                if analytics_cycle == 0:
                    await notify_detailed_trade_analytics()
                elif analytics_cycle == 1:
                    await notify_performance_comparison()
                else:
                    await notify_ai_trading_insights()

                logger.info("[ANALYTICS] Detailed analytics sent to trader")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Detailed analytics loop error: {e}")

        await asyncio.sleep(2 * 60 * 60)   # run every 2 hours


# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY DETECTION LOOP
# ─────────────────────────────────────────────────────────────────────────────

async def anomaly_detection_loop():
    """
    Scan portfolio for anomalies every 2 hours during market hours.
    Detects unusual price/volume movements and performance anomalies.
    """
    await asyncio.sleep(22 * 60)  # wait 22 min after startup
    while True:
        try:
            market_open = await asyncio.wait_for(
                asyncio.to_thread(broker.is_market_open), timeout=10
            )
            if market_open:
                from anomaly_detector import scan_portfolio_anomalies
                results = await asyncio.wait_for(
                    scan_portfolio_anomalies(), timeout=120
                )

                if results.get("critical_count", 0) > 0 or results.get("high_count", 0) > 0:
                    # Send alert
                    lines = ["🚨 <b>Anomaly Detected</b>", "━━━━━━━━━━━━━━━"]

                    critical = [a for a in results.get("anomalies", []) if a["severity"] == "critical"]
                    high = [a for a in results.get("anomalies", []) if a["severity"] == "high"]

                    for anomaly in (critical + high)[:5]:
                        emoji = "🔴" if anomaly["severity"] == "critical" else "🟠"
                        lines.append(f"{emoji} {anomaly['ticker']}: {anomaly['description']}")

                    from smart_notifications import notify_high
                    await notify_high(
                        title="Market Anomaly Detected",
                        message="\n".join(lines),
                        category="market",
                    )

                logger.info(f"[ANOMALY] Scanned: {results.get('count', 0)} anomalies found")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Anomaly detection loop error: {e}")

        await asyncio.sleep(2 * 60 * 60)   # every 2 hours


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH MONITORING LOOP
# ─────────────────────────────────────────────────────────────────────────────

async def health_monitoring_loop():
    """
    Monitor system health every 10 minutes.
    Sends alerts on critical issues.
    """
    await asyncio.sleep(2 * 60)  # wait 2 min after startup
    while True:
        try:
            from health_monitor import run_health_check, perform_auto_recovery
            report = await run_health_check()

            # Auto-recovery for critical issues
            if report.overall_status == "critical":
                recovery = await perform_auto_recovery(report)

                # Notify trader
                from smart_notifications import notify_critical
                issues_text = "\n".join(report.issues[:5])
                actions_text = "\n".join(recovery.get("actions_taken", []))

                await notify_critical(
                    title="System Health Critical",
                    message=f"<b>Issues:</b>\n{issues_text}\n\n<b>Auto-recovery:</b>\n{actions_text}",
                    category="system",
                )

            elif report.overall_status == "degraded":
                # Just log for degraded state
                logger.warning(f"[HEALTH] Degraded: {len(report.issues)} issues")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Health monitoring error: {e}")

        await asyncio.sleep(10 * 60)   # run every 10 minutes


# ─────────────────────────────────────────────────────────────────────────────
# NEWS CATALYST LOOP
# ─────────────────────────────────────────────────────────────────────────────

async def news_catalyst_loop():
    """
    Scan portfolio for news catalysts every 30 minutes.
    Alerts on breaking news.
    """
    await asyncio.sleep(20 * 60)  # wait 20 min after startup
    while True:
        try:
            market_open = await asyncio.wait_for(
                asyncio.to_thread(broker.is_market_open), timeout=10
            )
            if market_open:
                from news_intelligence import get_portfolio_news
                news_data = await get_portfolio_news()

                if "error" not in news_data:
                    catalysts = news_data.get("catalysts", [])
                    breaking = news_data.get("breaking_news", [])

                    # Alert on breaking high-impact news
                    if breaking:
                        from smart_notifications import notify_high
                        for article in breaking[:3]:
                            tickers_str = ", ".join(article.get("tickers", []))
                            await notify_high(
                                title=f"📰 Breaking News: {tickers_str}",
                                message=f"<b>{article['title']}</b>\nSentiment: {article.get('sentiment', 'neutral')}",
                                category="market",
                                subkey=tickers_str,
                            )

                    # Send catalyst summary
                    if catalysts:
                        lines = ["📰 <b>News Catalysts</b>", "━━━━━━━━━━━━━━"]
                        for cat in catalysts[:3]:
                            tickers_str = ", ".join(cat.get("tickers", []))
                            lines.append(f"  • {tickers_str}: {cat['title'][:80]}...")
                            lines.append(f"    Impact: {cat.get('impact', 0):.1f}/10 | {cat.get('sentiment', 'neutral')}")

                        from smart_notifications import notify_medium
                        await notify_medium(
                            title="News Catalysts Detected",
                            message="\n".join(lines),
                            category="market",
                        )

                logger.info(f"[NEWS] Scanned portfolio: {news_data.get('total_articles', 0)} articles")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"News catalyst loop error: {e}")

        await asyncio.sleep(30 * 60)   # run every 30 minutes


# ─────────────────────────────────────────────────────────────────────────────
# PAIRS TRADING SCANNER LOOP
# ─────────────────────────────────────────────────────────────────────────────

async def pairs_trading_loop():
    """
    Scan for pairs trading opportunities every 4 hours during market hours.
    """
    await asyncio.sleep(25 * 60)  # wait 25 min after startup
    while True:
        try:
            market_open = await asyncio.wait_for(
                asyncio.to_thread(broker.is_market_open), timeout=10
            )
            if market_open:
                from pairs_trading import scan_pairs_opportunities
                opportunities = await asyncio.wait_for(
                    scan_pairs_opportunities(),
                    timeout=180,
                )

                if opportunities:
                    lines = ["🔀 <b>Pairs Trading Opportunities</b>", "━━━━━━━━━━━━━━━━━"]
                    for opp in opportunities[:3]:
                        lines.append(
                            f"  • {opp['ticker1']}/{opp['ticker2']}: z={opp['z_score']:.2f}, conf={opp['confidence']:.0%}"
                        )
                        lines.append(f"    → {opp['expected_action']}")

                    from smart_notifications import notify_medium
                    await notify_medium(
                        title="Pairs Trading Opportunities",
                        message="\n".join(lines),
                        category="market",
                    )

                logger.info(f"[PAIRS] Found {len(opportunities)} opportunities")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Pairs trading loop error: {e}")

        await asyncio.sleep(4 * 60 * 60)   # every 4 hours


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK COMPARISON LOOP
# ─────────────────────────────────────────────────────────────────────────────

async def benchmark_comparison_loop():
    """
    Compare bot performance vs benchmarks daily.
    """
    await asyncio.sleep(35 * 60)  # wait 35 min after startup
    while True:
        try:
            from benchmark_compare import compare_to_benchmark

            spy_comparison = await compare_to_benchmark("SPY", days=30)

            if spy_comparison.bot_return_pct != 0 or spy_comparison.benchmark_return_pct != 0:
                lines = [
                    "📊 <b>Performance vs S&P 500 (30d)</b>",
                    "━━━━━━━━━━━━━━━━━━",
                    f"🤖 Bot Return: {spy_comparison.bot_return_pct:+.2f}%",
                    f"📈 SPY Return: {spy_comparison.benchmark_return_pct:+.2f}%",
                    f"⚡ Alpha: {spy_comparison.alpha_pct:+.2f}%",
                    f"📌 Status: {spy_comparison.win_rate_vs_market}",
                ]

                from smart_notifications import notify_medium
                await notify_medium(
                    title="Benchmark Comparison",
                    message="\n".join(lines),
                    category="learning",
                )

                logger.info(f"[BENCHMARK] Alpha vs SPY: {spy_comparison.alpha_pct:+.2f}%")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Benchmark loop error: {e}")

        await asyncio.sleep(24 * 60 * 60)   # daily


# ─────────────────────────────────────────────────────────────────────────────
# TRADE JOURNAL LOOP
# ─────────────────────────────────────────────────────────────────────────────

async def trade_journal_loop():
    """
    Generate trade journal reviews daily.
    """
    await asyncio.sleep(45 * 60)  # wait 45 min after startup
    while True:
        try:
            from trade_journal import generate_journal_summary

            summary = await generate_journal_summary(days=7)

            if summary.total_trades_reviewed > 0:
                lines = [
                    "📓 <b>Trade Journal (7 days)</b>",
                    "━━━━━━━━━━━━━━━━",
                    f"📊 Trades reviewed: {summary.total_trades_reviewed}",
                    f"⭐ Average grade: {summary.avg_quality_score:.0f}/100",
                    "",
                ]

                # Grade distribution
                dist = summary.grade_distribution
                lines.append(f"📋 Grades: A:{dist.get('A',0)} B:{dist.get('B',0)} C:{dist.get('C',0)} D:{dist.get('D',0)} F:{dist.get('F',0)}")

                # Strengths
                if summary.strengths:
                    lines.extend(["", "<b>💪 Strengths:</b>"])
                    for s in summary.strengths[:3]:
                        lines.append(f"  • {s}")

                # Improvement areas
                if summary.improvement_areas:
                    lines.extend(["", "<b>📈 To Improve:</b>"])
                    for i in summary.improvement_areas[:3]:
                        lines.append(f"  • {i}")

                from smart_notifications import notify_medium
                await notify_medium(
                    title="Trade Journal Review",
                    message="\n".join(lines),
                    category="learning",
                )

                logger.info(f"[JOURNAL] Reviewed {summary.total_trades_reviewed} trades")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Trade journal loop error: {e}")

        await asyncio.sleep(24 * 60 * 60)   # daily


# ─────────────────────────────────────────────────────────────────────────────
# AI DECISION ENGINE LOOP
# ─────────────────────────────────────────────────────────────────────────────

async def ai_decision_loop():
    """
    Use AI Decision Engine to evaluate current positions every hour.
    Combines all intelligence modules into unified decisions.
    """
    await asyncio.sleep(12 * 60)  # wait 12 min after startup
    while True:
        try:
            market_open = await asyncio.wait_for(
                asyncio.to_thread(broker.is_market_open), timeout=10
            )
            if market_open:
                from ai_decision_engine import make_trading_decision
                positions = await asyncio.to_thread(broker.get_positions)

                if positions:
                    high_risk_positions = []
                    for p in positions[:5]:  # Limit to 5 to avoid rate limits
                        try:
                            decision = await asyncio.wait_for(
                                make_trading_decision(
                                    ticker=p.symbol,
                                    current_price=float(p.current_price),
                                ),
                                timeout=30,
                            )

                            if decision.action in ("SELL", "STRONG_SELL"):
                                high_risk_positions.append((p.symbol, decision))

                        except Exception as e:
                            logger.debug(f"AI decision for {p.symbol} failed: {e}")

                    # Alert if any positions flagged for exit
                    if high_risk_positions:
                        from smart_notifications import notify_high
                        for ticker, decision in high_risk_positions:
                            await notify_high(
                                title=f"🔴 AI: Consider closing {ticker}",
                                message=f"{decision.final_explanation}\nRisk: {decision.risk_score:.0f}/100",
                                category="trade",
                                subkey=ticker,
                            )

                logger.info("[AI DECISION] Cycle complete")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"AI decision loop error: {e}")

        await asyncio.sleep(60 * 60)   # run every hour


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE ATTRIBUTION LOOP
# ─────────────────────────────────────────────────────────────────────────────

async def attribution_loop():
    """
    Generate performance attribution insights daily.
    Tells trader exactly what's working and what isn't.
    """
    await asyncio.sleep(15 * 60)  # wait 15 min after startup
    while True:
        try:
            from performance_attribution import get_actionable_insights
            insights = await get_actionable_insights()

            if insights:
                lines = [
                    "📊 <b>תובנות ביצועים מהשבוע האחרון</b>",
                    "━━━━━━━━━━━━━━━━━━━",
                ]
                lines.extend([f"  {insight}" for insight in insights])

                from smart_notifications import notify_medium
                await notify_medium(
                    title="Performance Insights",
                    message="\n".join(lines),
                    category="learning",
                )

                logger.info(f"[ATTRIBUTION] Sent {len(insights)} insights")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Attribution loop error: {e}")

        await asyncio.sleep(24 * 60 * 60)   # run daily


# ─────────────────────────────────────────────────────────────────────────────
# NOTIFICATION DIGEST LOOP
# ─────────────────────────────────────────────────────────────────────────────

async def notification_digest_loop():
    """
    Send batched notifications periodically (hourly digest + daily summary).
    """
    await asyncio.sleep(30 * 60)  # wait 30 min for batches to accumulate
    while True:
        try:
            from smart_notifications import send_hourly_digest, send_daily_digest

            now = _dt.datetime.now(_dt.timezone.utc)

            # Send daily digest at end of trading day (after EOD)
            if now.hour == 21:  # 4PM EST = 9PM UTC
                await send_daily_digest()
            else:
                # Hourly digest of low/medium priority items
                await send_hourly_digest()

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Notification digest error: {e}")

        await asyncio.sleep(60 * 60)   # run every hour


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-TIMEFRAME ANALYSIS LOOP
# ─────────────────────────────────────────────────────────────────────────────

async def multi_timeframe_loop():
    """
    Find high-confidence trading opportunities using multi-timeframe alignment.
    Runs every 2 hours during market hours.
    """
    await asyncio.sleep(18 * 60)  # wait 18 min after startup
    while True:
        try:
            market_open = await asyncio.wait_for(
                asyncio.to_thread(broker.is_market_open), timeout=10
            )
            if market_open:
                from multi_timeframe import find_confluence_opportunities

                # Check current positions for confluence
                positions = await asyncio.to_thread(broker.get_positions)
                if positions:
                    tickers = [p.symbol for p in positions[:5]]
                    opportunities = await asyncio.wait_for(
                        find_confluence_opportunities(tickers),
                        timeout=120,
                    )

                    if opportunities:
                        lines = ["🎯 <b>Multi-Timeframe Confluence</b>", "━━━━━━━━━━━━━━━"]
                        for opp in opportunities[:3]:
                            lines.append(
                                f"  • {opp['ticker']}: {opp['alignment_score']:.0%} alignment | {opp['recommendation']}"
                            )

                        from smart_notifications import notify_medium
                        await notify_medium(
                            title="MTF Analysis",
                            message="\n".join(lines),
                            category="market",
                        )

                logger.info("[MTF] Confluence analysis complete")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Multi-timeframe loop error: {e}")

        await asyncio.sleep(2 * 60 * 60)   # run every 2 hours


# ─────────────────────────────────────────────────────────────────────────────
# STALE POSITION GUARD LOOP
# ─────────────────────────────────────────────────────────────────────────────

async def stale_position_guard_loop():
    """
    Check for stale positions every 4 hours.
    Alerts on positions held too long without progress.
    """
    await asyncio.sleep(30 * 60)  # wait 30 min after startup
    while True:
        try:
            from stale_position_guard import scan_stale_positions, notify_stale_positions, tighten_wide_stop
            import database

            recommendations = await scan_stale_positions()

            if recommendations:
                # Auto-tighten wide stops
                open_trades = await asyncio.to_thread(database.get_open_trades)
                trade_map = {t["ticker"]: t for t in (open_trades or [])}

                for rec in recommendations:
                    if rec["action"] == "TIGHTEN_STOP":
                        trade = trade_map.get(rec["ticker"])
                        if trade:
                            await tighten_wide_stop(
                                rec["ticker"], trade, rec["current_price"]
                            )

                # Notify about urgent ones
                urgent = [r for r in recommendations if r["urgency"] == "high"]
                if urgent:
                    await notify_stale_positions(urgent)

                logger.info(f"[STALE GUARD] {len(recommendations)} stale positions found")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Stale position guard error: {e}")

        await asyncio.sleep(4 * 60 * 60)  # every 4 hours


# ─────────────────────────────────────────────────────────────────────────────
# FAST TRACK AUTO-PROGRESSION LOOP
# ─────────────────────────────────────────────────────────────────────────────

async def fast_track_progress_loop():
    """
    Auto-progress through Fast Track stages.
    Every 4 hours checks if criteria met for promotion.
    """
    try:
        from fast_track_live import auto_progress_check_loop
        await auto_progress_check_loop()
    except Exception as e:
        logger.error(f"Fast track loop error: {e}")
        await asyncio.sleep(60 * 60)
