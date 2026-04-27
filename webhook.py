import asyncio
import logging
import time
from fastapi import APIRouter, HTTPException, Request
from config import settings
from models import WebhookPayload, TradeAction, HealthResponse, BudgetStatus, BrokerSwitch
import broker
import budget
import database
import scanner
from signal_validator import validate_signal
from trade_logger import log_trade_open, log_trade_close, log_learning
from circuit_breaker import check_circuit_breaker, record_trade_result, get_status as cb_status
from trading_hours import get_status as hours_status
from iceberg import get_status as iceberg_status
from telegram_bot import (
    notify_trade_open, notify_trade_close, notify_error,
    notify_circuit_breaker_tripped, notify_budget_warning,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Rate limiting: track requests by IP/timestamp to prevent spam
_request_history = {}  # ip -> list of timestamps
_RATE_LIMIT_WINDOW = 60  # seconds
_RATE_LIMIT_MAX_REQUESTS = 100  # max requests per window

def _check_rate_limit(request: Request) -> bool:
    """Check if request exceeds rate limit. Returns True if allowed."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    if client_ip not in _request_history:
        _request_history[client_ip] = []

    # Remove old entries outside the window
    _request_history[client_ip] = [
        ts for ts in _request_history[client_ip]
        if now - ts < _RATE_LIMIT_WINDOW
    ]

    # Check if over limit
    if len(_request_history[client_ip]) >= _RATE_LIMIT_MAX_REQUESTS:
        return False

    # Add current timestamp
    _request_history[client_ip].append(now)

    # Cleanup: remove IPs with empty lists to prevent memory leak
    empty_ips = [ip for ip, ts in _request_history.items() if not ts]
    for ip in empty_ips:
        del _request_history[ip]

    return True


async def _shadow_eval(
    ticker: str,
    price: float,
    composite_score: float,
    sentiment_score: int,
    volume_ratio: float | None,
    live_blocked_by: str | None,
    live_block_reason: str,
    signal_source: str = "webhook",
) -> None:
    """Fire shadow.evaluate in a background thread — fire-and-forget."""
    try:
        import shadow as _shadow
        await asyncio.to_thread(
            _shadow.evaluate, ticker, price, composite_score,
            sentiment_score, volume_ratio, live_blocked_by,
            live_block_reason, signal_source,
        )
    except Exception as exc:
        logger.debug(f"[SHADOW] eval error for {ticker}: {exc}")


def _trade_duration_hours(entry_time_str: str | None) -> float:
    """Parse SQLite entry_time string and return elapsed hours."""
    if not entry_time_str:
        return 0.0
    try:
        from datetime import datetime, timezone
        fmt = "%Y-%m-%d %H:%M:%S"
        entry = datetime.strptime(str(entry_time_str)[:19], fmt).replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - entry).total_seconds() / 3600
    except Exception:
        return 0.0


@router.post("/webhook")
async def receive_webhook(payload: WebhookPayload, request: Request):
    """
    TradingView webhook receiver.
    Pipeline: authenticate -> validate -> sentiment -> budget -> execute -> log
    """
    # 0. Rate limiting check
    if not _check_rate_limit(request):
        logger.warning(f"Webhook rate limit exceeded from {request.client.host if request.client else 'unknown'}")
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Max 100 requests per 60 seconds")

    # 1. Authenticate — reject if secret is empty (misconfigured) or wrong
    if not settings.WEBHOOK_SECRET:
        logger.error("WEBHOOK_SECRET is not configured — rejecting all webhook requests")
        raise HTTPException(status_code=503, detail="Bot not configured: WEBHOOK_SECRET missing")
    if payload.secret != settings.WEBHOOK_SECRET:
        logger.warning(f"Webhook auth failed for {payload.ticker} from {request.client.host if request.client else 'unknown'}")
        raise HTTPException(status_code=401, detail="Invalid secret")

    # 2. Validate signal (wrapped — may call broker API)
    is_valid, reason = await asyncio.to_thread(validate_signal, payload.ticker, payload.action.value)
    if not is_valid:
        logger.info(f"Signal rejected: {payload.ticker} {payload.action.value} - {reason}")
        return {"status": "rejected", "reason": reason}

    # 3. Circuit breaker check (buys only)
    if payload.action == TradeAction.BUY:
        ok, cb_reason = check_circuit_breaker()
        if not ok:
            logger.warning(f"BUY blocked by circuit breaker: {cb_reason}")
            return {"status": "blocked_by_circuit_breaker", "reason": cb_reason}

    # 4. Process based on action
    if payload.action == TradeAction.BUY:
        return await _handle_buy(payload)
    else:
        return await _handle_sell(payload)


async def _handle_buy(payload: WebhookPayload) -> dict:
    """Handle a buy signal."""
    ticker = payload.ticker.upper()

    # Check if already have an open position
    existing = database.get_open_trade_by_ticker(ticker)
    if existing:
        return {"status": "skipped", "reason": f"Already have open position for {ticker}"}

    # Shadow mode tracking (updated as we learn more about the signal)
    _cscore:    float        = 50.0   # composite score — set after scoring step
    _vol_ratio: float | None = None   # volume ratio   — set after volume check
    _sent:      int          = 0      # sentiment score — set after sentiment step

    # Sentiment check (MANDATORY for buys)
    from sentiment import score_sentiment
    try:
        sentiment = await asyncio.wait_for(
            asyncio.to_thread(score_sentiment, ticker), timeout=45
        )
    except asyncio.TimeoutError:
        asyncio.ensure_future(notify_error("api_timeout", ticker, "Sentiment check timed out after 45s"))
        return {"status": "rejected", "reason": "Sentiment check timed out"}
    except Exception as e:
        logger.error(f"Sentiment check failed for {ticker}: {e}")
        asyncio.ensure_future(notify_error("sentiment_fail", ticker, str(e)))
        return {"status": "rejected", "reason": f"Sentiment check failed: {e}"}

    _sent = sentiment.score
    if sentiment.score < settings.SENTIMENT_MIN_SCORE:
        logger.info(
            f"BUY blocked by sentiment: {ticker} score={sentiment.score}/10 "
            f"(min={settings.SENTIMENT_MIN_SCORE})"
        )
        asyncio.ensure_future(_shadow_eval(
            ticker, payload.price, _cscore, _sent, None,
            "sentiment", f"score={sentiment.score} < {settings.SENTIMENT_MIN_SCORE}",
        ))
        return {
            "status": "blocked_by_sentiment",
            "ticker": ticker,
            "sentiment_score": sentiment.score,
            "reasoning": sentiment.reasoning,
        }

    # Composite scoring — combines ALL indicators into one score
    from scoring import get_composite_score
    from indicators import get_current_indicators
    from learning import should_override_buy
    try:
        result = await asyncio.to_thread(get_composite_score, ticker, sentiment.score)
        indicators = await asyncio.to_thread(get_current_indicators, ticker) or {}
    except Exception as e:
        logger.warning(f"Scoring failed for {ticker}: {e}")
        result = {"should_buy": True, "composite_score": 50}
        indicators = {}

    _cscore = result.get("composite_score", 50.0)
    if not result["should_buy"]:
        asyncio.ensure_future(_shadow_eval(
            ticker, payload.price, _cscore, _sent, None,
            "score", f"composite_score={_cscore:.0f} < {result.get('min_score', 65)}",
        ))
        return {
            "status": "blocked_by_score",
            "ticker": ticker,
            "composite_score": result["composite_score"],
            "min_score": result["min_score"],
            "reason": f"Score {result['composite_score']}/100 < min {result['min_score']}",
            "breakdown": result.get("breakdown", {}),
        }

    # Learning check - known loss patterns
    should_block, block_reason = await asyncio.to_thread(should_override_buy, ticker, indicators)
    if should_block:
        logger.info(f"BUY blocked by learning: {ticker} - {block_reason}")
        asyncio.ensure_future(_shadow_eval(
            ticker, payload.price, _cscore, _sent, None,
            "learning", block_reason,
        ))
        return {"status": "blocked_by_learning", "reason": block_reason}

    # Sanity check — price plausibility, velocity, data completeness
    from sanity_check import run_all as sanity_run
    try:
        sane, sane_reason = await asyncio.wait_for(
            asyncio.to_thread(sanity_run, ticker, payload.price, indicators or {}),
            timeout=20,
        )
        if not sane:
            logger.warning(f"BUY blocked by sanity check: {ticker} — {sane_reason}")
            asyncio.ensure_future(_shadow_eval(
                ticker, payload.price, _cscore, _sent, None,
                "sanity", sane_reason,
            ))
            return {"status": "blocked_by_sanity", "reason": sane_reason}
    except asyncio.TimeoutError:
        asyncio.ensure_future(_shadow_eval(
            ticker, payload.price, _cscore, _sent, None,
            "sanity", "sanity check timed out",
        ))
        return {"status": "blocked_by_sanity", "reason": "sanity check timed out"}

    # Market regime filter — skip trend-following buys in ranging markets
    from market_regime import get_regime as _get_regime
    try:
        _regime, _adx, _regime_det = await asyncio.wait_for(
            asyncio.to_thread(_get_regime, ticker), timeout=20
        )
        if _regime == "ranging":
            logger.info(
                f"BUY blocked by market regime: {ticker} ADX={_adx:.1f} "
                f"(ranging — trend-following disabled)"
            )
            asyncio.ensure_future(_shadow_eval(
                ticker, payload.price, _cscore, _sent, None,
                "market_regime", f"ranging ADX={_adx:.1f}",
            ))
            return {
                "status": "blocked_by_market_regime",
                "ticker": ticker,
                "regime": _regime,
                "adx":    round(_adx, 2),
                "reason": f"Ranging market (ADX={_adx:.1f} < {_regime_det.get('threshold', 25)}) — trend-following disabled",
            }
    except asyncio.TimeoutError:
        logger.warning(f"[ADX] {ticker} regime check timed out — proceeding (fail-open)")

    # Volume confirmation — reject low-volume signals
    from volume_confirm import check as vol_check
    try:
        vol_passed, vol_reason, vol_details = await asyncio.wait_for(
            asyncio.to_thread(vol_check, ticker), timeout=15
        )
        _vol_ratio = vol_details.get("ratio")
        if not vol_passed:
            logger.info(f"BUY blocked by volume: {ticker} — {vol_reason}")
            asyncio.ensure_future(_shadow_eval(
                ticker, payload.price, _cscore, _sent, _vol_ratio,
                "volume", vol_reason,
            ))
            return {
                "status": "blocked_by_volume",
                "ticker": ticker,
                "reason": vol_reason,
                "volume": vol_details,
            }
    except asyncio.TimeoutError:
        logger.warning(f"[VOLUME] {ticker} check timed out — proceeding (fail-open)")

    # Correlation filter — skip if too correlated with an open position
    from correlation import check as corr_check
    try:
        corr_blocked, corr_reason, corr_details = await asyncio.wait_for(
            asyncio.to_thread(corr_check, ticker), timeout=25
        )
        if corr_blocked:
            logger.info(f"BUY blocked by correlation: {corr_reason}")
            asyncio.ensure_future(_shadow_eval(
                ticker, payload.price, _cscore, _sent, _vol_ratio,
                "correlation", corr_reason,
            ))
            return {
                "status":      "blocked_by_correlation",
                "ticker":      ticker,
                "reason":      corr_reason,
                "correlation": corr_details,
            }
    except asyncio.TimeoutError:
        logger.warning(f"[CORR] {ticker} check timed out — proceeding (fail-open)")

    # Budget check
    can_buy, max_qty, budget_reason = await asyncio.to_thread(budget.check_can_buy, payload.price)
    if not can_buy:
        logger.info(f"BUY blocked by budget: {ticker} - {budget_reason}")
        try:
            acct = await asyncio.to_thread(broker.get_account)
            cash = float(acct.get("cash", 0))
        except Exception:
            cash = 0.0
        asyncio.ensure_future(notify_budget_warning(budget_reason, cash))
        asyncio.ensure_future(_shadow_eval(
            ticker, payload.price, _cscore, _sent, _vol_ratio,
            "budget", budget_reason,
        ))
        return {"status": "blocked_by_budget", "reason": budget_reason}

    # Execute buy — iceberg splits large orders automatically
    try:
        from iceberg import iceberg_buy
        order = await iceberg_buy(ticker, max_qty, payload.price)
    except asyncio.TimeoutError:
        asyncio.ensure_future(notify_error("api_timeout", ticker, "Buy order timed out after 15s"))
        return {"status": "error", "reason": "Buy order timed out"}
    except Exception as e:
        logger.error(f"BUY order failed for {ticker}: {e}")
        asyncio.ensure_future(notify_error("order_failed", ticker, str(e)))
        return {"status": "error", "reason": f"Order failed: {e}"}

    # Log trade
    actual_price = order.get("price") or payload.price
    trade_id = log_trade_open(payload, sentiment, order, max_qty)

    # Record actual slippage (signal price vs fill price, fire-and-forget)
    from slippage import record as _slippage_record
    asyncio.ensure_future(asyncio.to_thread(
        _slippage_record, payload.price, actual_price, max_qty, "buy", ticker
    ))

    # Set ATR trailing stop immediately after fill
    try:
        from atr_stop import compute_initial_stop
        atr_stop_price, stop_meta = await asyncio.to_thread(
            compute_initial_stop, ticker, actual_price
        )
        await asyncio.to_thread(
            database.update_trade_stop, trade_id, atr_stop_price, actual_price
        )
        logger.info(
            f"[ATR STOP] {ticker}: stop set @ ${atr_stop_price:.2f} "
            f"({stop_meta['stop_pct']:.2f}% from entry, ATR=${stop_meta['atr']:.4f})"
        )
    except Exception as stop_err:
        logger.warning(f"[ATR STOP] {ticker}: failed to set initial stop: {stop_err}")

    # Shadow: live also traded (live_blocked_by=None → both agree)
    asyncio.ensure_future(_shadow_eval(
        ticker, actual_price, _cscore, _sent, _vol_ratio,
        None, "",  # live_blocked_by=None means live also traded
    ))

    # Notify Telegram
    asyncio.ensure_future(notify_trade_open(
        ticker=ticker, qty=max_qty, price=actual_price,
        notional=round(actual_price * max_qty, 2),
        score=result.get("composite_score", 0),
        sentiment_score=sentiment.score,
        trade_id=trade_id,
        is_iceberg=order.get("iceberg", False),
        n_slices=len(order.get("slices", [])),
    ))

    return {
        "status": "executed",
        "action": "buy",
        "ticker": ticker,
        "qty": max_qty,
        "price": actual_price,
        "sentiment_score": sentiment.score,
        "trade_id": trade_id,
        "order_id": order.get("order_id", ""),
    }


async def _handle_sell(payload: WebhookPayload) -> dict:
    """Handle a sell signal."""
    ticker = payload.ticker.upper()

    # Find open trade in our log
    trade = database.get_open_trade_by_ticker(ticker)
    if not trade:
        return {"status": "skipped", "reason": f"No open position for {ticker}"}

    # Execute sell
    try:
        order = await asyncio.wait_for(
            asyncio.to_thread(broker.submit_sell, ticker), timeout=15
        )
    except asyncio.TimeoutError:
        asyncio.ensure_future(notify_error("api_timeout", ticker, "Sell order timed out after 15s"))
        return {"status": "error", "reason": "Sell order timed out"}
    except Exception as e:
        logger.error(f"SELL order failed for {ticker}: {e}")
        asyncio.ensure_future(notify_error("order_failed", ticker, str(e)))
        return {"status": "error", "reason": f"Order failed: {e}"}

    # Calculate PnL
    entry_price = trade["entry_price"]
    exit_price = order.get("price") or payload.price
    qty = trade["qty"]
    pnl_gross = (exit_price - entry_price) * qty
    fees = 0.0  # Alpaca has no commissions for stocks

    # Record actual slippage on sell (fire-and-forget)
    from slippage import record as _slippage_record
    asyncio.ensure_future(asyncio.to_thread(
        _slippage_record, payload.price, exit_price, qty, "sell", ticker
    ))

    # Tax logic
    from tax_tracker import process_trade_close
    tax_result = process_trade_close(trade["id"], pnl_gross)
    pnl_net = pnl_gross - tax_result["tax_amount"]

    # Log trade close
    log_trade_close(
        trade["id"], exit_price, pnl_gross, pnl_net,
        tax_result["tax_amount"], fees, "closed",
    )

    # Check circuit breaker (may have just tripped)
    was_ok, _ = check_circuit_breaker()
    record_trade_result(pnl_gross)
    is_ok, cb_reason = check_circuit_breaker()
    if not is_ok and was_ok:
        cb_st = cb_status()
        asyncio.ensure_future(notify_circuit_breaker_tripped(
            cb_st["daily_pnl"], cb_st["max_daily_loss"], cb_st["trip_reason"]
        ))

    # Log learning entry
    outcome = "profit" if pnl_gross > 0 else "loss"
    indicators = {
        "entry_rsi": trade.get("rsi"),
        "entry_macd": trade.get("macd"),
        "sentiment_score": trade.get("sentiment_score"),
    }
    description = (
        f"{outcome.upper()}: {ticker} bought@${entry_price:.2f} sold@${exit_price:.2f} "
        f"PnL=${pnl_gross:+.2f} sentiment={trade.get('sentiment_score')}"
    )
    log_learning(trade["id"], description, f"{outcome}_pattern", indicators, outcome, pnl_gross)

    # Cleanup per-ticker state so re-entry works correctly
    from heartbeat import _smart_sell_last_check, _position_alert_sent
    _smart_sell_last_check.pop(ticker, None)
    _position_alert_sent.pop(ticker, None)

    # Notify Telegram — full P&L breakdown
    duration_hours = _trade_duration_hours(trade.get("entry_time"))
    asyncio.ensure_future(notify_trade_close(
        ticker=ticker, qty=qty,
        entry_price=entry_price, exit_price=exit_price,
        pnl_gross=pnl_gross, pnl_net=pnl_net,
        tax_reserved=tax_result["tax_amount"],
        duration_hours=duration_hours,
        reason="TradingView signal",
        trade_id=trade["id"],
    ))

    return {
        "status": "executed",
        "action": "sell",
        "ticker": ticker,
        "qty": qty,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl_gross": round(pnl_gross, 2),
        "pnl_net": round(pnl_net, 2),
        "tax_reserved": round(tax_result["tax_amount"], 2),
        "trade_id": trade["id"],
    }


# ===== Dashboard Endpoints =====

@router.get("/health")
async def health_check():
    from main import get_uptime
    hb = database.get_last_heartbeat()
    open_trades = database.get_open_trades()
    budget_status = await asyncio.to_thread(budget.get_budget_status)

    return HealthResponse(
        status="running",
        uptime_seconds=round(get_uptime(), 1),
        last_heartbeat=hb["timestamp"] if hb else "never",
        open_positions=len(open_trades),
        budget_utilization_pct=round(budget_status.get("budget_used_pct", 0), 1),
    )


@router.get("/diagnose")
async def diagnose():
    """
    Full step-by-step diagnostic — shows EXACTLY why the bot is or isn't buying right now.
    Hit this URL in your browser to debug.
    """
    import random
    from scanner import get_watchlist
    from scoring import get_composite_score, MIN_BUY_SCORE
    from sentiment import score_sentiment
    from trading_hours import is_ok_to_trade, get_status as hours_status_fn
    from market_regime import get_regime as get_regime_fn
    from volume_confirm import check as vol_check
    from sanity_check import run_all as sanity_run
    from budget import compute_position_size, get_budget_status

    report = {}

    # ── 1. Market hours ────────────────────────────────────────────────────────
    hours_ok, hours_reason = is_ok_to_trade()
    mkt_open = await asyncio.to_thread(broker.is_market_open)
    report["step1_market_hours"] = {
        "broker_says_open": mkt_open,
        "hours_filter_ok": hours_ok,
        "reason": hours_reason,
        "verdict": "✅ PASS" if (mkt_open and hours_ok) else "❌ BLOCKED — market closed or outside session",
    }

    # ── 2. Budget ──────────────────────────────────────────────────────────────
    status = await asyncio.to_thread(get_budget_status)
    cash = float(status.get("cash_available", 0))
    open_trades = database.get_open_trades()
    report["step2_budget"] = {
        "cash_available": round(cash, 2),
        "open_positions": len(open_trades),
        "max_positions": settings.MAX_OPEN_POSITIONS,
        "verdict": "✅ PASS" if cash >= 10 and len(open_trades) < settings.MAX_OPEN_POSITIONS else f"❌ BLOCKED — cash=${cash:.2f}, positions={len(open_trades)}/{settings.MAX_OPEN_POSITIONS}",
    }

    # ── 3. Circuit breaker ──────────────────────────────────────────────────────
    from circuit_breaker import check_circuit_breaker
    cb_ok, cb_reason = check_circuit_breaker()
    report["step3_circuit_breaker"] = {
        "ok": cb_ok,
        "reason": cb_reason,
        "verdict": "✅ PASS" if cb_ok else f"❌ BLOCKED — {cb_reason}",
    }

    # ── 4. Score 3 stocks through every filter ─────────────────────────────────
    watchlist = get_watchlist()
    sample = random.sample(watchlist, min(3, len(watchlist)))
    stock_results = []

    for ticker in sample:
        result = {"ticker": ticker, "filters": {}}
        try:
            # Score
            sent = await asyncio.to_thread(score_sentiment, ticker)
            comp = await asyncio.to_thread(get_composite_score, ticker, sent.score)
            score = comp["composite_score"]
            result["score"] = round(score, 1)
            result["min_score"] = MIN_BUY_SCORE
            result["filters"]["scoring"] = "✅ PASS" if comp["should_buy"] else f"❌ BLOCKED — score={score:.0f} < {MIN_BUY_SCORE}"

            if comp["should_buy"]:
                # Price
                price = await asyncio.to_thread(broker.get_price, ticker)
                result["price"] = price

                # Regime
                try:
                    regime, adx, _ = await asyncio.wait_for(asyncio.to_thread(get_regime_fn, ticker), timeout=20)
                    result["filters"]["market_regime"] = "✅ PASS" if regime != "ranging" else f"❌ BLOCKED — ranging (ADX={adx:.1f})"
                except Exception as e:
                    result["filters"]["market_regime"] = f"⚠️ timeout/error — {e} (fail-open, continues)"

                # Sanity
                sane, sane_reason = await asyncio.wait_for(asyncio.to_thread(sanity_run, ticker, price, None), timeout=20)
                result["filters"]["sanity_check"] = "✅ PASS" if sane else f"❌ BLOCKED — {sane_reason}"

                # Volume
                try:
                    vol_ok, vol_reason, vol_details = await asyncio.wait_for(asyncio.to_thread(vol_check, ticker), timeout=15)
                    result["filters"]["volume"] = "✅ PASS" if vol_ok else f"❌ BLOCKED — {vol_reason} (ratio={vol_details.get('ratio', '?')})"
                except Exception as e:
                    result["filters"]["volume"] = f"⚠️ timeout/error — {e} (fail-open, continues)"

                # Position sizing
                qty, sizing_meta = await asyncio.to_thread(compute_position_size, price)
                result["filters"]["position_size"] = f"✅ qty={qty:.4f}" if qty > 0 else f"❌ BLOCKED — qty=0 at ${price:.2f} ({sizing_meta})"

        except Exception as e:
            result["error"] = str(e)

        stock_results.append(result)

    report["step4_stock_samples"] = stock_results

    # ── Summary ────────────────────────────────────────────────────────────────
    blockers = []
    if not mkt_open:   blockers.append("Market is closed (broker check)")
    if not hours_ok:   blockers.append(f"Trading hours blocked: {hours_reason}")
    if cash < 10:      blockers.append(f"Not enough cash: ${cash:.2f}")
    if len(open_trades) >= settings.MAX_OPEN_POSITIONS: blockers.append(f"Max positions reached: {len(open_trades)}/{settings.MAX_OPEN_POSITIONS}")
    if not cb_ok:      blockers.append(f"Circuit breaker: {cb_reason}")

    report["summary"] = {
        "verdict": "✅ Bot SHOULD be buying (check stock scores above)" if not blockers else "❌ Bot BLOCKED",
        "blockers": blockers if blockers else ["None — if still not buying, check stock scores above"],
    }

    return report


@router.get("/scan/now")
async def scan_now(secret: str = ""):
    """
    Force an immediate scan + buy cycle — skips the 5-minute wait.
    Requires secret param: /scan/now?secret=YOUR_SECRET
    Ignores market hours (useful for testing outside trading hours).
    """
    if not settings.WEBHOOK_SECRET or secret != settings.WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    from scanner import get_watchlist
    from scoring import get_composite_score, MIN_BUY_SCORE
    from sentiment import score_sentiment
    from budget import compute_position_size, get_budget_status
    from market_regime import get_regime as get_regime_fn
    from volume_confirm import check as vol_check
    from sanity_check import run_all as sanity_run
    from iceberg import iceberg_buy
    from slippage import estimate as slippage_estimate, record as slippage_record
    import random

    status_data = await asyncio.to_thread(get_budget_status)
    remaining = float(status_data.get("cash_available", 0))
    if remaining < 10:
        return {"status": "skip", "reason": f"Not enough cash: ${remaining:.2f}"}

    open_count = len(database.get_open_trades())
    if open_count >= settings.MAX_OPEN_POSITIONS:
        return {"status": "skip", "reason": f"Max positions reached ({open_count}/{settings.MAX_OPEN_POSITIONS})"}

    watchlist = get_watchlist()
    shuffled = watchlist.copy()
    random.shuffle(shuffled)
    # Filter already-held
    candidates = [t for t in shuffled[:30] if not database.get_open_trade_by_ticker(t)]

    bought = []
    skipped = []

    for ticker in candidates[:10]:
        if remaining < 10:
            break
        try:
            sent = await asyncio.wait_for(asyncio.to_thread(score_sentiment, ticker), timeout=25)
            comp = await asyncio.wait_for(asyncio.to_thread(get_composite_score, ticker, sent.score), timeout=25)
            score = comp["composite_score"]

            if not comp["should_buy"]:
                skipped.append({"ticker": ticker, "score": score, "reason": "score_too_low"})
                continue

            price = await asyncio.wait_for(asyncio.to_thread(broker.get_price, ticker), timeout=10)
            if not price or price <= 0:
                skipped.append({"ticker": ticker, "score": score, "reason": "no_price"})
                continue

            sane, sane_reason = await asyncio.wait_for(asyncio.to_thread(sanity_run, ticker, price, None), timeout=15)
            if not sane:
                skipped.append({"ticker": ticker, "score": score, "reason": f"sanity: {sane_reason}"})
                continue

            qty, sizing_meta = await asyncio.to_thread(compute_position_size, price)
            if qty <= 0:
                skipped.append({"ticker": ticker, "score": score, "reason": f"qty=0 at ${price:.2f}"})
                continue

            slip = await asyncio.to_thread(slippage_estimate, price, qty, "buy", ticker)
            order = await iceberg_buy(ticker, qty, price)
            actual_price = float(order.get("price") or price)
            remaining -= actual_price * qty

            from models import WebhookPayload, TradeAction
            fake_payload = WebhookPayload(
                secret=settings.WEBHOOK_SECRET,
                ticker=ticker, action=TradeAction.BUY, price=actual_price,
            )
            from trade_logger import log_trade_open
            log_trade_open(fake_payload, sent, order, qty, sizing_meta, slip)
            asyncio.ensure_future(notify_buy(ticker, qty, actual_price, score, sent.score))
            _create_background_task(asyncio.to_thread(slippage_record, price, actual_price, qty, "buy", ticker))

            bought.append({"ticker": ticker, "qty": round(qty, 4), "price": actual_price, "score": score})

        except Exception as e:
            skipped.append({"ticker": ticker, "reason": str(e)})

    return {
        "status": "done",
        "bought": bought,
        "skipped": skipped[:5],
        "cash_remaining": round(remaining, 2),
    }


@router.get("/scan/preview")
async def scan_preview():
    """
    Score 5 random large-cap stocks right now and show why bot buys/skips.
    Call this to understand what scores look like in current market conditions.
    """
    import random
    from scanner import get_watchlist
    from scoring import get_composite_score
    from sentiment import score_sentiment

    tickers = random.sample(get_watchlist(), min(5, len(get_watchlist())))
    results = []
    for ticker in tickers:
        try:
            sent = await asyncio.to_thread(score_sentiment, ticker)
            comp = await asyncio.to_thread(get_composite_score, ticker, sent.score)
            results.append({
                "ticker":    ticker,
                "score":     comp["composite_score"],
                "decision":  comp["decision"],
                "tech":      comp["scores"]["technicals"],
                "market":    comp["scores"]["market"],
                "sentiment": comp["scores"]["sentiment"],
                "vix":       comp.get("vix"),
            })
        except Exception as e:
            results.append({"ticker": ticker, "error": str(e)})

    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return {
        "min_score_to_buy": 50,
        "results": results,
        "tip": "אם כל הציונים מתחת ל-50 — השוק לא בכיוון טוב כרגע",
    }


@router.get("/status")
async def trading_status():
    status = await asyncio.to_thread(budget.get_budget_status)
    positions = await asyncio.to_thread(broker.get_positions)
    return {
        "budget": status,
        "positions": positions,
        "open_trades": database.get_open_trades(),
        "circuit_breaker":   cb_status(),
        "trading_hours":     hours_status(),
        "iceberg_active":    iceberg_status(),
        "max_budget":        settings.MAX_BUDGET,
        "max_position_pct":  getattr(settings, "MAX_POSITION_PCT", 20),
        "broker":            getattr(settings, "ACTIVE_BROKER", "alpaca_paper"),
    }


@router.get("/volume")
async def volume_check(ticker: str):
    """
    Check current vs MA20 volume for a ticker.
    ?ticker=AAPL
    Returns ratio, current volume, MA volume, and whether it passes the threshold.
    """
    from volume_confirm import check as vol_check
    passed, reason, details = await asyncio.to_thread(vol_check, ticker.upper())
    return {"passed": passed, "reason": reason, **details}


@router.get("/correlation")
async def correlation_matrix():
    """
    Full N×N Pearson correlation matrix for all currently open positions.
    Also checks each open position against the others and flags pairs above threshold.
    """
    from correlation import portfolio_matrix
    return await asyncio.to_thread(portfolio_matrix)


@router.get("/correlation/check")
async def correlation_check(ticker: str):
    """
    Check whether a specific ticker would be blocked by the correlation filter
    given the current open positions.
    ?ticker=NVDA
    """
    from correlation import check as corr_check
    blocked, reason, details = await asyncio.to_thread(corr_check, ticker.upper())
    return {"blocked": blocked, "reason": reason, **details}


@router.get("/performance")
async def performance_report(weeks: int = 4):
    """
    Full performance KPI report for the last `weeks` calendar weeks.
    Returns Sharpe Ratio, Max Drawdown, win rate per strategy, daily equity curve.
    """
    weeks = min(max(1, weeks), 52)  # clamp to [1, 52] weeks
    from performance import compute as perf_compute
    report = await asyncio.to_thread(perf_compute, weeks)
    return report.to_dict()


@router.get("/performance/csv")
async def performance_csv(weeks: int = 4):
    """
    Download a ZIP-style CSV bundle (trades + summary) for the last `weeks`.
    Returns the summary CSV as a downloadable file.
    """
    import io
    from fastapi.responses import StreamingResponse
    from performance import compute as perf_compute, export_csv
    import tempfile, os

    weeks    = min(max(1, weeks), 52)  # clamp to [1, 52] weeks
    report   = await asyncio.to_thread(perf_compute, weeks)
    tmp_dir  = tempfile.mkdtemp()
    csv_path = await asyncio.to_thread(export_csv, report, tmp_dir)

    def _iter_file():
        with open(csv_path, "rb") as f:
            while chunk := f.read(8192):
                yield chunk
        # cleanup
        try:
            os.remove(csv_path)
            trades_path = csv_path.replace("summary_", "trades_")
            if os.path.exists(trades_path):
                os.remove(trades_path)
            os.rmdir(tmp_dir)
        except Exception:
            pass

    filename = f"performance_{report.period_start}_{report.period_end}.csv"
    return StreamingResponse(
        _iter_file(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/trades")
async def trade_history(ticker: str | None = None, limit: int = 50):
    limit = min(max(1, limit), 500)   # clamp to [1, 500] — prevent memory exhaustion
    return database.get_trade_history(ticker, limit)


@router.get("/tax")
async def tax_report():
    return database.get_tax_summary()


@router.get("/learning")
async def learning_log(pattern_type: str | None = None, limit: int = 50):
    limit = min(max(1, limit), 200)   # clamp to [1, 200]
    return database.get_learning_entries(pattern_type, limit)


@router.post("/broker")
async def set_broker(payload: BrokerSwitch, request: Request):
    """
    Switch the active broker at runtime.
    Body: {"broker": "...", "secret": "WEBHOOK_SECRET"}
    """
    # Auth — accept secret from body OR from X-Webhook-Secret header
    secret = getattr(payload, "secret", None) or request.headers.get("X-Webhook-Secret", "")
    if not settings.WEBHOOK_SECRET or secret != settings.WEBHOOK_SECRET:
        logger.warning(f"Broker switch auth failed from {request.client.host if request.client else 'unknown'}")
        raise HTTPException(status_code=401, detail="Invalid secret")

    valid = {
        "tv_paper",
        "alpaca_paper", "alpaca_live", "ibkr", "oanda", "tradier", "tradier_live",
        "tastytrade", "schwab", "binance", "kraken", "coinbase",
        "robinhood", "webull", "bybit", "okx", "kucoin", "gemini", "tradestation",
    }
    if payload.broker not in valid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid broker '{payload.broker}'. Valid: {sorted(valid)}",
        )
    try:
        broker.switch_broker(payload.broker)
        logger.info(f"Broker switched to: {payload.broker}")
        return {"status": "ok", "active_broker": payload.broker}
    except Exception as e:
        logger.error(f"Broker switch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emergency-exit/{ticker}")
async def emergency_exit(ticker: str, request: Request):
    """Manual emergency exit for a ticker.
    Auth: pass secret in `?secret=...` query param OR X-Webhook-Secret header.
    """
    secret = request.query_params.get("secret", "") or request.headers.get("X-Webhook-Secret", "")
    if not settings.WEBHOOK_SECRET or secret != settings.WEBHOOK_SECRET:
        logger.warning(f"Emergency exit auth failed for {ticker} from {request.client.host if request.client else 'unknown'}")
        raise HTTPException(status_code=401, detail="Invalid secret")

    ticker = ticker.upper()
    trade = database.get_open_trade_by_ticker(ticker)
    if not trade:
        raise HTTPException(status_code=404, detail=f"No open position for {ticker}")

    try:
        position = await asyncio.to_thread(broker.get_position, ticker)
        if not position:
            raise HTTPException(status_code=404, detail=f"No broker position for {ticker}")

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

        # Cleanup per-ticker state so re-entry works correctly
        from heartbeat import _smart_sell_last_check, _position_alert_sent
        _smart_sell_last_check.pop(ticker, None)
        _position_alert_sent.pop(ticker, None)

        logger.warning(f"EMERGENCY EXIT: {ticker} | PnL=${pnl_gross:+.2f}")
        return {
            "status": "emergency_exit",
            "ticker": ticker,
            "pnl_gross": round(pnl_gross, 2),
            "order_id": order["order_id"],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Emergency exit failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scan")
async def scan_for_opportunity():
    """Scan watchlist and return the best buy opportunity right now."""
    try:
        import asyncio
        result = await asyncio.to_thread(scanner.get_top_pick)
        return result
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        return {"error": str(e), "ticker": None}


@router.post("/auto-invest")
async def auto_invest(data: dict):
    """
    סרוק ובחר את הנכסים הטובים ביותר לפי ציון, חלק תקציב יחסית לציון וקנה.
    הבוט מחליט לבד כמה חברות לקנות.
    Body: { "secret": "..." }
    """
    if data.get("secret") != settings.WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Invalid secret")

    import asyncio

    # 1. סרוק וקבל top 10 מועמדים
    try:
        all_picks = await asyncio.to_thread(scanner.scan_stocks, 10)
    except Exception as e:
        return {"status": "error", "reason": f"Scan failed: {e}"}

    if not all_picks:
        return {"status": "error", "reason": "No stocks passed screening"}

    # 2. בחר רק נכסים עם ציון >= 5
    MIN_SCORE = 5
    picks = [p for p in all_picks if p.get("score", 0) >= MIN_SCORE] or all_picks[:5]
    picks = picks[:10]  # עד 10 מועמדים

    # 3. קבל תקציב זמין
    budget_status = await asyncio.to_thread(budget.get_budget_status)
    remaining = float(budget_status.get("cash_available", budget_status.get("available_budget", 0)))
    initial_available = remaining

    if remaining < 1:
        return {"status": "error", "reason": f"אין תקציב זמין (${remaining:.2f})"}

    results = []

    # 4. הקצאה חכמה: עבור על המניות לפי ציון, קנה כמה שאפשר עם התקציב הנשאר
    for pick in picks:
        if remaining < 1:
            break
        ticker = pick["ticker"]
        price  = float(pick.get("price") or 0)
        if price <= 0:
            results.append({"ticker": ticker, "status": "skipped", "reason": "מחיר לא ידוע"})
            continue

        # Max 20% of total budget per position
        max_per_stock = settings.MAX_BUDGET * (settings.MAX_POSITION_PCT / 100)
        available_for_this = min(remaining, max_per_stock)
        qty = round(available_for_this / price, 6)
        if qty < 0.001:
            results.append({"ticker": ticker, "status": "skipped", "reason": f"תקציב נמוך (${available_for_this:.0f} < ${price:.0f})"})
            continue

        # בדוק אם כבר יש פוזיציה
        existing = database.get_open_trade_by_ticker(ticker)
        if existing:
            results.append({"ticker": ticker, "status": "skipped", "reason": "פוזיציה קיימת"})
            continue

        # קנה
        try:
            order = await asyncio.to_thread(broker.submit_buy, ticker, qty)
            from trade_logger import log_trade_open
            from models import WebhookPayload, TradeAction
            from sentiment import score_sentiment
            sentiment = await asyncio.to_thread(score_sentiment, ticker)
            fake_payload = WebhookPayload(
                secret=settings.WEBHOOK_SECRET,
                ticker=ticker,
                action=TradeAction.BUY,
                price=price,
            )
            trade_id = log_trade_open(fake_payload, sentiment, order, qty)
            results.append({
                "ticker": ticker,
                "status": "bought",
                "qty": qty,
                "price": price,
                "spent": round(qty * price, 2),
                "category": pick.get("category", ""),
                "trade_id": trade_id,
            })
            spent = qty * price
            remaining -= spent
            logger.info(f"Auto-invest: bought {qty}x {ticker} @ ${price:.2f} | נשאר: ${remaining:.2f}")
        except Exception as e:
            results.append({"ticker": ticker, "status": "error", "reason": str(e)})

    bought = [r for r in results if r["status"] == "bought"]
    total_spent = sum(r.get("spent", 0) for r in bought)

    return {
        "status": "done",
        "picks": len(picks),
        "bought": len(bought),
        "total_spent": round(total_spent, 2),
        "per_stock_budget": round(initial_available / len(picks) if picks else 0, 2),
        "results": results,
    }


@router.get("/news/{ticker}")
async def get_news(ticker: str):
    """Fetch recent news headlines for a ticker via news_service (RSS)."""
    import re
    # Validate ticker — only uppercase letters, digits, dots and dashes (e.g. BRK-B, BF.B)
    if not re.match(r'^[A-Za-z0-9.\-]{1,10}$', ticker):
        raise HTTPException(status_code=400, detail="Invalid ticker format")
    try:
        from news_service import get_headlines
        headlines = await asyncio.to_thread(get_headlines, ticker.upper(), 5)
        results = [{"title": h, "publisher": "", "link": "", "published": 0} for h in headlines]
        return {"ticker": ticker.upper(), "news": results}
    except Exception as e:
        return {"ticker": ticker.upper(), "news": [], "error": str(e)}


@router.get("/shadow")
async def shadow_compare():
    """
    Compare shadow (aggressive) strategy vs live (conservative) strategy.

    Returns:
      shadow       — aggregate P&L stats for all shadow trades
      live         — aggregate P&L stats for all live trades
      shadow_only  — stats for trades shadow took but live blocked
      agreement    — count of trades both strategies took
      filter_analysis — per-filter breakdown: how often each guard blocked, win-rate + avg P&L
      note         — interpretation guide

    Positive avg_pnl in shadow_only means live filters ARE protecting capital.
    Negative avg_pnl means live filters are blocking profitable trades.
    """
    import shadow as _shadow
    return await asyncio.to_thread(_shadow.compare)


@router.get("/shadow/trades")
async def shadow_trades(limit: int = 50):
    """
    List recent shadow trades (open and closed).
    ?limit=50  (default 50, max 1000)
    """
    import shadow as _shadow
    trades = await asyncio.to_thread(_shadow.get_trades, min(limit, 1000))
    return {"count": len(trades), "trades": trades}


@router.get("/regime")
async def market_regime_check(ticker: str):
    """
    Check ADX market regime for a ticker.
    ?ticker=AAPL
    Returns: regime (trending|ranging), ADX value, threshold, is_trending.
    """
    from market_regime import get_regime
    regime, adx, details = await asyncio.to_thread(get_regime, ticker.upper())
    return {"regime": regime, "adx": adx, **details}


@router.get("/kelly")
async def kelly_status():
    """
    Return the current Half-Kelly fraction derived from closed trade history.
    Shows the inputs (win_rate, profit_factor) and the recommended fraction.
    """
    from budget import kelly_fraction
    from database import get_win_trades, get_loss_trades

    wins   = await asyncio.to_thread(get_win_trades,  200)
    losses = await asyncio.to_thread(get_loss_trades, 200)
    n_wins, n_losses = len(wins), len(losses)
    total = n_wins + n_losses

    frac = await asyncio.to_thread(kelly_fraction)

    avg_win  = sum(t.get("pnl_gross", 0) or 0 for t in wins)  / n_wins  if n_wins  else 0
    avg_loss = sum(abs(t.get("pnl_gross", 0) or 0) for t in losses) / n_losses if n_losses else 0

    return {
        "kelly_fraction":    round(frac, 4),
        "kelly_pct":         round(frac * 100, 2),
        "win_rate":          round(n_wins / total * 100, 1) if total else 0,
        "profit_factor":     round(avg_win / avg_loss, 3) if avg_loss > 0 else None,
        "avg_win_usd":       round(avg_win, 2),
        "avg_loss_usd":      round(avg_loss, 2),
        "total_closed":      total,
        "wins":              n_wins,
        "losses":            n_losses,
        "note": (
            "Kelly fraction = 0 means either no edge or too few trades (<10). "
            "Positive fraction is used as % of equity per trade (half-Kelly applied)."
        ),
    }


@router.get("/slippage/summary")
async def slippage_summary():
    """
    Aggregate slippage statistics across all recorded trades.

    Tracks the difference between the signal price (what TradingView said)
    and the actual fill price (what the broker executed at).
    Positive slip_pct = we paid more (buy) or received less (sell) than expected.
    """
    from slippage import get_summary as _slip_summary
    import os as _os
    summary = await asyncio.to_thread(_slip_summary)
    return {
        **summary,
        "alert_threshold_pct": float(_os.getenv("SLIPPAGE_ALERT_PCT", "0.1")),
        "rolling_window":      int(_os.getenv("SLIPPAGE_ROLLING_N",   "20")),
        "note": (
            "avg_slip_pct < 0.1% is healthy. "
            "If consistently above, consider tuning SLIPPAGE_ATR_MULTIPLIER."
        ),
    }


@router.get("/slippage")
async def slippage_history(limit: int = 100):
    """
    Return the most recent slippage observations (newest first).
    Each row shows signal_price vs fill_price and the resulting cost in bps.
    """
    limit = min(max(1, limit), 500)  # clamp to [1, 500]
    rows = await asyncio.to_thread(database.get_slippage_history, limit)
    return {"count": len(rows), "records": rows}


@router.get("/settings/trading")
async def get_trading_settings():
    return {
        "stop_loss_pct": settings.STOP_LOSS_PCT,
        "take_profit_pct": settings.TAKE_PROFIT_PCT,
    }


@router.post("/settings")
async def update_settings(data: dict, request: Request):
    """Update bot settings from TradingView panel.
    Auth: pass `secret` in body OR X-Webhook-Secret header.
    """
    secret = data.get("secret", "") or request.headers.get("X-Webhook-Secret", "")
    if not settings.WEBHOOK_SECRET or secret != settings.WEBHOOK_SECRET:
        logger.warning(f"Settings update auth failed from {request.client.host if request.client else 'unknown'}")
        raise HTTPException(status_code=401, detail="Invalid secret")

    if "max_budget" in data:
        val = float(data["max_budget"])
        if val < 100:
            raise HTTPException(status_code=400, detail="Budget must be at least $100")
        settings.MAX_BUDGET = val
        import os as _os
        _os.environ["MAX_BUDGET"] = str(val)
        # Sync paper broker cash and persist to disk
        try:
            from broker_tv_paper import TVPaperBroker
            TVPaperBroker._cash = val
            TVPaperBroker._save_state()
        except Exception:
            pass
        logger.info(f"Budget updated to ${val:,.2f}")

    if "max_position_pct" in data:
        val = float(data["max_position_pct"])
        if 1 <= val <= 100:
            settings.MAX_POSITION_PCT = val
            logger.info(f"Position size updated to {val}%")

    if "broker" in data:
        broker_name = data["broker"]
        try:
            broker.switch_broker(broker_name)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid broker: {e}")
        logger.info(f"Broker switched to: {broker_name}")

    if "stop_loss_pct" in data:
        settings.STOP_LOSS_PCT = float(data["stop_loss_pct"])
        logger.info(f"Stop loss updated to {settings.STOP_LOSS_PCT}%")

    if "take_profit_pct" in data:
        settings.TAKE_PROFIT_PCT = float(data["take_profit_pct"])
        logger.info(f"Take profit updated to {settings.TAKE_PROFIT_PCT}%")

    return {
        "status": "ok",
        "max_budget": settings.MAX_BUDGET,
        "max_position_pct": getattr(settings, "MAX_POSITION_PCT", 20),
        "stop_loss_pct": settings.STOP_LOSS_PCT,
        "take_profit_pct": settings.TAKE_PROFIT_PCT,
        "broker": getattr(settings, "ACTIVE_BROKER", "alpaca_paper"),
    }


@router.get("/telegram/test")
async def test_telegram():
    """
    Sends a test message to Telegram and returns diagnostic info.
    Visit this URL in your browser to check if Telegram is connected.
    """
    from telegram_bot import send_message, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    token_set   = bool(TELEGRAM_BOT_TOKEN)
    chat_set    = bool(TELEGRAM_CHAT_ID)
    token_hint  = (TELEGRAM_BOT_TOKEN[:8] + "...") if token_set else "❌ לא מוגדר"
    chat_hint   = (TELEGRAM_CHAT_ID[:4]   + "...") if chat_set  else "❌ לא מוגדר"

    if not token_set or not chat_set:
        return {
            "status": "error",
            "message": "חסרים פרטי טלגרם ב-Render",
            "TELEGRAM_BOT_TOKEN": token_hint,
            "TELEGRAM_CHAT_ID":   chat_hint,
            "fix": "הוסף את המשתנים ב-Render → Environment",
        }

    ok = await send_message(
        "🤖 <b>בדיקת חיבור טלגרם</b>\n"
        "✅ הבוט מחובר ועובד!\n"
        "📡 מוכן לשלוח התראות."
    )
    return {
        "status": "ok" if ok else "failed",
        "sent": ok,
        "TELEGRAM_BOT_TOKEN": token_hint,
        "TELEGRAM_CHAT_ID":   chat_hint,
        "message": "הודעת בדיקה נשלחה בהצלחה ✅" if ok else "❌ שליחה נכשלה — בדוק לוגים",
    }
