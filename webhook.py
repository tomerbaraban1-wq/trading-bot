import logging
from fastapi import APIRouter, HTTPException
from config import settings
from models import WebhookPayload, TradeAction, HealthResponse, BudgetStatus, BrokerSwitch
import broker
import budget
import database
import scanner
from signal_validator import validate_signal
from trade_logger import log_trade_open, log_trade_close, log_learning

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/webhook")
async def receive_webhook(payload: WebhookPayload):
    """
    TradingView webhook receiver.
    Pipeline: authenticate -> validate -> sentiment -> budget -> execute -> log
    """
    # 1. Authenticate
    if payload.secret != settings.WEBHOOK_SECRET:
        logger.warning(f"Webhook auth failed for {payload.ticker}")
        raise HTTPException(status_code=401, detail="Invalid secret")

    # 2. Validate signal
    is_valid, reason = validate_signal(payload.ticker, payload.action.value)
    if not is_valid:
        logger.info(f"Signal rejected: {payload.ticker} {payload.action.value} - {reason}")
        return {"status": "rejected", "reason": reason}

    # 3. Process based on action
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

    # Sentiment check (MANDATORY for buys)
    from sentiment import score_sentiment
    try:
        sentiment = score_sentiment(ticker)
    except Exception as e:
        logger.error(f"Sentiment check failed for {ticker}: {e}")
        return {"status": "rejected", "reason": f"Sentiment check failed: {e}"}

    if sentiment.score < settings.SENTIMENT_MIN_SCORE:
        logger.info(
            f"BUY blocked by sentiment: {ticker} score={sentiment.score}/10 "
            f"(min={settings.SENTIMENT_MIN_SCORE})"
        )
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

    if not result["should_buy"]:
        return {
            "status": "blocked_by_score",
            "ticker": ticker,
            "composite_score": result["composite_score"],
            "min_score": result["min_score"],
            "reason": f"Score {result['composite_score']}/100 < min {result['min_score']}",
            "breakdown": result.get("breakdown", {}),
        }

    # Learning check - known loss patterns
    should_block, block_reason = should_override_buy(ticker, indicators)
    if should_block:
        logger.info(f"BUY blocked by learning: {ticker} - {block_reason}")
        return {"status": "blocked_by_learning", "reason": block_reason}

    # Budget check
    can_buy, max_qty, budget_reason = budget.check_can_buy(payload.price)
    if not can_buy:
        logger.info(f"BUY blocked by budget: {ticker} - {budget_reason}")
        return {"status": "blocked_by_budget", "reason": budget_reason}

    # Execute buy
    try:
        order = broker.submit_buy(ticker, max_qty)
    except Exception as e:
        logger.error(f"BUY order failed for {ticker}: {e}")
        return {"status": "error", "reason": f"Order failed: {e}"}

    # Log trade
    actual_price = order.get("price") or payload.price
    trade_id = log_trade_open(payload, sentiment, order, max_qty)

    return {
        "status": "executed",
        "action": "buy",
        "ticker": ticker,
        "qty": max_qty,
        "price": actual_price,
        "sentiment_score": sentiment.score,
        "trade_id": trade_id,
        "order_id": order["order_id"],
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
        order = broker.submit_sell(ticker)
    except Exception as e:
        logger.error(f"SELL order failed for {ticker}: {e}")
        return {"status": "error", "reason": f"Order failed: {e}"}

    # Calculate PnL
    entry_price = trade["entry_price"]
    exit_price = order.get("price") or payload.price
    qty = trade["qty"]
    pnl_gross = (exit_price - entry_price) * qty
    fees = 0.0  # Alpaca has no commissions for stocks

    # Tax logic
    from tax_tracker import process_trade_close
    tax_result = process_trade_close(trade["id"], pnl_gross)
    pnl_net = pnl_gross - tax_result["tax_amount"]

    # Log trade close
    log_trade_close(
        trade["id"], exit_price, pnl_gross, pnl_net,
        tax_result["tax_amount"], fees, "closed",
    )

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
    budget_status = budget.get_budget_status()

    return HealthResponse(
        status="running",
        uptime_seconds=round(get_uptime(), 1),
        last_heartbeat=hb["timestamp"] if hb else "never",
        open_positions=len(open_trades),
        budget_utilization_pct=round(budget_status.get("budget_used_pct", 0), 1),
    )


@router.get("/status")
async def trading_status():
    status = budget.get_budget_status()
    positions = broker.get_positions()
    return {
        "budget": status,
        "positions": positions,
        "open_trades": database.get_open_trades(),
    }


@router.get("/trades")
async def trade_history(ticker: str | None = None, limit: int = 50):
    return database.get_trade_history(ticker, limit)


@router.get("/tax")
async def tax_report():
    return database.get_tax_summary()


@router.get("/learning")
async def learning_log(pattern_type: str | None = None, limit: int = 50):
    return database.get_learning_entries(pattern_type, limit)


@router.post("/broker")
async def set_broker(payload: BrokerSwitch):
    """
    Switch the active broker at runtime.
    Body: {"broker": "alpaca_paper"|"alpaca_live"|"ibkr"|"oanda"|"tradier"|"tradier_live"}
    """
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
async def emergency_exit(ticker: str):
    """Manual emergency exit for a ticker."""
    ticker = ticker.upper()
    trade = database.get_open_trade_by_ticker(ticker)
    if not trade:
        raise HTTPException(status_code=404, detail=f"No open position for {ticker}")

    try:
        position = broker.get_position(ticker)
        if not position:
            raise HTTPException(status_code=404, detail=f"No broker position for {ticker}")

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
    budget_status = budget.get_budget_status()
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
        qty = int(available_for_this / price)
        if qty < 1:
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
    """Fetch recent news headlines for a ticker via yfinance."""
    import yfinance as yf
    try:
        t = yf.Ticker(ticker.upper())
        news = t.news or []
        results = []
        for item in news[:5]:
            results.append({
                "title": item.get("title", ""),
                "publisher": item.get("publisher", ""),
                "link": item.get("link", ""),
                "published": item.get("providerPublishTime", 0),
            })
        return {"ticker": ticker.upper(), "news": results}
    except Exception as e:
        return {"ticker": ticker.upper(), "news": [], "error": str(e)}


@router.get("/settings/trading")
async def get_trading_settings():
    return {
        "stop_loss_pct": settings.STOP_LOSS_PCT,
        "take_profit_pct": settings.TAKE_PROFIT_PCT,
    }


@router.post("/settings")
async def update_settings(data: dict):
    """Update bot settings from TradingView panel."""
    if "max_budget" in data:
        val = float(data["max_budget"])
        if val < 100:
            raise HTTPException(status_code=400, detail="Budget must be at least $100")
        settings.MAX_BUDGET = val
        # גם עדכן את הכסף הווירטואלי של tv_paper
        try:
            from broker_tv_paper import TVPaperBroker
            if TVPaperBroker._cash is None or TVPaperBroker._cash > val:
                TVPaperBroker._cash = val
        except Exception:
            pass
        logger.info(f"Budget updated to ${val:,.2f}")

    if "broker" in data:
        broker_name = data["broker"]
        if broker_name == "alpaca_live":
            settings.ALPACA_BASE_URL = "https://api.alpaca.markets"
        else:
            settings.ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
        logger.info(f"Broker switched to: {broker_name}")

    if "stop_loss_pct" in data:
        settings.STOP_LOSS_PCT = float(data["stop_loss_pct"])
        logger.info(f"Stop loss updated to {settings.STOP_LOSS_PCT}%")

    if "take_profit_pct" in data:
        settings.TAKE_PROFIT_PCT = float(data["take_profit_pct"])
        logger.info(f"Take profit updated to {settings.TAKE_PROFIT_PCT}%")

    return {"status": "ok", "max_budget": settings.MAX_BUDGET, "broker": settings.ALPACA_BASE_URL}
