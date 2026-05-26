"""
Analytics API Endpoints
========================

REST API endpoints for accessing all bot analytics:
- Performance reports
- Risk metrics
- AI decisions
- Multi-timeframe analysis
- News intelligence
- Health monitoring
- Tax optimization
- Portfolio optimization

All endpoints return JSON for easy consumption by frontends/dashboards.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Request, HTTPException, Header, Query
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["analytics"])


# ─────────────────────────────────────────────────────────────────────────────
# AUTHENTICATION
# ─────────────────────────────────────────────────────────────────────────────

def _verify_api_key(provided_key: Optional[str]) -> bool:
    """Verify API key for protected endpoints."""
    if not provided_key:
        return False

    expected = os.getenv("ANALYTICS_API_KEY", "")
    if not expected:
        # If not configured, allow access (development)
        return True

    import hmac
    return hmac.compare_digest(provided_key, expected)


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/performance")
async def get_performance(
    days: int = Query(30, ge=1, le=365),
    x_api_key: Optional[str] = Header(None)
):
    """Get performance attribution report."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        from performance_attribution import generate_attribution_report
        return await generate_attribution_report(days=days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/insights")
async def get_actionable_insights(x_api_key: Optional[str] = Header(None)):
    """Get actionable trading insights."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        from performance_attribution import get_actionable_insights
        insights = await get_actionable_insights()
        return {"insights": insights, "count": len(insights)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# RISK ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/risk/portfolio")
async def get_portfolio_risk(x_api_key: Optional[str] = Header(None)):
    """Get comprehensive portfolio risk analysis."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        from risk_engine import analyze_portfolio_risk
        return await analyze_portfolio_risk()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# AI DECISIONS
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/ai/decision/{ticker}")
async def get_ai_decision(ticker: str, x_api_key: Optional[str] = Header(None)):
    """Get AI trading decision for a ticker."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        import yfinance as yf
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        current_price = info.get("currentPrice") or info.get("regularMarketPrice") or 0

        from ai_decision_engine import make_trading_decision
        decision = await make_trading_decision(
            ticker=ticker.upper(),
            current_price=current_price,
        )

        return {
            "ticker": ticker.upper(),
            "action": decision.action,
            "confidence": decision.confidence,
            "expected_return": decision.expected_return,
            "risk_score": decision.risk_score,
            "position_size_pct": decision.position_size_pct,
            "stop_loss_price": decision.stop_loss_price,
            "take_profit_price": decision.take_profit_price,
            "holding_period_days": decision.holding_period_days,
            "reasoning": decision.reasoning,
            "warnings": decision.warnings,
            "explanation": decision.final_explanation,
            "signals": decision.signals,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-TIMEFRAME ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/mtf/{ticker}")
async def get_multi_timeframe_analysis(ticker: str, x_api_key: Optional[str] = Header(None)):
    """Get multi-timeframe analysis for a ticker."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        from multi_timeframe import analyze_multi_timeframe
        analysis = await analyze_multi_timeframe(ticker.upper())

        return {
            "ticker": analysis.ticker,
            "current_price": analysis.current_price,
            "timeframes": analysis.timeframes,
            "alignment_score": analysis.alignment_score,
            "overall_trend": analysis.overall_trend,
            "high_confidence": analysis.high_confidence,
            "actionable_setup": analysis.actionable_setup,
            "recommendation": analysis.recommendation,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# NEWS INTELLIGENCE
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/news/portfolio")
async def get_portfolio_news_endpoint(x_api_key: Optional[str] = Header(None)):
    """Get portfolio news summary."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        from news_intelligence import get_portfolio_news
        return await get_portfolio_news()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/news/{ticker}")
async def get_ticker_news(
    ticker: str,
    max_articles: int = Query(10, ge=1, le=50),
    x_api_key: Optional[str] = Header(None)
):
    """Get news for specific ticker."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        from news_intelligence import fetch_yahoo_news, detect_catalysts

        articles = await fetch_yahoo_news(ticker.upper(), max_articles)
        catalysts = await detect_catalysts(ticker.upper())

        return {
            "ticker": ticker.upper(),
            "total_articles": len(articles),
            "articles": [
                {
                    "title": a.title,
                    "source": a.source,
                    "sentiment": a.sentiment,
                    "sentiment_score": a.sentiment_score,
                    "impact_score": a.impact_score,
                    "topics": a.topics,
                    "is_breaking": a.is_breaking,
                    "is_catalyst": a.is_catalyst,
                    "url": a.url,
                }
                for a in articles
            ],
            "catalysts": catalysts,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# MARKET INTELLIGENCE
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/market/intelligence")
async def get_market_intelligence(x_api_key: Optional[str] = Header(None)):
    """Get comprehensive market intelligence."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        from market_intelligence import get_market_intelligence_report
        return await get_market_intelligence_report()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# TAX OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/tax/ytd")
async def get_ytd_tax(x_api_key: Optional[str] = Header(None)):
    """Get year-to-date tax summary."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        from tax_optimization import get_ytd_tax_summary
        return await get_ytd_tax_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tax/harvest")
async def get_tax_harvest_opportunities(x_api_key: Optional[str] = Header(None)):
    """Get tax loss harvesting opportunities."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        from tax_optimization import find_tax_loss_harvest_opportunities
        opportunities = await find_tax_loss_harvest_opportunities()
        return {
            "opportunities": opportunities,
            "count": len(opportunities),
            "total_potential_benefit": sum(o.get("tax_benefit_estimate", 0) for o in opportunities),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tax/wash-sales")
async def get_wash_sales(x_api_key: Optional[str] = Header(None)):
    """Get detected wash sales."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        from tax_optimization import detect_wash_sales
        wash_sales = await detect_wash_sales()
        return {
            "wash_sales": wash_sales,
            "count": len(wash_sales),
            "total_disallowed_losses": sum(w.get("loss_amount", 0) for w in wash_sales),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tax/efficiency")
async def get_tax_efficiency(x_api_key: Optional[str] = Header(None)):
    """Get tax efficiency score."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        from tax_optimization import calculate_tax_efficiency
        return await calculate_tax_efficiency()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/portfolio/optimize")
async def optimize_portfolio_endpoint(
    strategy: str = Query("max_sharpe", regex="^(max_sharpe|min_variance|risk_parity)$"),
    x_api_key: Optional[str] = Header(None)
):
    """Optimize portfolio using MPT."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        import broker
        positions = await asyncio.to_thread(broker.get_positions)
        tickers = [p.symbol for p in positions]

        if not tickers:
            return {"error": "No positions to optimize"}

        from portfolio_optimizer import optimize_portfolio
        result = await optimize_portfolio(tickers, strategy=strategy)

        return {
            "strategy": strategy,
            "tickers": result.tickers,
            "weights": result.weights,
            "expected_return_pct": result.expected_return,
            "expected_volatility_pct": result.expected_volatility,
            "sharpe_ratio": result.sharpe_ratio,
            "diversification_score": result.diversification_score,
            "recommendation": result.recommendation,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolio/rebalance")
async def analyze_rebalancing(x_api_key: Optional[str] = Header(None)):
    """Analyze portfolio rebalancing needs."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        from portfolio_optimizer import analyze_rebalancing_needs
        return await analyze_rebalancing_needs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# BACKTESTING
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/backtest/{ticker}")
async def quick_backtest(
    ticker: str,
    days: int = Query(180, ge=30, le=730),
    x_api_key: Optional[str] = Header(None)
):
    """Run quick backtest on a ticker."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        from backtesting_engine import StrategyConfig, run_backtest

        config = StrategyConfig(name="default")
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        result = await run_backtest(ticker.upper(), config, start_date, end_date)

        return {
            "ticker": ticker.upper(),
            "period_days": days,
            "initial_capital": result.initial_capital,
            "final_capital": result.final_capital,
            "total_return_pct": result.total_return_pct,
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown_pct": result.max_drawdown_pct,
            "profit_factor": result.profit_factor,
            "avg_win": result.avg_win,
            "avg_loss": result.avg_loss,
            "longest_winning_streak": result.longest_winning_streak,
            "longest_losing_streak": result.longest_losing_streak,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# CONTINUOUS LEARNING
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/learning/insights")
async def get_learning_insights(x_api_key: Optional[str] = Header(None)):
    """Get continuous learning insights."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        from continuous_learner import run_continuous_learning_cycle
        return await run_continuous_learning_cycle()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED DASHBOARD DATA
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/dashboard")
async def get_dashboard_data(x_api_key: Optional[str] = Header(None)):
    """Get all data needed for dashboard in one call."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        # Gather data in parallel
        from performance_attribution import generate_attribution_report
        from risk_engine import analyze_portfolio_risk
        from health_monitor import run_health_check
        from market_intelligence import get_market_intelligence_report

        results = await asyncio.gather(
            generate_attribution_report(days=30),
            analyze_portfolio_risk(),
            run_health_check(),
            get_market_intelligence_report(),
            return_exceptions=True,
        )

        perf, risk, health, market = results

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "performance": perf if not isinstance(perf, Exception) else {"error": str(perf)},
            "risk": risk if not isinstance(risk, Exception) else {"error": str(risk)},
            "health": {
                "status": health.overall_status if not isinstance(health, Exception) else "error",
                "issues": health.issues if not isinstance(health, Exception) else [],
            },
            "market": market if not isinstance(market, Exception) else {"error": str(market)},
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
