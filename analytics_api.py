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

# ── Security Headers Middleware ────────────────────────────────────────────────
# Injected on every analytics API response
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",          # don't cache sensitive financial data
}

def _secure_response(data: dict) -> JSONResponse:
    """Return a JSONResponse with security headers added."""
    return JSONResponse(content=data, headers=_SECURITY_HEADERS)


# ─────────────────────────────────────────────────────────────────────────────
# AUTHENTICATION
# ─────────────────────────────────────────────────────────────────────────────

def _verify_api_key(provided_key: Optional[str]) -> bool:
    """Verify API key for protected endpoints."""
    if not provided_key:
        return False

    expected = os.getenv("ANALYTICS_API_KEY", "")
    if not expected:
        # ⚠️ ANALYTICS_API_KEY not configured — DENY access (security fix)
        logger.warning("ANALYTICS_API_KEY not set — denying analytics access")
        return False

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
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


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
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


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
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


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
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


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
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


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
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


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
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


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
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


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
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


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
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


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
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/tax/efficiency")
async def get_tax_efficiency(x_api_key: Optional[str] = Header(None)):
    """Get tax efficiency score."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        from tax_optimization import calculate_tax_efficiency
        return await calculate_tax_efficiency()
    except Exception as e:
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


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
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/portfolio/rebalance")
async def analyze_rebalancing(x_api_key: Optional[str] = Header(None)):
    """Analyze portfolio rebalancing needs."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        from portfolio_optimizer import analyze_rebalancing_needs
        return await analyze_rebalancing_needs()
    except Exception as e:
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


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
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


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
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


# ─────────────────────────────────────────────────────────────────────────────
# PAIRS TRADING & HEDGING
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/pairs/opportunities")
async def get_pairs_opportunities(x_api_key: Optional[str] = Header(None)):
    """Get pairs trading opportunities."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        from pairs_trading import scan_pairs_opportunities
        opportunities = await scan_pairs_opportunities()
        return {"opportunities": opportunities, "count": len(opportunities)}
    except Exception as e:
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/pairs/{ticker1}/{ticker2}")
async def analyze_pair(ticker1: str, ticker2: str, x_api_key: Optional[str] = Header(None)):
    """Analyze a specific pair."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        from pairs_trading import generate_pairs_signal
        pair = await generate_pairs_signal(ticker1.upper(), ticker2.upper())
        return {
            "ticker1": pair.ticker1,
            "ticker2": pair.ticker2,
            "correlation": pair.correlation,
            "z_score": pair.z_score,
            "signal": pair.signal,
            "confidence": pair.confidence,
        }
    except Exception as e:
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/portfolio/beta")
async def get_portfolio_beta(x_api_key: Optional[str] = Header(None)):
    """Get portfolio beta vs SPY."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        from pairs_trading import calculate_portfolio_beta
        return await calculate_portfolio_beta()
    except Exception as e:
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


# ─────────────────────────────────────────────────────────────────────────────
# COMPOUND GROWTH
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/compound/strategy")
async def get_compound_strategy(x_api_key: Optional[str] = Header(None)):
    """Get personalized compounding strategy."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        from compound_engine import get_compounding_strategy
        return await get_compounding_strategy()
    except Exception as e:
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/compound/project")
async def project_compound_growth(
    initial: float = Query(10000),
    monthly: float = Query(500),
    annual_return: float = Query(15),
    years: int = Query(10, ge=1, le=50),
    x_api_key: Optional[str] = Header(None)
):
    """Project compound growth with given parameters."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        from compound_engine import calculate_compound_growth, compare_growth_scenarios

        projection = calculate_compound_growth(initial, monthly, annual_return, years)
        scenarios = compare_growth_scenarios(initial, monthly, years)

        return {
            "projection": {
                "initial": projection.initial_amount,
                "monthly": projection.monthly_contribution,
                "annual_return": projection.annual_return_pct,
                "years": projection.years,
                "final_amount": projection.final_amount,
                "total_contributions": projection.total_contributions,
                "total_growth": projection.total_growth,
                "growth_pct": projection.growth_pct,
            },
            "scenarios": scenarios,
        }
    except Exception as e:
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT: /benchmark/all MUST be defined BEFORE /benchmark/{ticker}
# otherwise "/benchmark/all" gets matched as ticker="all"

@router.get("/benchmark/all")
async def get_all_benchmarks(
    days: int = Query(90, ge=7, le=365),
    x_api_key: Optional[str] = Header(None)
):
    """Compare to all major benchmarks."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        from benchmark_compare import compare_to_all_benchmarks
        return await compare_to_all_benchmarks(days)
    except Exception as e:
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/benchmark/{ticker}")
async def get_benchmark_comparison(
    ticker: str,
    days: int = Query(90, ge=7, le=365),
    x_api_key: Optional[str] = Header(None)
):
    """Compare bot performance to a specific benchmark."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Guard: reject 'all' explicitly - should go to /benchmark/all
    if ticker.lower() == "all":
        raise HTTPException(status_code=400, detail="Use /benchmark/all for all benchmarks")

    try:
        from benchmark_compare import compare_to_benchmark
        comparison = await compare_to_benchmark(ticker.upper(), days)
        return {
            "benchmark_ticker": comparison.benchmark_ticker,
            "benchmark_name": comparison.benchmark_name,
            "period_days": days,
            "bot_return_pct": comparison.bot_return_pct,
            "benchmark_return_pct": comparison.benchmark_return_pct,
            "alpha_pct": comparison.alpha_pct,
            "beta": comparison.beta,
            "correlation": comparison.correlation,
            "sharpe_difference": comparison.sharpe_difference,
            "information_ratio": comparison.information_ratio,
            "status": comparison.win_rate_vs_market,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


# ─────────────────────────────────────────────────────────────────────────────
# SOCIAL SENTIMENT
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/sentiment/{ticker}")
async def get_unified_sentiment_endpoint(ticker: str, x_api_key: Optional[str] = Header(None)):
    """Get unified sentiment for a ticker."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        from social_sentiment import get_unified_sentiment
        sentiment = await get_unified_sentiment(ticker.upper())
        return {
            "ticker": sentiment.ticker,
            "overall_score": sentiment.overall_score,
            "overall_label": sentiment.overall_label,
            "confidence": sentiment.confidence,
            "source_agreement": sentiment.source_agreement,
            "total_mentions": sentiment.total_mentions,
            "interpretation": sentiment.interpretation,
            "actionable": sentiment.actionable,
            "sources": [
                {
                    "source": s.source,
                    "score": s.score,
                    "mentions": s.mention_count,
                    "confidence": s.confidence,
                }
                for s in sentiment.sources
            ],
        }
    except Exception as e:
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


# ─────────────────────────────────────────────────────────────────────────────
# TRADE JOURNAL
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/journal/recent")
async def get_recent_journal(
    days: int = Query(7, ge=1, le=90),
    x_api_key: Optional[str] = Header(None)
):
    """Get recent trade reviews."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        from trade_journal import review_recent_trades
        reviews = await review_recent_trades(days)
        return {
            "period_days": days,
            "count": len(reviews),
            "reviews": [
                {
                    "trade_id": r.trade_id,
                    "ticker": r.ticker,
                    "entry_date": r.entry_date,
                    "exit_date": r.exit_date,
                    "pnl": r.pnl,
                    "pnl_pct": r.pnl_pct,
                    "outcome": r.outcome,
                    "overall_grade": r.overall_grade,
                    "entry_grade": r.entry_grade,
                    "exit_grade": r.exit_grade,
                    "quality_score": r.quality_score,
                    "lessons": r.lessons_learned,
                    "mistakes": r.mistakes_made,
                }
                for r in reviews
            ],
        }
    except Exception as e:
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/journal/summary")
async def get_journal_summary(
    days: int = Query(30, ge=1, le=365),
    x_api_key: Optional[str] = Header(None)
):
    """Get journal summary statistics."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        from trade_journal import generate_journal_summary
        summary = await generate_journal_summary(days)
        return {
            "period_days": summary.period_days,
            "total_trades_reviewed": summary.total_trades_reviewed,
            "avg_quality_score": summary.avg_quality_score,
            "grade_distribution": summary.grade_distribution,
            "most_common_mistakes": summary.most_common_mistakes,
            "most_repeated_patterns": summary.most_repeated_patterns,
            "improvement_areas": summary.improvement_areas,
            "strengths": summary.strengths,
        }
    except Exception as e:
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


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
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


# ─────────────────────────────────────────────────────────────────────────────
# SMART EXECUTION (TWAP/VWAP/Iceberg)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/execution/analyze")
async def analyze_execution(
    ticker: str = Query(...),
    quantity: float = Query(...),
    side: str = Query("buy", regex="^(buy|sell)$"),
    urgency: str = Query("normal", regex="^(low|normal|high)$"),
    x_api_key: Optional[str] = Header(None)
):
    """Analyze optimal execution strategy for an order."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        from smart_execution import analyze_optimal_execution
        return await analyze_optimal_execution(ticker.upper(), quantity, side, urgency)
    except Exception as e:
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/optimizer/run")
async def run_strategy_optimizer(
    ticker: str = Query("SPY"),
    generations: int = Query(5, ge=1, le=20),
    x_api_key: Optional[str] = Header(None)
):
    """Run genetic algorithm to optimize trading strategy parameters."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        from strategy_optimizer import GeneticOptimizer
        optimizer = GeneticOptimizer(generations=generations, population_size=10)
        result = await optimizer.optimize(ticker.upper())

        return {
            "ticker": ticker.upper(),
            "optimization_score": result.optimization_score,
            "confidence": result.confidence,
            "optimal_params": {
                "min_buy_score": result.min_buy_score,
                "stop_loss_pct": result.stop_loss_pct,
                "take_profit_pct": result.take_profit_pct,
                "max_position_size_pct": result.max_position_size_pct,
                "rsi_oversold": result.rsi_oversold,
                "rsi_overbought": result.rsi_overbought,
                "min_volume_ratio": result.min_volume_ratio,
            },
        }
    except Exception as e:
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC PROTECTION (Stop Loss Optimization)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/protection/analyze")
async def analyze_protection(x_api_key: Optional[str] = Header(None)):
    """Analyze stop loss protection for all open positions."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        from dynamic_protection import analyze_all_position_protections
        recommendations = await analyze_all_position_protections()
        return {
            "recommendations": recommendations,
            "count": len(recommendations),
        }
    except Exception as e:
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


# ─────────────────────────────────────────────────────────────────────────────
# TRANSLATION SERVICE
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/translation/stats")
async def get_translation_stats_endpoint(x_api_key: Optional[str] = Header(None)):
    """Get translation service statistics."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        from translation_service import get_translation_stats
        return get_translation_stats()
    except Exception as e:
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/translation/test")
async def test_translation(request: Request, x_api_key: Optional[str] = Header(None)):
    """Test translation with custom text. Body: {'text': '...'}"""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        data = await request.json()
        text = data.get("text", "")

        if not text:
            raise HTTPException(status_code=400, detail="text required")

        from translation_service import translate_message
        translated = await translate_message(text)

        return {
            "original": text,
            "translated": translated,
            "changed": text != translated,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/translation/toggle")
async def toggle_translation(request: Request, x_api_key: Optional[str] = Header(None)):
    """Enable/disable translation. Body: {'enabled': true/false}"""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        data = await request.json()
        enabled = data.get("enabled", True)

        from translation_service import enable_translation, disable_translation
        if enabled:
            enable_translation()
        else:
            disable_translation()

        return {"enabled": enabled}
    except Exception as e:
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/anomalies/scan")
async def scan_anomalies(x_api_key: Optional[str] = Header(None)):
    """Scan portfolio for anomalies."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        from anomaly_detector import scan_portfolio_anomalies
        return await scan_portfolio_anomalies()
    except Exception as e:
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/anomalies/{ticker}")
async def detect_ticker_anomaly(
    ticker: str,
    anomaly_type: str = Query("price", regex="^(price|volume|both)$"),
    x_api_key: Optional[str] = Header(None)
):
    """Detect anomalies for a specific ticker."""
    if not _verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        from anomaly_detector import detect_price_anomaly, detect_volume_anomaly

        results = {}
        if anomaly_type in ("price", "both"):
            price_anomaly = await detect_price_anomaly(ticker.upper())
            results["price"] = (
                {
                    "severity": price_anomaly.severity,
                    "z_score": price_anomaly.z_score,
                    "deviation_pct": price_anomaly.deviation_pct,
                    "description": price_anomaly.description,
                } if price_anomaly else None
            )

        if anomaly_type in ("volume", "both"):
            volume_anomaly = await detect_volume_anomaly(ticker.upper())
            results["volume"] = (
                {
                    "severity": volume_anomaly.severity,
                    "z_score": volume_anomaly.z_score,
                    "deviation_pct": volume_anomaly.deviation_pct,
                    "description": volume_anomaly.description,
                } if volume_anomaly else None
            )

        return {
            "ticker": ticker.upper(),
            "anomalies_detected": any(results.values()),
            **results,
        }

    except Exception as e:
        logger.error(f"API error: {e}"); raise HTTPException(status_code=500, detail="Internal server error")
