"""
Backtesting Engine
==================

Test trading strategies on historical data BEFORE deploying.

Features:
1. Strategy backtesting with realistic execution
2. Walk-forward analysis (train/test splits)
3. Monte Carlo simulation
4. Strategy comparison (A/B testing)
5. Parameter optimization via grid search
6. Realistic costs (slippage, commissions)
7. Performance metrics matching live trading
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """A single trade in backtest."""
    ticker: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    exit_reason: str  # "stop_loss", "take_profit", "signal_exit"
    holding_days: float


@dataclass
class BacktestResult:
    """Comprehensive backtest results."""
    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown_pct: float
    longest_winning_streak: int
    longest_losing_streak: int
    avg_holding_days: float
    trades: list[BacktestTrade]
    equity_curve: list[float]


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyConfig:
    """Configuration for a backtest strategy."""
    name: str
    min_buy_score: float = 75
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 4.5
    max_position_size_pct: float = 10.0
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    use_macd_filter: bool = True
    use_volume_filter: bool = True
    min_volume_ratio: float = 0.5
    max_holding_days: int = 14


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

async def run_backtest(
    ticker: str,
    config: StrategyConfig,
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 10000,
    slippage_pct: float = 0.05,
    commission_per_trade: float = 0.0,
) -> BacktestResult:
    """
    Run a backtest on historical data.

    Args:
        ticker: Stock symbol
        config: Strategy configuration
        start_date: Backtest start
        end_date: Backtest end
        initial_capital: Starting balance
        slippage_pct: Slippage per trade (0.05% default)
        commission_per_trade: Commission per trade
    """
    try:
        import yfinance as yf

        # Get historical data
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return _empty_backtest_result(config.name, start_date, end_date, initial_capital)

        # Calculate indicators
        prices = data["Close"].values
        volumes = data["Volume"].values

        # RSI
        rsi_values = _calculate_rsi_series(prices, period=14)

        # MACD
        macd_line, signal_line = _calculate_macd_series(prices)

        # Run simulation
        capital = initial_capital
        position = None
        trades = []
        equity_curve = [initial_capital]

        for i in range(30, len(prices) - 1):  # Skip warmup period
            current_price = float(prices[i])
            current_date = data.index[i].strftime("%Y-%m-%d")

            # Skip if not enough data for indicators
            if i >= len(rsi_values) or i >= len(macd_line):
                continue

            current_rsi = rsi_values[i]
            current_macd = macd_line[i]
            current_signal = signal_line[i] if i < len(signal_line) else 0
            current_volume = volumes[i]
            avg_volume = float(np.mean(volumes[max(0, i-20):i]))
            volume_ratio = (current_volume / avg_volume) if avg_volume > 0 else 1.0

            # Check exit conditions if position open
            if position is not None:
                exit_signal = None
                exit_price = current_price

                # Stop loss
                stop_loss_price = position["entry_price"] * (1 - config.stop_loss_pct / 100)
                if current_price <= stop_loss_price:
                    exit_signal = "stop_loss"
                    exit_price = stop_loss_price * (1 - slippage_pct / 100)

                # Take profit
                tp_price = position["entry_price"] * (1 + config.take_profit_pct / 100)
                if current_price >= tp_price:
                    exit_signal = "take_profit"
                    exit_price = tp_price * (1 - slippage_pct / 100)

                # Max holding period
                holding_days = (data.index[i] - data.index[position["entry_idx"]]).days
                if holding_days >= config.max_holding_days:
                    exit_signal = "time_exit"

                # RSI overbought exit
                if current_rsi > config.rsi_overbought:
                    exit_signal = "rsi_overbought"

                if exit_signal:
                    # Close position
                    pnl = (exit_price - position["entry_price"]) * position["quantity"] - commission_per_trade
                    pnl_pct = ((exit_price - position["entry_price"]) / position["entry_price"]) * 100

                    capital += position["quantity"] * exit_price - commission_per_trade

                    trade = BacktestTrade(
                        ticker=ticker,
                        entry_date=position["entry_date"],
                        exit_date=current_date,
                        entry_price=position["entry_price"],
                        exit_price=exit_price,
                        quantity=position["quantity"],
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason=exit_signal,
                        holding_days=holding_days,
                    )
                    trades.append(trade)
                    position = None

            # Check entry conditions if no position
            if position is None:
                buy_signal = False

                # Strategy: Buy when oversold + MACD bullish + volume confirms
                rsi_ok = current_rsi < (config.rsi_oversold + 10) and current_rsi > 25
                macd_ok = current_macd > current_signal if config.use_macd_filter else True
                volume_ok = volume_ratio >= config.min_volume_ratio if config.use_volume_filter else True

                if rsi_ok and macd_ok and volume_ok:
                    buy_signal = True

                if buy_signal:
                    # Calculate position size
                    position_value = capital * (config.max_position_size_pct / 100)
                    entry_price = current_price * (1 + slippage_pct / 100)
                    quantity = position_value / entry_price

                    if quantity > 0:
                        capital -= quantity * entry_price + commission_per_trade
                        position = {
                            "entry_price": entry_price,
                            "quantity": quantity,
                            "entry_date": current_date,
                            "entry_idx": i,
                        }

            # Update equity curve
            current_equity = capital
            if position:
                current_equity += position["quantity"] * current_price
            equity_curve.append(current_equity)

        # Close any open position at end
        if position is not None:
            final_price = float(prices[-1])
            pnl = (final_price - position["entry_price"]) * position["quantity"]
            pnl_pct = ((final_price - position["entry_price"]) / position["entry_price"]) * 100
            capital += position["quantity"] * final_price

            trades.append(BacktestTrade(
                ticker=ticker,
                entry_date=position["entry_date"],
                exit_date=data.index[-1].strftime("%Y-%m-%d"),
                entry_price=position["entry_price"],
                exit_price=final_price,
                quantity=position["quantity"],
                pnl=pnl,
                pnl_pct=pnl_pct,
                exit_reason="end_of_period",
                holding_days=(data.index[-1] - data.index[position["entry_idx"]]).days,
            ))

        # Calculate metrics
        return _calculate_backtest_metrics(
            config.name, start_date, end_date, initial_capital, capital,
            trades, equity_curve
        )

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return _empty_backtest_result(config.name, start_date, end_date, initial_capital)


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────

def _calculate_rsi_series(prices, period: int = 14):
    """Calculate RSI series."""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    rsi = np.zeros(len(prices))
    rsi[:period] = 50  # warmup

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(prices)):
        if i < len(gains):
            avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period

        if avg_loss == 0:
            rsi[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))

    return rsi


def _calculate_macd_series(prices, fast=12, slow=26, signal=9):
    """Calculate MACD and signal line series."""
    def ema(data, period):
        alpha = 2 / (period + 1)
        result = np.zeros(len(data))
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result

    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)

    return macd_line, signal_line


# ─────────────────────────────────────────────────────────────────────────────
# METRICS CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def _calculate_backtest_metrics(
    strategy_name, start_date, end_date, initial_capital, final_capital,
    trades, equity_curve
) -> BacktestResult:
    """Calculate comprehensive metrics from trades."""

    total_return_pct = ((final_capital - initial_capital) / initial_capital * 100)

    if not trades:
        return BacktestResult(
            strategy_name=strategy_name,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return_pct=total_return_pct,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            avg_win=0,
            avg_loss=0,
            profit_factor=0,
            sharpe_ratio=0,
            max_drawdown_pct=0,
            longest_winning_streak=0,
            longest_losing_streak=0,
            avg_holding_days=0,
            trades=[],
            equity_curve=equity_curve,
        )

    winning_trades = [t for t in trades if t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl < 0]

    win_rate = (len(winning_trades) / len(trades) * 100) if trades else 0
    avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0

    gross_profit = sum(t.pnl for t in winning_trades)
    gross_loss = abs(sum(t.pnl for t in losing_trades))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0

    # Calculate Sharpe ratio
    returns = [(equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
               for i in range(1, len(equity_curve))]
    sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if returns and np.std(returns) > 0 else 0

    # Max drawdown
    peak = equity_curve[0]
    max_dd = 0
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (value - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd

    # Streaks
    streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    current_type = None

    for t in trades:
        is_win = t.pnl > 0
        if (is_win and current_type == "win") or (not is_win and current_type == "loss"):
            streak += 1
        else:
            streak = 1
            current_type = "win" if is_win else "loss"

        if current_type == "win":
            max_win_streak = max(max_win_streak, streak)
        else:
            max_loss_streak = max(max_loss_streak, streak)

    return BacktestResult(
        strategy_name=strategy_name,
        start_date=start_date.isoformat() if hasattr(start_date, "isoformat") else str(start_date),
        end_date=end_date.isoformat() if hasattr(end_date, "isoformat") else str(end_date),
        initial_capital=initial_capital,
        final_capital=final_capital,
        total_return_pct=total_return_pct,
        total_trades=len(trades),
        winning_trades=len(winning_trades),
        losing_trades=len(losing_trades),
        win_rate=win_rate,
        avg_win=float(avg_win),
        avg_loss=float(avg_loss),
        profit_factor=float(profit_factor),
        sharpe_ratio=float(sharpe),
        max_drawdown_pct=float(max_dd),
        longest_winning_streak=max_win_streak,
        longest_losing_streak=max_loss_streak,
        avg_holding_days=float(np.mean([t.holding_days for t in trades])),
        trades=trades,
        equity_curve=equity_curve,
    )


def _empty_backtest_result(strategy_name, start_date, end_date, initial_capital) -> BacktestResult:
    return BacktestResult(
        strategy_name=strategy_name,
        start_date=start_date.isoformat() if hasattr(start_date, "isoformat") else str(start_date),
        end_date=end_date.isoformat() if hasattr(end_date, "isoformat") else str(end_date),
        initial_capital=initial_capital,
        final_capital=initial_capital,
        total_return_pct=0,
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        win_rate=0,
        avg_win=0,
        avg_loss=0,
        profit_factor=0,
        sharpe_ratio=0,
        max_drawdown_pct=0,
        longest_winning_streak=0,
        longest_losing_streak=0,
        avg_holding_days=0,
        trades=[],
        equity_curve=[initial_capital],
    )


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────

async def optimize_strategy_parameters(
    ticker: str,
    start_date: datetime,
    end_date: datetime,
    parameter_grid: Optional[dict] = None,
) -> dict:
    """
    Grid search to find optimal strategy parameters.

    Tests combinations of:
    - stop_loss_pct: [1.5, 2.0, 2.5, 3.0]
    - take_profit_pct: [3.0, 4.5, 6.0, 8.0]
    - rsi_oversold: [25, 30, 35]
    """
    if parameter_grid is None:
        parameter_grid = {
            "stop_loss_pct": [1.5, 2.0, 2.5, 3.0],
            "take_profit_pct": [3.0, 4.5, 6.0, 8.0],
            "rsi_oversold": [25, 30, 35],
        }

    best_result = None
    best_params = None
    all_results = []

    # Grid search
    for sl in parameter_grid["stop_loss_pct"]:
        for tp in parameter_grid["take_profit_pct"]:
            for rsi in parameter_grid["rsi_oversold"]:
                config = StrategyConfig(
                    name=f"SL{sl}-TP{tp}-RSI{rsi}",
                    stop_loss_pct=sl,
                    take_profit_pct=tp,
                    rsi_oversold=rsi,
                )

                result = await run_backtest(ticker, config, start_date, end_date)

                # Score: combination of return + Sharpe + low drawdown
                score = result.total_return_pct + (result.sharpe_ratio * 10) + (abs(result.max_drawdown_pct) * -1)

                all_results.append({
                    "params": {"sl": sl, "tp": tp, "rsi": rsi},
                    "return_pct": result.total_return_pct,
                    "sharpe": result.sharpe_ratio,
                    "win_rate": result.win_rate,
                    "max_dd": result.max_drawdown_pct,
                    "score": score,
                })

                if best_result is None or score > best_result["score"]:
                    best_result = all_results[-1]
                    best_params = config

    # Sort all results by score
    all_results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "best_params": best_result,
        "top_5": all_results[:5],
        "total_combinations_tested": len(all_results),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

async def compare_strategies(
    ticker: str,
    strategies: list[StrategyConfig],
    start_date: datetime,
    end_date: datetime,
) -> dict:
    """
    Run multiple strategies and compare results side-by-side.
    """
    results = []

    for strategy in strategies:
        result = await run_backtest(ticker, strategy, start_date, end_date)
        results.append({
            "name": strategy.name,
            "return_pct": result.total_return_pct,
            "win_rate": result.win_rate,
            "sharpe": result.sharpe_ratio,
            "max_dd": result.max_drawdown_pct,
            "total_trades": result.total_trades,
            "profit_factor": result.profit_factor,
        })

    # Find winner
    results.sort(key=lambda x: x["sharpe"], reverse=True)

    return {
        "ticker": ticker,
        "period": f"{start_date.date()} to {end_date.date()}",
        "strategies_tested": len(strategies),
        "winner": results[0] if results else None,
        "all_results": results,
    }
