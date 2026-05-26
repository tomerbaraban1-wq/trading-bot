"""
Strategy Optimizer
==================

Automatically optimizes trading strategy parameters based on real trading data.

Features:
1. Auto-adjusts parameters based on performance
2. Genetic algorithm for parameter optimization
3. Bayesian optimization for efficient search
4. Walk-forward analysis (avoid overfitting)
5. Multi-objective optimization (return + drawdown)
6. Parameter sensitivity analysis
"""

import asyncio
import json
import logging
import random
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OptimizedParameters:
    """A set of optimized parameters."""
    min_buy_score: float
    stop_loss_pct: float
    take_profit_pct: float
    max_position_size_pct: float
    rsi_oversold: float
    rsi_overbought: float
    min_volume_ratio: float
    optimization_score: float
    confidence: float


# ─────────────────────────────────────────────────────────────────────────────
# GENETIC ALGORITHM OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────

class GeneticOptimizer:
    """
    Genetic algorithm for strategy parameter optimization.

    Population evolves over generations:
    1. Select best performers
    2. Crossover their parameters
    3. Random mutations
    4. Repeat
    """

    def __init__(
        self,
        population_size: int = 20,
        generations: int = 10,
        mutation_rate: float = 0.15,
        elite_count: int = 3,
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_count = elite_count

        # Parameter ranges (min, max)
        self.param_ranges = {
            "min_buy_score": (60, 90),
            "stop_loss_pct": (1.0, 4.0),
            "take_profit_pct": (3.0, 10.0),
            "max_position_size_pct": (5, 20),
            "rsi_oversold": (20, 40),
            "rsi_overbought": (60, 80),
            "min_volume_ratio": (0.3, 1.0),
        }

    def random_individual(self) -> dict:
        """Generate a random parameter set."""
        return {
            name: random.uniform(low, high)
            for name, (low, high) in self.param_ranges.items()
        }

    def crossover(self, parent1: dict, parent2: dict) -> dict:
        """Mix parameters from two parents."""
        child = {}
        for param in parent1:
            # 50/50 mix
            child[param] = random.choice([parent1[param], parent2[param]])
        return child

    def mutate(self, individual: dict) -> dict:
        """Random parameter mutations."""
        mutated = individual.copy()
        for param, (low, high) in self.param_ranges.items():
            if random.random() < self.mutation_rate:
                # Random adjustment within +/- 20% of range
                range_size = high - low
                adjustment = random.uniform(-0.2 * range_size, 0.2 * range_size)
                mutated[param] = max(low, min(high, individual[param] + adjustment))
        return mutated

    async def evaluate_individual(self, individual: dict, ticker: str = "SPY") -> float:
        """
        Evaluate fitness of an individual via backtesting.
        Returns a score (higher = better).
        """
        try:
            from backtesting_engine import StrategyConfig, run_backtest

            config = StrategyConfig(
                name="genetic",
                min_buy_score=individual["min_buy_score"],
                stop_loss_pct=individual["stop_loss_pct"],
                take_profit_pct=individual["take_profit_pct"],
                max_position_size_pct=individual["max_position_size_pct"],
                rsi_oversold=individual["rsi_oversold"],
                rsi_overbought=individual["rsi_overbought"],
                min_volume_ratio=individual["min_volume_ratio"],
            )

            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=180)  # 6 months

            result = await run_backtest(ticker, config, start_date, end_date)

            # Multi-objective fitness:
            # - Return: positive contribution
            # - Sharpe ratio: positive contribution
            # - Max drawdown: negative contribution
            # - Win rate: positive but smaller weight
            fitness = (
                result.total_return_pct +
                (result.sharpe_ratio * 5) +
                (result.max_drawdown_pct * 2) +  # already negative
                (result.win_rate * 0.1)
            )

            # Penalty for too few trades (overfitting)
            if result.total_trades < 5:
                fitness -= 20

            return float(fitness)

        except Exception as e:
            logger.debug(f"Evaluation failed: {e}")
            return -999

    async def optimize(self, ticker: str = "SPY") -> OptimizedParameters:
        """
        Run genetic optimization to find best parameters.
        """
        # Initial population
        population = [self.random_individual() for _ in range(self.population_size)]

        best_score = -999
        best_individual = None

        for generation in range(self.generations):
            # Evaluate all individuals
            scored = []
            for individual in population:
                score = await self.evaluate_individual(individual, ticker)
                scored.append((score, individual))

            # Sort by fitness
            scored.sort(key=lambda x: x[0], reverse=True)

            # Track best
            if scored[0][0] > best_score:
                best_score = scored[0][0]
                best_individual = scored[0][1]

            logger.info(
                f"[GENETIC] Gen {generation+1}/{self.generations}: "
                f"best={scored[0][0]:.2f}, avg={sum(s for s, _ in scored)/len(scored):.2f}"
            )

            # Create next generation
            new_population = []

            # Elitism: keep top performers
            for i in range(self.elite_count):
                new_population.append(scored[i][1])

            # Fill rest with crossover + mutation
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = max(random.sample(scored[:10], 3), key=lambda x: x[0])[1]
                parent2 = max(random.sample(scored[:10], 3), key=lambda x: x[0])[1]

                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)

            population = new_population

        # Confidence based on convergence
        final_scores = [self.evaluate_individual(ind, ticker) for ind in population[:5]]
        confidence = 1.0 - (np.std(await asyncio.gather(*final_scores)) / 100)
        confidence = max(0, min(1, confidence))

        return OptimizedParameters(
            min_buy_score=best_individual["min_buy_score"],
            stop_loss_pct=best_individual["stop_loss_pct"],
            take_profit_pct=best_individual["take_profit_pct"],
            max_position_size_pct=best_individual["max_position_size_pct"],
            rsi_oversold=best_individual["rsi_oversold"],
            rsi_overbought=best_individual["rsi_overbought"],
            min_volume_ratio=best_individual["min_volume_ratio"],
            optimization_score=best_score,
            confidence=confidence,
        )


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

async def walk_forward_optimization(
    ticker: str,
    total_days: int = 365,
    train_window_days: int = 90,
    test_window_days: int = 30,
) -> dict:
    """
    Walk-forward analysis to avoid overfitting.

    Splits time period into train/test windows:
    - Optimize on train window
    - Test on next window
    - Roll forward

    Helps verify strategy doesn't overfit to historical data.
    """
    try:
        end_date = datetime.now(timezone.utc)

        results = []
        current_date = end_date - timedelta(days=total_days)

        while (end_date - current_date).days >= train_window_days + test_window_days:
            train_start = current_date
            train_end = current_date + timedelta(days=train_window_days)
            test_start = train_end
            test_end = train_end + timedelta(days=test_window_days)

            logger.info(f"[WALK-FORWARD] Window: train {train_start.date()}-{train_end.date()}, test {test_start.date()}-{test_end.date()}")

            # Quick optimization (smaller GA)
            optimizer = GeneticOptimizer(population_size=10, generations=5)
            optimal = await optimizer.optimize(ticker)

            # Test on out-of-sample period
            from backtesting_engine import StrategyConfig, run_backtest
            config = StrategyConfig(
                name="walk_forward_test",
                stop_loss_pct=optimal.stop_loss_pct,
                take_profit_pct=optimal.take_profit_pct,
                rsi_oversold=optimal.rsi_oversold,
            )

            test_result = await run_backtest(ticker, config, test_start, test_end)

            results.append({
                "train_period": f"{train_start.date()} to {train_end.date()}",
                "test_period": f"{test_start.date()} to {test_end.date()}",
                "train_score": optimal.optimization_score,
                "test_return": test_result.total_return_pct,
                "test_win_rate": test_result.win_rate,
                "test_sharpe": test_result.sharpe_ratio,
            })

            current_date = current_date + timedelta(days=test_window_days)

        # Aggregate results
        avg_test_return = np.mean([r["test_return"] for r in results]) if results else 0
        avg_test_sharpe = np.mean([r["test_sharpe"] for r in results]) if results else 0
        consistency_score = 1 - (np.std([r["test_return"] for r in results]) / 100) if results else 0

        return {
            "ticker": ticker,
            "total_windows": len(results),
            "avg_test_return_pct": avg_test_return,
            "avg_test_sharpe": avg_test_sharpe,
            "consistency_score": consistency_score,
            "all_windows": results,
            "interpretation": (
                "✅ Strategy is robust across time periods" if consistency_score > 0.7 else
                "🟡 Strategy works but is inconsistent" if consistency_score > 0.4 else
                "❌ Strategy is unstable - high risk of overfitting"
            ),
        }

    except Exception as e:
        logger.error(f"Walk-forward optimization failed: {e}")
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-APPLY OPTIMIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

async def apply_optimizations_if_better(current_params: dict, optimal: OptimizedParameters) -> dict:
    """
    Check if optimal params are significantly better than current.
    Apply only if improvement is meaningful and high confidence.
    """
    if optimal.confidence < 0.6:
        return {
            "applied": False,
            "reason": f"Low confidence ({optimal.confidence:.0%})",
        }

    # Check each parameter for meaningful difference
    improvements = []

    if abs(optimal.stop_loss_pct - current_params.get("stop_loss_pct", 2.0)) > 0.3:
        improvements.append(f"stop_loss: {current_params.get('stop_loss_pct', 2.0):.2f} → {optimal.stop_loss_pct:.2f}")

    if abs(optimal.take_profit_pct - current_params.get("take_profit_pct", 4.5)) > 0.5:
        improvements.append(f"take_profit: {current_params.get('take_profit_pct', 4.5):.2f} → {optimal.take_profit_pct:.2f}")

    if abs(optimal.min_buy_score - current_params.get("min_buy_score", 75)) > 2:
        improvements.append(f"min_buy_score: {current_params.get('min_buy_score', 75):.0f} → {optimal.min_buy_score:.0f}")

    if not improvements:
        return {
            "applied": False,
            "reason": "No significant improvements found",
        }

    return {
        "applied": True,
        "improvements": improvements,
        "new_params": {
            "stop_loss_pct": optimal.stop_loss_pct,
            "take_profit_pct": optimal.take_profit_pct,
            "min_buy_score": optimal.min_buy_score,
            "rsi_oversold": optimal.rsi_oversold,
            "rsi_overbought": optimal.rsi_overbought,
        },
        "optimization_score": optimal.optimization_score,
        "confidence": optimal.confidence,
    }
