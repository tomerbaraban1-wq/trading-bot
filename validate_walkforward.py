"""
Walk-Forward Validation — does the learner's tuning actually GENERALIZE?
=======================================================================

The learning loop (backtest_learner) optimizes MIN_BUY_SCORE on historical
signals, then heartbeat applies it live. The danger — and the documented
real failure of this bot (a 90% → 56% collapse) — is OVERFITTING: a threshold
that looks great on the data it was tuned on, but falls apart on data it never
saw. In-sample performance is NOT evidence; out-of-sample performance is.

This tool answers that, honestly and reproducibly:

  1. Pool the same backtest signals the learner uses (reusing _analyze_ticker).
  2. Split them CHRONOLOGICALLY: an older TRAIN slice and a recent TEST slice.
  3. Insert an EMBARGO gap of HOLD_PERIOD days between the two so a TRAIN
     signal's forward-return window can never bleed into TEST (no lookahead).
  4. Pick the "optimal" threshold on TRAIN ONLY (exactly as the live learner
     would, via _find_optimal_threshold).
  5. Measure that threshold's expectancy on TRAIN vs on TEST.

The TRAIN→TEST gap is the overfitting magnitude. We also compute the TEST-set
"oracle" threshold (best in hindsight) and a no-selection baseline, so we can
say plainly whether tuning the buy-bar adds REAL out-of-sample value or not.

This is read-only analysis. It does NOT touch live trading, the DB, or .env.

Run:
    python validate_walkforward.py                 # quick: top 40 names
    python validate_walkforward.py --max 107       # full watchlist
    python validate_walkforward.py --lookback 504  # ~2y of history
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta

import numpy as np

# Reuse the EXACT signal generation + objective the live learner uses, so this
# validation reflects reality rather than a parallel re-implementation.
from backtest_learner import (
    _analyze_ticker,
    _find_optimal_threshold,
    MIN_SAMPLES,
    HOLD_PERIOD,
)

# The analysis-inclusion floor in _analyze_ticker: signals below this score are
# never generated, so "take everything" == threshold 45.
_FLOOR = 45


def _expectancy(signals: list[dict], threshold: int) -> dict | None:
    """Risk-adjusted expectancy of the bucket scoring >= threshold.

    Mirrors _find_optimal_threshold's metric (mean forward return minus half a
    standard error) and adds win-rate + count for reporting. None if empty.
    """
    bucket = [s["forward_return"] for s in signals if s["simple_score"] >= threshold]
    n = len(bucket)
    if n == 0:
        return None
    arr = np.asarray(bucket, dtype=float)
    mean = float(arr.mean())
    stderr = float(arr.std()) / (n ** 0.5) if n > 1 else 0.0
    winrate = 100.0 * float((arr > 0).sum()) / n
    return {
        "n": n,
        "mean": mean,
        "stderr": stderr,
        "metric": mean - 0.5 * stderr,
        "winrate": winrate,
    }


def _add_days(date_str: str, days: int) -> str:
    return (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=days)).strftime("%Y-%m-%d")


def collect_signals(tickers: list[str], lookback_days: int) -> list[dict]:
    """Run the real per-ticker analyzer over the universe; pool all signals."""
    pooled: list[dict] = []
    t0 = time.time()
    for i, tk in enumerate(tickers, 1):
        try:
            sigs = _analyze_ticker(tk, lookback_days)
            pooled.extend(sigs)
        except Exception as e:  # one bad ticker must not sink the run
            print(f"  [skip] {tk}: {e}")
        if i % 10 == 0 or i == len(tickers):
            print(f"  …analyzed {i}/{len(tickers)} tickers, "
                  f"{len(pooled)} signals so far ({time.time()-t0:.0f}s)")
    return pooled


def walk_forward(tickers: list[str], lookback_days: int,
                 train_frac: float, embargo_days: int) -> dict:
    pooled = collect_signals(tickers, lookback_days)
    if len(pooled) < 4 * MIN_SAMPLES:
        return {"error": f"only {len(pooled)} signals — too few to validate "
                         f"(need ≥ {4 * MIN_SAMPLES})."}

    pooled.sort(key=lambda s: s["date"])
    dates = [s["date"] for s in pooled]
    cutoff = dates[int(len(dates) * train_frac)]
    embargo_end = _add_days(cutoff, embargo_days)

    train = [s for s in pooled if s["date"] < cutoff]
    test = [s for s in pooled if s["date"] >= embargo_end]
    dropped = len(pooled) - len(train) - len(test)

    if not train or not test:
        return {"error": "split produced an empty side — widen the lookback."}

    # Pick the threshold on TRAIN only — exactly what the live learner does.
    chosen = _find_optimal_threshold(train)
    is_exp = _expectancy(train, chosen)
    oos_exp = _expectancy(test, chosen)

    # Reference points on TEST: the hindsight-best ("oracle") threshold, and the
    # no-selection baseline (take every signal). These bound what's achievable.
    oracle = _find_optimal_threshold(test)
    oracle_exp = _expectancy(test, oracle)
    base_exp = _expectancy(test, _FLOOR)

    return {
        "total": len(pooled),
        "train_n": len(train), "test_n": len(test), "embargo_dropped": dropped,
        "cutoff": cutoff, "embargo_end": embargo_end,
        "train_span": (train[0]["date"], train[-1]["date"]),
        "test_span": (test[0]["date"], test[-1]["date"]),
        "chosen": chosen, "is_exp": is_exp, "oos_exp": oos_exp,
        "oracle": oracle, "oracle_exp": oracle_exp, "base_exp": base_exp,
    }


def _fmt(e: dict | None) -> str:
    if not e:
        return "n=0 (no signals at this bar)"
    return (f"n={e['n']:<4d} expectancy={e['mean']:+.2f}%  "
            f"(±{e['stderr']:.2f} se)  win-rate={e['winrate']:.0f}%")


def report(r: dict) -> None:
    print("\n" + "=" * 68)
    print("  WALK-FORWARD VALIDATION — does threshold tuning generalize?")
    print("=" * 68)
    if "error" in r:
        print(f"\n  ⚠️  {r['error']}\n")
        return

    print(f"\n  Signals: {r['total']} total | TRAIN {r['train_n']} | "
          f"TEST {r['test_n']} | embargo-dropped {r['embargo_dropped']}")
    print(f"  TRAIN window: {r['train_span'][0]} → {r['train_span'][1]}")
    print(f"  TEST  window: {r['test_span'][0]} → {r['test_span'][1]}  "
          f"(embargo gap of {HOLD_PERIOD} trading days at the seam)")

    print(f"\n  Threshold chosen on TRAIN only: MIN_BUY_SCORE = {r['chosen']}")
    print(f"    • IN-SAMPLE  @ {r['chosen']}: {_fmt(r['is_exp'])}")
    print(f"    • OUT-SAMPLE @ {r['chosen']}: {_fmt(r['oos_exp'])}")

    is_m = r["is_exp"]["mean"] if r["is_exp"] else None
    oos_m = r["oos_exp"]["mean"] if r["oos_exp"] else None
    if is_m is not None and oos_m is not None:
        print(f"    → IN→OUT gap: {is_m - oos_m:+.2f}%  "
              f"(how much the in-sample number flattered itself)")

    print(f"\n  TEST-set reference points:")
    print(f"    • no-selection (take all, bar={_FLOOR}): {_fmt(r['base_exp'])}")
    print(f"    • hindsight-best bar on TEST = {r['oracle']}: {_fmt(r['oracle_exp'])}")

    # ── Honest verdict ────────────────────────────────────────────────────────
    print("\n  " + "-" * 64)
    verdict = []
    test_n = r["test_n"]
    if test_n < 30:
        verdict.append("⚠️  TEST sample is small (<30) — treat as SUGGESTIVE, not proof.")

    if r["oos_exp"] and r["base_exp"]:
        added = r["oos_exp"]["mean"] - r["base_exp"]["mean"]
        if added > 0.10:
            verdict.append(f"✅ The trained bar ADDS out-of-sample value "
                           f"(+{added:.2f}% vs taking everything) — it generalizes.")
        elif added < -0.10:
            verdict.append(f"❌ The trained bar HURTS out-of-sample "
                           f"({added:.2f}% vs taking everything) — sign of overfitting.")
        else:
            verdict.append("➖ The trained bar is ≈ no-selection out-of-sample — "
                           "threshold tuning adds little real edge here.")

    if is_m is not None and oos_m is not None and (is_m - oos_m) > 0.5:
        verdict.append(f"⚠️  Large in→out drop ({is_m - oos_m:+.2f}%) — the in-sample "
                       "result is NOT a reliable promise of live performance.")

    for v in verdict:
        print(f"  {v}")
    print("  " + "-" * 64 + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Walk-forward validation of buy-bar tuning.")
    ap.add_argument("--max", type=int, default=40, help="max tickers from the watchlist (default 40)")
    ap.add_argument("--lookback", type=int, default=365, help="calendar days of history (default 365)")
    ap.add_argument("--train-frac", type=float, default=0.70, help="fraction of signals for TRAIN (default 0.70)")
    args = ap.parse_args()

    # The Windows console is often cp1255 (Hebrew) and can't encode the arrows /
    # ± / emoji in the report. Force UTF-8 so the output never crashes on a glyph.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from scanner import get_watchlist
    universe = get_watchlist()[: args.max]
    embargo_days = HOLD_PERIOD + 2  # cover the 5 trading-day forward window in calendar days

    print(f"Universe: {len(universe)} tickers | lookback {args.lookback}d | "
          f"train_frac {args.train_frac} | embargo {embargo_days}d")
    r = walk_forward(universe, args.lookback, args.train_frac, embargo_days)
    report(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
