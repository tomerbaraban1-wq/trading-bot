"""
Exit-Aware Walk-Forward Validation — the FULL strategy, out-of-sample.
=====================================================================

validate_walkforward.py validates only the ENTRY bar using a raw 5-day forward
return. But this bot's real edge lives in its asymmetric EXITS (an ATR trailing
stop + a 10% take-profit + a max-hold), which turn a ~54% win-rate into profit.
This tool measures the WHOLE strategy — entry + those real exits — out-of-sample.

How it stays faithful instead of inventing a toy model:
  • It reuses the SAME signal/score generation as the live learner (_quick_score).
  • It reuses the SAME indicators pipeline (add_all_indicators → atr_14), so the
    ATR is computed AS-OF each historical bar — never lookahead, never live ATR.
  • It applies the EXACT exit math from atr_stop.py (atr×MULTIPLIER, clamped to
    [MIN_STOP_PCT, MAX_STOP_PCT], a stop that only trails UP) plus the live
    TAKE_PROFIT_PCT, pulling those constants straight from the real modules so it
    auto-tracks any .env change.

⚠️ HONEST LIMITATION — granularity. The live bot monitors INTRADAY (≈hourly holds,
5-minute confirmation candles). This backtest runs on DAILY bars, so it cannot
see the intraday path. Same-bar ties are resolved CONSERVATIVELY (assume the stop
is hit before the take-profit). Therefore: use this for RELATIVE comparison of
exit-parameter choices out-of-sample — NOT as an absolute predictor of live P&L.
The win-rate / payoff it prints should be sanity-checked against the live numbers;
a large divergence is the granularity gap talking, and is reported honestly.

Read-only. Touches no live trading, no DB, no .env.

Run:
    python validate_exits.py                       # quick: top 40 names
    python validate_exits.py --max 107             # full watchlist
    python validate_exits.py --max-hold 5          # vary the hold-day cap
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

# Real signal/score generation + objective (same as the live learner).
from backtest_learner import _quick_score, _find_optimal_threshold, MIN_SAMPLES, HOLD_PERIOD
# Real exit constants — pulled live so this stays faithful to production config.
from atr_stop import MULTIPLIER, MIN_STOP_PCT, MAX_STOP_PCT
from config import settings
# Reuse the walk-forward helpers so there is ONE implementation of each.
from validate_walkforward import _expectancy, _add_days, _FLOOR

TAKE_PROFIT_PCT: float = float(getattr(settings, "TAKE_PROFIT_PCT", 10.0))


def _simulate_exit(hist, i: int, max_hold_days: int) -> tuple[float, str]:
    """Faithfully simulate the bot's ATR-trailing + take-profit + max-hold exit
    for an entry at bar i (entry = that bar's close). Returns (return_pct, reason).

    Conservative daily-bar convention: within a bar the adverse move (stop) is
    assumed to trigger before the favorable one (take-profit); the trailing stop
    ratchets up only AFTER a bar's high sets a new watermark (it cannot protect
    against the same bar that made the high).
    """
    entry = float(hist.iloc[i]["close"])
    atr = float(hist.iloc[i].get("atr_14") or 0.0)
    if entry <= 0:
        return 0.0, "invalid"

    # ATR stop distance, clamped exactly as atr_stop.compute_initial_stop does.
    raw_dist = atr * MULTIPLIER
    stop_dist = max(entry * MIN_STOP_PCT / 100, min(entry * MAX_STOP_PCT / 100, raw_dist))
    stop = entry - stop_dist
    wm = entry
    tp_price = entry * (1 + TAKE_PROFIT_PCT / 100)

    n = len(hist)
    last_j = min(i + max_hold_days, n - 1)
    for j in range(i + 1, last_j + 1):
        bar = hist.iloc[j]
        hi, lo = float(bar["high"]), float(bar["low"])
        if lo <= stop:                      # trailing/initial stop breached
            return (stop - entry) / entry * 100.0, "stop"
        if hi >= tp_price:                  # take-profit reached
            return TAKE_PROFIT_PCT, "take_profit"
        if hi > wm:                         # new high → trail the stop up (only up)
            wm = hi
            stop = max(stop, round(wm - stop_dist, 4))
    # Max-hold reached → exit at the last available close (time exit).
    last_close = float(hist.iloc[last_j]["close"])
    return (last_close - entry) / entry * 100.0, "time"


def _analyze_ticker_exits(ticker: str, lookback_days: int, max_hold_days: int) -> list[dict]:
    """Like backtest_learner._analyze_ticker, but each signal's return is the
    FULL-STRATEGY modeled return (with real exits), stored under 'forward_return'
    so the shared threshold/expectancy helpers work unchanged."""
    import yfinance as yf
    from indicators import add_all_indicators

    hist = yf.Ticker(ticker).history(period=f"{lookback_days + 30}d", auto_adjust=True)
    if hist.empty or len(hist) < 60:
        return []
    hist.columns = [c.lower() for c in hist.columns]
    hist = hist[["open", "high", "low", "close", "volume"]].dropna()
    hist = add_all_indicators(hist)

    out: list[dict] = []
    for i in range(len(hist) - max_hold_days - 1):
        row = hist.iloc[i]
        if float(row["close"]) <= 0:
            continue
        simple_score = _quick_score(row)
        if simple_score < 45:               # same inclusion floor as the live analyzer
            continue
        ret, reason = _simulate_exit(hist, i, max_hold_days)
        out.append({
            "ticker": ticker,
            "date": str(hist.index[i])[:10],
            "forward_return": round(ret, 2),   # FULL-strategy return (with exits)
            "simple_score": simple_score,
            "exit_reason": reason,
        })
    return out


def _payoff(signals: list[dict], threshold: int) -> dict | None:
    """Win-rate and payoff (avg win / |avg loss|) for the bucket >= threshold —
    the metrics that actually describe an asymmetric exit strategy."""
    rets = [s["forward_return"] for s in signals if s["simple_score"] >= threshold]
    if not rets:
        return None
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r < 0]
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    payoff = (avg_win / abs(avg_loss)) if losses else float("inf")
    return {
        "n": len(rets),
        "winrate": 100.0 * len(wins) / len(rets),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff": payoff,
    }


def _exit_mix(signals: list[dict], threshold: int) -> dict:
    """How the modeled trades exited (stop / take_profit / time) — a faithfulness
    check against the live exit distribution."""
    rets = [s for s in signals if s["simple_score"] >= threshold]
    mix: dict[str, int] = {}
    for s in rets:
        mix[s["exit_reason"]] = mix.get(s["exit_reason"], 0) + 1
    total = max(1, len(rets))
    return {k: f"{100*v/total:.0f}%" for k, v in sorted(mix.items(), key=lambda kv: -kv[1])}


def run(tickers: list[str], lookback_days: int, train_frac: float,
        embargo_days: int, max_hold_days: int) -> dict:
    pooled: list[dict] = []
    t0 = time.time()
    for idx, tk in enumerate(tickers, 1):
        try:
            pooled.extend(_analyze_ticker_exits(tk, lookback_days, max_hold_days))
        except Exception as e:
            print(f"  [skip] {tk}: {e}")
        if idx % 10 == 0 or idx == len(tickers):
            print(f"  …{idx}/{len(tickers)} tickers, {len(pooled)} signals ({time.time()-t0:.0f}s)")

    if len(pooled) < 4 * MIN_SAMPLES:
        return {"error": f"only {len(pooled)} signals — too few (need ≥ {4*MIN_SAMPLES})."}

    pooled.sort(key=lambda s: s["date"])
    dates = [s["date"] for s in pooled]
    cutoff = dates[int(len(dates) * train_frac)]
    embargo_end = _add_days(cutoff, embargo_days)
    train = [s for s in pooled if s["date"] < cutoff]
    test = [s for s in pooled if s["date"] >= embargo_end]
    if not train or not test:
        return {"error": "empty split side — widen lookback."}

    chosen = _find_optimal_threshold(train)
    return {
        "total": len(pooled), "train_n": len(train), "test_n": len(test),
        "dropped": len(pooled) - len(train) - len(test),
        "train_span": (train[0]["date"], train[-1]["date"]),
        "test_span": (test[0]["date"], test[-1]["date"]),
        "chosen": chosen,
        "is_exp": _expectancy(train, chosen), "oos_exp": _expectancy(test, chosen),
        "base_exp": _expectancy(test, _FLOOR),
        "oos_payoff": _payoff(test, chosen), "base_payoff": _payoff(test, _FLOOR),
        "oos_mix": _exit_mix(test, chosen),
        "params": {"atr_mult": MULTIPLIER, "min_stop%": MIN_STOP_PCT,
                   "max_stop%": MAX_STOP_PCT, "tp%": TAKE_PROFIT_PCT,
                   "max_hold_days": max_hold_days},
    }


def _fmt(e: dict | None) -> str:
    if not e:
        return "n=0"
    return (f"n={e['n']:<4d} per-trade={e['mean']:+.2f}%  (±{e['stderr']:.2f} se)  "
            f"win-rate={e['winrate']:.0f}%")


def report(r: dict, max_hold_days: int) -> None:
    print("\n" + "=" * 70)
    print("  EXIT-AWARE WALK-FORWARD — full strategy (entry + real exits), OOS")
    print("=" * 70)
    if "error" in r:
        print(f"\n  ⚠️  {r['error']}\n")
        return
    p = r["params"]
    print(f"\n  Exit model (from live config): ATR×{p['atr_mult']} stop, clamp "
          f"[{p['min_stop%']}%,{p['max_stop%']}%], take-profit {p['tp%']}%, "
          f"max-hold {p['max_hold_days']}d")
    print(f"  Signals: {r['total']} | TRAIN {r['train_n']} | TEST {r['test_n']} "
          f"| embargo-dropped {r['dropped']}")
    print(f"  TRAIN {r['train_span'][0]}→{r['train_span'][1]} | "
          f"TEST {r['test_span'][0]}→{r['test_span'][1]}")

    print(f"\n  Buy-bar chosen on TRAIN only: MIN_BUY_SCORE = {r['chosen']}")
    print(f"    • IN-SAMPLE : {_fmt(r['is_exp'])}")
    print(f"    • OUT-SAMPLE: {_fmt(r['oos_exp'])}")
    if r["is_exp"] and r["oos_exp"]:
        print(f"    → in→out gap: {r['is_exp']['mean'] - r['oos_exp']['mean']:+.2f}%")

    op, bp = r["oos_payoff"], r["base_payoff"]
    if op:
        print(f"\n  OUT-OF-SAMPLE strategy shape @ bar {r['chosen']}:")
        print(f"    win-rate {op['winrate']:.0f}%  |  avg win {op['avg_win']:+.2f}%  |  "
              f"avg loss {op['avg_loss']:+.2f}%  |  payoff {op['payoff']:.2f}:1")
        print(f"    exit mix: {r['oos_mix']}")
    if bp:
        print(f"  Reference — no selection (bar {_FLOOR}): win-rate {bp['winrate']:.0f}%, "
              f"payoff {bp['payoff']:.2f}:1, per-trade {r['base_exp']['mean']:+.2f}%")

    # ── Honest verdict ────────────────────────────────────────────────────────
    print("\n  " + "-" * 66)
    if r["test_n"] < 30:
        print("  ⚠️  TEST sample <30 — SUGGESTIVE, not proof.")
    if r["oos_exp"] and r["base_exp"]:
        added = r["oos_exp"]["mean"] - r["base_exp"]["mean"]
        if added > 0.10:
            print(f"  ✅ The trained buy-bar adds OOS value (+{added:.2f}% vs taking all).")
        elif added < -0.10:
            print(f"  ❌ The trained buy-bar HURTS OOS ({added:.2f}%) — overfitting sign.")
        else:
            print("  ➖ Buy-bar ≈ no-selection OOS — entry tuning adds little even WITH exits.")
    print("  ℹ️  Daily-bar model of an intraday strategy — RELATIVE tool, not absolute P&L.")
    print("  " + "-" * 66 + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Exit-aware walk-forward validation.")
    ap.add_argument("--max", type=int, default=40)
    ap.add_argument("--lookback", type=int, default=365)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--max-hold", type=int, default=5, help="max hold in trading days (daily-bar approx)")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from scanner import get_watchlist
    universe = get_watchlist()[: args.max]
    embargo = HOLD_PERIOD + 2
    print(f"Universe {len(universe)} | lookback {args.lookback}d | train_frac "
          f"{args.train_frac} | max-hold {args.max_hold}d | embargo {embargo}d")
    r = run(universe, args.lookback, args.train_frac, embargo, args.max_hold)
    report(r, args.max_hold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
