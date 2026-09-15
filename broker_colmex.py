"""
Colmex Pro broker adapter — via a MetaTrader 4 bridge.

The bot talks to this exactly as it talks to broker_ibkr: same BrokerBase
methods, same return keys, same guards. Only the transport underneath differs.

WHY A BRIDGE AND NOT HTTP
-------------------------
Colmex Pro publishes no programmatic API (re-verified 2026-08-31). Its site
lists three platforms and nothing else:

  * Colmex Pro 2.0 (web/desktop/mobile) — GUI only.
  * TradingView — an ORDER-ENTRY integration. A human links the account and
    clicks buy in the TradingView UI. TradingView does not auto-execute
    strategies or alerts into a linked broker, so it cannot drive this bot.
  * MetaTrader 4 — runs Expert Advisors. This is the only automation-capable
    route Colmex offers.

MT4 hosts MQL4 code, not Python, and exposes no REST endpoint. So the
integration is two halves: an Expert Advisor (mt4/ColmexBridge.mq4) running
inside Colmex's MT4 terminal, and this module, which speaks to it through JSON
files in MT4's sandbox directory (MQL4/Files).

Files rather than a socket on purpose: MQL4 can only open sockets by importing
a Windows DLL, which requires enabling "Allow DLL imports" — a terminal-wide
setting that lets ANY EA call arbitrary native code. File I/O inside the sandbox
needs no such permission. The bridge is slower (polling, ~1s granularity), which
is irrelevant for a bot holding positions for hours.

PROTOCOL
--------
  EA  -> Python   colmex_state.json      rewritten every timer tick:
                                         account, positions, pending sells,
                                         market-open flag, and a `ts` heartbeat.
  Python -> EA    colmex_req_<id>.json   one order request.
  EA  -> Python   colmex_res_<id>.json   that order's outcome.

Staleness is treated as failure, never as "flat". If the EA stops, MT4 closes,
or the terminal loses its connection, colmex_state.json stops being refreshed —
and a stale file still parses fine, still contains an account, and still lists
positions that may no longer be true. Every read therefore checks `ts` and
raises ConnectionError past COLMEX_STATE_MAX_AGE rather than returning empty.
That direction matters: main.py's startup reconciliation reads an exception as
"no evidence" and leaves the database alone, but would read an empty list as
possible truth. Returning [] from a dead bridge is precisely how 17 real
positions were force-closed as 'stale_restart' on the IBKR side.

COLMEX-SPECIFIC ADAPTATIONS
---------------------------
These are the places where Colmex/MT4 differs from IBKR and the code has to
compensate. They are also the values that must be confirmed against the live
account before this is trusted:

  1. SYMBOLS — MT4 symbols are broker-named and rarely bare tickers. Colmex may
     list Apple as "AAPL", "AAPL.US", "#AAPL" or "AAPL.NAS". Set
     COLMEX_SYMBOL_SUFFIX, or COLMEX_SYMBOL_MAP for per-ticker exceptions. The
     EA reports the resolved name back so a mismatch is visible immediately.
  2. LOTS vs SHARES — MT4 orders are sized in lots, not shares, and one lot is
     MarketInfo(sym, MODE_LOTSIZE) units, which for equity instruments is often
     1 but can be 100. The EA does the conversion and speaks SHARES to this
     module, so the rest of the bot keeps working in shares like it does on IBKR.
  3. CURRENCY — the MT4 account may be denominated in ILS, not USD. The EA
     reports its account currency; equity figures are passed through as-is, the
     same way broker_ibkr prefers IBKR's BASE currency.
  4. INSTRUMENT TYPE — Colmex stopped offering CFDs to new clients on
     2026-01-30 and its TradingView listing shows real Stocks/ETFs. If this
     account predates that and still holds CFDs, overnight financing applies to
     every night a position is held (this bot holds up to 72h) and must be set
     in broker_costs.py. Confirm which one the account actually is.
  5. COMMISSION — Colmex publishes $1.75-2.50 per trade minimum plus
     $0.0035-0.007/share, i.e. ~$3.50-5.00 round trip. Not the $1.00/order
     figure this file previously recorded.

     This number is load-bearing, not trivia. Measured gross return is +0.291%
     per trade over 138 real strategy exits (61.6% win rate, +1.49% average
     win, -1.63% average loss). On a $1,000 position that edge is ~$2.91, so a
     $3.50-5.00 round trip erases it while the previously assumed $2.00 did
     not. Whether Colmex beats IBKR here depends entirely on the real schedule
     for this account tier — get it in writing before switching.

     Do NOT quote -0.031% as this strategy's return. That figure comes from
     averaging in 'stale_restart' rows, which are forced closes from bot
     restarts rather than strategy decisions, and it is wrong by an order of
     magnitude. Always filter with database._REAL_EXIT_STATUSES, and drop rows
     where exit_price == entry_price (iceberg slices that never traded).

BEFORE LIVE USE
---------------
COLMEX_ALLOW_LIVE gates every order. It defaults to False, so a misconfigured
run cannot place a real trade — the adapter reads state and refuses to order.
Run the full 30-trade validation on Colmex's demo account first, exactly as the
bot is doing on IBKR paper.
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime, time as dtime, timezone
from pathlib import Path

from broker_base import BrokerBase

logger = logging.getLogger(__name__)


# ─── CONFIGURE ───────────────────────────────────────────────────────────────
# MT4's sandbox. Typically:
#   %APPDATA%\MetaQuotes\Terminal\<32-hex-instance-id>\MQL4\Files
# Find it in MT4 via File -> Open Data Folder, then MQL4\Files.
COLMEX_MT4_FILES_DIR: str = os.getenv("COLMEX_MT4_FILES_DIR", "")

# Appended to every ticker to form the MT4 symbol (e.g. ".US" -> "AAPL.US").
COLMEX_SYMBOL_SUFFIX: str = os.getenv("COLMEX_SYMBOL_SUFFIX", "")

# JSON object for tickers the suffix rule gets wrong, e.g. {"BRK.B": "BRKB.US"}
COLMEX_SYMBOL_MAP: str = os.getenv("COLMEX_SYMBOL_MAP", "")

# Reject state older than this. The EA refreshes on a 1s timer, so 30s means
# roughly 30 missed ticks — long enough to ride out a slow terminal, short
# enough that a dead EA is caught within one heartbeat cycle.
COLMEX_STATE_MAX_AGE: float = float(os.getenv("COLMEX_STATE_MAX_AGE", "30"))

# How long to wait for the EA to answer an order request.
COLMEX_ORDER_TIMEOUT: float = float(os.getenv("COLMEX_ORDER_TIMEOUT", "30"))

# Master safety switch — no order leaves this module while False.
COLMEX_ALLOW_LIVE: bool = os.getenv("COLMEX_ALLOW_LIVE", "false").lower() == "true"
# ─────────────────────────────────────────────────────────────────────────────

_STATE_FILE = "colmex_state.json"
_REQ_PREFIX = "colmex_req_"
_RES_PREFIX = "colmex_res_"


class ColmexNotConfigured(RuntimeError):
    """Raised when the adapter cannot reach a usable MT4 bridge."""


class ColmexBroker(BrokerBase):
    """Colmex Pro via an MT4 Expert Advisor bridge."""

    def __init__(self):
        if not COLMEX_MT4_FILES_DIR:
            raise ColmexNotConfigured(
                "Colmex: COLMEX_MT4_FILES_DIR is not set. Open MT4 -> File -> "
                "Open Data Folder, and point it at that folder's MQL4\\Files."
            )
        self._dir = Path(COLMEX_MT4_FILES_DIR)
        if not self._dir.is_dir():
            raise ColmexNotConfigured(
                f"Colmex: COLMEX_MT4_FILES_DIR does not exist: {self._dir}"
            )

        try:
            self._symbol_map = json.loads(COLMEX_SYMBOL_MAP) if COLMEX_SYMBOL_MAP else {}
            if not isinstance(self._symbol_map, dict):
                raise ValueError("COLMEX_SYMBOL_MAP must be a JSON object")
        except Exception as e:
            raise ColmexNotConfigured(f"Colmex: bad COLMEX_SYMBOL_MAP ({e})") from e
        self._symbol_map = {k.upper(): v for k, v in self._symbol_map.items()}

        if not COLMEX_ALLOW_LIVE:
            logger.warning(
                "Colmex adapter is READ-ONLY (COLMEX_ALLOW_LIVE is not 'true'). "
                "Account and positions will be read; every order will be refused."
            )
        logger.info(f"Colmex bridge using {self._dir}")

    # ── Symbol translation ───────────────────────────────────────────────────

    def _mt4_symbol(self, ticker: str) -> str:
        """Plain ticker -> the symbol Colmex's MT4 lists it under."""
        t = ticker.upper().strip()
        return self._symbol_map.get(t, t + COLMEX_SYMBOL_SUFFIX)

    def _from_mt4_symbol(self, symbol: str) -> str:
        """
        MT4 symbol -> plain ticker. The inverse of _mt4_symbol().

        Both directions are needed. The bridge reports positions under Colmex's
        own names ("AAPL.US"), but every other part of the bot — the database,
        the scanner, the Telegram messages — works in bare tickers ("AAPL").
        Without translating back, get_position("AAPL") never matches the held
        "AAPL.US" row: the bot would believe it is flat while holding shares,
        buy the same position repeatedly, and never exit any of them.
        """
        s = symbol.upper().strip()
        for ticker, mapped in self._symbol_map.items():
            if mapped.upper() == s:
                return ticker
        if COLMEX_SYMBOL_SUFFIX and s.endswith(COLMEX_SYMBOL_SUFFIX.upper()):
            return s[: -len(COLMEX_SYMBOL_SUFFIX)]
        return s

    # ── Bridge I/O ───────────────────────────────────────────────────────────

    def _read_state(self) -> dict:
        """
        Latest EA snapshot, or ConnectionError.

        Raising rather than returning an empty dict is deliberate — see the
        module docstring. A caller must never be able to mistake a dead bridge
        for an account holding nothing.
        """
        path = self._dir / _STATE_FILE
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise ConnectionError(
                f"Colmex: no {_STATE_FILE} in {self._dir} — is ColmexBridge.mq4 "
                f"attached to a chart with AutoTrading enabled?"
            ) from None
        except OSError as e:
            raise ConnectionError(f"Colmex: cannot read {_STATE_FILE}: {e}") from e

        try:
            state = json.loads(raw)
        except json.JSONDecodeError as e:
            # The EA rewrites this file in place, so a read can land mid-write.
            # One retry after a short pause covers that without masking a file
            # that is genuinely malformed.
            time.sleep(0.3)
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                raise ConnectionError(f"Colmex: {_STATE_FILE} is not valid JSON: {e}") from e

        ts = float(state.get("ts", 0))
        age = time.time() - ts
        if age > COLMEX_STATE_MAX_AGE:
            raise ConnectionError(
                f"Colmex: bridge state is {age:.0f}s old (limit {COLMEX_STATE_MAX_AGE:.0f}s) — "
                f"MT4 or the EA has stopped updating. Refusing to report stale positions."
            )
        return state

    def _request(self, payload: dict) -> dict:
        """Send one order request to the EA and wait for its verdict."""
        if not COLMEX_ALLOW_LIVE:
            raise ColmexNotConfigured(
                "Colmex: refusing to send an order — COLMEX_ALLOW_LIVE is not "
                "'true'. Validate on the demo account before enabling this."
            )
        # Prove the EA is alive before writing a request; otherwise the order
        # file sits unread in the sandbox and silently executes whenever MT4
        # next starts, which could be hours later at a completely different price.
        self._read_state()

        req_id = uuid.uuid4().hex[:12]
        payload = {**payload, "id": req_id, "ts": time.time()}
        req_path = self._dir / f"{_REQ_PREFIX}{req_id}.json"
        res_path = self._dir / f"{_RES_PREFIX}{req_id}.json"

        # Write to a temporary name and rename into place, so the EA can never
        # pick up a half-written request.
        tmp = req_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(req_path)

        deadline = time.time() + COLMEX_ORDER_TIMEOUT
        while time.time() < deadline:
            if res_path.exists():
                try:
                    result = json.loads(res_path.read_text(encoding="utf-8"))
                except Exception:
                    time.sleep(0.3)
                    continue
                try:
                    res_path.unlink()
                except OSError:
                    pass
                if not result.get("ok"):
                    raise RuntimeError(
                        f"Colmex rejected {payload.get('action')} "
                        f"{payload.get('ticker')}: {result.get('error', 'unknown error')}"
                    )
                return result
            time.sleep(0.25)

        # Timed out. The EA may still execute the request afterwards, so remove
        # it — an order that fills minutes late at an unknown price is worse
        # than no order, and the bot will simply retry on the next cycle.
        try:
            req_path.unlink()
        except OSError:
            pass
        raise TimeoutError(
            f"Colmex: no response to {payload.get('action')} {payload.get('ticker')} "
            f"within {COLMEX_ORDER_TIMEOUT:.0f}s — request withdrawn"
        )

    # ── Interface implementation ─────────────────────────────────────────────

    def get_account(self) -> dict:
        """-> {equity, buying_power, cash, portfolio_value, status}"""
        try:
            acct = self._read_state().get("account", {})
            return {
                "equity": float(acct.get("equity", 0.0)),
                "buying_power": float(acct.get("buying_power", 0.0)),
                "cash": float(acct.get("cash", 0.0)),
                "portfolio_value": float(acct.get("portfolio_value", 0.0)),
                "status": "active",
            }
        except Exception as e:
            # Mirrors broker_ibkr.get_account: a status the caller can see
            # rather than an exception, since equity is read on many paths.
            logger.error(f"Colmex get_account failed: {e}")
            return {"equity": 0.0, "buying_power": 0.0, "cash": 0.0,
                    "portfolio_value": 0.0, "status": "unavailable"}

    def get_positions(self) -> list[dict]:
        """
        -> [{ticker, qty, avg_entry_price, current_price, market_value,
              unrealized_pl, unrealized_plpc}]

        qty is negative for shorts — the bot's short detection and the sell
        guard both key off the sign. Raises rather than returning [] when the
        bridge is unreachable.
        """
        state = self._read_state()          # ConnectionError propagates
        out = []
        for p in state.get("positions", []):
            try:
                qty = float(p["qty"])
                avg = float(p["avg_entry_price"])
                cur = float(p.get("current_price") or avg) or avg
            except (KeyError, TypeError, ValueError):
                logger.warning(f"Colmex: skipping malformed position row {p!r}")
                continue
            out.append({
                # Translated back to a bare ticker — the bridge reports Colmex's
                # MT4 symbol, the rest of the bot works in plain tickers.
                "ticker": self._from_mt4_symbol(str(p.get("ticker", ""))),
                "qty": qty,
                "avg_entry_price": avg,
                "current_price": cur,
                "market_value": qty * cur,
                "unrealized_pl": (cur - avg) * qty,
                "unrealized_plpc": ((cur - avg) / avg) if avg else 0.0,
            })
        return out

    def get_position(self, ticker: str) -> dict | None:
        for p in self.get_positions():
            if p["ticker"] == ticker.upper():
                return p
        return None

    @staticmethod
    def _whole_shares(ticker: str, qty: float) -> int:
        """
        Round down to whole shares. Same rule as broker_ibkr: position sizing
        produces fractions, and rounding up could overspend the budget. MT4
        additionally cannot express a fractional share below its lot step.
        """
        whole = int(qty)
        if whole < 1:
            raise ValueError(
                f"{ticker}: position size {qty:.4f} rounds to 0 whole shares — "
                f"share price exceeds the per-position budget"
            )
        return whole

    def submit_buy(self, ticker: str, qty: float, price: float | None = None) -> dict:
        """-> {order_id, symbol, qty, price, status, type}"""
        # Refuse to buy into an existing short, exactly as broker_ibkr does.
        # The bot is long-only, so a short means the broker and the DB have
        # diverged; an ordinary buy would partially cover it in a way the DB
        # never records. Covering is a separate, deliberate action.
        existing = self.get_position(ticker)
        if existing and existing["qty"] < 0:
            raise RuntimeError(
                f"Refusing to BUY {ticker}: broker shows an existing short "
                f"position ({existing['qty']:g} shares) — cover it explicitly "
                f"instead of buying through the normal entry path"
            )

        shares = self._whole_shares(ticker, qty)
        result = self._request({
            "action": "BUY",
            "ticker": ticker.upper(),
            "symbol": self._mt4_symbol(ticker),
            "shares": shares,
        })
        logger.info(f"Colmex BUY accepted: {ticker} x{shares} (ticket={result.get('order_id')})")
        return {
            "order_id": str(result.get("order_id", "")),
            "symbol": ticker.upper(),
            "qty": float(result.get("shares", shares)),
            "price": float(result["price"]) if result.get("price") else None,
            "status": str(result.get("status", "filled")),
            "type": "market",
        }

    def submit_sell(self, ticker: str, qty: float | None = None, price: float | None = None) -> dict:
        """-> {order_id, symbol, qty, price, status}"""
        # Never sell more than is actually held net of in-flight sells. On IBKR
        # the absence of this guard turned repeated exit attempts into a short
        # that reached -372 shares. MT4 behaves the same way: a sell beyond the
        # holding opens a new short ticket rather than failing.
        state = self._read_state()
        want = ticker.upper()
        held = 0.0
        for p in state.get("positions", []):
            if self._from_mt4_symbol(str(p.get("ticker", ""))) == want:
                held = float(p.get("qty", 0.0))
                break
        # pending_sells is keyed by MT4 symbol, so translate every key rather
        # than looking up the bare ticker and silently finding nothing.
        pending_sell = 0.0
        for sym, amount in (state.get("pending_sells", {}) or {}).items():
            if self._from_mt4_symbol(str(sym)) == want:
                pending_sell += float(amount)

        available = held - pending_sell
        if available <= 0:
            raise ValueError(
                f"Refusing to SELL {ticker}: {held:g} held minus {pending_sell:g} "
                f"already pending leaves {available:g} — selling would open a short"
            )

        requested = available if qty is None else float(qty)
        if requested > available:
            logger.warning(
                f"SELL {ticker}: requested {requested:g} but only {available:g} sellable "
                f"({held:g} held − {pending_sell:g} pending) — capping to avoid going short"
            )
            requested = available

        shares = self._whole_shares(ticker, requested)
        result = self._request({
            "action": "SELL",
            "ticker": ticker.upper(),
            "symbol": self._mt4_symbol(ticker),
            "shares": shares,
        })
        logger.info(f"Colmex SELL accepted: {ticker} x{shares} (ticket={result.get('order_id')})")
        return {
            "order_id": str(result.get("order_id", "")),
            "symbol": ticker.upper(),
            "qty": float(result.get("shares", shares)),
            "price": float(result["price"]) if result.get("price") else None,
            "status": str(result.get("status", "filled")),
        }

    def is_market_open(self) -> bool:
        """
        Prefer the EA's answer: it knows whether Colmex is actually quoting the
        instrument, including holidays and any broker-specific session. Falls
        back to US regular hours, and reports closed whenever the bridge is
        unreachable — never claim tradeable on a dead connection.
        """
        try:
            state = self._read_state()
        except Exception as e:
            logger.warning(f"Colmex is_market_open: bridge unavailable ({e}) — reporting closed")
            return False

        flag = state.get("market_open")
        if isinstance(flag, bool):
            return flag

        now = datetime.now(timezone.utc)
        if now.weekday() >= 5:
            return False
        return dtime(13, 30) <= now.time() < dtime(20, 0)

    def get_asset(self, ticker: str) -> dict | None:
        """
        -> {symbol, name, tradable, fractionable}

        Tradability is decided by whether Colmex actually lists the symbol in
        MT4, which the EA reports. An unknown ticker returns None rather than a
        guess, so the scanner skips it instead of ordering something the broker
        cannot fill.
        """
        try:
            symbols = self._read_state().get("symbols", {})
        except Exception as e:
            logger.warning(f"Colmex get_asset({ticker}): bridge unavailable ({e})")
            return None

        mt4_symbol = self._mt4_symbol(ticker)
        info = symbols.get(mt4_symbol)
        if not info:
            return None
        return {
            "symbol": ticker.upper(),
            "name": info.get("description", ticker.upper()),
            "tradable": bool(info.get("tradable", False)),
            # MT4 sizes in lot steps, so anything below one share is unavailable.
            "fractionable": False,
        }
