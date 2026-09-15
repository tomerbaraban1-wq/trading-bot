import asyncio
import itertools
import logging
import threading
import time
from datetime import datetime, time as dtime
from broker_base import BrokerBase

logger = logging.getLogger(__name__)


class IBKRBroker(BrokerBase):
    """Interactive Brokers broker via ib_insync."""

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        self._host = host
        self._port = port
        self._client_id = client_id
        # ib_insync ties each IB() and its asyncio event loop to a single thread.
        # The bot calls broker methods from a thread pool (bot-io_*), so a single
        # shared IB() fails with "no current event loop in thread" the moment a
        # different worker touches it. We therefore keep one connection PER THREAD
        # (thread-local) — each with its own event loop and a unique clientId.
        self._local = threading.local()
        self._cid_counter = itertools.count(client_id)
        self._cid_lock = threading.Lock()
        # Serialize connection ATTEMPTS across all threads. Root cause of the
        # "Cannot run the event loop while another loop is running" failures
        # (seen repeatedly in production, including the retry also failing):
        # several worker threads used to race to open a fresh IB() connection
        # to the same Gateway at once. asyncio's "is a loop running in this
        # thread?" check is thread-local so it should not itself cross
        # threads — but IB Gateway's API can stall/refuse a socket accept
        # when hit with several near-simultaneous connect handshakes, and a
        # stalled connect() (util.run -> run_until_complete) that overruns
        # its own timeout leaves that pooled thread's loop marked "running"
        # indefinitely (Python cannot forcibly abort a blocked thread), so a
        # LATER, unrelated call reusing that same pooled thread then hits the
        # exact "another loop is running" error. One lock means only one
        # connect handshake happens at a time — no more races, no more stuck
        # threads poisoning later calls.
        self._connect_lock = threading.Lock()
        self._down_until = 0.0  # circuit breaker: monotonic time until which we fail fast

    def _next_client_id(self) -> int:
        with self._cid_lock:
            return next(self._cid_counter)

    def _connect_once(self):
        """One connection attempt on the current thread with a guaranteed-fresh loop."""
        # A brand-new loop every attempt — never reuse asyncio.get_event_loop(),
        # which can hand back a stale/broken loop left over from a previous
        # connection on this same pooled thread (e.g. after Gateway drops the
        # socket) and trigger "Cannot run the event loop while another loop is
        # running". A fresh loop removes that failure mode entirely.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            from ib_insync import IB
            ib = IB()
            cid = self._next_client_id()
            ib.connect(self._host, self._port, clientId=cid, timeout=10)
            logger.info(
                f"IBKR connected to {self._host}:{self._port} "
                f"(clientId={cid}, thread={threading.current_thread().name})"
            )
            return ib
        except BaseException:
            # Never leave a half-open loop/socket behind on failure — that is
            # exactly what poisons the NEXT attempt on this pooled thread.
            try:
                loop.close()
            except Exception:
                pass
            asyncio.set_event_loop(None)
            raise

    # How long to stop trying after Gateway refuses/times out a connection.
    _DOWN_COOLDOWN_SEC = 60

    def _get_ib(self):
        ib = getattr(self._local, "ib", None)
        if ib is not None and ib.isConnected():
            return ib
        self._local.ib = None  # drop any stale/disconnected handle before retrying

        # Circuit breaker. When IB Gateway is down (it logs out daily), every
        # broker call used to make 3 slow attempts (~10s) while holding the
        # connect lock. Dozens of background callers then queued on that lock,
        # exhausting the worker threads and freezing the whole bot (/health and
        # Telegram stopped responding). A refused connection means Gateway is
        # not there — fail fast for everyone until the cooldown expires.
        if time.monotonic() < self._down_until:
            raise ConnectionError("IBKR unavailable: Gateway down (retrying in a minute)")

        last_err: Exception | None = None
        with self._connect_lock:  # one connect handshake at a time, bot-wide
            if time.monotonic() < self._down_until:  # a queued caller just found it down
                raise ConnectionError("IBKR unavailable: Gateway down (retrying in a minute)")
            for attempt in range(1, 4):  # retries only for transient loop errors
                try:
                    ib = self._connect_once()
                    self._local.ib = ib
                    if self._down_until:
                        logger.info("IBKR Gateway reachable again — connection restored")
                        self._down_until = 0.0
                    return ib
                except (OSError, TimeoutError, asyncio.TimeoutError) as e:
                    # Refused / unreachable / timed out: Gateway itself is down.
                    # Retrying right away cannot help — open the breaker.
                    last_err = e
                    break
                except Exception as e:
                    last_err = e
                    if attempt < 3:
                        logger.warning(
                            f"IBKR connect attempt {attempt}/3 failed "
                            f"({type(e).__name__}: {e}) — retrying"
                        )
                        time.sleep(1.5 * attempt)  # 1.5s, then 3s
            self._down_until = time.monotonic() + self._DOWN_COOLDOWN_SEC
        logger.error(
            f"IBKR connection failed ({self._host}:{self._port}): {last_err} — "
            f"pausing connection attempts for {self._DOWN_COOLDOWN_SEC}s"
        )
        if isinstance(last_err, RuntimeError) and "another loop is running" in str(last_err):
            # DIAGNOSTIC: identify which caller invokes the sync broker API from
            # inside an already-running event loop (retries can never fix that).
            import traceback
            frames = "".join(traceback.format_stack(limit=12)[:-1])
            logger.error(
                f"[IBKR-DIAG] thread={threading.current_thread().name} caller stack:\n{frames}"
            )
        raise ConnectionError(f"IBKR unavailable: {last_err}") from last_err

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def get_account_id(self) -> str | None:
        """The connected account's ID (e.g. 'DUQ938552'). IBKR paper accounts
        always start with 'D' — this is the only reliable live-vs-paper check,
        since ACTIVE_BROKER='ibkr' alone doesn't say which."""
        try:
            ib = self._get_ib()
            accounts = ib.managedAccounts()
            return accounts[0] if accounts else None
        except Exception as e:
            logger.error(f"IBKR get_account_id failed: {e}")
            return None

    def get_account(self) -> dict:
        try:
            ib = self._get_ib()
            raw = ib.accountValues()
            # IBKR מדווח כל ערך בכמה מטבעות + סיכום מאוחד 'BASE' (מטבע הבסיס של
            # החשבון — למשל ILS לחשבון ישראלי). הקוד הישן קרא רק USD, ולכן החזיר 0
            # לחשבון שאינו דולרי (זה מה שהראה $0 למרות שיש כסף). כאן מעדיפים BASE,
            # אחר כך USD, ואחרת את המטבע היחיד שקיים.
            def pick(tag: str) -> float:
                rows = [v for v in raw if v.tag == tag]
                for pref in ("BASE", "USD", ""):
                    for v in rows:
                        if v.currency == pref:
                            try:
                                return float(v.value)
                            except (TypeError, ValueError):
                                return 0.0
                for v in rows:  # fallback: המטבע הראשון הזמין (למשל ILS)
                    try:
                        return float(v.value)
                    except (TypeError, ValueError):
                        continue
                return 0.0

            return {
                "equity": pick("NetLiquidation"),
                "buying_power": pick("BuyingPower"),
                "cash": pick("CashBalance"),
                "portfolio_value": pick("GrossPositionValue"),
                "status": "active",
            }
        except Exception as e:
            logger.error(f"IBKR get_account failed: {e}")
            return {"equity": 0.0, "buying_power": 0.0, "cash": 0.0,
                    "portfolio_value": 0.0, "status": "unavailable"}

    @staticmethod
    def _first_real_price(*candidates) -> float | None:
        """
        First usable price from IBKR's ticker fields.

        `a or b or c` is wrong here: IBKR reports missing prices as float('nan'),
        and NaN is truthy in Python, so the chain returns NaN instead of falling
        through. That NaN then flowed into market_value → account equity →
        position sizing, which blew up with "cannot convert float NaN to integer"
        and stopped the bot from sizing any trade.
        """
        for value in candidates:
            if value is None:
                continue
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if v == v and v > 0:   # v == v filters NaN
                return v
        return None

    def get_positions(self) -> list[dict]:
        try:
            ib = self._get_ib()
            # Paper accounts often lack live-data subscriptions (error 10089).
            # Type 3 = delayed quotes, which they do get — without this the
            # ticker comes back all-NaN and every price below is unusable.
            try:
                ib.reqMarketDataType(3)
            except Exception:
                pass
            result = []
            for pos in ib.positions():
                contract = pos.contract
                qty = pos.position
                avg_cost = pos.avgCost
                ticker_data = ib.reqMktData(contract, "", True, False)
                ib.sleep(1)
                current_price = self._first_real_price(
                    ticker_data.last,
                    ticker_data.close,
                    getattr(ticker_data, "marketPrice", lambda: None)(),
                    avg_cost,
                ) or avg_cost
                market_value = qty * current_price
                unrealized_pl = (current_price - avg_cost) * qty
                unrealized_plpc = ((current_price - avg_cost) / avg_cost) if avg_cost else 0.0
                result.append({
                    "ticker": contract.symbol,
                    "qty": float(qty),
                    "avg_entry_price": float(avg_cost),
                    "current_price": float(current_price),
                    "market_value": float(market_value),
                    "unrealized_pl": float(unrealized_pl),
                    "unrealized_plpc": float(unrealized_plpc),
                })
            return result
        except Exception as e:
            logger.error(f"IBKR get_positions failed: {e}")
            return []

    def get_position(self, ticker: str) -> dict | None:
        try:
            positions = self.get_positions()
            for p in positions:
                if p["ticker"].upper() == ticker.upper():
                    return p
            return None
        except Exception as e:
            logger.error(f"IBKR get_position({ticker}) failed: {e}")
            return None

    # Order statuses that mean IBKR refused the order outright.
    _REJECTED_STATUSES = {"Cancelled", "ApiCancelled", "Inactive"}

    @staticmethod
    def _whole_shares(ticker: str, qty: float) -> int:
        """
        IBKR refuses fractional quantities over the API (error 10243:
        "Fractional-sized order cannot be placed via API"), even though the
        desktop app allows them. Position sizing here produces fractions
        (e.g. 14.005602 shares), so round DOWN to whole shares — rounding up
        could overspend the position budget.
        """
        whole = int(qty)   # truncates toward zero
        if whole < 1:
            raise ValueError(
                f"{ticker}: position size {qty:.4f} rounds to 0 whole shares — "
                f"share price exceeds the per-position budget"
            )
        return whole

    def _place_and_confirm(self, ib, contract, order, side: str, ticker: str) -> dict:
        """
        Place an order and confirm IBKR actually accepted it.

        placeOrder() returns a Trade immediately, before IBKR has validated
        anything — so a rejected order still comes back looking like a normal
        object. Without this check the bot logged "BUY submitted" for orders
        IBKR had already cancelled, and the iceberg slicer kept "filling"
        slices that never existed.
        """
        trade = ib.placeOrder(contract, order)
        # Give IBKR a moment to accept or reject (ib.sleep pumps the event loop)
        for _ in range(10):
            ib.sleep(0.5)
            if trade.orderStatus.status not in ("PendingSubmit", "PreSubmitted"):
                break

        status = str(trade.orderStatus.status)
        if status in self._REJECTED_STATUSES:
            # Surface IBKR's own explanation rather than a bare status word
            detail = ""
            for entry in reversed(getattr(trade, "log", []) or []):
                if getattr(entry, "message", ""):
                    detail = entry.message
                    break
            raise RuntimeError(
                f"IBKR rejected {side} {ticker} x{order.totalQuantity}: "
                f"{status}{' — ' + detail if detail else ''}"
            )

        fill_price = getattr(trade.orderStatus, "avgFillPrice", None)
        logger.info(f"IBKR {side} accepted: {ticker} x{order.totalQuantity} (status={status})")
        return {
            "order_id": str(trade.order.orderId),
            "symbol": ticker.upper(),
            "qty": float(order.totalQuantity),
            "price": float(fill_price) if fill_price else None,
            "status": status,
        }

    def submit_buy(self, ticker: str, qty: float, price: float | None = None) -> dict:
        from ib_insync import Stock, MarketOrder

        # Refuse to buy into an existing unexpected short. The bot is long-only —
        # a short here means a prior bug or manual action left the broker out of
        # sync with the DB, and an ordinary auto-invest buy would partially cover
        # it in a way the DB never records. Covering a short is a deliberate,
        # separate action (manual, or AUTO_FLATTEN_SHORTS), not a side effect
        # of the normal buy path.
        existing = self.get_position(ticker)
        if existing and float(existing["qty"]) < 0:
            raise RuntimeError(
                f"Refusing to BUY {ticker}: broker shows an existing short "
                f"position ({existing['qty']:g} shares) — cover it explicitly "
                f"instead of buying through the normal entry path"
            )

        ib = self._get_ib()
        contract = Stock(ticker.upper(), "SMART", "USD")
        ib.qualifyContracts(contract)
        order = MarketOrder("BUY", self._whole_shares(ticker, qty))
        result = self._place_and_confirm(ib, contract, order, "BUY", ticker)
        result["type"] = "market"
        return result

    def submit_sell(self, ticker: str, qty: float | None = None, price: float | None = None) -> dict:
        from ib_insync import Stock, MarketOrder
        ib = self._get_ib()

        # Never sell more than the broker actually holds. A sell beyond the held
        # quantity does not fail — IBKR happily opens a SHORT. The bot's exit
        # logic re-fires while its DB still shows a position (e.g. the stagnant
        # sweep retried MPC 24 times), so without this the first sell closes the
        # position and every later one shorts: MPC reached -372 shares
        # (-$111,697) on paper. On a live account that is a naked short.
        position = self.get_position(ticker)
        held = float(position["qty"]) if position else 0.0

        # Subtract sells already in flight. IBKR reports positions as SETTLED, so
        # an order sitting at PreSubmitted/Submitted has not reduced `held` yet.
        # Checking only `held` let the caller re-sell the same shares every cycle
        # while the first order was still working — SCHW was sold 1 share at a
        # time roughly every 2 minutes and ran to -280. Netting off open sells
        # makes the check reflect the position the account is actually heading to.
        pending_sell = 0.0
        try:
            for tr in ib.openTrades():
                if (tr.contract.symbol.upper() == ticker.upper()
                        and tr.order.action.upper() == "SELL"
                        and tr.orderStatus.status not in self._REJECTED_STATUSES
                        and tr.orderStatus.status != "Filled"):
                    pending_sell += float(tr.order.totalQuantity) - float(tr.orderStatus.filled or 0)
        except Exception as e:
            logger.warning(f"SELL {ticker}: could not read open orders ({e}) — assuming none")

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

        contract = Stock(ticker.upper(), "SMART", "USD")
        ib.qualifyContracts(contract)
        order = MarketOrder("SELL", self._whole_shares(ticker, requested))
        return self._place_and_confirm(ib, contract, order, "SELL", ticker)

    def is_market_open(self) -> bool:
        try:
            now = datetime.utcnow()
            # NYSE hours: Mon-Fri 13:30-20:00 UTC
            if now.weekday() >= 5:
                return False
            market_open = dtime(13, 30)
            market_close = dtime(20, 0)
            current = now.time()
            return market_open <= current < market_close
        except Exception as e:
            logger.error(f"IBKR is_market_open failed: {e}")
            return False

    def get_asset(self, ticker: str) -> dict | None:
        try:
            from ib_insync import Stock
            ib = self._get_ib()
            contract = Stock(ticker.upper(), "SMART", "USD")
            details = ib.reqContractDetails(contract)
            if not details:
                return None
            d = details[0]
            return {
                "symbol": ticker.upper(),
                "name": d.longName,
                "tradable": True,
                "fractionable": False,
            }
        except Exception as e:
            logger.error(f"IBKR get_asset({ticker}) failed: {e}")
            return None
