import sqlite3
import json
import threading
import logging
from datetime import datetime, timedelta
from config import settings

logger = logging.getLogger(__name__)

_local = threading.local()


def get_connection() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        from pathlib import Path
        Path(settings.DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(settings.DATABASE_PATH)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")

        # Apply durability setting based on configuration
        if settings.HARDENED_DURABILITY:
            conn.execute("PRAGMA synchronous=FULL")  # Hardened durability for critical data
        else:
            conn.execute("PRAGMA synchronous=NORMAL")  # Balanced performance/safety

        conn.row_factory = sqlite3.Row
        _local.conn = conn
    return _local.conn


def flush_database():
    """Flush WAL checkpoint to consolidate all pending writes before shutdown."""
    if hasattr(_local, "conn") and _local.conn is not None:
        try:
            _local.conn.execute("PRAGMA wal_checkpoint(RESTART)")
            logger.info("Database WAL checkpoint completed")
        except Exception as e:
            logger.warning(f"WAL checkpoint failed (non-critical): {e}")


def check_database_integrity():
    """Verify database integrity. If corrupt, attempts auto-recovery (backup + recreate)."""
    conn = get_connection()
    try:
        # Check for database corruption
        result = conn.execute("PRAGMA integrity_check").fetchone()
        if result[0] != "ok":
            logger.warning(f"Database integrity check failed: {result[0]}")
            # AUTO-RECOVERY: backup corrupt DB and create fresh one
            _attempt_recovery(reason=str(result[0])[:100])
            return False

        # Verify critical tables exist
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('trade_log','heartbeat_log','tax_events')"
        ).fetchall()
        if len(tables) < 3:
            logger.warning("Missing critical database tables")
            return False

        # Check for recent data
        last_hb = conn.execute("SELECT MAX(timestamp) FROM heartbeat_log").fetchone()[0]
        if last_hb:
            logger.info(f"✓ Database OK | Last heartbeat: {last_hb}")
        else:
            logger.info("✓ Database OK | No heartbeat history yet")

        return True
    except sqlite3.DatabaseError as e:
        # File-level corruption (e.g., "database disk image is malformed")
        logger.error(f"Database corruption detected: {e}")
        _attempt_recovery(reason=str(e)[:100])
        return False
    except Exception as e:
        logger.error(f"Database integrity check failed: {e}")
        return False


def _attempt_recovery(reason: str) -> bool:
    """Backup corrupt DB and create fresh one. Returns True on success."""
    try:
        from pathlib import Path
        import shutil, datetime
        db_path = Path(settings.DATABASE_PATH)
        if not db_path.exists():
            return False

        # Backup with timestamp
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = db_path.parent / f"trading_CORRUPT_{ts}.db"
        try:
            close_connections()  # release file lock
        except Exception:
            pass
        shutil.copy(str(db_path), str(backup_path))
        logger.warning(f"[RECOVERY] Backed up corrupt DB → {backup_path.name}")

        # Delete corrupt file (next get_connection will recreate it)
        try:
            db_path.unlink()
            # Also remove WAL/SHM
            (db_path.parent / (db_path.name + "-wal")).unlink(missing_ok=True)
            (db_path.parent / (db_path.name + "-shm")).unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"[RECOVERY] Failed to remove corrupt files: {e}")
            return False

        logger.warning(f"[RECOVERY] Fresh database will be created on next access (reason: {reason})")

        # Send Telegram alert
        try:
            import os, requests
            token = os.getenv("TELEGRAM_BOT_TOKEN", "")
            chat = os.getenv("TELEGRAM_CHAT_ID", "")
            if token and chat:
                requests.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    json={
                        "chat_id": chat,
                        "text": (
                            f"🔴 <b>DATABASE RECOVERY</b>\n"
                            f"מסד הנתונים היה פגום ושוחזר.\n"
                            f"גיבוי: <code>{backup_path.name}</code>\n"
                            f"סיבה: <code>{reason[:80]}</code>"
                        ),
                        "parse_mode": "HTML",
                    },
                    timeout=5,
                )
        except Exception:
            pass

        return True
    except Exception as e:
        logger.critical(f"[RECOVERY] Failed: {e}")
        return False


def close_connections():
    if hasattr(_local, "conn") and _local.conn is not None:
        try:
            flush_database()  # Ensure all writes are flushed before closing
            _local.conn.close()
        except Exception:
            pass
        _local.conn = None


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trade_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            action TEXT NOT NULL,
            qty REAL NOT NULL,
            entry_price REAL NOT NULL,
            entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            exit_price REAL,
            exit_time DATETIME,
            trailing_stop_pct REAL,
            rsi REAL,
            macd REAL,
            macd_signal REAL,
            bb_position REAL,
            volume_ratio REAL,
            sentiment_score INTEGER,
            sentiment_reasoning TEXT,
            pnl_gross REAL,
            pnl_net REAL,
            tax_reserved REAL,
            fees REAL DEFAULT 0,
            status TEXT DEFAULT 'open'
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_ticker ON trade_log(ticker)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_status ON trade_log(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_time ON trade_log(entry_time DESC)")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS learning_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id INTEGER NOT NULL,
            pattern_type TEXT,
            description TEXT NOT NULL,
            indicators_snapshot TEXT,
            outcome TEXT,
            pnl REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (trade_id) REFERENCES trade_log(id)
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_learning_pattern ON learning_log(pattern_type)")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tax_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            amount REAL NOT NULL,
            balance_after REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (trade_id) REFERENCES trade_log(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS heartbeat_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            open_positions INTEGER,
            budget_used_pct REAL,
            total_equity REAL,
            notes TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS shadow_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            signal_source TEXT,
            entry_price REAL NOT NULL,
            qty REAL NOT NULL,
            entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            exit_price REAL,
            exit_time DATETIME,
            composite_score REAL,
            sentiment_score INTEGER,
            volume_ratio REAL,
            atr_stop_price REAL,
            high_watermark REAL,
            live_blocked_by TEXT,
            live_block_reason TEXT,
            pnl_gross REAL,
            pnl_pct REAL,
            status TEXT DEFAULT 'open',
            close_reason TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_shadow_ticker ON shadow_trades(ticker)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_shadow_status ON shadow_trades(status)")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS slippage_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            side TEXT NOT NULL,
            qty REAL NOT NULL,
            signal_price REAL NOT NULL,
            fill_price REAL NOT NULL,
            slip_pct REAL NOT NULL,
            abs_slip_pct REAL NOT NULL,
            slip_bps REAL NOT NULL,
            slip_per_share REAL NOT NULL,
            total_slip_usd REAL NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_slip_ticker ON slippage_log(ticker)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_slip_time   ON slippage_log(created_at DESC)")

    # ── Schema migrations (safe to run repeatedly) ────────────────────────────
    # Add ATR trailing stop columns introduced in v2
    # Add created_at, exit_reason columns introduced in v3
    _migrations = [
        "ALTER TABLE trade_log ADD COLUMN atr_stop_price REAL",
        "ALTER TABLE trade_log ADD COLUMN high_watermark  REAL",
        "ALTER TABLE trade_log ADD COLUMN created_at     DATETIME",
        "ALTER TABLE trade_log ADD COLUMN exit_reason    TEXT",
    ]
    for ddl in _migrations:
        try:
            conn.execute(ddl)
        except Exception:
            pass  # column already exists — sqlite3.OperationalError is expected

    # Backfill created_at from entry_time for existing rows
    try:
        conn.execute("UPDATE trade_log SET created_at = entry_time WHERE created_at IS NULL AND entry_time IS NOT NULL")
    except Exception:
        pass

    # Fix bb_position: SQLite stored it as TEXT but it should be REAL.
    # SQLite cannot ALTER COLUMN type, so we CAST all existing values in-place.
    # This is a no-op if the column is already numeric.
    try:
        conn.execute("UPDATE trade_log SET bb_position = CAST(bb_position AS REAL) WHERE bb_position IS NOT NULL")
    except Exception:
        pass

    # Trade journal — user notes on positions and decisions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trade_journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            note TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_journal_ticker ON trade_journal(ticker)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_journal_time ON trade_journal(created_at DESC)")

    # Migrate qty columns to REAL (fractional share support)
    # Whitelist table/column names to prevent SQL injection
    _ALLOWED_MIGRATIONS = {
        ("trade_log",    "qty"),
        ("shadow_trades","qty"),
        ("slippage_log", "qty"),
    }
    for tbl, col in _ALLOWED_MIGRATIONS:
        try:
            # Safe: tbl and col are whitelisted constants, not user input
            conn.execute(f"UPDATE {tbl} SET {col} = CAST({col} AS REAL) WHERE {col} IS NOT NULL")
        except Exception:
            pass

    conn.commit()
    logger.info("Database initialized")


# ===== Trade Log =====

def save_trade(trade: dict) -> int:
    conn = get_connection()
    cursor = conn.execute(
        """INSERT INTO trade_log
        (ticker, action, qty, entry_price, trailing_stop_pct,
         rsi, macd, macd_signal, bb_position, volume_ratio,
         sentiment_score, sentiment_reasoning)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (trade["ticker"], trade["action"], trade["qty"], trade["entry_price"],
         trade.get("trailing_stop_pct"), trade.get("rsi"), trade.get("macd"),
         trade.get("macd_signal"), trade.get("bb_position"), trade.get("volume_ratio"),
         trade.get("sentiment_score"), trade.get("sentiment_reasoning")),
    )
    conn.commit()
    return cursor.lastrowid


def close_trade(trade_id: int, exit_price: float, pnl_gross: float,
                pnl_net: float, tax_reserved: float, fees: float = 0.0,
                status: str = "closed", exit_reason: str = ""):
    conn = get_connection()
    conn.execute(
        """UPDATE trade_log SET
        exit_price=?, exit_time=CURRENT_TIMESTAMP,
        pnl_gross=?, pnl_net=?, tax_reserved=?, fees=?, status=?,
        exit_reason=COALESCE(NULLIF(?, ''), status)
        WHERE id=?""",
        (exit_price, pnl_gross, pnl_net, tax_reserved, fees, status,
         exit_reason, trade_id),
    )
    conn.commit()


def update_trade_stop(trade_id: int, atr_stop_price: float, high_watermark: float) -> None:
    """Persist the current trailing stop price and high-watermark for an open trade."""
    conn = get_connection()
    conn.execute(
        "UPDATE trade_log SET atr_stop_price=?, high_watermark=? WHERE id=?",
        (atr_stop_price, high_watermark, trade_id),
    )
    conn.commit()


def get_open_trades() -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM trade_log WHERE status='open' ORDER BY entry_time DESC"
    ).fetchall()
    return [dict(row) for row in rows]


def get_open_trade_by_ticker(ticker: str) -> dict | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM trade_log WHERE ticker=? AND status='open' ORDER BY entry_time DESC LIMIT 1",
        (ticker,),
    ).fetchone()
    return dict(row) if row else None


def get_trade_history(ticker: str | None = None, limit: int = 50) -> list[dict]:
    conn = get_connection()
    if ticker:
        rows = conn.execute(
            "SELECT * FROM trade_log WHERE ticker=? ORDER BY entry_time DESC LIMIT ?",
            (ticker, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM trade_log ORDER BY entry_time DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


# ===== Learning Log =====

def save_learning_entry(entry: dict):
    conn = get_connection()
    conn.execute(
        """INSERT INTO learning_log (trade_id, pattern_type, description, indicators_snapshot, outcome, pnl)
        VALUES (?, ?, ?, ?, ?, ?)""",
        (entry["trade_id"], entry.get("pattern_type"), entry["description"],
         json.dumps(entry.get("indicators_snapshot", {})),
         entry.get("outcome"), entry.get("pnl")),
    )
    conn.commit()


def get_learning_entries(pattern_type: str | None = None, limit: int = 50) -> list[dict]:
    conn = get_connection()
    if pattern_type:
        rows = conn.execute(
            "SELECT * FROM learning_log WHERE pattern_type=? ORDER BY created_at DESC LIMIT ?",
            (pattern_type, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM learning_log ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


_CLOSED_STATUSES = "('closed','stop_loss','take_profit','smart_sell','emergency_exit','time_exit','stale_restart','momentum_exit','partial_tp','news_exit','earnings_miss')"


def get_loss_trades(limit: int = 20) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        f"SELECT * FROM trade_log WHERE status IN {_CLOSED_STATUSES} AND pnl_gross < 0 ORDER BY entry_time DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_win_trades(limit: int = 20) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        f"SELECT * FROM trade_log WHERE status IN {_CLOSED_STATUSES} AND pnl_gross > 0 ORDER BY entry_time DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_total_trades_count() -> dict:
    """
    סופר את כל העסקאות שהבוט עשה אי פעם.
    מחזיר:
      {
        "total":   סה"כ עסקאות (פתוחות + סגורות)
        "open":    עסקאות פתוחות
        "closed":  עסקאות סגורות
        "wins":    מהסגורות — כמה ברווח
        "losses":  מהסגורות — כמה בהפסד
        "today":   נפתחו היום
      }
    """
    conn = get_connection()
    total  = conn.execute("SELECT COUNT(*) FROM trade_log").fetchone()[0] or 0
    open_  = conn.execute("SELECT COUNT(*) FROM trade_log WHERE status = 'open'").fetchone()[0] or 0
    closed = conn.execute(
        f"SELECT COUNT(*) FROM trade_log WHERE status IN {_CLOSED_STATUSES}"
    ).fetchone()[0] or 0
    wins   = conn.execute(
        f"SELECT COUNT(*) FROM trade_log WHERE status IN {_CLOSED_STATUSES} AND pnl_gross > 0"
    ).fetchone()[0] or 0
    losses = conn.execute(
        f"SELECT COUNT(*) FROM trade_log WHERE status IN {_CLOSED_STATUSES} AND pnl_gross < 0"
    ).fetchone()[0] or 0
    today  = conn.execute(
        "SELECT COUNT(*) FROM trade_log WHERE DATE(entry_time) = DATE('now')"
    ).fetchone()[0] or 0
    return {
        "total": total, "open": open_, "closed": closed,
        "wins": wins, "losses": losses, "today": today,
    }


# ===== Tax Events =====

def save_tax_event(trade_id: int, event_type: str, amount: float):
    conn = get_connection()
    current = get_tax_balance()
    if event_type == "tax_reserved":
        balance_after = current["tax_reserved"] + amount
    else:
        balance_after = current["tax_credit"] + amount
    conn.execute(
        "INSERT INTO tax_events (trade_id, event_type, amount, balance_after) VALUES (?, ?, ?, ?)",
        (trade_id, event_type, amount, balance_after),
    )
    conn.commit()


def get_tax_balance() -> dict:
    conn = get_connection()
    reserved = conn.execute(
        "SELECT COALESCE(SUM(amount), 0) FROM tax_events WHERE event_type='tax_reserved'"
    ).fetchone()[0]
    credit = conn.execute(
        "SELECT COALESCE(SUM(amount), 0) FROM tax_events WHERE event_type='tax_credit'"
    ).fetchone()[0]
    return {"tax_reserved": reserved, "tax_credit": credit}


def get_tax_summary() -> dict:
    conn = get_connection()
    row = conn.execute(
        f"""SELECT
            COALESCE(SUM(pnl_gross), 0) as realized_pnl_gross,
            COALESCE(SUM(CASE WHEN pnl_gross > 0 THEN tax_reserved ELSE 0 END), 0) as tax_reserved,
            COALESCE(SUM(pnl_net), 0) as realized_pnl_net
        FROM trade_log WHERE status IN {_CLOSED_STATUSES}"""
    ).fetchone()
    tax_bal = get_tax_balance()
    return {
        "realized_pnl_gross": row[0],
        "tax_reserved": tax_bal["tax_reserved"],
        "tax_credit": tax_bal["tax_credit"],
        "realized_pnl_net": row[2],
    }


# ===== Trade Journal =====

def update_trade_qty(trade_id: int, new_qty: float) -> None:
    """Update qty on an open trade after a partial sell."""
    conn = get_connection()
    conn.execute(
        "UPDATE trade_log SET qty = ? WHERE id = ?",
        (new_qty, trade_id),
    )
    conn.commit()


def add_journal_entry(note: str, ticker: str | None = None) -> int:
    """Add a note to the trade journal. Returns the new entry id."""
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO trade_journal (ticker, note) VALUES (?, ?)",
        (ticker.upper() if ticker else None, note.strip()),
    )
    conn.commit()
    return cur.lastrowid


def get_journal_entries(ticker: str | None = None, limit: int = 10) -> list[dict]:
    """Return recent journal entries, optionally filtered by ticker."""
    conn = get_connection()
    if ticker:
        rows = conn.execute(
            "SELECT * FROM trade_journal WHERE ticker=? ORDER BY created_at DESC LIMIT ?",
            (ticker.upper(), limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM trade_journal ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


# ===== Heartbeat =====

def save_heartbeat(open_positions: int, budget_used_pct: float,
                   total_equity: float, notes: str = ""):
    conn = get_connection()
    conn.execute(
        "INSERT INTO heartbeat_log (open_positions, budget_used_pct, total_equity, notes) VALUES (?, ?, ?, ?)",
        (open_positions, budget_used_pct, total_equity, notes),
    )
    conn.commit()


def get_last_heartbeat() -> dict | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM heartbeat_log ORDER BY timestamp DESC LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


# ===== Cleanup =====

def cleanup_old_heartbeats(days: int = 7):
    conn = get_connection()
    # Use UTC to match SQLite CURRENT_TIMESTAMP which is always UTC
    from datetime import timezone as _tz
    cutoff = (datetime.now(_tz.utc) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    result = conn.execute("DELETE FROM heartbeat_log WHERE timestamp < ?", (cutoff,))
    conn.commit()
    if result.rowcount > 0:
        logger.info(f"Cleaned up {result.rowcount} heartbeat entries older than {days} days")


def cleanup_old_data(days: int = 30):
    """
    Clean up old records from shadow_trades, slippage_log, and learning_log.
    Keeps the last `days` days of data.  Prevents unbounded table growth.
    """
    conn = get_connection()
    from datetime import timezone as _tz
    cutoff = (datetime.now(_tz.utc) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    total = 0

    for table, col in [
        ("shadow_trades", "created_at"),
        ("slippage_log",  "created_at"),
        ("learning_log",  "created_at"),
    ]:
        try:
            r = conn.execute(f"DELETE FROM {table} WHERE {col} < ?", (cutoff,))
            if r.rowcount > 0:
                logger.info(f"Cleanup: removed {r.rowcount} rows from {table}")
                total += r.rowcount
        except Exception as e:
            logger.debug(f"Cleanup {table} skipped: {e}")

    conn.commit()
    if total > 0:
        logger.info(f"Total cleanup: {total} old rows removed (>{days} days)")


# ===== Shadow Trades =====

def save_shadow_trade(row: dict) -> int:
    """Insert a new shadow trade and return its row id."""
    conn = get_connection()
    cursor = conn.execute(
        """INSERT INTO shadow_trades
        (ticker, signal_source, entry_price, qty, composite_score, sentiment_score,
         volume_ratio, atr_stop_price, high_watermark, live_blocked_by, live_block_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            row["ticker"], row.get("signal_source"), row["entry_price"], row["qty"],
            row.get("composite_score"), row.get("sentiment_score"), row.get("volume_ratio"),
            row.get("atr_stop_price"), row.get("high_watermark"),
            row.get("live_blocked_by"), row.get("live_block_reason"),
        ),
    )
    conn.commit()
    return cursor.lastrowid


def get_shadow_trade(shadow_id: int) -> dict | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM shadow_trades WHERE id=?", (shadow_id,)
    ).fetchone()
    return dict(row) if row else None


def close_shadow_trade(
    shadow_id: int,
    exit_price: float,
    pnl_gross: float,
    pnl_pct: float,
    status: str,
    reason: str,
) -> None:
    conn = get_connection()
    conn.execute(
        """UPDATE shadow_trades SET
        exit_price=?, exit_time=CURRENT_TIMESTAMP,
        pnl_gross=?, pnl_pct=?, status=?, close_reason=?
        WHERE id=?""",
        (exit_price, pnl_gross, pnl_pct, status, reason, shadow_id),
    )
    conn.commit()


def get_open_shadow_trades() -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM shadow_trades WHERE status='open' ORDER BY entry_time DESC"
    ).fetchall()
    return [dict(row) for row in rows]


def get_open_shadow_trade_by_ticker(ticker: str) -> dict | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM shadow_trades WHERE ticker=? AND status='open' ORDER BY entry_time DESC LIMIT 1",
        (ticker,),
    ).fetchone()
    return dict(row) if row else None


def get_shadow_trade_history(limit: int = 100) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM shadow_trades ORDER BY entry_time DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


def update_shadow_stop(shadow_id: int, new_stop: float, new_wm: float) -> None:
    """Update trailing stop price and high watermark for an open shadow trade."""
    conn = get_connection()
    conn.execute(
        "UPDATE shadow_trades SET atr_stop_price=?, high_watermark=? WHERE id=?",
        (new_stop, new_wm, shadow_id),
    )
    conn.commit()


# ===== Slippage Log =====

def save_slippage(row: dict) -> int:
    """Persist one slippage observation. Returns the new row id."""
    conn = get_connection()
    cur = conn.execute(
        """INSERT INTO slippage_log
           (ticker, side, qty, signal_price, fill_price,
            slip_pct, abs_slip_pct, slip_bps, slip_per_share, total_slip_usd)
           VALUES (:ticker, :side, :qty, :signal_price, :fill_price,
                   :slip_pct, :abs_slip_pct, :slip_bps, :slip_per_share, :total_slip_usd)""",
        row,
    )
    conn.commit()
    return cur.lastrowid


def get_rolling_slippage(n: int = 20) -> float:
    """Return the mean abs_slip_pct (%) of the most recent n slippage records."""
    conn = get_connection()
    row = conn.execute(
        """SELECT AVG(abs_slip_pct) FROM (
               SELECT abs_slip_pct FROM slippage_log
               ORDER BY created_at DESC LIMIT ?
           )""",
        (n,),
    ).fetchone()
    return float(row[0] or 0.0)


def get_slippage_history(limit: int = 100) -> list[dict]:
    """Return the most recent slippage records, newest first."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM slippage_log ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_slippage_summary() -> dict:
    """Return aggregate slippage statistics across all recorded trades."""
    conn = get_connection()
    row = conn.execute(
        """SELECT
               COUNT(*)                                      AS total_records,
               ROUND(AVG(abs_slip_pct), 4)                  AS avg_slip_pct,
               ROUND(MAX(abs_slip_pct), 4)                  AS max_slip_pct,
               ROUND(MIN(abs_slip_pct), 4)                  AS min_slip_pct,
               ROUND(AVG(slip_bps), 2)                      AS avg_bps,
               ROUND(SUM(total_slip_usd), 4)                AS total_cost_usd,
               ROUND(AVG(CASE WHEN side='buy'  THEN abs_slip_pct END), 4) AS avg_buy_slip_pct,
               ROUND(AVG(CASE WHEN side='sell' THEN abs_slip_pct END), 4) AS avg_sell_slip_pct
           FROM slippage_log"""
    ).fetchone()
    return dict(row) if row else {}
