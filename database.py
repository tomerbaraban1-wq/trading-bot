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
        conn.row_factory = sqlite3.Row
        _local.conn = conn
    return _local.conn


def close_connections():
    if hasattr(_local, "conn") and _local.conn is not None:
        try:
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
            qty INTEGER NOT NULL,
            entry_price REAL NOT NULL,
            entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            exit_price REAL,
            exit_time DATETIME,
            trailing_stop_pct REAL,
            rsi REAL,
            macd REAL,
            macd_signal REAL,
            bb_position TEXT,
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
                status: str = "closed"):
    conn = get_connection()
    conn.execute(
        """UPDATE trade_log SET
        exit_price=?, exit_time=CURRENT_TIMESTAMP,
        pnl_gross=?, pnl_net=?, tax_reserved=?, fees=?, status=?
        WHERE id=?""",
        (exit_price, pnl_gross, pnl_net, tax_reserved, fees, status, trade_id),
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


def get_loss_trades(limit: int = 20) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM trade_log WHERE status IN ('closed','emergency_exit') AND pnl_gross < 0 ORDER BY entry_time DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_win_trades(limit: int = 20) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM trade_log WHERE status IN ('closed','emergency_exit') AND pnl_gross > 0 ORDER BY entry_time DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


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
        """SELECT
            COALESCE(SUM(pnl_gross), 0) as realized_pnl_gross,
            COALESCE(SUM(CASE WHEN pnl_gross > 0 THEN tax_reserved ELSE 0 END), 0) as tax_reserved,
            COALESCE(SUM(pnl_net), 0) as realized_pnl_net
        FROM trade_log WHERE status IN ('closed', 'emergency_exit')"""
    ).fetchone()
    tax_bal = get_tax_balance()
    return {
        "realized_pnl_gross": row[0],
        "tax_reserved": tax_bal["tax_reserved"],
        "tax_credit": tax_bal["tax_credit"],
        "realized_pnl_net": row[2],
    }


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
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    result = conn.execute("DELETE FROM heartbeat_log WHERE timestamp < ?", (cutoff,))
    conn.commit()
    if result.rowcount > 0:
        logger.info(f"Cleaned up {result.rowcount} heartbeat entries older than {days} days")
