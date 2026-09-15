"""
db_backup.py — consistent online backup of a live SQLite database.

Usage:  python db_backup.py <source.db> <dest.db>

Uses sqlite3.Connection.backup() (the official online-backup API), which
produces a consistent snapshot even while another process (the bot) is
actively writing — unlike a raw file copy, which fails on SQLite's
byte-range write locks.
"""
import sqlite3
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python db_backup.py <source.db> <dest.db>", file=sys.stderr)
        return 2
    src_path, dst_path = sys.argv[1], sys.argv[2]
    src = sqlite3.connect(src_path)
    try:
        dst = sqlite3.connect(dst_path)
        try:
            with dst:
                src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
