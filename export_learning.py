"""
export_learning.py — מייצא את כל מה שהבוט למד.

מוציא כל טבלה מ-data/trading.db לקובץ CSV נפרד (אחד לכל טבלה), לתוך
data/learning_export_<תאריך>/. קריאה-בלבד (mode=ro, רק SELECT), אז בטוח
להריץ גם כשהבוט פעיל.

הרצה:  python export_learning.py
"""
import csv
import os
import sqlite3
from datetime import date

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "data", "trading.db")
OUT = os.path.join(BASE, "data", f"learning_export_{date.today():%Y-%m-%d}")


def main():
    if not os.path.exists(DB):
        print(f"DB not found: {DB}")
        return
    os.makedirs(OUT, exist_ok=True)
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=10)
    cur = con.cursor()
    tables = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]

    manifest, grand_total = [], 0
    for t in tables:
        cols = [c[1] for c in cur.execute(f'PRAGMA table_info("{t}")')]
        rows = cur.execute(f'SELECT * FROM "{t}"').fetchall()
        # utf-8-sig so Hebrew opens correctly in Excel
        with open(os.path.join(OUT, f"{t}.csv"), "w", newline="",
                  encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(rows)
        manifest.append((t, len(rows)))
        grand_total += len(rows)
        print(f"  {t}: {len(rows)} rows -> {t}.csv")

    with open(os.path.join(OUT, "MANIFEST.txt"), "w", encoding="utf-8") as f:
        f.write(f"Bot learning export — {date.today():%Y-%m-%d}\n")
        f.write(f"Source: {DB}\n")
        f.write(f"{len(tables)} tables, {grand_total} total rows\n\n")
        for t, n in manifest:
            f.write(f"{t}: {n} rows\n")

    con.close()
    print(f"\nExported {len(tables)} tables ({grand_total} rows) to:\n  {OUT}")


if __name__ == "__main__":
    main()
