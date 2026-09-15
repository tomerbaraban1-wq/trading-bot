# -*- coding: utf-8 -*-
"""
test_ibkr_connection.py — בדיקת חיבור בטוחה ל-Interactive Brokers.

קורא בלבד: יתרה, כוח קנייה ופוזיציות. לא שולח שום פקודת מסחר.
מזהה אם החשבון המחובר הוא דמו (Paper) או אמיתי (Live) ומתריע בהתאם.

הרצה (כש-TWS / IB Gateway פתוח ומחובר):
    python test_ibkr_connection.py
"""

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config import settings

# clientId נפרד מזה של הבוט (1) כדי לא להתנגש בחיבור פעיל
TEST_CLIENT_ID = 99


def main() -> int:
    host = settings.IBKR_HOST
    port = settings.IBKR_PORT
    mode = "Paper (דמו)" if port == 7497 else "Live (כסף אמיתי!)" if port == 7496 else f"לא מוכר (פורט {port})"

    print("=" * 60)
    print("בדיקת חיבור ל-Interactive Brokers")
    print(f"יעד: {host}:{port} — מצב לפי פורט: {mode}")
    print("=" * 60)

    try:
        from ib_insync import IB
    except ImportError:
        print("[X] הספרייה ib_insync לא מותקנת. הרץ: pip install ib_insync")
        return 1

    ib = IB()
    try:
        ib.connect(host, port, clientId=TEST_CLIENT_ID, timeout=15)
    except Exception as e:
        print(f"[X] החיבור נכשל: {e}")
        print()
        print("בדוק את הדברים הבאים:")
        print("  1. TWS או IB Gateway פתוח ומחובר לחשבון?")
        print("  2. ב-TWS: File > Global Configuration > API > Settings")
        print("     - Enable ActiveX and Socket Clients מסומן?")
        print("     - Socket port תואם? (7497=Paper, 7496=Live)")
        print("     - Read-Only API כבוי?")
        print("  3. אחרי שינוי הגדרות — צריך Restart ל-TWS.")
        return 1

    try:
        accounts = ib.managedAccounts()
        account_id = accounts[0] if accounts else "?"
        # חשבונות דמו של IBKR מתחילים ב-D (למשל DU1234567)
        is_paper_account = account_id.upper().startswith("D")

        print(f"[V] מחובר! מספר חשבון: {account_id}")
        if is_paper_account:
            print("[V] זהו חשבון דמו (Paper) — אין כסף אמיתי בסיכון.")
        else:
            print("[!] אזהרה: זה נראה כמו חשבון LIVE — כסף אמיתי!")
            print("[!] אם התכוונת לדמו: התחבר ב-TWS עם משתמש ה-Paper (Simulated Trading).")

        raw = ib.accountValues()
        # מעדיפים את מטבע הבסיס של החשבון (BASE), אחרת USD, אחרת המטבע היחיד
        # שקיים (למשל ILS) — כדי לא להציג 0 לחשבון שאינו דולרי.
        def _pick(tag):
            rows = [v for v in raw if v.tag == tag]
            for pref in ("BASE", "USD", ""):
                for v in rows:
                    if v.currency == pref:
                        try:
                            return float(v.value)
                        except (TypeError, ValueError):
                            return 0.0
            for v in rows:
                try:
                    return float(v.value)
                except (TypeError, ValueError):
                    continue
            return 0.0
        equity = _pick("NetLiquidation")
        buying_power = _pick("BuyingPower")
        cash = _pick("CashBalance")

        print()
        print(f"שווי חשבון (Net Liquidation): ${equity:,.2f}")
        print(f"כוח קנייה:                    ${buying_power:,.2f}")
        print(f"מזומן:                        ${cash:,.2f}")

        positions = ib.positions()
        print(f"פוזיציות פתוחות: {len(positions)}")
        for pos in positions:
            print(f"  - {pos.contract.symbol}: {pos.position} יח' במחיר ממוצע ${pos.avgCost:,.2f}")

        print()
        print("=" * 60)
        print("[V] הבדיקה עברה — הבוט יכול לדבר עם IBKR.")
        if not is_paper_account:
            print("[!] אבל שים לב: החשבון המחובר הוא LIVE. במסלול המדורג")
            print("    מתחילים עם חשבון דמו (פורט 7497 + משתמש Paper).")
        print("=" * 60)
        return 0
    finally:
        ib.disconnect()


if __name__ == "__main__":
    sys.exit(main())
