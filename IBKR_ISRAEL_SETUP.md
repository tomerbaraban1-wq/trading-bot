# 🇮🇱 חיבור הבוט ל-Interactive Brokers Israel

## ✅ מה כבר יש בבוט

הבוט תומך מלא ב-IBKR (`broker_ibkr.py`):
- ✅ קנייה/מכירה דרך IBKR
- ✅ בדיקת פוזיציות
- ✅ בדיקת חשבון ומזומן
- ✅ Stop Loss / Take Profit
- ✅ עובד עם מניות + ETFs

---

## 🔧 איך מחברים — 3 שלבים

### **שלב 1: התקן TWS (Trader Workstation)**

1. הורד מ-Interactive Brokers:
   - https://www.interactivebrokers.com/en/trading/tws.php
   - או IB Gateway (קל יותר): https://www.interactivebrokers.com/en/trading/ibgateway-stable.php

2. התקן ופתח את TWS/IB Gateway

3. התחבר עם המשתמש של Interactive Israel שלך

---

### **שלב 2: הפעל את ה-API ב-TWS**

ב-TWS:
1. **File → Global Configuration**
2. **API → Settings**
3. הפעל:
   - ✅ `Enable ActiveX and Socket Clients`
   - ✅ `Allow connections from localhost only` (לבטיחות)
4. **Socket port:**
   - `7497` = Paper Trading
   - `7496` = Live Trading
5. **Read-Only API: OFF** (כדי שהבוט יוכל לקנות)
6. לחץ **OK** ו-restart TWS

---

### **שלב 3: הגדר את הבוט**

ערוך את `.env`:

```bash
# החלף את ACTIVE_BROKER:
ACTIVE_BROKER=ibkr

# הגדרות IBKR (כבר קיימות):
IBKR_HOST=127.0.0.1
IBKR_PORT=7497          # 7497=Paper, 7496=Live
```

**זהו! הבוט עכשיו מסחר דרך IBKR שלך.**

---

## 🌐 בעיה: הבוט רץ ב-Render, TWS במחשב שלי

זו בעיה אמיתית. TWS חייב להיות פתוח על אותו רשת כמו הבוט.

### **פתרון 1: הפעל את הבוט מקומית** (הכי פשוט)
```bash
# במחשב שלך (עם TWS פתוח):
cd "C:\Users\תומר\Pictures\קלוד קוד\trading-bot"
python main.py
```

**יתרון:** עובד מיד עם TWS  
**חסרון:** המחשב חייב להיות דלוק 24/7

### **פתרון 2: VPS עם TWS** (מומלץ לlive)
1. שכור VPS זול ($5/חודש - DigitalOcean/Hetzner)
2. התקן TWS על ה-VPS
3. הרץ את הבוט על אותו VPS
4. סגור את הלוקאל

### **פתרון 3: IB Gateway Cloud-Friendly**
IB Gateway קל יותר מ-TWS — צורך פחות זיכרון, אפשר להריץ headless.

---

## 📊 הבדלים: Alpaca vs IBKR Israel

| תכונה | Alpaca | IBKR Israel |
|-------|--------|-------------|
| **מחיר** | חינם | $0-1/חודש |
| **עמלות** | $0 | $0.005/share min $1 |
| **מס בארץ** | מורכב | פשוט (חברה ישראלית) |
| **דיווח מס** | אתה לבד | IBKR ישראל מנפיק טופס |
| **שעות מסחר** | רגיל | רגיל + Extended |
| **מטבעות** | USD | USD/EUR/ILS |
| **מינימום** | $0 | $0 |

---

## 🚀 התוכנית המעודכנת — IBKR Israel

### **שלב 1 — Paper דרך IBKR (3 ימים)**
```
1. פתח TWS Paper
2. הגדר ACTIVE_BROKER=ibkr  
3. IBKR_PORT=7497 (paper)
4. הרץ את הבוט
5. /validate בטלגרם
```

### **שלב 2 — Live דרך IBKR (יום 4+)**
```
1. סגור Paper TWS
2. פתח Live TWS עם Interactive Israel
3. שנה IBKR_PORT=7496 (live!)
4. שנה MAX_BUDGET=500 (להתחיל קטן)
5. הבוט יסחר עם הכסף האמיתי שלך!
```

---

## ⚠️ הערות חשובות

1. **TWS חייב להיות פתוח** כל הזמן שהבוט רץ
2. **API צריך להיות מאופשר** ב-TWS settings
3. **לא לסגור את ה-TWS** באמצע יום מסחר!
4. **2FA ב-TWS**: אם יש 2FA, הוא יתבקש פעם ביום (TWS dialog)
5. **Render לא יכול להתחבר ל-TWS המקומי** — חייב VPS או מחשב מקומי

---

## 💡 המלצה שלי

עבור שלב 1 (Paper trading + validation):
**הרץ את הבוט מקומית עם TWS Paper שלך.**

```bash
# במחשב שלך:
cd "C:\Users\תומר\Pictures\קלוד קוד\trading-bot"
# פתח TWS paper
# הגדר ACTIVE_BROKER=ibkr ב-.env
python main.py
```

זה הכי פשוט והכי מהיר.

עבור Live (שלב 2):
שקול VPS — זול ($5/חודש) ובטוח יותר.
