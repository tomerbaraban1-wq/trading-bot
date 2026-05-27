# 📈 TradingView → Bot → IBKR Israel

מדריך מלא לחיבור TradingView אל הבוט שלך.

---

## 🎯 איך זה עובד

```
TradingView (גרף + Pine Script)
        ↓ alert webhook
   הבוט שלך ב-Render
        ↓ validates + adds filters
   Interactive Brokers Israel
        ↓ executes trade
   💰 כסף אמיתי בחשבון שלך
```

---

## 🚀 שלב 1: TradingView Premium (Essential)

TradingView **חינמי** מאפשר רק התראות **דרך push notifications** (לא webhooks).

צריך **TradingView Essential** ($14.95/חודש) או יותר עבור webhooks.

🔗 https://www.tradingview.com/gopro/

---

## 🔧 שלב 2: הגדרת Alert ב-TradingView

### **דרך 1: Alert בסיסי (לחיצה ידנית)**

1. פתח גרף TradingView
2. **Alert → Create Alert** (Alt+A)
3. **Condition:** מהאינדיקטור שלך
4. **Webhook URL:** `https://your-render-url.onrender.com/webhook`
5. **Message** — JSON:
```json
{
  "secret": "tradebot_wh_2026_secure",
  "ticker": "{{ticker}}",
  "action": "buy",
  "price": {{close}}
}
```

### **דרך 2: Pine Script Strategy אוטומטי**

ראה דוגמאות בהמשך.

---

## 📜 שלב 3: Pine Script דוגמאות

### **אסטרטגיה 1: Pullback Buy (חיברת לנתוני הבוט שלנו)**

```pine
//@version=5
strategy("Bot Pullback Buy", overlay=true)

// Inputs
rsiPeriod = input.int(14, "RSI Period")
rsiBuyZone = input.int(40, "RSI Buy Zone (max)")
rsiOversold = input.int(30, "RSI Oversold (min)")
sma50Required = input.bool(true, "Require above SMA50")
volMultiplier = input.float(0.85, "Volume Ratio min")
secretKey = input.string("tradebot_wh_2026_secure", "Webhook Secret")

// Indicators
rsi = ta.rsi(close, rsiPeriod)
sma50 = ta.sma(close, 50)
sma200 = ta.sma(close, 200)
volAvg = ta.sma(volume, 20)
volRatio = volume / volAvg

// Conditions matching our bot's filters
buyCondition = 
    rsi >= rsiOversold and 
    rsi <= rsiBuyZone and
    (not sma50Required or close > sma50) and
    close > sma200 and
    volRatio >= volMultiplier

// Plot signals
plotshape(buyCondition, "BUY", shape.triangleup, location.belowbar, color.green, size=size.normal)

// Alert message (sent to bot)
if buyCondition
    alert('{"secret":"' + secretKey + '","ticker":"' + syminfo.ticker + '","action":"buy","price":' + str.tostring(close) + '}', alert.freq_once_per_bar)
```

### **אסטרטגיה 2: Breakout Buy**

```pine
//@version=5
strategy("Bot Breakout Buy", overlay=true)

// Inputs
period = input.int(20, "Breakout Period")
volSurge = input.float(1.5, "Volume Surge multiplier")
secretKey = input.string("YOUR_SECRET", "Webhook Secret")

// Calculate
highestHigh = ta.highest(high, period)[1]   // Previous N-day high
volAvg = ta.sma(volume, 20)

// Conditions
breakout = close > highestHigh and volume > volAvg * volSurge

// Alert
if breakout
    alert('{"secret":"' + secretKey + '","ticker":"' + syminfo.ticker + '","action":"buy","price":' + str.tostring(close) + '}', alert.freq_once_per_bar)
```

### **אסטרטגיה 3: Sell Signal**

```pine
//@version=5
strategy("Bot Sell Signal", overlay=true)

// Inputs
secretKey = input.string("YOUR_SECRET", "Webhook Secret")
rsiOverbought = input.int(70, "RSI Overbought")

// Indicators
rsi = ta.rsi(close, 14)
sma20 = ta.sma(close, 20)

// Sell conditions
sellCondition = 
    rsi > rsiOverbought or
    close < sma20

if sellCondition
    alert('{"secret":"' + secretKey + '","ticker":"' + syminfo.ticker + '","action":"sell","price":' + str.tostring(close) + '}', alert.freq_once_per_bar)
```

---

## 🛡️ שלב 4: הגנה דו-שכבתית

הבוט שלך **לא** קונה רק בגלל איתות מ-TradingView!  
הוא מוסיף בדיקות נוספות:

```
TradingView signal arrives
  ↓
1. ✅ Validate webhook secret
2. ✅ Check circuit breaker
3. ✅ Sentiment check (sentiment >= 5)
4. ✅ Composite score >= MIN_BUY_SCORE (70)
5. ✅ SMA50 filter
6. ✅ Volume confirmation
7. ✅ Sector check
8. ✅ Pre-buy checklist
9. ✅ Pro Entry grade A/B
10. 💰 Execute on IBKR Israel
```

**אם משהו לא בסדר → הבוט יסרב גם אם TradingView אמר לקנות!**

---

## 📊 שלב 5: 3 דרכי שימוש

### **דרך A: TradingView מחליט, הבוט מבצע** (אגרסיבי)
```
TradingView Pine Script → Webhook → Bot executes
המתאים אם אתה רוצה לסחור על אסטרטגיות מתוחכמות שלך
```

### **דרך B: הבוט מחליט בלבד, TradingView לreference** (שמרני)
```
הבוט סורק לבד עם הפילטרים שלו
TradingView משמש רק לצפייה ולניתוח ידני
```

### **דרך C: היברידי — שני המקורות** (מומלץ!)
```
הבוט סורק כל 5 דקות (default)
+
TradingView שולח signals כשפיין סקריפט מצלצל
+
הבוט בודק שניהם — קונה רק אם שניהם מסכימים
```

---

## 🔥 דוגמה לתסריט מומלץ

### **המסחר היומי שלך:**

1. **בוקר** (16:00 שעון ישראל)
   - הבוט שולח /briefing
   - TradingView פתוח על מסך שני

2. **בזמן יום מסחר**
   - הבוט קונה אוטומטית לפי הפילטרים
   - TradingView שולח alerts על breakouts שראית

3. **כשהגרף שלך אומר BUY**
   - Pine Script שולח webhook
   - הבוט מקבל + מאמת + קונה דרך IBKR

4. **בסגירה**
   - הבוט שולח /digest
   - אתה רואה את העסקאות

---

## 🎯 הגדרה מהירה (5 דקות)

### **שלב 1: הוסף ל-.env (במחשב + ב-Render):**
```bash
ACTIVE_BROKER=ibkr             # ← לא tv_paper יותר
ALLOW_TRADINGVIEW_WEBHOOKS=true
TV_REQUIRE_BOT_VALIDATION=true  # ← חשוב לבטיחות
```

### **שלב 2: TradingView (Premium)**
1. עלה ל-Essential ($14.95/חודש)
2. צור Alert
3. URL: `https://your-bot.onrender.com/webhook`
4. JSON message (ראה למעלה)

### **שלב 3: הדבק את ה-Pine Script**
1. Pine Editor → Open
2. הדבק את האסטרטגיה הרצויה
3. Save → Add to Chart
4. אם פעיל → אישור Webhook נשלח אוטומטית

---

## ⚠️ סיכונים והגנות

### **סיכון 1: TradingView יזייף signal**
✅ **הגנה:** WEBHOOK_SECRET מאומת — רק עם הסיסמא המדויקת מתקבל

### **סיכון 2: רובוטים זדוניים שולחים signals**
✅ **הגנה:** IP blocking אחרי 3 כשלי auth

### **סיכון 3: Pine Script באג שגורם להפסדים**
✅ **הגנה:** הבוט בודק את הסיגנל **שוב** עם הפילטרים שלו

### **סיכון 4: TradingView signal מתקבל בעלות גבוהה**
✅ **הגנה:** Min Position $1,500 (commission < 0.3%)

---

## 💎 יתרון מערכת היברידית

```
TradingView Premium ($15/חודש):
  ✅ גרפים מקצועיים
  ✅ Pine Script מתוחכם
  ✅ Backtest מהיר

+ הבוט שלך:
  ✅ פילטרים נוספים
  ✅ Risk management
  ✅ ניהול תיק
  ✅ Telegram alerts

+ IBKR Israel:
  ✅ עמלות נמוכות
  ✅ דיווח מס פשוט
  ✅ ETFs ומניות

= מערכת ברמת fund hedge מקצועי!
```

---

## 🚀 מה אני אעשה עכשיו

ארחיב את ה-webhook כדי לתמוך ב-TradingView signals מתקדמים יותר.
