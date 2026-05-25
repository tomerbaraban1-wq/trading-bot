# הגדרת Discord Bot - מדריך שלם

## שלב 1: יצירת Discord Application

1. עבור לפורטל המפתחים: https://discord.com/developers/applications
2. לחץ "New Application"
3. תן לה שם (לדוגמה: "TradingBot")
4. לחץ "Create"

## שלב 2: הפוך ל-Bot

1. בתפריט הצד, לחץ "Bot"
2. לחץ "Add Bot"
3. תחת "TOKEN", לחץ "Reset Token"
4. **העתק את ה-TOKEN** - זה יהיה `DISCORD_BOT_TOKEN` שלך

## שלב 3: הגדר הרשאות

1. תחת "Intents", הפעל:
   - ✅ Message Content Intent
   - ✅ Server Members Intent
   - ✅ Guild Messages
   - ✅ Direct Messages

2. תחת "OAuth2" → "URL Generator":
   - בחר scopes: `bot`
   - בחר permissions: `Send Messages`, `Read Messages/View Channels`
   - **העתק את ה-URL**

## שלב 4: הוסף את ה-Bot ל-Server

1. העתק את ה-URL מ-URL Generator
2. פתח בדפדפן ובחר את ה-Server שלך
3. לחץ "Authorize"

## שלב 5: קבל Channel ID

1. ב-Discord, הפעל Developer Mode: 
   - Settings → Advanced → Developer Mode (כבה/הדלק)
   
2. לחץ ימין על הערוץ שבו תרצה הודעות
3. בחר "Copy Channel ID"
4. זה יהיה `DISCORD_CHANNEL_ID` שלך

## שלב 6: הוסף ל-.env

```bash
DISCORD_BOT_TOKEN=your_token_here
DISCORD_CHANNEL_ID=your_channel_id_here
DISCORD_GUILD_ID=882265638784090182
```

## בדיקה

כאשר הבוט מופעל, יראה בתפקידים שלו את "TradingBot" offline/online.
