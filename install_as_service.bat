@echo off
REM ═══════════════════════════════════════════════════════════════════════════
REM הפעלת הבוט כ-Windows Service דרך NSSM
REM יתרון: לא תלוי בטרמינל, לא נופל מ-Ctrl+C, לא תלוי בLogin של המשתמש
REM ═══════════════════════════════════════════════════════════════════════════

echo.
echo ====================================================================
echo  Trading Bot - Install as Windows Service
echo ====================================================================
echo.

REM בדיקה אם NSSM מותקן
where nssm >nul 2>nul
if errorlevel 1 (
    echo [ERROR] NSSM not installed!
    echo.
    echo Please download NSSM:
    echo 1. Go to https://nssm.cc/download
    echo 2. Download nssm 2.24+
    echo 3. Extract nssm.exe to C:\Windows\System32\
    echo.
    pause
    exit /b 1
)

REM הסר שירות קיים אם יש
nssm stop TradingBot >nul 2>nul
nssm remove TradingBot confirm >nul 2>nul

REM התקן שירות חדש
echo Installing TradingBot service...
nssm install TradingBot "C:\Python\python.exe" "-m uvicorn main:app --host 0.0.0.0 --port 8000"
nssm set TradingBot AppDirectory "C:\Users\תומר\Pictures\קלוד קוד\מנהל ההשקעות שלך 💼"
nssm set TradingBot AppStdout "C:\Users\תומר\Pictures\קלוד קוד\מנהל ההשקעות שלך 💼\service_stdout.log"
nssm set TradingBot AppStderr "C:\Users\תומר\Pictures\קלוד קוד\מנהל ההשקעות שלך 💼\service_stderr.log"
nssm set TradingBot AppRotateFiles 1
nssm set TradingBot AppRotateBytes 10485760
nssm set TradingBot Start SERVICE_AUTO_START
nssm set TradingBot AppExit Default Restart
nssm set TradingBot AppRestartDelay 5000

REM הפעל את השירות
echo Starting TradingBot service...
nssm start TradingBot

echo.
echo ====================================================================
echo  Done! TradingBot installed as Windows Service.
echo ====================================================================
echo.
echo Commands:
echo   nssm status TradingBot    - Check status
echo   nssm restart TradingBot   - Restart bot
echo   nssm stop TradingBot      - Stop bot
echo   nssm remove TradingBot    - Uninstall service
echo.
echo The bot will now run 24/7 without needing a terminal!
echo Logs: service_stdout.log + service_stderr.log
echo.
pause
