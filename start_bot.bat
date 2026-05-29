@echo off
title Trading Bot Supervisor - KEEP OPEN
chcp 65001 > nul
cd /d "%~dp0"

echo.
echo ============================================
echo   Trading Bot Supervisor v2
echo ============================================
echo   Auto-restart if crash
echo   Restart proactive every 8 hours
echo   Sleep prevention ACTIVE
echo ============================================
echo.

REM ═══════════════════════════════════════════════════════════════════════
REM CRITICAL: Prevent Windows from sleeping
REM Last night's crash was caused by Windows sleeping at 23:02!
REM These commands don't require admin — they apply only to current user.
REM ═══════════════════════════════════════════════════════════════════════
echo Configuring power settings (no admin needed)...
powercfg /change standby-timeout-ac 0 2>nul
powercfg /change monitor-timeout-ac 0 2>nul
powercfg /change hibernate-timeout-ac 0 2>nul
echo Power: AC standby/monitor/hibernate disabled

:loop
echo.
echo [%date% %time%] Starting supervisor...
python bot_supervisor.py
echo.
echo [%date% %time%] Supervisor stopped — restarting in 5 seconds...
timeout /t 5 /nobreak
goto loop
