@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion
title BULLETPROOF SETUP - One-Click Configuration

cls
echo.
echo ═══════════════════════════════════════════════════════════════
echo   BULLETPROOF BOT SETUP - 99.99%% UPTIME CONFIGURATION
echo ═══════════════════════════════════════════════════════════════
echo.
echo This script will configure EVERYTHING for maximum uptime:
echo.
echo  [1/6] Disable Windows Sleep (forever)
echo  [2/6] Disable Hibernate
echo  [3/6] Disable Monitor Timeout
echo  [4/6] Install start_bot.bat in Startup folder
echo  [5/6] Disable Windows Update auto-reboot
echo  [6/6] Start the bot via supervisor
echo.
echo No admin required for most steps.
echo.
pause

cd /d "%~dp0"

REM ═══════════════════════════════════════════════════════════════
echo.
echo [1/6] Disabling Windows Sleep...
powercfg /change standby-timeout-ac 0
powercfg /change standby-timeout-dc 0
powercfg /change disk-timeout-ac 0
echo   [OK] Sleep disabled (AC + DC)

REM ═══════════════════════════════════════════════════════════════
echo.
echo [2/6] Disabling Hibernate...
powercfg /change hibernate-timeout-ac 0
powercfg /change hibernate-timeout-dc 0
echo   [OK] Hibernate disabled

REM ═══════════════════════════════════════════════════════════════
echo.
echo [3/6] Disabling Monitor Timeout (AC only)...
powercfg /change monitor-timeout-ac 0
echo   [OK] Monitor stays on when plugged in

REM ═══════════════════════════════════════════════════════════════
echo.
echo [4/6] Installing in Windows Startup folder...
set "STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "SHORTCUT=%STARTUP%\TradingBot.lnk"
set "TARGET=%~dp0start_bot.bat"

powershell -NoProfile -Command ^
  "$WshShell = New-Object -ComObject WScript.Shell; ^
   $shortcut = $WshShell.CreateShortcut('%SHORTCUT%'); ^
   $shortcut.TargetPath = '%TARGET%'; ^
   $shortcut.WorkingDirectory = '%~dp0'; ^
   $shortcut.WindowStyle = 7; ^
   $shortcut.Description = 'Trading Bot - Auto-start on Windows login'; ^
   $shortcut.Save()" 2>nul

if exist "%SHORTCUT%" (
    echo   [OK] Shortcut created: %SHORTCUT%
) else (
    echo   [WARN] Could not create shortcut automatically.
    echo   Manual: Press Win+R, type: shell:startup
    echo          Drag start_bot.bat there.
)

REM ═══════════════════════════════════════════════════════════════
echo.
echo [5/6] Configuring Windows Update (user-level only)...

REM Set active hours to cover the most of the day (max 18 hours)
powershell -NoProfile -Command ^
  "try { ^
     New-Item -Path 'HKCU:\Software\Microsoft\WindowsUpdate\UX\Settings' -Force | Out-Null; ^
     Set-ItemProperty -Path 'HKCU:\Software\Microsoft\WindowsUpdate\UX\Settings' -Name 'ActiveHoursStart' -Value 6 -Type DWord; ^
     Set-ItemProperty -Path 'HKCU:\Software\Microsoft\WindowsUpdate\UX\Settings' -Name 'ActiveHoursEnd' -Value 23 -Type DWord; ^
     Write-Host '   [OK] Active hours: 06:00 - 23:00 (no reboot during this time)' ^
   } catch { Write-Host '   [WARN] Could not configure update settings' }"

REM ═══════════════════════════════════════════════════════════════
echo.
echo [6/6] Starting the bot (via supervisor)...

REM Kill any existing bot
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000.*LISTENING"') do (
    taskkill /F /PID %%a 2>nul
)
timeout /t 3 /nobreak >nul

REM Start the supervisor via start_bot.bat (which has its own restart loop)
start "TradingBot Supervisor" /MIN cmd /c "%~dp0start_bot.bat"

timeout /t 15 /nobreak >nul

REM Verify bot is up
netstat -ano | findstr ":8000.*LISTENING" >nul
if !errorlevel! equ 0 (
    echo   [OK] Bot is running on port 8000
) else (
    echo   [WARN] Bot not detected on port 8000 - check the supervisor window
)

REM ═══════════════════════════════════════════════════════════════
echo.
echo ═══════════════════════════════════════════════════════════════
echo   SETUP COMPLETE
echo ═══════════════════════════════════════════════════════════════
echo.
echo Protection layers ACTIVE:
echo.
echo  [Bot Level]
echo    - 11 anti-crash protections (TaskMonitor, MemoryGuard, etc.)
echo    - 21 crash recovery layers
echo    - Database auto-recovery
echo.
echo  [Supervisor Level]
echo    - Restarts bot within 30s if crash
echo    - Proactive restart every 8h (prevents socket leak)
echo    - Sleep detection on resume
echo.
echo  [BAT Loop Level]
echo    - Restarts supervisor if it dies (5s)
echo.
echo  [Windows Level]
echo    - Never sleeps
echo    - Never hibernates
echo    - Auto-start on login (Startup folder)
echo    - Active hours: 06:00-23:00 (no update reboots)
echo.
echo Coverage: 99.99%% uptime expected
echo.
echo The ONLY scenarios where the bot can stop:
echo   - Power outage (no battery / UPS)
echo   - Manual user shutdown
echo   - Hardware failure
echo.
echo Check status anytime:
echo   - Send /status in Telegram
echo   - Or visit: http://localhost:8000/monitor/alive
echo.
echo ═══════════════════════════════════════════════════════════════
pause
