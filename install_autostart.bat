@echo off
REM ============================================================
REM Installs the Trading Bot Watchdog as a Windows Scheduled Task
REM The watchdog will start automatically when Windows boots up.
REM ============================================================

setlocal enabledelayedexpansion

set BOT_DIR=%~dp0
set BOT_DIR=%BOT_DIR:~0,-1%
set PYTHON_EXE=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python313\python.exe

echo.
echo =============================================================
echo  Trading Bot Auto-Start Installation
echo =============================================================
echo.
echo Bot directory: %BOT_DIR%
echo Python: %PYTHON_EXE%
echo.

REM Verify Python exists
if not exist "%PYTHON_EXE%" (
    echo ERROR: Python not found at %PYTHON_EXE%
    echo Please update PYTHON_EXE in this script.
    pause
    exit /b 1
)

REM Create the scheduled task
schtasks /Create /F /SC ONLOGON /TN "TradingBotWatchdog" ^
    /TR "\"%PYTHON_EXE%\" \"%BOT_DIR%\watchdog.py\"" ^
    /RL HIGHEST ^
    /RU "%USERNAME%"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo SUCCESS: Watchdog will now start automatically on Windows logon.
    echo.
    echo To start it NOW without waiting for next logon, run:
    echo   schtasks /Run /TN "TradingBotWatchdog"
    echo.
    echo To uninstall:
    echo   schtasks /Delete /F /TN "TradingBotWatchdog"
    echo.
) else (
    echo.
    echo FAILED to create scheduled task. Try running as Administrator.
    echo.
)

pause
