@echo off
echo Stopping TradeBot...
taskkill /f /fi "WINDOWTITLE eq TradeBot Server" >nul 2>&1
taskkill /f /im python.exe >nul 2>&1
echo Bot stopped.
timeout /t 2 >nul
