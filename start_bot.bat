@echo off
title TradeBot - Auto Restart
cd /d "%~dp0"

:loop
echo.
echo [%time%] מפעיל את הבוט...
py main.py
echo.
echo [%time%] הבוט נפל - מאתחל תוך 5 שניות...
ping -n 6 127.0.0.1 ^> /dev/null
goto loop
