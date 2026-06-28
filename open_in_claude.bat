@echo off
REM Opens Claude Code inside the trading-bot folder and registers it at the
REM top of the recent-projects list. Uses %~dp0 so no Hebrew path is needed.
cd /d "%~dp0"
call claude
