@echo off
REM One-click play: double-click play.bat or run from terminal
REM Starts the Riftbound simulator and opens it in the browser.

cd /d "%~dp0backend"

echo.
echo ===================================
echo   Riftbound Simulator
echo   http://localhost:8000
echo ===================================
echo.
echo Open two browser tabs to play.
echo Press Ctrl+C to stop.
echo.

start http://localhost:8000

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
