@echo off
REM Card Test Lab: pre-built scenarios with infinite resources and reset controls.
REM Starts the server, waits for it to be ready, then opens the browser.

cd /d "%~dp0backend"

echo.
echo ===================================
echo   Riftbound Card Test Lab
echo ===================================
echo.
echo Starting server...

start /b python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

:wait_loop
timeout /t 1 /nobreak >nul
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo   Waiting for server...
    goto wait_loop
)

echo   Server ready!
echo   Opening http://localhost:8000/?mode=testlab
echo.
echo Press Ctrl+C to stop.
echo.

start http://localhost:8000/?mode=testlab

REM Keep the window open so the server keeps running
cmd /k
