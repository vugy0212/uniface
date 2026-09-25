@echo off
title UniFace Studio - Face Recognition Web UI
echo ===================================================
echo     Starting UniFace Studio (Local Web UI)
echo ===================================================
echo.

cd /d "%~dp0"

:: Free port 7860 if occupied by previous session
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":7860" ^| findstr "LISTENING"') do taskkill /f /pid %%a >nul 2>&1

:: Find python executable in .venv
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY_CMD=%~dp0.venv\Scripts\python.exe"
) else if exist "%~dp0..\.venv\Scripts\python.exe" (
    set "PY_CMD=%~dp0..\.venv\Scripts\python.exe"
) else (
    set "PY_CMD=python"
)

echo Starting Web UI on http://127.0.0.1:7860
echo Browser will open automatically.
echo.

if exist "run_studio.py" (
    "%PY_CMD%" run_studio.py
) else if exist "src\app.py" (
    "%PY_CMD%" src\app.py
)

if errorlevel 1 (
    echo.
    echo ===================================================
    echo  An error occurred while running the application.
    echo ===================================================
    pause
)
