@echo off
title UniFace Studio - Face Recognition Web UI
echo ===================================================
echo     Starting UniFace Studio (Local Web UI)
echo ===================================================
echo.
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else if exist "..\.venv\Scripts\activate.bat" (
    call "..\.venv\Scripts\activate.bat"
)

echo Starting Web UI on http://127.0.0.1:7860 ...
python run_studio.py
if errorlevel 1 (
    python src\app.py
)
pause
