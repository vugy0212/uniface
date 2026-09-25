@echo off
title UniFace Studio - Sustav za prepoznavanje lica
echo ===================================================
echo     Pokretanje UniFace Studio (Lokalno)
echo ===================================================
echo.

cd /d "%~dp0"

:: 1. Oslobodi port 7860 ako je zaostala stara instanca u pozadini
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":7860" ^| findstr "LISTENING"') do taskkill /f /pid %%a >nul 2>&1

:: 2. Pronadi Python unutar virtualnog okruzenja (.venv)
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY_CMD=%~dp0.venv\Scripts\python.exe"
) else if exist "%~dp0..\.venv\Scripts\python.exe" (
    set "PY_CMD=%~dp0..\.venv\Scripts\python.exe"
) else (
    set "PY_CMD=python"
)

echo Pokrecem graficko sucelje na http://127.0.0.1:7860
echo Web preglednik ce se automatski otvoriti.
echo.

if exist "run_studio.py" (
    "%PY_CMD%" run_studio.py
) else if exist "src\app.py" (
    "%PY_CMD%" src\app.py
)

if errorlevel 1 (
    echo.
    echo ===================================================
    echo  Doslo je do greske prilikom pokretanja aplikacije.
    echo ===================================================
    pause
)
