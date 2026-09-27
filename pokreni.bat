@echo off
title UniFace Studio - Sustav za prepoznavanje lica
echo ===================================================
echo     Pokretanje UniFace Studio (Lokalno)
echo ===================================================
echo.

cd /d "%~dp0"

:: 1. Oslobodi port 7860 ako je zaostala stara instanca u pozadini
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":7860" ^| findstr "LISTENING"') do taskkill /f /pid %%a >nul 2>&1

:: 2. Pronadi Python: ugradeni samostalni runtime, .venv ili sistemski
if exist "python\python.exe" (
    set "PY_CMD=python\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PY_CMD=.venv\Scripts\python.exe"
) else (
    set "PY_CMD=python"
)

echo Pokrecem UniFace Studio u samostalnom radnom prozoru...
echo (Za pokretanje u obicnom web pregledniku pokrenite: pokreni.bat --browser)
echo.

"%PY_CMD%" "run_studio.py" %*

if errorlevel 1 (
    echo.
    echo ===================================================
    echo  Doslo je do greske prilikom pokretanja aplikacije.
    echo ===================================================
    pause
)
