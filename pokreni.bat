@echo off
title ArgusFace Studio
cd /d "%~dp0"

:: 1. Oslobodi port 7860 ako je zaostala stara instanca
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":7860" ^| findstr "LISTENING"') do taskkill /f /pid %%a >nul 2>&1

:: 2. Provjeri zeli li korisnik eksplicitno vidjeti konzolu (npr. pokreni.bat --console)
if "%1"=="--console" goto run_console
if "%1"=="--browser" goto run_console

:: 3. Zadano tiho pokretanje: ako postoji pythonw, pokreni u pozadini i odmah ugasi ovaj terminal
if exist "python\pythonw.exe" (
    start "" "python\pythonw.exe" "run_studio.py" %*
    exit
) else if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" "run_studio.py" %*
    exit
)

:run_console
if exist "python\python.exe" (
    set "PY_CMD=python\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PY_CMD=.venv\Scripts\python.exe"
) else (
    set "PY_CMD=python"
)

"%PY_CMD%" "run_studio.py" %*
