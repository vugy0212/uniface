@echo off
chcp 65001 >nul
title UniFace Multi-Camera 2x2 Grid
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo ===================================================
echo     Pokretanje UniFace 2x2 Multi-Cam Nadzorne Mreze
echo ===================================================
echo [Tipka 1-4] Povecanje pojedine kamere (Solo mod)
echo [Tipka 0/ESC] Povratak na 2x2 prikaz
echo [Tipka S] Spremanje snimke kadra mreze
echo [Tipka E] Ukljucivanje/iskljucivanje evidencije
echo [Tipka Q] Izlaz
echo.

"%PYTHON_EXE%" run_multicam.py --log-events

pause
