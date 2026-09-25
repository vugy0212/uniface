@echo off
title UniFace - Live Prepoznavanje Lica (Logitech C270)
echo ===================================================
echo     UniFace Live Camera - Logitech C270
echo ===================================================
echo.

cd /d "%~dp0"

:: Pronadi Python unutar virtualnog okruzenja (.venv)
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY_CMD=%~dp0.venv\Scripts\python.exe"
) else if exist "%~dp0..\.venv\Scripts\python.exe" (
    set "PY_CMD=%~dp0..\.venv\Scripts\python.exe"
) else (
    set "PY_CMD=python"
)

echo Pokrecem video prozor za prepoznavanje uzivo...
echo.
echo Tipkovnicke kratice u prozoru:
echo   [Q] ili [ESC] - Zatvaranje prozora
echo   [S]           - Spremi trenutni kadar
echo   [+] / [-]     - Promjena praga osjetljivosti
echo   [R]           - Osvjezi bazu osoba
echo   [SPACE]       - Zamrzni / nastavi sliku
echo.

if exist "run_live_cam.py" (
    "%PY_CMD%" run_live_cam.py
) else if exist "src\live_cam.py" (
    "%PY_CMD%" src\live_cam.py
)

if errorlevel 1 (
    echo.
    echo Doslo je do greske prilikom pokretanja kamere.
    pause
)
