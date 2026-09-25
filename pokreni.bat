@echo off
title UniFace Studio - Sustav za prepoznavanje lica
echo ===================================================
echo     Pokretanje UniFace Studio (Lokalno)
echo ===================================================
echo.
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else if exist "..\.venv\Scripts\activate.bat" (
    call "..\.venv\Scripts\activate.bat"
)

echo Pokrecem graficko sucelje na http://127.0.0.1:7860 ...
python run_studio.py
if errorlevel 1 (
    python src\app.py
)
pause
