@echo off
chcp 65001 >nul
title UniFace 24/7 NVR Video Snimanje i Biometrijska Evidencija
cls

echo =========================================================================
echo       UniFace 24/7 NVR Servis - Snimanje i Evidencija Lica
echo =========================================================================
echo.
echo  Kamera 1: USB Web Kamera (0)
echo  Kamera 2: Denver IP Nadzorna Kamera (192.168.50.236)
echo  Segmenti: 5 minuta po MP4 datoteci
echo  Arhiva:   data/recordings/ (FIFO automatsko ciscenje do 20 GB)
echo.
echo  Pritisnite Ctrl+C za zaustavljanje servisa.
echo =========================================================================
echo.

cd /d "%~dp0"
call .venv\Scripts\activate.bat

python run_nvr.py --cam1 0 --name1 "USB Web Kamera" --cam2 "rtsp://admin:admin@192.168.50.236:554/11" --name2 "Denver IP Kamera" --segment-min 5 --max-gb 20 --threshold 0.45 --cooldown 30

pause
