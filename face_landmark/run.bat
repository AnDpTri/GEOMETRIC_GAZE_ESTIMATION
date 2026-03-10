@echo off
:: Script chay nhanh Landmark Detection su dung moi truong tu yolov8_head

call ..\venv\Scripts\activate
if %errorlevel% neq 0 (
    echo [!] Khong tim thay moi truong venv tai goc MMPOSE.
    pause
    exit /b 1
)

:: Chay bang python trong venv de dam bao dependency
python landmark_detect.py %*
