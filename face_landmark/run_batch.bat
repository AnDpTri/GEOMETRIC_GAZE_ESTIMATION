@echo off
:: Script chay nhanh Batch Landmark Processing su dung moi truong tu yolov8_head

if not exist "..\venv\Scripts\python.exe" (
    echo [!] Khong tim thay moi truong venv tai goc MMPOSE.
    pause
    exit /b 1
)

echo [*] Dang khoi chay Batch Landmark Processing...
"..\venv\Scripts\python.exe" batch_landmark.py %*

if %errorlevel% neq 0 (
    echo.
    echo [!] Da co loi xay ra trong qua trinh xu ly.
)
pause
