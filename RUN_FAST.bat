@echo off
setlocal
echo ===================================================
echo   GAZE ESTIMATION - FAST RUN (1-CLICK)
echo ===================================================

:: Check if environment exists
if not exist venv (
    echo [!] venv environment NOT detected. Starting official setup...
    call setup_pc.bat
)

:: Activation and Run
echo [*] Activating venv and starting Gaze Estimation...
call venv\Scripts\activate
python gaze_estimation.py

if %errorlevel% neq 0 (
    echo [!] Session closed with error code: %errorlevel%
    pause
)
