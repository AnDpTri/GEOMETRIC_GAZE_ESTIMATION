@echo off
echo ===================================================
echo   GAZE ESTIMATION - WINDOWS SETUP (PC/CUDA)
echo ===================================================

echo [*] Kiem tra Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Khong tim thay Python. Vui long cai dat tu python.org
    pause
    exit /b
)

echo [*] Tao moi truong ao (venv)...
python -m venv venv

echo [*] Kich hoat moi truong ao...
call venv\Scripts\activate

echo [*] Cap nhat pip...
python -m pip install --upgrade pip

echo [*] Cai dat cac thu vien (requirements.txt)...
pip install -r requirements.txt

echo [*] Kiem tra CUDA (Optional)...
python -c "import torch; print('CUDA Ready:', torch.cuda.is_available())"

echo ===================================================
echo   DA XONG! Chay bang: python gaze_estimation.py
echo ===================================================
pause
