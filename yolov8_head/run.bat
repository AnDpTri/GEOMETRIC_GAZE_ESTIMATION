@echo off
title YOLOv8 Face Detection
echo ==========================================
echo  YOLOv8 Face Detection - Real-time Webcam
echo ==========================================
echo.
echo [*] Kich hoat moi truong ao trung tam...
call "%~dp0..\venv\Scripts\activate.bat"
echo [v] Moi truong ao da kich hoat!
echo.
echo [*] Khoi dong nhan dien dau (Head Detection)...
python "%~dp0head_detect.py"
echo.
pause
