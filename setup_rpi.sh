#!/bin/bash
echo "========================================================="
echo "   GAZE ESTIMATION - RASPBERRY PI 4 SETUP (ARM/CPU)   "
echo "========================================================="

# 1. Thu vien he thong
echo "[*] Cap nhat bo cai dat (apt)..."
sudo apt-get update
sudo apt-get install -y libopencv-dev libatlas-base-dev libhdf5-dev libqt5gui5 libqt5test5 python3-pip python3-venv

# 2. Moi truong ao
echo "[*] Tao moi truong ao (venv)..."
python3 -m venv venv
source venv/bin/activate

# 3. Cai dat thu vien Python
echo "[*] Cap nhat pip..."
pip install --upgrade pip

echo "[*] Cai dat thu vien tu requirements_rpi.txt..."
pip install -r requirements_rpi.txt

echo "========================================================="
echo "   DA XONG! De chay, hay dung lenh: "
echo "   source venv/bin/activate"
echo "   python3 gaze_estimation.py"
echo "========================================================="
chmod +x setup_rpi.sh
