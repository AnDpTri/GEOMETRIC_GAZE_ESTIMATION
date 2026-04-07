#!/bin/bash
echo "========================================================="
echo "   GAZE ESTIMATION - FAST RUN (1-CLICK)              "
echo "   RPi 4 / Debian / ARM64 Compatible                 "
echo "========================================================="

# Check for environment
if [ ! -d "venv" ]; then
    echo "[!] venv environment NOT detected. Starting official setup..."
    sudo bash setup_rpi.sh
fi

# Activation and Run
echo "[*] Activating venv and starting Gaze Estimation..."
source venv/bin/activate
python3 gaze_estimation.py

# Cleanup (optional)
# deactivate
