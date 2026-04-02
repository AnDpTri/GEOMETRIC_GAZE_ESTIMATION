"""
Secondary Model Extraction: MediaPipe FaceMesh with Attention (478 landmarks)
This script finds the exact face_landmark_with_attention.tflite model and converts it to ONNX.
"""
import os, sys, shutil, subprocess
from pathlib import Path

# Config
CWD = Path(__file__).parent
MODELS = CWD / "models"
MODELS.mkdir(exist_ok=True)
DEST_ONNX = MODELS / "face_mesh_attention.onnx"

# Find MediaPipe in the venv
VENV_PYTHON = CWD.parent / "venv" / "Scripts" / "python.exe"
if not VENV_PYTHON.exists():
    VENV_PYTHON = sys.executable

def run_cmd(args):
    print(f"Running: {' '.join(args)}")
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"ERR: {r.stderr}")
    return r.returncode == 0

# 1. Extraction Script (to be run via venv)
extract_script = f'''
import mediapipe as mp
import os, shutil
from pathlib import Path

mp_dir = Path(mp.__file__).parent
print(f"MediaPipe found at: {{mp_dir}}")

# Search for the attention model
# Usually in modules/face_landmark/face_landmark_with_attention.tflite
models = list(mp_dir.rglob("face_landmark_with_attention.tflite"))
if not models:
    # Fallback to standard mesh if attention is not found
    models = list(mp_dir.rglob("face_landmark.tflite"))

if models:
    src = models[0]
    dst = Path(r"{str(MODELS)}") / "temp_mesh.tflite"
    shutil.copy2(src, dst)
    print(f"Extracted: {{src.name}} -> {{dst}}")
else:
    print("CRITICAL: No FaceMesh TFLite models found in mediapipe package.")
    exit(1)
'''

print("-" * 50)
print("1. Extracting TFLite model from MediaPipe...")
with open("tmp_extract.py", "w") as f: f.write(extract_script)
if not run_cmd([str(VENV_PYTHON), "tmp_extract.py"]):
    sys.exit(1)

# 2. Conversion using tf2onnx
print("\n" + "-" * 50)
print("2. Converting TFLite to ONNX (478 pts support)...")
tflite_path = MODELS / "temp_mesh.tflite"
if tflite_path.exists():
    # Attempt conversion
    # Note: We use opset 13 for better compatibility with ORT C++
    conv_success = run_cmd([
        str(VENV_PYTHON), "-m", "tf2onnx.convert",
        "--tflite", str(tflite_path),
        "--output", str(DEST_ONNX),
        "--opset", "13"
    ])
    
    if conv_success:
        print(f"\n[OK] FaceMesh ONNX created: {DEST_ONNX}")
        # Clean up
        os.remove("tmp_extract.py")
        os.remove(tflite_path)
    else:
        print("\n[FAIL] tf2onnx conversion failed. Check logs.")
else:
    print("\n[FAIL] TFLite model was not extracted.")

print("-" * 50)
print("Done!")
