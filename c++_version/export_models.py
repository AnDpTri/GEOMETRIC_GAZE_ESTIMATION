"""
Export FaceMesh from MediaPipe TFLite -> ONNX (cv::dnn compatible)
Uses the parent project's venv which has mediapipe installed.
"""
import os, sys, shutil, subprocess, struct
from pathlib import Path

CWD = Path(__file__).parent
MODELS = CWD / "models"
MODELS.mkdir(exist_ok=True)

# Find the venv Python that has mediapipe
VENV_PYTHON = CWD.parent / "venv" / "Scripts" / "python.exe"
SYS_PYTHON = sys.executable

def run_py(script_text, python_exe=None):
    """Run a Python script string with the given interpreter."""
    py = str(python_exe or SYS_PYTHON)
    r = subprocess.run([py, "-c", script_text], capture_output=True, text=True, cwd=str(CWD))
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr[-1000:] if len(r.stderr) > 1000 else r.stderr)
    return r.returncode == 0

# ============ 1. YOLO ============
print("=" * 50)
print("1. YOLO Face -> ONNX")
yolo_dest = MODELS / "yolov8_face.onnx"
if yolo_dest.exists():
    print(f"  [SKIP] Already exists: {yolo_dest.name} ({yolo_dest.stat().st_size//1024} KB)")
else:
    try:
        from ultralytics import YOLO
        pt = CWD.parent / "yolov8_head" / "yolov8n-face.pt"
        if pt.exists():
            m = YOLO(str(pt))
            ep = m.export(format='onnx', imgsz=640, simplify=True)
            shutil.move(ep, yolo_dest)
            print(f"  [OK] {yolo_dest.name}")
    except Exception as e:
        print(f"  [ERR] {e}")

# ============ 2. FaceMesh TFLite -> ONNX ============
print("\n" + "=" * 50)
print("2. MediaPipe FaceMesh -> ONNX")
mesh_dest = MODELS / "face_mesh.onnx"

if mesh_dest.exists():
    print(f"  [SKIP] Already exists: {mesh_dest.name} ({mesh_dest.stat().st_size//1024} KB)")
else:
    # Determine which Python has mediapipe
    py_to_use = None
    for py_candidate in [SYS_PYTHON, str(VENV_PYTHON)]:
        if not Path(py_candidate).exists():
            continue
        r = subprocess.run([py_candidate, "-c", "import mediapipe; print('OK')"], 
                          capture_output=True, text=True)
        if r.returncode == 0 and "OK" in r.stdout:
            py_to_use = py_candidate
            print(f"  [INFO] Found mediapipe in: {py_candidate}")
            break
    
    if not py_to_use:
        print("  [ERR] mediapipe not found in any Python environment!")
        print(f"  Tried: {SYS_PYTHON}")
        print(f"  Tried: {VENV_PYTHON}")
        print("  Please run: pip install mediapipe tf2onnx")
        sys.exit(1)
    
    # Install tf2onnx in the mediapipe environment
    print("  [INFO] Installing tf2onnx...")
    subprocess.run([py_to_use, "-m", "pip", "install", "tf2onnx", "flatbuffers", "-q"],
                   capture_output=True)
    
    # Extract built-in TFLite model and convert to ONNX
    export_script = f'''
import mediapipe as mp
import os, glob, shutil
from pathlib import Path

# Find the face_landmark TFLite model inside mediapipe package
mp_dir = Path(mp.__file__).parent
print(f"MediaPipe dir: {{mp_dir}}")

# Search for face landmark models
candidates = []
for pattern in ["**/face_landmark*.tflite", "**/face_mesh*.tflite"]:
    candidates.extend(mp_dir.rglob(pattern.replace("**", "*")))
    # Deep search
    for root, dirs, files in os.walk(mp_dir):
        for f in files:
            if "face_landmark" in f and f.endswith(".tflite"):
                candidates.append(Path(root) / f)
            if "face_mesh" in f and f.endswith(".tflite"):
                candidates.append(Path(root) / f)

# Deduplicate
candidates = list(set(candidates))
print(f"Found {{len(candidates)}} TFLite model(s):")
for c in candidates:
    print(f"  - {{c.name}} ({{c.stat().st_size//1024}} KB)")

# Pick the best one (with_attention preferred, then largest)
tflite = None
for c in candidates:
    if "with_attention" in c.name:
        tflite = c; break
if not tflite:
    candidates.sort(key=lambda x: x.stat().st_size, reverse=True)
    tflite = candidates[0] if candidates else None

if not tflite:
    print("ERROR: No TFLite model found")
    exit(1)

print(f"\\nUsing: {{tflite.name}}")

# Copy to temp location
tmp = Path(r"{str(MODELS)}") / "temp_face_mesh.tflite"
shutil.copy2(tflite, tmp)

# Convert using tf2onnx
import subprocess, sys
dest = Path(r"{str(mesh_dest)}")
result = subprocess.run([
    sys.executable, "-m", "tf2onnx.convert",
    "--tflite", str(tmp),
    "--output", str(dest),
    "--opset", "13",
    "--inputs-as-nchw", "input_1"
], capture_output=True, text=True)

print("tf2onnx stdout:", result.stdout[-500:])
if result.returncode != 0:
    print("tf2onnx stderr:", result.stderr[-500:])

# Cleanup
if tmp.exists():
    os.remove(tmp)

if dest.exists():
    print(f"\\n[OK] Saved: {{dest}} ({{dest.stat().st_size//1024}} KB)")
else:
    print("\\n[FAIL] Conversion failed")
'''
    
    print("  [INFO] Extracting and converting FaceMesh model...")
    success = run_py(export_script, py_to_use)
    
    if not success or not mesh_dest.exists():
        print("  [WARN] tf2onnx conversion failed. Trying alternative method...")
        
        # Alternative: Use onnxruntime to verify and re-export
        alt_script = f'''
import mediapipe as mp
import numpy as np
import os
from pathlib import Path

# Use MediaPipe's Python API to find the model
mp_dir = Path(mp.__file__).parent
print(f"Scanning {{mp_dir}} for face landmark models...")

for root, dirs, files in os.walk(mp_dir):
    for f in files:
        if f.endswith(".tflite") and ("face_landmark" in f or "face_mesh" in f):
            full = Path(root) / f
            print(f"  Found: {{full}} ({{full.stat().st_size//1024}}KB)")
            # Copy it so we can convert it later
            dest = Path(r"{str(MODELS)}") / "face_landmark.tflite"
            import shutil
            shutil.copy2(full, dest)
            print(f"  Copied to: {{dest}}")
'''
        run_py(alt_script, py_to_use)

# ============ 3. Verify ============
print("\n" + "=" * 50)
print("Verification:")
for name in ["yolov8_face.onnx", "face_mesh.onnx", "face_landmark.tflite"]:
    p = MODELS / name
    if p.exists():
        print(f"  [OK] {name} ({p.stat().st_size // 1024} KB)")
    else:
        print(f"  [--] {name}")

if not (MODELS / "face_mesh.onnx").exists() and (MODELS / "face_landmark.tflite").exists():
    print("\n[INFO] ONNX conversion failed but TFLite model was extracted.")
    print("       The C++ app will need onnxruntime or tflite runtime to use it.")
    print("       OR install tf2onnx: pip install tf2onnx")
    print(f"       Then run: python -m tf2onnx.convert --tflite {MODELS/'face_landmark.tflite'} --output {MODELS/'face_mesh.onnx'} --opset 13")

print("\nDone!")
