"""
MMPOSE Central Configuration
============================
Quản lý các tham số tập trung cho toàn bộ dự án.
"""

import torch
from pathlib import Path

# ─── ĐƯỜNG DẪN GỐC ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# ─── THIẾT BỊ XỬ LÝ (CPU/GPU) ────────────────────────────────────────────────
# 'cpu' : Chạy trên CPU
# '0'   : Chạy trên GPU NVIDIA thứ nhất (nếu có CUDA)
# 'auto': Tự động tìm kiếm GPU, nếu không có sẽ dùng CPU
DEVICE = 'cpu' 

# ─── CẤU HÌNH YOLOv8 HEAD ────────────────────────────────────────────────────
YOLO_MODEL_SIZE = 'n'        # n, s, m, l, x
YOLO_CONF_THRESHOLD = 0.50
YOLO_MODEL_DIR = BASE_DIR / "yolov8_head"

# ─── CẤU HÌNH MEDIAPIPE ──────────────────────────────────────────────────────
MP_MAX_FACES = 5
MP_REFINE_LANDMARKS = True
MP_STATIC_MODE = False       # False cho real-time, True cho batch processing

# ─── CẤU HÌNH CROP & PADDING ─────────────────────────────────────────────────
FACE_PAD_RATIO = 0.15        # 15% mỗi phía (tổng 30% diện tích)

# ─── THƯ MỤC DỮ LIỆU ─────────────────────────────────────────────────────────
INPUT_DIR = BASE_DIR / "face_landmark" / "input"
OUTPUT_DIR = BASE_DIR / "face_landmark" / "output"
DATA_DIR = BASE_DIR / "face_landmark" / "data"

def get_device():
    """Trả về thiết bị xử lý chuẩn xác nhất dựa trên cấu hình."""
    target_device = DEVICE
    
    # Nếu là chuỗi số, chuyển sang int (GPU index)
    if isinstance(target_device, str) and target_device.isdigit():
        target_device = int(target_device)
        
    if target_device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Kiểm tra nếu yêu cầu GPU (số nguyên hoặc 'cuda') nhưng không có CUDA
    if target_device != 'cpu':
        if not torch.cuda.is_available():
            print(f"[!] Cảnh báo: Yêu cầu thiết bị '{target_device}' nhưng CUDA không khả dụng. Chuyển về 'cpu'.")
            return 'cpu'
            
    return target_device
