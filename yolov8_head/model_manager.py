"""
Model Manager - Quản lý tải và chọn model YOLOv8 Face
"""

import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Danh sách model tích hợp (Face & Head)
MODELS = {
    # YOLOv8-Head (Abcfsa)
    "hn": {
        "filename": "yolov8n-head.pt",
        "url": "https://github.com/Abcfsa/YOLOv8_head_detector/raw/main/nano.pt",
        "size": "~6 MB",
        "type": "Head",
        "desc": "Head Nano - Nhanh, phù hợp webcam",
    },
    "hm": {
        "filename": "yolov8m-head.pt",
        "url": "https://github.com/Abcfsa/YOLOv8_head_detector/raw/main/medium.pt",
        "size": "~50 MB",
        "type": "Head",
        "desc": "Head Medium - Chính xác cao",
    },
    # YOLOv8-Face (Lindevs)
    "fn": {
        "filename": "yolov8n-face.pt",
        "url": "https://github.com/lindevs/yolov8-face/releases/download/1.0.1/yolov8n-face-lindevs.pt",
        "size": "~6 MB",
        "type": "Face",
        "desc": "Face Nano - Nhanh, chuẩn cho mặt",
    },
}
# Alias cho thuận tiện (mặc định n -> head nano)
MODELS["n"] = MODELS["hn"]
MODELS["m"] = MODELS["hm"]


def list_models():
    print("\n  Model  | Kích thước | Mô tả")
    print("  " + "─" * 60)
    for key, info in MODELS.items():
        exists = "✓" if (BASE_DIR / info["filename"]).exists() else " "
        print(f"  [{exists}] -{key}  | {info['size']:10} | {info['desc']}")
    print()


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 / total_size)
        mb  = downloaded / 1e6
        tot = total_size / 1e6
        print(f"\r    {pct:.1f}%  ({mb:.1f} / {tot:.1f} MB)", end="", flush=True)


def get_model_info(model_size: str) -> dict:
    """Trả về thông tin metadata của model."""
    size = model_size.lower().strip()
    return MODELS.get(size, MODELS["n"])

def get_model_path(model_size: str) -> Path:
    """Trả về đường dẫn model, tự tải nếu chưa có."""
    model_size = model_size.lower().strip()
    if model_size not in MODELS:
        valid = ", ".join(f"-{k}" for k in MODELS)
        raise ValueError(f"Model không hợp lệ: '{model_size}'. Chọn trong: {valid}")

    info = MODELS[model_size]
    path = BASE_DIR / info["filename"]

    if path.exists():
        print(f"[✓] Model: {info['filename']} ({info['size']}) - đã tồn tại")
        return path

    print(f"[↓] Đang tải {info['filename']} ({info['size']}) ...")
    print(f"    URL: {info['url']}")
    try:
        urllib.request.urlretrieve(info["url"], path, _progress)
        print(f"\n[✓] Tải xong: {path}")
    except Exception as e:
        print(f"\n[✗] Không tải được: {e}")
        print(f"    Tải thủ công tại: {info['url']}")
        raise
    return path
