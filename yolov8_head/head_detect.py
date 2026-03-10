"""
YOLOv8 Face Detection - Real-time Webcam
=========================================
Sử dụng model YOLOv8 Face để phát hiện khuôn mặt theo thời gian thực.

Cách dùng:
  python face_detect.py                    # model Nano (mặc định)
  python face_detect.py --model s          # model Small cho độ chính xác cao hơn
  python face_detect.py --model m --conf 0.4
  python face_detect.py --models           # liệt kê tất cả model

Phím điều khiển:
  Q / ESC  : Thoát
  S        : Chụp ảnh → lưu vào captures/
  F        : Bật/tắt FPS
  +/-      : Tăng/giảm ngưỡng tin cậy
"""

import argparse
import sys
import io

# Đảm bảo in ra đúng định dạng UTF-8 trên Windows
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except Exception:
        pass
import cv2
import time
import urllib.request
from pathlib import Path

# Thêm đường dẫn gốc vào sys.path để import config
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import config
from ultralytics import YOLO
from model_manager import get_model_path, list_models, get_model_info

# ─── Cấu hình ────────────────────────────────────────────────────────────────
CAPTURES_DIR = Path(__file__).parent / "captures"
CAMERA_INDEX = 0
WINDOW_TITLE = "YOLOv8 Face Detection"

# Màu sắc (BGR)
COLOR_BOX   = (0, 220, 100)
COLOR_LABEL = (0, 0, 0)
COLOR_BG    = (0, 220, 100)
COLOR_FPS   = (0, 200, 255)

# ─── Vẽ detection ─────────────────────────────────────────────────────────────
def draw_detection(frame, box, conf, label_text="Object"):
    x1, y1, x2, y2 = map(int, box)
    label = f"{label_text} {conf:.0%}"

    # Bounding box với góc bo tròn nhẹ
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)

    # Nền nhãn
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), COLOR_BG, -1)

    # Chữ nhãn
    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_LABEL, 1, cv2.LINE_AA)

# ─── Hiển thị thông tin ───────────────────────────────────────────────────────
def draw_info(frame, fps, count, label_type, show_fps):
    h, w = frame.shape[:2]

    # Số lượng
    info = f"{label_type}s: {count}"
    cv2.putText(frame, info, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # FPS (nếu bật)
    if show_fps:
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_FPS, 2, cv2.LINE_AA)

    # Hướng dẫn ở góc dưới
    guide = "Q/ESC: Thoat  |  S: Chup anh  |  F: FPS"
    cv2.putText(frame, guide, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Face Detection - Real-time Webcam")
    parser.add_argument("--model", default="n", choices=["n","s","m","l","x"],
                        metavar="SIZE",
                        help="Kích thước model: n s m l x (mặc định: n)")
    parser.add_argument("--models", action="store_true",
                        help="Liệt kê tất cả model và thoát")
    parser.add_argument("--conf", type=float, default=0.50,
                        help="Ngưỡng tin cậy (mặc định: 0.50)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Index camera (mặc định: 0)")
    args = parser.parse_args()

    if args.models:
        list_models()
        sys.exit(0)

    conf_threshold = args.conf
    camera_index   = args.camera

    # Tải & nạp model
    model_key = args.model if args.model != 'n' else config.YOLO_MODEL_SIZE
    model_info = get_model_info(model_key)
    model_path = get_model_path(model_key)
    label_type = model_info["type"]

    print(f"[*] Đang nạp model (Device: {config.get_device()}) ...")
    model = YOLO(str(model_path)).to(config.get_device())
    print(f"[✓] Model sẵn sàng! (conf={conf_threshold})")

    # Mở camera
    print(f"[*] Mở camera (index={camera_index}) ...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[✗] Không mở được camera (index={camera_index}). Thử --camera 1 hoặc 2.")
        return

    # Cấu hình camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[✓] Camera: {actual_w}×{actual_h} @ "
          f"{int(cap.get(cv2.CAP_PROP_FPS))} FPS")

    CAPTURES_DIR.mkdir(exist_ok=True)

    # Vòng lặp chính
    show_fps   = True
    fps        = 0.0
    prev_time  = time.time()
    frame_cnt  = 0

    print(f"\n[▶] Đang chạy nhận diện khuôn mặt real-time ...")
    print(f"    Nhấn Q hoặc ESC để thoát\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Không đọc được frame từ camera.")
            break

        # Inference
        results = model(frame, conf=conf_threshold, verbose=False)[0]

        # Vẽ kết quả
        count = 0
        if results.boxes is not None:
            for box in results.boxes:
                conf = float(box.conf[0])
                draw_detection(frame, box.xyxy[0].tolist(), conf, label_text=label_type)
                count += 1

        # Tính FPS (trung bình 15 frame)
        frame_cnt += 1
        if frame_cnt % 15 == 0:
            cur_time = time.time()
            fps = 15.0 / (cur_time - prev_time + 1e-9)
            prev_time = cur_time

        draw_info(frame, fps, count, label_type, show_fps)

        cv2.imshow(WINDOW_TITLE, frame)

        # Xử lý phím bấm
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):   # Q hoặc ESC
            print("\n[✓] Đã thoát.")
            break
        elif key == ord('f'):
            show_fps = not show_fps
        elif key == ord('+') or key == ord('='):
            conf_threshold = min(0.95, round(conf_threshold + 0.05, 2))
            print(f"[*] Conf: {conf_threshold:.2f}")
        elif key == ord('-'):
            conf_threshold = max(0.05, round(conf_threshold - 0.05, 2))
            print(f"[*] Conf: {conf_threshold:.2f}")
        elif key == ord('s'):
            ts   = time.strftime("%Y%m%d_%H%M%S")
            path = CAPTURES_DIR / f"capture_{ts}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"[📷] Đã lưu ảnh: {path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
