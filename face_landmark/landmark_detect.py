"""
Face Landmark Detection - Real-time Webcam
===========================================
Kết hợp YOLOv8 để phát hiện khuôn mặt và MediaPipe Face Mesh
để dự đoán 468/478 điểm trên mặt.

Cách dùng:
  python landmark_detect.py
  python landmark_detect.py --model s --conf 0.5
  python landmark_detect.py --show-mesh False
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
from pathlib import Path

# Thêm đường dẫn gốc vào sys.path để import config
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import config
import cv2
import time
import mediapipe as mp

# Thêm path yolov8_head để dùng lại các hàm tải model YOLO
sys.path.append(str(ROOT_DIR / "yolov8_head"))
from ultralytics import YOLO
from model_manager import get_model_path, list_models, get_model_info

from landmark_model import get_face_mesh

# ─── Cấu hình ────────────────────────────────────────────────────────────────
CAPTURES_DIR = Path(__file__).parent / "captures"
WINDOW_TITLE = "YOLOv8 + MediaPipe Face Landmark"

# MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# Màu sắc (BGR)
COLOR_BBOX  = (0, 220, 100)
COLOR_LABEL = (0, 0, 0)
COLOR_BG    = (0, 220, 100)
COLOR_FPS   = (0, 200, 255)

# ─── Hàm phụ trợ ─────────────────────────────────────────────────────────────
def draw_bbox(frame, box, conf, label_text="Object"):
    x1, y1, x2, y2 = map(int, box)
    label = f"{label_text} {conf:.0%}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BBOX, 2)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), COLOR_BG, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_LABEL, 1, cv2.LINE_AA)


def draw_info(frame, fps, count, label_type, show_fps):
    h, w = frame.shape[:2]
    info = f"{label_type}s: {count}"
    cv2.putText(frame, info, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    if show_fps:
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_FPS, 2, cv2.LINE_AA)

    guide = "Q/ESC: Thoat | S: Chup | F: FPS | L: Bat/Tat Mesh | +/-: Conf"
    cv2.putText(frame, guide, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="YOLOv8 + MediaPipe Face Landmark")
    parser.add_argument("--model", default="n", choices=["n","s","m","l","x"],
                        help="Kích thước model YOLOv8 (mặc định: n)")
    parser.add_argument("--models", action="store_true",
                        help="Liệt kê tất cả model YOLOv8 và thoát")
    parser.add_argument("--conf", type=float, default=0.50,
                        help="Ngưỡng tin cậy YOLOv8 (mặc định: 0.50)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--max-faces", type=int, default=5, help="Số mặt tối đa để dự đoán landmark")
    parser.add_argument("--no-mesh", action="store_true", help="Ẩn lưới landmark lúc khởi động")
    parser.add_argument("--no-refine", action="store_true", help="Tắt refine landmarks (ít điểm hơn nhưng nhanh hơn đôi chút)")
    args = parser.parse_args()

    if args.models:
        list_models()
        sys.exit(0)

    conf_threshold = args.conf
    camera_index   = args.camera
    show_mesh      = not args.no_mesh
    refine         = not args.no_refine
    max_faces      = args.max_faces

    # 1. Tải YOLOv8 Model
    model_key = args.model if args.model != 'n' else config.YOLO_MODEL_SIZE
    model_info = get_model_info(model_key)
    model_path = get_model_path(model_key)
    label_type = model_info["type"]

    print(f"[*] Đang nạp YOLOv8 model (Device: {config.get_device()}) ...")
    yolo_model = YOLO(str(model_path)).to(config.get_device())
    print(f"[✓] YOLOv8 sẵn sàng! (conf={args.conf})")

    # 2. Khởi tạo MediaPipe Face Mesh
    print(f"[*] Đang nạp MediaPipe Face Mesh (max_faces={config.MP_MAX_FACES}, refine={config.MP_REFINE_LANDMARKS}) ...")
    face_mesh = get_face_mesh(
        max_num_faces=config.MP_MAX_FACES, 
        refine_landmarks=config.MP_REFINE_LANDMARKS
    )
    print(f"[✓] MediaPipe sẵn sàng!")

    # 3. Mở camera
    print(f"[*] Mở camera (index={camera_index}) ...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[✗] Không mở được camera. Thử --camera 1 hoặc 2.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    CAPTURES_DIR.mkdir(exist_ok=True)

    # 4. Biến trạng thái
    show_fps   = True
    fps        = 0.0
    prev_time  = time.time()
    frame_cnt  = 0

    print(f"\n[▶] Đang chạy phát hiện Landmark thời gian thực ...")
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        # ---- BƯỚC 1: YOLOv8 Face Detection ----
        results = yolo_model(frame, conf=conf_threshold, verbose=False)[0]
        
        count = 0
        if results.boxes is not None and len(results.boxes) > 0:
            count = len(results.boxes)
            
            # --- Chạy riêng MediaPipe thay vì cắt từng box ---
            # Vẽ bounding box từ YOLOv8 và chạy MediaPipe trên từng vùng crop
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                draw_bbox(frame, [x1, y1, x2, y2], conf, label_text=label_type)

                # Tính toán padding (mặc định 15% mỗi phía từ config)
                w_box, h_box = x2 - x1, y2 - y1
                pad_w = int(w_box * config.FACE_PAD_RATIO)
                pad_h = int(h_box * config.FACE_PAD_RATIO)
                
                # Mở rộng vùng crop nhưng giữ nguyên trọng tâm
                px1 = max(0, x1 - pad_w)
                py1 = max(0, y1 - pad_h)
                px2 = min(frame.shape[1], x2 + pad_w)
                py2 = min(frame.shape[0], y2 + pad_h)

                face_crop = frame[py1:py2, px1:px2]
                if face_crop.size == 0: continue

                # Chạy MediaPipe trên vùng crop
                rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                mp_results = face_mesh.process(rgb_crop)

                # Vẽ landmark nếu tìm thấy
                if show_mesh and mp_results.multi_face_landmarks:
                    face_landmarks = mp_results.multi_face_landmarks[0]
                    h_c, w_c = face_crop.shape[:2]
                    h_f, w_f = frame.shape[:2]

                    # Ánh xạ tọa độ về frame gốc
                    for lm in face_landmarks.landmark:
                        lm.x = (lm.x * w_c + px1) / w_f
                        lm.y = (lm.y * h_c + py1) / h_f
                        lm.z = (lm.z * w_c) / w_f

                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    
                    if refine:
                        mp_drawing.draw_landmarks(
                            image=frame, landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                        mp_drawing.draw_landmarks(
                            image=frame, landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

        # Tính FPS
        frame_cnt += 1
        if frame_cnt % 15 == 0:
            cur_time = time.time()
            fps = 15.0 / (cur_time - prev_time + 1e-9)
            prev_time = cur_time

        draw_info(frame, fps, count, label_type, show_fps)
        cv2.imshow(WINDOW_TITLE, frame)

        # Xử lý phím bấm
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('f'):
            show_fps = not show_fps
        elif key == ord('l'):
            show_mesh = not show_mesh
        elif key == ord('+') or key == ord('='):
            conf_threshold = min(0.95, round(conf_threshold + 0.05, 2))
            print(f"[*] YOLO Conf: {conf_threshold:.2f}")
        elif key == ord('-'):
            conf_threshold = max(0.05, round(conf_threshold - 0.05, 2))
            print(f"[*] YOLO Conf: {conf_threshold:.2f}")
        elif key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = CAPTURES_DIR / f"landmark_{ts}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"[📷] Đã lưu ảnh: {path}")

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()


if __name__ == "__main__":
    main()
