"""
Face Landmark Detection - Batch Image Processing
================================================
Xử lý tất cả ảnh trong thư mục input và lưu kết quả (vẽ landmark) vào thư mục output.

Cách dùng:
  python batch_landmark.py                          
  python batch_landmark.py --model s                
  python batch_landmark.py -i ../yolov8_face/input -o ./output

Tham số:
  -i / --input   : Thư mục chứa ảnh gốc   (mặc định: ./input)
  -o / --output  : Thư mục lưu ảnh kết quả (mặc định: ./output)
  --model        : Kích thước model YOLOv8 n/s/m/l/x  (mặc định: n)
  --conf         : Ngưỡng tin cậy YOLOv8 0.0-1.0  (mặc định: 0.50)
  --show         : Hiện preview từng ảnh
"""

import argparse
import os
import sys
import io

# Đảm bảo in ra đúng định dạng UTF-8 trên Windows
if sys.platform == "win32":
    try:
        # Kiểm tra nếu stdout có buffer (tránh lỗi trong môi trường đặc biệt)
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except Exception:
        pass

import time
import json
import csv
from pathlib import Path

# Thêm đường dẫn gốc vào sys.path để import config
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import config
import cv2
import numpy as np
import mediapipe as mp

# Thêm path yolov8_head để dùng lại các hàm tải model YOLO
sys.path.append(str(ROOT_DIR / "yolov8_head"))

from ultralytics import YOLO
from model_manager import get_model_path, list_models, get_model_info

from landmark_model import get_face_mesh_static

# ─── Cấu hình ────────────────────────────────────────────────────────────────
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

COLOR_BBOX  = (0, 220, 100)
COLOR_LABEL = (0, 0, 0)
COLOR_BG    = (0, 220, 100)

def draw_bbox(frame, box, conf, label_text="Face"):
    x1, y1, x2, y2 = map(int, box)
    label = f"{label_text} {conf:.0%}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BBOX, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), COLOR_BG, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_LABEL, 1, cv2.LINE_AA)
 
def get_indices(connections):
    return sorted(list(set([idx for conn in connections for idx in conn])))

# ─── Xử lý 1 ảnh ─────────────────────────────────────────────────────────────
def process_image(yolo_model, face_mesh, img_path: Path, out_path: Path,
                  conf: float, show: bool, refine: bool, label_text: str = "Object") -> dict:
    
    img = cv2.imread(str(img_path))
    if img is None:
        return {"file": img_path.name, "status": "ERROR (không đọc được ảnh)", "faces": 0, "landmarks": []}

    orig_h, orig_w = img.shape[:2]

    # 1. Bounding box YOLOv8 (Toàn khung hình để tìm mặt)
    results = yolo_model(img, conf=conf, verbose=False)[0]
    count = 0
    all_faces_landmarks = []
    
    if results.boxes is not None and len(results.boxes) > 0:
        count = len(results.boxes)
        
        # Sắp xếp các box từ trái sang phải để ID ổn định (tùy chọn)
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        # 2. Với mỗi khuôn mặt tìm được, crop và chạy MediaPipe
        for i, (box, b_conf) in enumerate(zip(boxes, confs)):
            x1, y1, x2, y2 = map(int, box)
            
            # Tính toán vùng nới rộng (Padding) dựa trên config
            w_box, h_box = x2 - x1, y2 - y1
            pad_w = int(w_box * config.FACE_PAD_RATIO)
            pad_h = int(h_box * config.FACE_PAD_RATIO)
            
            # Mở rộng box nhưng giữ nguyên trọng tâm
            px1 = max(0, x1 - pad_w)
            py1 = max(0, y1 - pad_h)
            px2 = min(orig_w, x2 + pad_w)
            py2 = min(orig_h, y2 + pad_h)
            
            # Crop vùng mặt đã nới rộng
            face_crop = img[py1:py2, px1:px2]
            if face_crop.size == 0: continue
            
            # Chạy MediaPipe trên vùng crop
            rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            mp_results = face_mesh.process(rgb_crop)
            
            # Vẽ bounding box GỐC lên ảnh chính
            draw_bbox(img, [x1, y1, x2, y2], float(b_conf), label_text=label_text)

            # Nếu tìm thấy landmark trên vùng crop
            if mp_results.multi_face_landmarks:
                face_landmarks = mp_results.multi_face_landmarks[0]
                
                crop_h, crop_w = face_crop.shape[:2]
                landmarks_list = []
                
                for lm in face_landmarks.landmark:
                    # Mapping: Từ vùng crop nới rộng -> Tọa độ ảnh gốc
                    abs_x = lm.x * crop_w + px1
                    abs_y = lm.y * crop_h + py1
                    abs_z = lm.z * crop_w
                    
                    landmarks_list.append({
                        "x": round(abs_x / orig_w, 6),
                        "y": round(abs_y / orig_h, 6),
                        "z": round(abs_z / orig_w, 6)
                    })

                # Phân loại bộ phận dựa trên hằng số MediaPipe
                # Chúng ta lấy các indices từ các cặp kết nối (connections)

                categorized = {
                    "lips": get_indices(mp_face_mesh.FACEMESH_LIPS),
                    "left_eye": get_indices(mp_face_mesh.FACEMESH_LEFT_EYE),
                    "right_eye": get_indices(mp_face_mesh.FACEMESH_RIGHT_EYE),
                    "left_eyebrow": get_indices(mp_face_mesh.FACEMESH_LEFT_EYEBROW),
                    "right_eyebrow": get_indices(mp_face_mesh.FACEMESH_RIGHT_EYEBROW),
                    "face_oval": get_indices(mp_face_mesh.FACEMESH_FACE_OVAL),
                    "iris": get_indices(mp_face_mesh.FACEMESH_IRISES)
                }

                parts_data = {pn: [landmarks_list[idx] for idx in ids if idx < len(landmarks_list)] 
                             for pn, ids in categorized.items()}
                
                all_faces_landmarks.append({
                    "all": landmarks_list,
                    "parts": parts_data
                })

                # --- VẼ LÊN ẢNH GỐC (Dùng tọa độ đã mapping) ---
                # Vì các hàm vẽ của MP nhận đối tượng LandmarkList, ta cần tạo bản sao đã map tọa độ
                # Tuy nhiên cách đơn giản nhất là vẽ trực tiếp bằng OpenCV hoặc map lại tay.
                # Để giữ thẩm mỹ của MediaPipe Drawing, ta vẽ lên img gốc:
                
                # Tạo một bản sao landmarks đã được mapping để dùng với mp_drawing
                for idx, lm in enumerate(face_landmarks.landmark):
                    lm.x = (lm.x * crop_w + px1) / orig_w
                    lm.y = (lm.y * crop_h + py1) / orig_h
                    lm.z = (lm.z * crop_w) / orig_w 

                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                
                if refine:
                    mp_drawing.draw_landmarks(
                        image=img, landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        image=img, landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

    # Ghi thông tin
    info = f"{label_text}s: {count} (Crop-mode) | {img_path.name}"
    cv2.putText(img, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)

    if show:
        cv2.imshow(f"Preview - {img_path.name}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {"file": img_path.name, "status": "OK", "count": count, "landmarks": all_faces_landmarks}


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="YOLOv8 + MediaPipe Landmark Batch")
    parser.add_argument("-i", "--input",  default="input", help="Thư mục ảnh đầu vào")
    parser.add_argument("-o", "--output", default="output", help="Thư mục ảnh đầu ra")
    parser.add_argument("--model", default="n", choices=["n","s","m","l","x"], help="YOLOv8 model size (n/s/m/l/x)")
    parser.add_argument("--models", action="store_true", help="Liệt kê tất cả model YOLOv8")
    parser.add_argument("--conf", type=float, default=0.50, help="YOLO conf threshold")
    parser.add_argument("--max-faces", type=int, default=10, help="Max faces per image")
    parser.add_argument("--no-refine", action="store_true", help="Turn off refine landmarks")
    parser.add_argument("--show", action="store_true", help="Hiện preview từng ảnh")
    parser.add_argument("--export", choices=["json", "csv", "none"], default="json", help="Export format (json/csv/none)")
    parser.add_argument("--data-dir", default="data", help="Thư mục lưu tọa độ (mặc định: ./data)")
    args = parser.parse_args()

    if args.models:
        list_models()
        sys.exit(0)

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    data_dir   = Path(args.data_dir)
    refine     = not args.no_refine
    export_fmt = args.export.lower()

    if not input_dir.exists():
        input_dir.mkdir(parents=True)
        print(f"[!] Đã tạo thư mục input: {input_dir.resolve()}")
        print(f"    Hãy đặt ảnh vào thư mục trên rồi chạy lại.")
        sys.exit(0)

    images = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXT])

    if not images:
        print(f"[!] Không tìm thấy ảnh trong: {input_dir.resolve()}")
        sys.exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)
    if export_fmt != "none":
        data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Nạp YOLOv8
    model_key = args.model if args.model != 'n' else config.YOLO_MODEL_SIZE
    model_info = get_model_info(model_key)
    model_path = get_model_path(model_key)
    label_type = model_info["type"]

    print(f"[*] Đang nạp YOLOv8 (Device: {config.get_device()}) (Type: {label_type}) ...")
    yolo_model = YOLO(str(model_path)).to(config.get_device())

    # 2. Nạp MediaPipe tĩnh
    print(f"[*] Đang nạp MediaPipe Face Mesh Static (refine={config.MP_REFINE_LANDMARKS}) ...")
    face_mesh = get_face_mesh_static(
        max_num_faces=config.MP_MAX_FACES, 
        refine_landmarks=config.MP_REFINE_LANDMARKS
    )

    input_dir = Path(args.input) if args.input != "input" else config.INPUT_DIR
    output_dir = Path(args.output) if args.output != "output" else config.OUTPUT_DIR
    data_dir = config.DATA_DIR
    print(f"\n{'─'*55}")
    print(f"  Landmark : Tĩnh (Static Image Mode)")
    print(f"  Input    : {input_dir.resolve()}")
    print(f"  Output   : {output_dir.resolve()}")
    print(f"  Ảnh      : {len(images)} file")
    print(f"{'─'*55}\n")

    results_log = []
    t_start = time.time()
    total_count = 0

    for idx, img_path in enumerate(images, 1):
        out_path = output_dir / img_path.name
        result = process_image(yolo_model, face_mesh, img_path, out_path,
                               args.conf, args.show, refine, label_text=label_type)
        results_log.append(result)
        total_count += result["count"]

        # Xuất dữ liệu
        if export_fmt != "none" and result["landmarks"]:
            base_name = img_path.stem
            if export_fmt == "json":
                json_path = data_dir / f"{base_name}.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result["landmarks"], f, indent=2)
            elif export_fmt == "csv":
                csv_path = data_dir / f"{base_name}.csv"
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["face_id", "landmark_id", "x", "y", "z"])
                    for f_id, face_data in enumerate(result["landmarks"]):
                        for l_id, lm in enumerate(face_data["all"]):
                            writer.writerow([f_id, l_id, lm['x'], lm['y'], lm['z']])

        status_icon = "✓" if result["status"] == "OK" else "✗"
        print(f"  [{idx:>3}/{len(images)}] [{status_icon}] {result['file']:<35}  {result['count']} {label_type.lower()}")

    elapsed = time.time() - t_start
    ok_count = sum(1 for r in results_log if r["status"] == "OK")

    print(f"\n{'─'*55}")
    print(f"  ✅ Hoàn thành! ({elapsed:.1f}s)")
    print(f"  Đã xử lý : {ok_count}/{len(images)} ảnh")
    print(f"  Tổng {label_type.lower()} : {total_count} {label_type.lower()}")
    print(f"  Lưu tại  : {output_dir.resolve()}")
    print(f"{'─'*55}")
    
    face_mesh.close()

if __name__ == "__main__":
    main()
