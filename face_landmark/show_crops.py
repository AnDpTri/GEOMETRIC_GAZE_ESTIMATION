"""
Show Crops - Face Landmark Debugger
===================================
Hiển thị trực tiếp vùng ảnh sau khi được YOLOv8 crop và nới rộng 30%, 
kèm theo các điểm landmark được phát hiện bởi MediaPipe.

Cách dùng:
    python show_crops.py --input path/to/images
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path

# Thêm đường dẫn gốc vào sys.path để import config
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import config
import mediapipe as mp

# Thêm path yolov8_head để dùng lại các hàm tải model YOLO
sys.path.append(str(ROOT_DIR / "yolov8_head"))
from ultralytics import YOLO
from model_manager import get_model_path

from landmark_model import get_face_mesh_static

# Drawing Utils
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def main():
    parser = argparse.ArgumentParser(description="Show Crops & Landmarks Debugger")
    parser.add_argument("--input", default="input", help="Thư mục ảnh đầu vào")
    parser.add_argument("--model", default="n", help="Model YOLOv8 (n, s, m, l, x)")
    parser.add_argument("--conf", type=float, default=0.5, help="Ngưỡng YOLOv8")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"[✗] Không thấy thư mục: {input_dir}")
        return

    # 1. Load Models
    model_path = get_model_path(args.model if args.model != 'n' else config.YOLO_MODEL_SIZE)
    print(f"[*] Đang nạp YOLOv8 (Device: {config.get_device()}) ...")
    yolo_model = YOLO(str(model_path)).to(config.get_device())
    
    face_mesh = get_face_mesh_static(
        max_num_faces=config.MP_MAX_FACES, 
        refine_landmarks=config.MP_REFINE_LANDMARKS
    )

    input_dir = Path(args.input) if args.input != 'input' else config.INPUT_DIR
    
    print(f"[*] Tìm thấy {len(img_files)} ảnh. Nhấn phím bất kỳ để xem ảnh tiếp theo, ESC để thoát.")

    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        orig_h, orig_w = img.shape[:2]
        results = yolo_model(img, conf=args.conf, verbose=False)[0]

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                
                # Tính toán padding dựa trên config
                w_box, h_box = x2 - x1, y2 - y1
                pad_w = int(w_box * config.FACE_PAD_RATIO)
                pad_h = int(h_box * config.FACE_PAD_RATIO)
                
                px1 = max(0, x1 - pad_w)
                py1 = max(0, y1 - pad_h)
                px2 = min(orig_w, x2 + pad_w)
                py2 = min(orig_h, y2 + pad_h)
                
                # Cắt mặt
                face_crop = img[py1:py2, px1:px2].copy()
                if face_crop.size == 0: continue
                
                # Chạy MediaPipe trên vùng crop
                rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                mp_results = face_mesh.process(rgb_crop)
                
                if mp_results.multi_face_landmarks:
                    for face_landmarks in mp_results.multi_face_landmarks:
                        # Vẽ lưới tam giác
                        mp_drawing.draw_landmarks(
                            image=face_crop,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                        
                        # Vẽ viền và Iris
                        mp_drawing.draw_landmarks(
                            image=face_crop,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                        
                        mp_drawing.draw_landmarks(
                            image=face_crop,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

                # Hiển thị
                title = f"Crop {i+1}: {img_path.name} (15% padding)"
                cv2.imshow(title, face_crop)
                
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if key == 27: # ESC
            break

    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == "__main__":
    main()
