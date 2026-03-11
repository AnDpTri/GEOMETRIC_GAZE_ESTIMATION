"""
Batch Landmark Processor
========================
Xử lý hàng loạt ảnh: YOLO (Face/Head) -> MediaPipe -> Triangulation Mesh -> JSON Export.
"""

import cv2
import sys
import json
import torch
from pathlib import Path
import mediapipe as mp

# Setup Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import config
from ultralytics import YOLO
from face_landmark.landmark_model import get_face_mesh
from face_landmark.face_parts import FACE_PARTS_INDICES, FACE_PARTS_COLORS, MESH_CONNECTIONS, IRIS_CONNECTIONS

def process_batch():
    # 1. Setup Environment
    device = config.get_device()
    input_dir = config.INPUT_DIR
    output_dir = config.CURRENT_OUTPUT_DIR
    data_dir = config.CURRENT_DATA_DIR
    
    print(f"[*] Starting Batch Processing (Type: {config.MODEL_TYPE})")
    
    # 2. Load Detection Model
    model_path = config.YOLO_FACE_MODEL if config.MODEL_TYPE == 'face' else config.YOLO_HEAD_MODEL
    print(f"[*] Using Detection Model: {model_path.name}")
    yolo_model = YOLO(str(model_path)).to(device)
    
    # 3. Load MediaPipe (Static Mode)
    face_mesh = get_face_mesh(static_image_mode=True)
    
    # 4. Get Image List
    images = list(input_dir.glob("*.[jJ][pP][gG]")) + list(input_dir.glob("*.[pP][nN][gG]"))
    print(f"[*] Found {len(images)} images to process.")

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None: continue
        orig_h, orig_w = frame.shape[:2]
        
        # --- PHASE 1: Detection & Fallback ---
        results = yolo_model(frame, conf=config.YOLO_CONF_THRESHOLD, imgsz=640, verbose=False)[0]
        
        # Smart Fallback: If Head fails, try Face and expand upwards
        if (results.boxes is None or len(results.boxes) == 0) and config.MODEL_TYPE == 'head':
            face_results = YOLO(str(config.YOLO_FACE_MODEL)).to(device)(frame, conf=0.3, imgsz=640, verbose=False)[0]
            if face_results.boxes is not None and len(face_results.boxes) > 0:
                print(f"[Fallback] Face found for {img_path.name}, expanding box.")
                results = face_results
                for box in results.boxes:
                    coords = box.xyxy[0].tolist()
                    h_box = coords[3] - coords[1]
                    coords[1] = max(0, coords[1] - h_box * 0.4) # Expand 40% up
                    box.xyxy[0] = torch.tensor(coords).to(device)

        # --- PHASE 2: Landmark Processing ---
        all_landmarks_data = []
        if results.boxes is not None and len(results.boxes) > 0:
            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(orig_w, x2), min(orig_h, y2)
                
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: continue
                
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                mp_results = face_mesh.process(rgb_crop)
                
                if mp_results.multi_face_landmarks:
                    face_landmarks = mp_results.multi_face_landmarks[0]
                    h_c, w_c = crop.shape[:2]
                    
                    # 2.1 Categorize and Scale Landmarks
                    parts_json = {}
                    pixel_coords = {}
                    
                    for part_name, indices in FACE_PARTS_INDICES.items():
                        part_list = []
                        for idx in indices:
                            # a. Regular Landmarks -> ĐƯA VỀ CHUẨN PIXEL ĐỂ X, Y, Z ĐỒNG NHẤT TỈ LỆ
                            if idx < len(face_landmarks.landmark):
                                lm = face_landmarks.landmark[idx]
                                
                                # X, Y tính theo Pixel trên ảnh gốc
                                px = (lm.x * w_c) + x1
                                py = (lm.y * h_c) + y1
                                # Z của MediaPipe tỉ lệ thuận với width của crop, quy đổi ra Pixel
                                pz = lm.z * w_c 
                                
                                part_list.append({"x": px, "y": py, "z": pz})
                                pixel_coords[idx] = (int(px), int(py))
                        
                        parts_json[part_name] = part_list
                    
                    all_landmarks_data.append({
                        "id": i, 
                        "parts": parts_json
                    })
                    
                    # 2.2 Draw Mesh Triangulation
                    for conn in MESH_CONNECTIONS:
                        s_idx, e_idx = conn
                        if s_idx in pixel_coords and e_idx in pixel_coords:
                            # Color by starting point's part
                            color = (128, 128, 128) # Gray default
                            for part, p_indices in FACE_PARTS_INDICES.items():
                                if s_idx in p_indices:
                                    color = FACE_PARTS_COLORS[part]
                                    break
                            cv2.line(frame, pixel_coords[s_idx], pixel_coords[e_idx], color, 1)
                    
                    # 2.3 Drawing IRIS Connections (Làm nổi bật con ngươi)
                    for conn in IRIS_CONNECTIONS:
                        s_idx, e_idx = conn
                        if s_idx in pixel_coords and e_idx in pixel_coords:
                            cv2.line(frame, pixel_coords[s_idx], pixel_coords[e_idx], FACE_PARTS_COLORS["iris"], 1)
                            # Vẽ tâm con ngươi to hơn
                            if s_idx in [468, 473]: 
                                cv2.circle(frame, pixel_coords[s_idx], 2, (255, 255, 255), -1) # Tâm trắng

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # --- PHASE 3: Save Results ---
        cv2.imwrite(str(output_dir / img_path.name), frame)
        if all_landmarks_data:
            with open(data_dir / f"{img_path.stem}.json", 'w') as f:
                json.dump(all_landmarks_data, f, indent=2)
        
        print(f"[✓] Processed: {img_path.name}")

    face_mesh.close()
    print("[*] Batch Processing Completed.")

if __name__ == "__main__":
    process_batch()
