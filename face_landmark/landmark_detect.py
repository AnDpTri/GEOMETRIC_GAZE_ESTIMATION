"""
Live Landmark Detection (Webcam)
================================
Xử lý thời gian thực: Camera -> YOLO -> MediaPipe -> Mesh + Gaze Visualization.
"""

import time
import cv2
import sys
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
# Import hàm Gaze Vector dùng chung
from face_landmark.batch_landmark import calculate_gaze_vector, draw_gaze_line_2d

def run_live():
    # 1. Setup Environment
    device = config.get_device()
    print(f"[*] Starting Live Mode (Type: {config.MODEL_TYPE}, Device: {device})")
    
    # 2. Load Models
    model_path = config.YOLO_FACE_MODEL if config.MODEL_TYPE == 'face' else config.YOLO_HEAD_MODEL
    yolo_model = YOLO(str(model_path)).to(device)
    face_mesh = get_face_mesh(static_image_mode=False) # Video stream mode
    
    # 3. Setup Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[✗] Cannot open webcam.")
        return

    print("[*] Streaming! Press 'ESC' to quit.")
    pTime = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        orig_h, orig_w = frame.shape[:2]
        
        # --- Detection & Smart Fallback ---
        results = yolo_model(frame, conf=config.YOLO_CONF_THRESHOLD, imgsz=640, verbose=False)[0]
        
        if (results.boxes is None or len(results.boxes) == 0) and config.MODEL_TYPE == 'head':
            face_results = YOLO(str(config.YOLO_FACE_MODEL)).to(device)(frame, conf=0.3, imgsz=640, verbose=False)[0]
            if face_results.boxes is not None and len(face_results.boxes) > 0:
                results = face_results
                for box in results.boxes:
                    coords = box.xyxy[0].tolist()
                    h_box = coords[3] - coords[1]
                    coords[1] = max(0, coords[1] - h_box * 0.4)
                    box.xyxy[0] = torch.tensor(coords).to(device)

        # --- Processing & Visualization ---
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(orig_w, x2), min(orig_h, y2)
                
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: continue
                h_c, w_c = crop.shape[:2]
                
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                mp_results = face_mesh.process(rgb_crop)
                
                if mp_results.multi_face_landmarks:
                    face_landmarks = mp_results.multi_face_landmarks[0]
                    world_landmarks = getattr(mp_results, 'multi_face_world_landmarks', None)
                    world_landmarks = world_landmarks[0] if world_landmarks else None
                    pixel_coords = {}
                    world_coords = {}
                    
                    # Lưu và Vẽ điểm Landmark
                    for part_name, indices in FACE_PARTS_INDICES.items():
                        for idx in indices:
                            if idx < len(face_landmarks.landmark):
                                lm = face_landmarks.landmark[idx]
                                px = int(lm.x * w_c + x1)
                                py = int(lm.y * h_c + y1)
                                pixel_coords[idx] = (px, py)
                                
                                # Highlight Iris
                                if part_name == "iris":
                                    cv2.circle(frame, (px, py), 2, FACE_PARTS_COLORS[part_name], -1)
                                else:
                                    cv2.circle(frame, (px, py), 1, FACE_PARTS_COLORS[part_name], -1)
                            
                            # Lưu World Landmarks cho Gaze
                            if world_landmarks and idx < len(world_landmarks.landmark):
                                w_lm = world_landmarks.landmark[idx]
                                world_coords[idx] = (w_lm.x, w_lm.y, w_lm.z)

                    # Vẽ Mesh
                    for conn in MESH_CONNECTIONS:
                        s_idx, e_idx = conn
                        if s_idx in pixel_coords and e_idx in pixel_coords:
                            color = (130, 130, 130)
                            for part, p_indices in FACE_PARTS_INDICES.items():
                                if s_idx in p_indices:
                                    color = FACE_PARTS_COLORS[part]
                                    break
                            cv2.line(frame, pixel_coords[s_idx], pixel_coords[e_idx], color, 1)

                    # Vẽ Iris
                    for conn in IRIS_CONNECTIONS:
                        s_idx, e_idx = conn
                        if s_idx in pixel_coords and e_idx in pixel_coords:
                            cv2.line(frame, pixel_coords[s_idx], pixel_coords[e_idx], FACE_PARTS_COLORS["iris"], 1)
                            if s_idx in [468, 473]:
                                cv2.circle(frame, pixel_coords[s_idx], 2, (255, 255, 255), -1)

                    # Vẽ Gaze Arrow (Mũi tên hướng nhìn Real-time)
                    # Tạo fallback map cho gaze nếu world_coords rỗng
                    norm_coords = {}
                    if not world_coords:
                        nose_indices = FACE_PARTS_INDICES.get("nose", [])
                        for idx in nose_indices:
                            if idx in pixel_coords:
                                # Lấy lại giá trị normalized từ landmark (đã lưu trong loop trên)
                                lm = face_landmarks.landmark[idx]
                                norm_coords[idx] = (lm.x, lm.y, lm.z)
                    
                    draw_gaze_line_2d(frame, world_coords, pixel_coords, w_c, h_c, x1, y1, orig_w, orig_h, normalized_map=norm_coords)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Calculate and Display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)

        # Show Output
        cv2.imshow("MMPOSE: Live Gaze Ready Landmarks", frame)
        if cv2.waitKey(1) & 0xFF == 27: # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == "__main__":
    run_live()
