"""
MediaPipe FaceMesh Helper - Persistent process for C++ IPC.
Protocol: C++ writes image path to stdin, Python writes landmarks to stdout.
Format: "OK x1 y1 z1 wx1 wy1 wz1|x2 y2 z2 wx2 wy2 wz2|..." (468 landmarks)
     or "FAIL" if no face found.
"""
import sys, cv2
import numpy as np

# Preprocess (giong preprocess_face trong Python)
def preprocess_face(crop, min_dim=384):
    if crop is None or crop.size == 0:
        return crop, 0, 0
    orig_h, orig_w = crop.shape[:2]
    current_max = max(orig_h, orig_w)
    if current_max < min_dim:
        scale = min_dim / current_max
        crop = cv2.resize(crop, (int(orig_w*scale), int(orig_h*scale)), interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    cl = clahe.apply(l)
    crop = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    kernel = np.array([[0,-0.5,0],[-0.5,3,-0.5],[0,-0.5,0]])
    crop = cv2.filter2D(crop, -1, kernel)
    return crop, orig_w, orig_h

def main():
    import mediapipe as mp
    fm = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5,
        min_tracking_confidence=0.60
    )
    
    # Signal ready
    print("READY", flush=True)
    
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            path = line.strip()
            if path == "EXIT":
                break
            
            img = cv2.imread(path)
            if img is None:
                print("FAIL", flush=True)
                continue
            
            proc, ow, oh = preprocess_face(img)
            result = fm.process(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))
            
            if not result.multi_face_landmarks:
                print("FAIL", flush=True)
                continue
            
            face = result.multi_face_landmarks[0]
            wface = None
            if hasattr(result, 'multi_face_world_landmarks') and result.multi_face_world_landmarks:
                wface = result.multi_face_world_landmarks[0]
            
            parts = []
            for i, lm in enumerate(face.landmark):
                if wface:
                    wlm = wface.landmark[i]
                    parts.append(f"{lm.x} {lm.y} {lm.z} {wlm.x} {wlm.y} {wlm.z}")
                else:
                    parts.append(f"{lm.x} {lm.y} {lm.z} {lm.x} {-lm.y} {lm.z}")
            
            print("OK " + "|".join(parts), flush=True)
            
        except Exception as e:
            print("FAIL", flush=True)
    
    fm.close()

if __name__ == "__main__":
    main()
