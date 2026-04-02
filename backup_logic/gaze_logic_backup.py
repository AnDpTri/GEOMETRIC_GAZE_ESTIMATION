import numpy as np

"""
BẢN SAO LƯU LOGIC CỐT LÕI - GEOMETRIC GAZE ESTIMATION (3D MODEL)
--------------------------------------------------------------
Tài liệu này lưu trữ các hàm tính toán hình học quan trọng nhất, 
tách biệt khỏi phần giao diện (OpenCV/Menu) để bảo soát thuật toán.

Mô hình hiện tại: True 3D Eyeball Model (Raycasting qua Tâm Nhãn Cầu).
"""

# ------------------------------------------------------------------ #
#  1. HỆ TỌA ĐỘ KHUÔN MẶT (FACE BASIS)                                #
# ------------------------------------------------------------------ #

def step1_get_face_basis(p168, p2, p331, p102):
    """
    Tạo bộ ba trực chuẩn [Vf, Rf, Uf] xác định hướng của khuôn mặt.
    - p168: Ấn đường (Glabella)
    - p2: Chóp mũi (Nose tip)
    - p331, p102: Hai điểm mốc hốc mắt (để lấy pháp tuyến mặt)
    """
    # 1. Trục Dọc Tham Chiếu (Từ Chóp mũi 2 -> Ấn đường 168)
    U_ref = p168 - p2
    U_ref /= (np.linalg.norm(U_ref) + 1e-9)
    
    # 2. Vector Nhìn Thẳng Cơ Sở (Pháp tuyến mặt phẳng mặt)
    nf = np.cross(p331 - p168, p102 - p168)
    Vf = -nf / np.linalg.norm(nf) if np.linalg.norm(nf) > 1e-6 else np.array([0,0,-1.])
    if Vf[2] > 0: Vf = -Vf # Z hướng âm (về phía camera)
    
    # 3. Trục OX - Ngang Mặt (Vuông góc với Dọc tham chiếu và Hướng nhìn)
    Rf = np.cross(U_ref, Vf)
    Rf /= (np.linalg.norm(Rf) + 1e-9)
    
    # 4. Trục OY - Dọc Mặt (Vuông góc với Ngang mặt và Hướng nhìn)
    Uf = np.cross(Vf, Rf)
    Uf /= (np.linalg.norm(Uf) + 1e-9)
    
    return Vf, Rf, Uf

# ------------------------------------------------------------------ #
#  2. TÂM NHÃN CẦU 3D (EYEBALL CENTER)                                #
# ------------------------------------------------------------------ #

def step2_find_true_eyeball_center(P_top, P_bottom, P_inner, P_outer, V_face):
    """
    Ước lượng tâm nhãn cầu O nằm sâu trong hốc mắt.
    V_face được dùng để xác định hướng lùi vào (chiều sâu).
    """
    # Tâm bề mặt khe hở (trung bình 4 điểm mí mắt)
    O_surface = (P_top + P_bottom + P_inner + P_outer) / 4.0
    
    # Ước lượng bán kính nhãn cầu (tương đối theo độ rộng mắt)
    iris_radius_approx = np.linalg.norm(P_outer - P_inner) * 0.4
    
    # Lùi sâu vào trong (ngược hướng V_face)
    O_eyeball = O_surface - V_face * iris_radius_approx
    
    return O_eyeball

# ------------------------------------------------------------------ #
#  3. TÍNH TOÁN GAZE (YAW / PITCH)                                    #
# ------------------------------------------------------------------ #

def calculate_gaze(world_landmarks):
    """
    Logic Raycasting: Tia nhìn = Đồng tử - Tâm nhãn cầu.
    world_landmarks: Dict chứa tọa độ 3D (idx: np.array([x,y,z]))
    """
    def _pt(idx): return world_landmarks.get(idx)

    p = {'g': _pt(168), 'n': _pt(2), 'a': _pt(331), 'b': _pt(102)}
    if any(v is None for v in p.values()): return None

    # Lấy hệ trục mặt
    V_face, Rf, Uf = step1_get_face_basis(p['g'], p['n'], p['a'], p['b'])
    
    # Landmark mắt trái (163, 157, 161, 154 | Đồng tử: 468)
    # Landmark mắt phải (390, 384, 388, 381 | Đồng tử: 473)
    eyeL = [_pt(i) for i in (163, 157, 161, 154, 468)]
    eyeR = [_pt(i) for i in (390, 384, 388, 381, 473)]
    
    if any(pt is None for pt in eyeL) or any(pt is None for pt in eyeR):
        return None

    # Tính hướng nhìn từng mắt
    O_L = step2_find_true_eyeball_center(eyeL[0], eyeL[1], eyeL[2], eyeL[3], V_face)
    gaze_L = (eyeL[4] - O_L); gaze_L /= np.linalg.norm(gaze_L)

    O_R = step2_find_true_eyeball_center(eyeR[0], eyeR[1], eyeR[2], eyeR[3], V_face)
    gaze_R = (eyeR[4] - O_R); gaze_R /= np.linalg.norm(gaze_R)

    # Trung bình cộng vector hướng nhìn
    V_final = (gaze_L + gaze_R) / 2.0
    V_final /= np.linalg.norm(V_final)
    
    # Chuyển sang Yaw/Pitch (độ)
    # Yaw: Xoay ngang (arctan2 X / -Z)
    # Pitch: Ngước lên/xuống (arcsin -Y)
    pitch = np.degrees(np.arcsin(-V_final[1]))
    yaw = np.degrees(np.arctan2(V_final[0], -V_final[2]))
    
    return yaw, pitch, V_final

# ------------------------------------------------------------------ #
#  4. TRACKING LOGIC (KALMAN + IOU)                                   #
# ------------------------------------------------------------------ #

class KalmanBoxTracker:
    """Theo dõi Bounding Box bằng bộ lọc Kalman để mượt khung cắt (Crop)."""
    def __init__(self, bbox):
        from filterpy.kalman import KalmanFilter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # Ma trận chuyển trạng thái F, quan sát H (chi tiết xem trong code chính)
        # [x, y, s, r, vx, vy, vs]
        self.kf.x[:4] = self._box_to_z(bbox)
        self.id = -1 # Sẽ được gán bởi FaceTracker
        self.smooth_gaze = None

    def _box_to_z(self, b):
        w, h = b[2]-b[0], b[3]-b[1]
        return np.array([b[0]+w/2., b[1]+h/2., w*h, w/float(h)]).reshape((4,1))

    # predict() và update() thực hiện dự báo vị trí hộp ở frame tiếp theo

def calculate_iou(b1, b2):
    """Giao trên diện tích (Intersection over Union) để khớp ID."""
    xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
    xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1, area2 = (b1[2]-b1[0])*(b1[3]-b1[1]), (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / float(area1 + area2 - inter + 1e-6)

if __name__ == "__main__":
    print("BACKUP LOGIC: Đã cập nhật mô hình 3D Eyeball chuẩn.")
    print("Sử dụng các hàm step1, step2 và calculate_gaze cho các dự án khác.")
