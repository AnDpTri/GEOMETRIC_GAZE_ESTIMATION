"""
MediaPipe Face Mesh Factory
===========================
Hàm tiện ích khởi tạo MediaPipe Face Mesh từ cấu hình config.
"""

import mediapipe as mp
import config

def get_face_mesh(static_image_mode: bool = False):
    """
    Khởi tạo đối tượng FaceMesh.
    Args:
        static_image_mode: True cho xử lý ảnh tĩnh (Batch), False cho video (Webcam).
    """
    mp_face_mesh = mp.solutions.face_mesh
    
    return mp_face_mesh.FaceMesh(
        static_image_mode=static_image_mode,
        max_num_faces=config.MP_MAX_FACES,
        refine_landmarks=config.MP_REFINE_LANDMARKS,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
