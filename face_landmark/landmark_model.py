"""
MediaPipe Face Mesh Manager
===========================
Quản lý tải và khởi tạo model MediaPipe Face Mesh.
"""

import mediapipe as mp

def get_face_mesh(max_num_faces: int = 1, refine_landmarks: bool = True):
    """
    Khởi tạo và trả về đối tượng MediaPipe FaceMesh.
    
    Args:
        max_num_faces: Số lượng khuôn mặt tối đa cần phát hiện trong 1 frame.
        refine_landmarks: Nếu True, sẽ xuất ra 478 landmarks (thêm 10 điểm cho mắt/môi chính xác hơn).
                          Nếu False, xuất 468 landmarks.
                          
    Returns:
        mp.solutions.face_mesh.FaceMesh object.
    """
    mp_face_mesh = mp.solutions.face_mesh
    
    # Khởi tạo model
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,        # Dành cho video stream (True cho ảnh tĩnh)
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    return face_mesh

def get_face_mesh_static(max_num_faces: int = 1, refine_landmarks: bool = True):
    """
    Khởi tạo FaceMesh chuyên dụng cho ảnh tĩnh (batch processing).
    Độ trễ cao hơn nhưng chính xác hơn trên từng ảnh đơn lẻ.
    """
    mp_face_mesh = mp.solutions.face_mesh
    
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,         # Tính toán độc lập từng ảnh
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=0.5
    )
    
    return face_mesh
