"""
Face Parts Definitions for MediaPipe Face Mesh
==============================================
Định nghĩa các chỉ số (indices) và kết nối (connections) cho từng bộ phận khuôn mặt.
Cung cấp dữ liệu hỗ trợ vẽ lưới tam giác và nhận diện con ngươi (Iris).
"""

import mediapipe as mp

# MediaPipe face mesh connections shortcut
mp_face_mesh = mp.solutions.face_mesh

# 1. Chỉ số điểm (Indices) phân loại theo bộ phận
FACE_PARTS_INDICES = {
    "lips": [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
        185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78
    ],
    "left_eye": [
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
    ],
    "left_eyebrow": [
        70, 63, 105, 66, 107, 55, 65, 52, 53, 46
    ],
    "right_eye": [
        362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
    ],
    "right_eyebrow": [
        336, 296, 334, 293, 300, 276, 283, 282, 295, 285
    ],
    "nose": [
        1, 2, 98, 327, 168, 6, 197, 195, 5, 4, 45, 275, 440, 220, 115, 344, 102, 331, 218, 438
    ],
    "face_oval": [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 
        152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    ],
    "iris": [
        468, 469, 470, 471, 472, # Left iris
        473, 474, 475, 476, 477  # Right iris
    ]
}

# 2. Màu sắc hiển thị gợi ý cho từng bộ phận (BGR)
FACE_PARTS_COLORS = {
    "lips": (0, 0, 255),          # Đỏ
    "left_eye": (255, 0, 0),      # Xanh dương
    "left_eyebrow": (0, 255, 255), # Vàng
    "right_eye": (255, 0, 0),
    "right_eyebrow": (0, 255, 255),
    "nose": (0, 255, 0),          # Xanh lá
    "face_oval": (255, 255, 255),  # Trắng
    "iris": (255, 255, 0)         # Cyan (Xanh lơ)
}

# 3. Connections (Kết nối) để vẽ lưới Mesh
# Chúng ta sử dụng TESSELATION mặc định của MediaPipe cho lưới tam giác
MESH_CONNECTIONS = mp_face_mesh.FACEMESH_TESSELATION
IRIS_CONNECTIONS = mp_face_mesh.FACEMESH_IRISES
