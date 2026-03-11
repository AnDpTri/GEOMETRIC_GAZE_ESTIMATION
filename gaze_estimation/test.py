"""
MMPOSE 3D Visualizer (Clean Rebuild)
====================================
Render Face Landmarks 3D dựa trên hệ thống Face Parts.
Đồng bộ hoàn toàn màu sắc và cấu trúc với batch_landmark.py.
"""

import json
import csv
import argparse
import sys
import io
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Setup Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import config
from face_landmark.face_parts import FACE_PARTS_INDICES, FACE_PARTS_COLORS, MESH_CONNECTIONS, IRIS_CONNECTIONS

# Fix Windows UTF-8
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def load_data(file_path: Path):
    """Tải dữ liệu JSON Gaze-Ready."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # data là list các mặt: [{"id": 0, "parts": {...}}]
            return data
    except Exception as e:
        print(f"[✗] Lỗi đọc file: {e}")
        return None

def calculate_target_point(p168, p331, p102, k=0.1):
    """Tính điểm đích của Vector pháp tuyến từ 3 điểm."""
    try:
        p168, p331, p102 = np.array(p168), np.array(p331), np.array(p102)
        v1 = p331 - p168
        v2 = p102 - p168
        # Tích có hướng v1 x v2 theo chuẩn Right-Hand-Rule (với hệ tọa độ mới) sẽ hướng ra phía trước (phía camera)
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm == 0: return p168
        n_unit = n / norm
        return p168 + k * n_unit
    except:
        return p168

def plot_3d(faces_data, title, show_indices=False):
    """Render 3D chuyên nghiệp với Gaze Vector."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('#050505')
    ax.set_facecolor('#050505')

    def bgr_to_hex(bgr):
        return '#{:02x}{:02x}{:02x}'.format(bgr[2], bgr[1], bgr[0])

    PART_HEX_COLORS = {k: bgr_to_hex(v) for k, v in FACE_PARTS_COLORS.items()}

    for face in faces_data:
        parts = face.get("parts", {})
        if not parts: continue

        # Tạo map index -> tọa độ (Chuyển sang Tọa độ Vật lý Chuẩn)
        all_points_map = {}
        for part_name, landmarks in parts.items():
            indices = FACE_PARTS_INDICES.get(part_name, [])
            for i, lm in enumerate(landmarks):
                if i < len(indices):
                    idx = indices[i]
                    # --- BÍ QUYẾT TẠO 3D THỰC TẾ ---
                    # MediaPipe: X (Phải+), Y (Xuống+), Z (Gần-)
                    # Chuẩn 3D Toán học/Matplotlib: X (Ngang/Phải+), Y (Sâu/Tiến+), Z (Cao/Lên+)
                    std_x = lm['x']
                    std_y = -lm['z'] # Đảo Z thành Y: Z âm (gần camera) -> Y dương (nhô ra trước)
                    std_z = -lm['y'] # Đảo Y thành Z: Y lớn (xuống cằm) -> Z âm (thấp xuống)
                    
                    all_points_map[idx] = (std_x, std_y, std_z)

        # --- BƯỚC 1: Tính và Vẽ Gaze Vector (Normal Vector từ 168) ---
        if 168 in all_points_map and 331 in all_points_map and 102 in all_points_map:
            p168 = all_points_map[168]
            p331 = all_points_map[331]
            p102 = all_points_map[102]
            
            # Hệ tọa độ hiện tại là Tọa độ Pixel nguyên gốc. 
            # k_value cần lớn (ví dụ 100-150 pixels) để đường thẳng dài tương xứng trong không gian 3D.
            k_value = 150
            target = calculate_target_point(p168, p331, p102, k=k_value)
            
            # Vẽ đường thẳng pháp tuyến (Màu Vàng Neon)
            ax.plot([p168[0], target[0]], [p168[1], target[1]], [p168[2], target[2]], 
                    c='#FFFF00', linewidth=3, alpha=1.0, label='Normal Vector (Gaze)')
            
            # --- VẼ ĐIỂM ĐÍCH (TARGET POINT) ---
            ax.scatter([target[0]], [target[1]], [target[2]], c='#FF0000', s=30, edgecolors='white', zorder=11, label='Target Point')
            
            # --- VẼ TAM GIÁC THAM CHIẾU (168, 331, 102) XANH MỜ ---
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            verts = [[(p168[0], p168[1], p168[2]), 
                      (p331[0], p331[1], p331[2]), 
                      (p102[0], p102[1], p102[2])]]
            tri = Poly3DCollection(verts, alpha=0.2, facecolors='#00FFFF', edgecolors='#00FFFF', linewidths=0.5)
            ax.add_collection3d(tri)
            
            # Vẽ điểm mồi súng 168 to hơn
            ax.scatter([p168[0]], [p168[1]], [p168[2]], c='#FFFF00', s=50, edgecolors='white', zorder=10)

        # --- BƯỚC 2: Vẽ điểm và Chỉ số ---
        for part_name, indices in FACE_PARTS_INDICES.items():
            p_xs, p_ys, p_zs = [], [], []
            color = PART_HEX_COLORS.get(part_name, "#ffffff")
            s = 15 if part_name == "iris" else 4
            
            for idx in indices:
                if idx in all_points_map:
                    p = all_points_map[idx]
                    p_xs.append(p[0])
                    p_ys.append(p[1])
                    p_zs.append(p[2])
                    if show_indices:
                        ax.text(p[0], p[1], p[2], str(idx), color='white', fontsize=6, alpha=0.7)
            
            if p_xs:
                ax.scatter(p_xs, p_ys, p_zs, c=color, s=s, edgecolors='white', linewidths=0.2, alpha=0.9)

        # --- BƯỚC 3: Vẽ lưới Tesselation ---
        for conn in MESH_CONNECTIONS:
            s_idx, e_idx = conn
            if s_idx in all_points_map and e_idx in all_points_map:
                p1, p2 = all_points_map[s_idx], all_points_map[e_idx]
                color = "#222222"
                for part_name, part_idx_list in FACE_PARTS_INDICES.items():
                    if s_idx in part_idx_list:
                        color = PART_HEX_COLORS.get(part_name, color)
                        break
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                        c=color, linewidth=0.3, alpha=0.3)

        # --- BƯỚC 4: Vẽ Iris Connections ---
        for conn in IRIS_CONNECTIONS:
            s_idx, e_idx = conn
            if s_idx in all_points_map and e_idx in all_points_map:
                p1, p2 = all_points_map[s_idx], all_points_map[e_idx]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                        c=PART_HEX_COLORS["iris"], linewidth=1.5, alpha=1.0)

    # Cấu hình không gian 3D chuẩn XYZ
    ax.set_title(f"MMPOSE 3D GAZE ENGINE (Standard XYZ)\n{title}", color='white', fontsize=12, fontweight='bold', pad=30)
    
    # Gán nhãn trục chuẩn
    ax.set_xlabel('X (Right / Left)', color='gray')
    ax.set_ylabel('Y (Depth / Forward)', color='gray')
    ax.set_zlabel('Z (Vertical / Up)', color='gray')
    
    # Góc nhìn mặc định (Nhìn chéo từ trên xuống như các phần mềm 3D)
    ax.view_init(elev=20, azim=-45) 
    
    # --- CÀI ĐẶT ISOMETRIC 3D (CHỐNG MÉO TỈ LỆ VÀ GÓC KHI XOAY) ---
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range) / 2.0

    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)

    # Đảo ngược mảng đỉnh cho trục Y (chuẩn ảnh 0 ở trên)
    xb = [mid_x - max_range, mid_x + max_range]
    yb = [mid_y + max_range, mid_y - max_range]
    zb = [mid_z - max_range, mid_z + max_range]

    # Vẽ bounding box tàng hình (alpha=0.0) ở 8 góc để ép AutoScaler của Matplotlib tạo khối lập phương hoàn hảo
    for xi in xb:
        for yi in yb:
            for zi in zb:
                ax.plot([xi], [yi], [zi], 'w', alpha=0.0)

    # Đưa tất cả các trục về cùng một biên độ (Max Range)
    ax.set_xlim3d(xb)
    ax.set_ylim3d(yb)
    ax.set_zlim3d(zb)
    
    # Ép khung 3D thành hình lập phương (tỉ lệ 1:1:1) và TẮT TÍNH NĂNG TỰ ĐỘNG CO DÃN
    ax.set_box_aspect([1, 1, 1])
    ax.autoscale(False)
    
    # Hiển thị lưới trục mờ để dễ định hướng
    ax.grid(True, linestyle='--', alpha=0.1)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # --- TÍNH NĂNG ZOOM BẰNG CUỘN CHUỘT ---
    def on_scroll(event):
        base_scale = 1.1
        if event.button == 'up':
            # Zoom in
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            # Zoom out
            scale_factor = base_scale
        else:
            return

        # Lấy giới hạn hiện tại
        curr_xlim = ax.get_xlim()
        curr_ylim = ax.get_ylim()
        curr_zlim = ax.get_zlim()

        def get_new_lims(curr_lim, scale):
            center = (curr_lim[0] + curr_lim[1]) / 2
            half_width = (curr_lim[1] - curr_lim[0]) / 2
            new_half_width = half_width * scale
            return center - new_half_width, center + new_half_width

        ax.set_xlim3d(get_new_lims(curr_xlim, scale_factor))
        ax.set_ylim3d(get_new_lims(curr_ylim, scale_factor))
        ax.set_zlim3d(get_new_lims(curr_zlim, scale_factor))
        
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    
    print(f"[*] Rendering Completed: {title}")
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize Face Landmarks in 3D Wireframe")
    parser.add_argument("--file", type=str, help="File JSON hoặc CSV")
    parser.add_argument("--indices", action="store_true", help="Hiển thị số thứ tự của các điểm")
    args = parser.parse_args()

    base_data_dir = config.DATA_BASE
    categories = ["face", "head"]
    all_files = []

    print("\n" + "="*45)
    print(" MMPOSE 3D DATA SELECTOR ".center(45, " "))
    print("="*45)
    
    for cat in categories:
        cat_dir = base_data_dir / cat
        if cat_dir.exists():
            for f in sorted(cat_dir.glob("*.json")):
                all_files.append((cat, f))

    if not all_files:
        print("[!] Không thấy file JSON nào trong data/face hoặc data/head.")
        return

    for idx, (cat, f) in enumerate(all_files):
        print(f"  [{idx}]  [{cat.upper():<5}] {f.name}")
    
    print("="*45)
    try:
        choice = input(f"\nNhập số thứ tự file (0-{len(all_files)-1}) [Mặc định 0]: ").strip()
        file_idx = int(choice) if choice else 0
        cat, file_path = all_files[file_idx]
        
        # Hỏi thêm về việc hiển thị số thứ tự
        show_idx_input = input("Hiển thị số thứ tự điểm? (y/n) [Mặc định n]: ").strip().lower()
        show_indices = (show_idx_input == 'y')
        
    except (ValueError, IndexError):
        print("[!] Lựa chọn không hợp lệ, lấy file đầu tiên, không hiện số thứ tự.")
        cat, file_path = all_files[0]
        show_indices = False

    print(f"\n[*] Đang nạp dữ liệu {cat.upper()}...")
    data = load_data(file_path)
    if data:
        plot_3d(data, f"{cat.upper()} - {file_path.name}", show_indices=show_indices)

if __name__ == "__main__":
    main()
