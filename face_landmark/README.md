# Face Landmark Detection Module

Module này tận dụng **YOLOv8** (có sẵn tại `../yolov8_face/`) để cắt khuôn mặt với độ chính xác cao, sau đó chuyển vùng mặt cho **MediaPipe Face Mesh** vẽ 468 (hoặc 478) điểm chi tiết.

## Yêu cầu

1. Đã cài đặt thành công và cấu hình `yolov8_face` (có thư mục `venv`).
2. Cài thêm MediaPipe cho thư mục ảo đó:

```bash
cd d:\MMPOSE\yolov8_face
.\venv\Scripts\activate
pip install mediapipe
```

## Cách sử dụng

### 1. Real-time (Webcam)
Vào thư mục `face_landmark` và chạy:
```bash
run.bat
```
Hoặc cấu hình nâng cao:
```bash
run.bat --model s --conf 0.6 --no-refine
```

**Phím tắt**:
- `Q` / `ESC`: Thoát
- `S`: Chụp màn hình (Lưu vào `captures/`)
- `F`: Ẩn/hiển thị FPS
- `L`: Ẩn/hiển thị lưới Landmark
- `+` / `-`: Tăng/giảm ngưỡng niềm tin (Confidence) của YOLOv8

### 2. Batch Processing (Xử lý ảnh tĩnh)

Chạy trên toàn bộ ảnh đầu vào trong thư mục `input/` và lưu kết quả vào `output/`:
```bash
# Phải kích hoạt môi trường ảo trước
..\yolov8_face\venv\Scripts\activate

python batch_landmark.py
```

Sử dụng tham số để dùng chung thư mục input/output với YOLOv8:
```bash
python batch_landmark.py -i ..\yolov8_face\input -o .\output
```

### 3. Visualizer 3D (Nghiên cứu & Phát triển)
Để xem mô hình 3D của các điểm landmark đã xuất ra:
```bash
python visualize_3d.py
```
*(Mặc định sẽ lấy file đầu tiên trong thư mục `data/`)*

Hoặc chỉ định file cụ thể:
```bash
python visualize_3d.py --file data/image.json
```
**Tính năng**: Dùng chuột xoay, phóng to/thu nhỏ để quan sát chiều sâu (Z-axis) của các điểm trên khuôn mặt.
