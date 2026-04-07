# Hướng dẫn Cấu hình & Chạy Gaze Estimation (PC & Raspberry Pi 4)

Dự án hình học ước lượng hướng nhìn (Geometric Gaze Estimation) hỗ trợ đa nền tảng, tự động nhận diện phần cứng để tối ưu hiệu năng.

## 1. Yêu cầu Hệ thống

### Trên Máy tính (PC / Laptop)
- **HĐH**: Windows 10/11 hoặc Linux x86_64.
- **GPU**: NVIDIA (Khuyên dùng để chạy ONNX GPU).
- **Yêu cầu**: Cài đặt CUDA Toolkit 11.8+ và cuDNN nếu muốn dùng GPU.

### Trên Raspberry Pi 4
- **HĐH**: Raspberry Pi OS (Hệ điều hành 64-bit khuyên dùng) hoặc Ubuntu 22.04 AArch64.
- **RAM**: Tối thiểu 4GB.
- **Yêu cầu**: Cài đặt các thư viện hệ thống cho OpenCV (xem mục 3).

---

## 2. Cài đặt nhanh trên Windows (PC)

Bạn có thể chạy tệp `setup_pc.bat` hoặc thực hiện thủ công:

1. Tạo môi trường ảo:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
2. Cài đặt thư viện:
   ```bash
   pip install -r requirements.txt
   ```
3. Chạy chương trình:
   ```bash
   python gaze_estimation.py
   ```

---

## 3. Cài đặt trên Raspberry Pi 4

Dự án tự động nhận diện RPi để chuyển sang chế độ tiết kiệm tài nguyên (CPU-only, High FPS Mode).

1. Cài đặt thư viện hệ thống (Bắt buộc):
   ```bash
   sudo apt-get update
   sudo apt-get install -y libopencv-dev libatlas-base-dev libhdf5-dev libqt5gui5 libqt5test5
   ```
2. Tạo môi trường ảo và cài đặt:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install opencv-python mediapipe ultralytics filterpy scipy
   ```
3. Chạy chương trình:
   ```bash
   python3 gaze_estimation.py
   ```

---

## 4. Các tính năng chính

- **Tự động cấu hình (Auto-Setup)**: 
  - Khi chạy trên PC có CUDA: Hệ thống bật ONNX-GPU cho tốc độ cực cao.
  - Khi chạy trên RPi: Hệ thống tự chuyển sang MediaPipe TFLite và bật `High FPS Mode` để đảm bảo video không bị giật.
- **Chế độ Video**: Vì RPi có thể không có camera, bạn nên dùng phím **5** để xử lý các tệp video trong thư mục `input/video/`.
- **Thông số chi tiết**: Nhấn phím **D** trong menu chính để xem báo cáo về phần cứng đang chạy.

## 5. Cấu trúc thư mục
- `input/`: Chứa ảnh đầu vào.
- `input/video/`: Chứa các video `.mp4`, `.avi` để xử lý theo lô.
- `output/`: Kết quả xử lý ảnh và video.
- `data/`: Nhật ký tọa độ hướng nhìn (CSV).

---
*Chúc bạn có trải nghiệm tốt nhất với dự án!*
