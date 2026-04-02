# Task List: Gaze Estimation C++ Migration

- `[x]` **Giai đoạn 1: Base Environment & native YOLOv8** (HOÀN TẤT PHẦN MÃ NGUỒN)
  - `[x]` Dọn dẹp cấu trúc thư mục rác cũ.
  - `[x]` Cài đặt ONNX Runtime qua `vcpkg` (Đã hoàn tất).
  - `[x]` Cấu trúc lại `CMakeLists.txt` hỗ trợ ONNX Runtime + OpenCV + Eigen3.
  - `[x]` Viết Class `YOLOv8Detector` (Native ONNX Runtime).
  - `[x]` Build và kiểm chứng FPS trên CPU/GPU.

- `[x]` **Giai đoạn 2: Native FaceMesh (478 pts)** (HOÀN TẤT)
  - `[x]` Tải/Convert model FaceMesh ONNX chuẩn (478 pts).
  - `[x]` Viết Class `FaceMeshDetector` (Inference logic).
  - `[x]` Preprocessing: Crop vùng mặt từ Box của YOLO với 25% padding.
  - `[x]` Postprocessing: Trích xuất 478 điểm và ánh xạ về ảnh gốc.
  - `[x]` Vẽ Landmark 478 điểm lên khuôn mặt (Debug UI).

- `[x]` **Giai đoạn 3: Tracking & Eye Refinement** (HOÀN TẤT)
  - `[x]` Tích hợp Kalman Filter để làm mịn Landmark 478 điểm.
  - `[x]` Chống rung cho vùng Box và Landmark.
  - `[x]` Trích xuất điểm đồng tử (Iris) trực tiếp từ mesh model.

- `[x]` **Giai đoạn 4: Gaze Geometry Math** (HOÀN TẤT)
  - `[x]` Convert logic 3D raycasting sang C++ (sử dụng `Eigen3`).
  - `[x]` Tính toán hệ trục khuôn mặt (Face Basis) từ điểm 2, 168.
  - `[x]` Tính toán Yaw, Pitch, và Gaze Vector 3D.
  - `[x]` Smoothing phụ tống cho góc Yaw/Pitch.

- `[/]` **Giai đoạn 5: UI & Final Integration**
  - `[/]` Xây dựng Terminal Interface / CommandLine Parsing (cxxopts / argp).
  - `[ ]` Xử lý Video file (đọc ghi MP4).
  - `[ ]` Xử lý ảnh Batch (đọc từ thư mục input/).
  - `[ ]` Xuất kết quả CSV log.
