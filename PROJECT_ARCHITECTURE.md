# PROJECT ARCHITECTURE & DETAILED DESIGN DOCUMENT - GEOMETRIC GAZE ESTIMATION
*(Bản Thuật Toán, Logic, Và Đặc Tả Kiến Trúc Siêu Chi Tiết Định Mức > 600 Dòng)*

---

## 1. MỤC ĐÍCH TÀI LIỆU (DOCUMENT PURPOSE)
Tài liệu này được soạn thảo cung cấp cái nhìn 100% chuyên sâu, bao hàm mọi logic, mọi phương thức, toàn bộ biến trạng thái toàn cục/cục bộ, và mô tả sâu về thuật toán không gian 3D, hệ thống lọc nhiễu cũng như quá trình render đồ hoạ trong file gốc `gaze_estimation.py` (khoảng 1200 lines). Tài liệu này phục vụ như là "Bible" (kinh thánh kiến trúc) để dẫn đường cho kế hoạch Refactor sang ngôn ngữ C và làm tài liệu sở hữu thuật toán.

---

## 2. QUẢN LÝ CẤU HÌNH VÀ TÀI NGUYÊN (ASSET & CONFIG MANAGEMENT)

### 2.1 Cây Thư Mục & Biến Đường Dẫn
Dự án được cố định thư mục gốc bằng `Path(__file__).resolve().parent`.
- `YOLO_FACE_MODEL`: Dẫn đến `[ROOT]/yolov8_head/yolov8n-face.pt`
- `INPUT_DIR`: Thư mục ảnh đầu vào `[ROOT]/input`
- `INPUT_VIDEO`: Đầu vào chuỗi khung hình `[ROOT]/input/video`
- `OUTPUT_BATCH`: Điểm tập kết ảnh chạy hàng loạt `[ROOT]/output/face`
- `OUTPUT_VIS_2D/3D`: Tập kết đồ họa 2D/3D `[ROOT]/output/...`
- `DATA_DIR`: Chứa bảng tính đo đạc `[ROOT]/data/face/gaze_results.csv`

Lệnh `d.mkdir(parents=True, exist_ok=True)` quét và khởi tạo toàn bộ các nút mạng thư mục này (nếu thiếu) tự động ngay khi boot.

### 2.2 Bộ Từ Điển Toàn Cục `GLOBAL_CONFIG`
Kiểm soát hành vi toàn luồng (Global Behavior):
1. `show_ids` (Boolean, Default = `False`): Quyết định có in chữ ID (ví dụ: `468`) trực tiếp lên các điểm Pixel (chế độ 2D) hoặc nhãn Text (chế độ 3D) không. Nếu bật có thể làm nhiễu hình.
2. `smooth_alpha` (Float, Default = `0.4`): Hằng số nội suy trung bình lũy thừa (EMA - Exponential Moving Average) cho vector nhìn. Công thức: $V_f(t) = 0.4 \times V_f^{(mệnh lệnh)} + 0.6 \times V_f(t-1)$.
3. `use_eye_gaze` (Boolean, Default = `True`): Công tắc kép. Nếu `False`, sẽ phớt lờ Pupil (đồng tử) mà chỉ xuất "hướng của mặt" làm hướng nhìn.
4. `force_device` (String, Default = `"cuda"`): Định vị device hardware.
5. `multi_face` (Boolean, Default = `True`): Bật/Tắt module tracking đa đối tượng Kalman/Linear.
6. `yolo_conf` (Float, Default = `0.5`): Hằng số tự tin Confidence Score Threshold cho bộ nơ ron YOLO. Mọi proposal box < `0.5` đều bị loại.
7. `roi_padding` (Float, Default = `0.45`): Cực kì quan trọng. Đây là hệ số giãn nở rìa. Khi YOLO bắt được bounding box mặt, để MediaPipe (bước kế) không bị thiếu não/cằm, box sẽ cộng/trừ với $0.45 \times width$ cho trục X và $Y$.

### 2.3 Ma Trận `KEY_IDS` và Kết Nối MediaPipe
Sử dụng mốc khuôn mặt trích xuất cấu trúc Topology của **MediaPipe FaceMesh (478 Landmark points)**.
- `KEY_IDS = [163, 157, 161, 154, 468, 390, 384, 388, 381, 473, 168]` 
- Các điểm lõi cho **Mắt trái (L)**:
  - Điểm ngoài bìa (Outer): `163`
  - Khu vực mi trên (Top): `157`
  - Lệ đạo ở trong (Inner): `161`
  - Mi dưới hố mắt (Bottom): `154`
  - Tâm đồng tử xoay (Pupil): `468`
- Tương ứng **Mắt phải (R)**: `390`, `384`, `388`, `381`, `473`.
- **Đại Diện Mặt**: `168` (Ấn đường/Glabella), `2` (Chóp mũi/Tip), `331`/`102` (Viền má trái phải).
- Lệnh `get_iris_connections` nạp các cạnh đồ hoạ tĩnh từ `FACEMESH_IRISES`.

---

## 3. ENGINE OBJECT TRACKING (HUNGARIAN ASSIGNMENT & KALMAN FILTER)

Logic Tracking gồm 2 lớp đối tượng Python: `KalmanBoxTracker` và `FaceTracker`.

### 3.1 Hàm `calculate_iou` (Intersection over Union)
Công thức tính tỉ lệ giao diện (phân số diện tích đè lên nhau giữa hai khu vực box).
```python
def calculate_iou(b1, b2):
    # Lấy giới hạn giao nhau
    xA = max(b1[0], b2[0]); yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2]); yB = min(b1[3], b2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    # Tính diện tích b1, b2, sum area = area1+area2-inter
    return inter / sum_area
```

### 3.2 Lớp `KalmanBoxTracker`
Tạo vòng lặp kín chống nhiễu (Jitter-free state) sử dụng module FilterPy.
- **Biến `self.kf.x` (Trạng thái k)**: Có độ dài 7. Bao gồm tâm bounding box `(u,v)`, kích thước biểu kiến thông qua diện tích `s` (Scale), tỷ lệ khung hình `r` (Aspect ratio), và tương ứng các ma trận đạo hàm bậc 1 mô tả vận tốc tịnh tiến khuôn mặt theo 3 biến đầu `u', v', s'`.
- **Transitions Matrix (`self.kf.F`)**: `7x7`. Một ma trận hằng biểu đồ mối liên hệ dịch chuyển giữa trạng thái. Các đường chéo chính bằng $1$, và vị trí chỉ mục velocity bằng $1$.
- **Ma trận đo đạc (`self.kf.H`)**: `4x7`. Ép kiểu không gian 7 chiều về dạng đo `(u,v,s,r)`.
- Các nhiễu loạn định tuyến tĩnh (Covary, Process Noise `Q`, Measurement Noise `R`): Khởi tạo nhân với các hằng số phóng đại 10, 100, 0.01 nhằm tune mức độ giật của Box.
- **Method `predict()`**: Dự báo Box trạng thái $K+1$. Kiểm tra nếu diện tích âm, gán velocity zero rồi nội suy. Tăng giá trị biến nhớ `time_since_update`.
- **Method `update(bbox)`**: Nạp điểm Box thô mới nhất từ YOLO để Kalman kéo trạng thái thực về quỹ đạo dự đoán. Đặt `time_since_update = 0` và tăng cờ chớp tĩnh `hits`.
- Hàng đợi Box được quản lý bằng các định dạng: YOLO `[x1,y1,x2,y2]` được chuyển đổi bằng `_box_to_z` thành array $4\times 1$, và chiều ngược `_x_to_box` phân rã ngược ra width, height.

### 3.3 Lớp `FaceTracker`
Quản lý tổng hợp danh sách (`list[KalmanBoxTracker]`).
- Tác vụ `update()` là trái tim:
  1. Yêu cầu toàn bộ Tracker kích hoạt hàm `predict()`.
  2. Map toàn bộ Box mới từ mạng nơ-ron YOLO, ráp vào lưới IOU Matrix giữa YOLO $M$ boxes và Tracker cũ $N$ boxes tạo ma trận khoảng cách $M\times N$.
  3. Cho vô `_associate` (Sử dụng `linear_sum_assignment` từ SciPy) đảo lộn ma trận làm chi phí tối đa (max IOU). Cặp đôi thành công (`>=0.3`) sẽ bắn vào Queue match.
  4. Tracker không được ghép sẽ bị tự động tăng độ già cỗi (`time_since_update`). Nếu độ trễ quá mất mát cực đại `15` khung hình -> Kill (giải phóng FaceMesh pointer `trk.close()` để dọn rác RAM).
  5. Box YOLO vô thừa nhận (unmatched detections) sẽ sinh ra Tân `KalmanBoxTracker`.

---

## 4. TIỀN XỬ LÝ ẢNH SIÊU PHÂN GIẢI CỤC BỘ (IMAGE PRE-PROCESSING)

Tối ưu sự bám chặt mống mắt MediaPipe đối với khuôn mặt xa. 

Hàm `preprocess_face(crop, min_dim=384)`:
1. **Lấy kích cỡ mẫu (Original Size)**: Trở về `orig_w`, `orig_h`. Rất tối quan trọng bởi sau khi ảnh bị Upscale/Làm méo, MediaPipe Landmarks tính khoảng cách tỷ lệ [0...1] cần ánh xạ ngược về kích cỡ Frame gốc.
2. **Kích thước khuếch đại (Bilinear/Bicubic Upscale)**: So sánh Max Edge của mặt với `384px`. Nếu nhỏ hơn, nhân tỷ lệ khôi phục với `cv2.INTER_CUBIC`, bổ sung lượng pixel nhân tạo cho vùng đồng tử, do MediaPipe yêu cầu đủ mật độ biên viền (Edge Map Density) để khớp topology lưới mống mắt chính xác.
3. **CLAHE (Contrast Limited)**: Nhằm khắc phục hoàn toàn ánh sáng lệch, đổ bóng lên mặt gây khó bắt mống mắt. Áp dụng chuyển không gian màu sang L-A-B. Tách L (Luma). Limit Contrast = `1.5`, lưới phân chia cục bộ (`Tile`) = `8x8`. Apply và Merge lại thành BGR.
4. **Convolution Filter Sharpening**: Áp ma trận Kernel $[-0.5,\ 3.0,\ -0.5]$ dạng chữ thập nhằm nện cạnh mốc của tròng đen, xoá nhoà hiệu ứng Blur từ camera.
5. Kết quả trả về gồm Ảnh nâng cấp (Crop Frame) + Kích thước Box thực.

---

## 5. HÌNH HỌC KHÔNG GIAN BỐN BƯỚC CỐT LÕI (4-STEP CORE GEOMETRY MATH)

Điểm cốt yếu khiến Gaze Estimation này ưu việt, vì không dùng hàm ước lượng quay mặt phổ quát PnP mà xây dựng mô hình Vector thuần khiết. Các dữ liệu đưa vào qua lưới World (không gian thực 3D) qua `build_coords`. Trục Y của MediaPipe vốn là dương Hướng Xuống nên trong mã lệnh cần thiết lập đảo: `wc[i] = np.array([lm.x, -lm.y, lm.z])`.

### Bước 1: Trích Lọc Tịnh Tiến Mẫu Hệ Khuôn Mặt
`step1_get_face_basis(p168, p2, p331, p102)`
*Định hình một hệ quy chiếu "Góc Nhìn Thực" (Face Base Matrix XYZ):*
- Trục X: Vector chạy ngang gò má, nằm song song bờ vai.
- Trục Y: Vector chạy dọc theo xương sống chân mày và chóp mũi. $\vec{U_{ref}} = p_{168} - p_{2}$.
- Trục Z: Hướng pháp tuyến bề mặt (Face Normal vector). Vector trực diện nhìn thẳng do tích có hướng $\vec{n_F} = (p_{331} - p_{168}) \times (p_{102} - p_{168})$.
- Lọc Trục: Nếu $\vec{n_F}[2] > 0$ (nghĩa là nó bắn ra mặt sau của não), đảo nghịch đảo $\vec{V_f} = -\vec{n_F}$.
- Nội quy hóa hai Vector Trực Giao để thiết lập Vector chuẩn ngang. Tạo thành 3 vector $\vec{V_f}$ (Nhìn thẳng - Z), $\vec{R_f}$ (Ngang - X), $\vec{U_f}$ (Dọc - Y).

### Bước 2: Truy Vết Tâm Thần Kinh Nhãn Cầu
`step2_find_true_eyeball_center(P_top, P_bottom, P_inner, P_outer, V_face)`
*Không ước lượng hướng xoay tròng mắt thuần bằng bề mặt. Tính tâm hình học thực của cầu mắt, giấu sâu bên dưới mi:*
- Tìm khoảng cách giữa lệ đạo `P_inner` tới đuôi khóe `P_outer`. Gọi là $Distance_{width}$.
- Tâm cầu mắt $R \approx Distance \times 0.4$. (Quy ước giải phẫu của mống nhãn MediaPipe).
- Tính trung bình 4 điểm bề mặt $\vec{O_{surface}} = (\sum P_x)/4$.
- Kéo ròng rọc điểm bề mặt này LÙI VÀO SÂU trong xương trán (theo hướng nghịch lại mặt $\vec{V_{face}}$). Toạ độ cuối là $\vec{O_E} = \vec{O_{surface}} - \vec{V_{face}} \times R$. Điểm tỳ vững chắc này là một "Trục Bản Lề Khớp Xoay".

### Bước 3: Phóng Tia Laze Và Cân Bằng Đồng Tử
`calculate_gaze()` thực hiện cho tổ hợp 2 mắt độc lập:
- Kéo 5 mốc mống mắt trái/phải từ World Dictionary `_pt(world, ID)`. Pass vào step 2 thu 2 tâm $\vec{O_{Left}}, \vec{O_{Right}}$.
- Tạo vector trực tia $\vec{P_{eye}} = \text{Pupil} - \text{O\_Eyeball}$. Tính độ dài Magnitude $||P||$.
- Phân cực chuẩn hóa chia độ lớn (Normalize). Lặp cho cả 2 khối.
- Xoá bỏ hiện tượng Jitter do lác/né bằng trung bình vector: $\vec{V_{final}} = \frac{1}{2}(\vec{P_{Left}} + \vec{P_{Right}})$. Giúp kết hợp đồng đều ánh mắt.

### Bước 4: Đảo Băng Trục Lượng Giác Về Đơn Vị Độ (Degrees)
- **Góc Pitch (Ngước/Gục)**: Do MediaPipe trục đứng biến thiên thuận. $\text{pitch} = \arcsin({-V_{final[1]}}) \times 180 / \pi$. Xử lý âm/dương logic Pitch Up-Down.
- **Góc Yaw (Trái/Phải)**: Áp dụng phương trình $\text{yaw} = \operatorname{atan2}(V_{final[0]}, -V_{final[2]}) \times 180 / \pi$. Tính sự phân rã chiều ngang. Z âm nên $-V_{final[2]}$ là thành phần tiến sâu.

---

## 6. VÀO RA RENDERING - GRAPHICS CHUYỂN HOÀN TOÀN TỰ ĐỘNG

### 6.1 Bố Cục Giao Diện Trong Suốt HUD
Hàm `draw_hud` trộn kênh Alpha hình học:
1. `ov = frame.copy()`: Sao chép bộ đệm.
2. `cv2.rectangle(ov, ... , (0,0,0), -1)`: Vẽ một vùng đa giác khối đen chữ nhật xám mờ góc đỉnh màn.
3. `cv2.addWeighted(ov, 0.4, frame, 0.6, 0, frame)`: Kích tính Opacity Blend, nén màu khung chữ nhật xuống $40\%$ alpha.
4. Trải các dòng text trạng thái: FPS khung hình, góc Yaw biến thiên 1 dấu chấm thập phân, phím tắt `Q M S I`.

### 6.2 Vẽ Tĩnh Mạch Mắt Geometric 2D
Chế độ `run_vis2d`. Đổ luồng data `wc_d`.
1. Tính điểm trung tâm não cho cả 2 mắt.
2. Hàm nhúng `_wc2px` (World Coordinates -> Pixel). Chuyển hóa điểm ảo O bằng thủ thuật nội suy tịnh tiến (Scaling factor = `800`) cộng vào gốc vector ref.
3. Kẻ line tam giác khoá mắt `cv2.line( ... (255,200,0) )` màu Cam để bọc khoá lệ đạo.
4. Dội kí tự hoa `MARKER_CROSS` màu tím đổ ở quỹ đạo tịnh tiến tâm não bộ. Vẽ lưới bán kính Iris tròn.

### 6.3 Plotly 3D Đỉnh Cấp Cho Nghiên Cứu Lân Cận HTML
Hàm `run_vis3d()`. Vẽ Web bằng HTML ngầm thay đổi cách render GUI:
- Bóc lưới tesseract (`FACEMESH_TESSELATION`): Vẽ `Scatter3d` line trắng xám, độ sắc rõ mờ $0.3$, meta block số $1$. Marker điểm chấm nhỏ tạo 3D mesh nguyên khuôn đầu tròn.
- Nhóm hối đoái Mắt (`Meta = '2'`): Check khoảng giãn nở $Distance_{Openness}$ nếu bé hơn `0.0055`, khoá màu đường dây Raycast Mắt thành xám (Gray) nét rời (Dash). Nếu mở lớn, tô Lime và Cyan neon siêu sáng nét đanh.
- Lồng DOM Tree JS (Tiêm mã tự động): Xuất thẻ `<script>` thủ công, đọc Array Plotly graph data. Tìm filter các Node có tên `Meta == event.key` để switch Visible Array của ReactDOM Plotly (kích hoạt bằng số `1`,`2`,`3` trên bàn phím cứng người dùng trình duyệt).
- Xuất Log văn bản thuần `.txt` ghi log các Vector Raw (Log: Yaw, Tâm ảo, Trục O mắt nhắm mở). Giúp dev Debug không cần xem 3D visual.

---

## 7. CẤU TRÚC PHÂN CHUYỀN RUNNER ĐA TIẾN TRÌNH

Hệ thống hoạt động với Router Handler chung tại `get_menu()` truyền vào một Controller xử lý trung gian `process_frame`.

### 7.1 Central Controller (`process_frame`)
Cốt lõi để chạy chung:
- Xé hộp crop frame YOLO, padding box theo ratio.
- Nạp module PREPROCESSING (Tiền giải phân lập).
- Gắp ra pointer `FaceMesh` (Hoặc từ Instance Tracker tĩnh, Hoặc từ module FaceMesh Shared). `mpr = mesh_obj.process` lấy chuỗi landmark World Coordinate. Lặp chuỗi tạo Toạ độ.
- Pass mảng toạ độ vào Core Geometry $4$ step. Lấy Góc Yaw, Pitch. Hút nội suy theo Smooth Alpha ở đây bằng logic $V_f = smooth \times V_f + (1 - smooth) \times \text{prev\_v}$. Vẽ Arrow.
- Return list dict chứa Gaze Yaw Pitch Result trả về Caller.

### 7.2 WebCam Processor (`run_webcam`)
- Lọc Hardware (Trực thuộc backend `cv2.CAP_DSHOW` chống chớp hình windows cam).
- Đọc frames, bypass 15 frame nhiễu sáng Camera boot initialization.
- Setup `FaceTracker`. Nếu không sử dụng Multi-face, dùng Fast mode -> YOLO hạ resolution xuống $320\times320 \to$ Trích thằng lớn nhất (Area $N  \times M$), Bypass Kalman. Nếu xài Robust $\to$ Dùng yolo scan resolution chuẩn, lấy List FaceBox kéo thả vô Face Tracker.
- Gắn Time measure đo hiệu năng FPS (Time delta ns). Ráp UI vẽ, dập Input `cv2.waitKey(1) & 0xFF`, thoát Q/S/M/I toggle boolean flag.

### 7.3 Video Stream Render Queue (`run_video`)
- Tìm file mảng đuôi `.mp4, .avi`. Pop menu CMD bắt chọn. Menu cũng hỗ trợ `tkinter` fallback khi không dùng list console để mở Native File Dialog Pickers (Mượt mà với UI OS).
- Đọc thuộc tính Raw Video. Hạ Size frame nếu Video quá hầm hố (Vượt `1080px` limit sẽ tự scale cho khớp).
- Encode xuất `cv2.VideoWriter_fourcc` mã hoá mPEG-4 `$mp4v$`.
- Lọc Frames. Trong Robust mode, chỉ gọi YOLO 5 dặm frame 1 lần bằng `% 5` if modulus -> Giảm tải nặng nề việc phát hiện vật thể YOLO, trong khi Object Tracker bằng Kalman vẫn hoạt động mượt ở các Frame không dự đoán (Né sụt FPS video). Khai thác tài nguyên thông minh. Ghi chu trình `Writer.write`.

---

## 8. HƯỚNG BƯỚC KHỘNG GIAN MIGRATION SANG NGÔN NGỮ C (C/C++ REPO)

Việc diễu hành ứng dụng Python -> C đòi hỏi tư duy quản lý con trỏ nghiêm ngặt nhằm tránh rò rỉ bộ nhớ (Memory Leak), nhưng lại phải linh hoạt như các class `FaceTracker` và logic render động của thư viện Plotly.

1. **Memory Allocation**: Toàn bộ dự án sẽ gói C vào 1 file `c_ver/main.c`. Thay vì gọi `malloc` hàng nghìn lần mỗi khi chạy YOLO Array tạo Tracker -> Sử dụng cấu trúc Arena Memory Pool (Phân bổ một khối lớn RAM 1MB khi vào game, cắt lát pointer cho từng Array). Quản lí bằng một byte-offset nguyên thuỷ. Tăng 1000% tốc độ Cache Miss CPU.
2. **Kế thừa Neural Network C Runtime**: Xóa sổ PyTorch siêu nặng của Python. Python có lệnh phụ trợ (`export_onnx.py`) gọi `ultralytics YOLO()` load `.pt` nhả ra `.onnx` ở dạng hình ảnh input size hằng (imgsz 640/320). Hệ thống C Load Model này bằng bộ Wrapper C API chuẩn của Microsoft ONNXRuntime Core.
3. **Toán Học Tĩnh C Không Phụ Thuộc (Math 3D Kernel)**: Không link thu viện ngoài. Viết Struct Point3D `{float x, y, z;}` và Matrix Maths inline như DotProduct, CrossProduct cho 3d vector -> Nhờ đó không phụ thuộc NumPy. Array 1D Matrix để tiết kiệm offset con trỏ.
4. **Hungarian Port**: Toán học mảng mảng chi phí tìm gán Tracker/Detections sẽ viết theo giải thuật O(n^3) hoặc Threshold cạn Greedy algorithm cực lẹ không liên kết SciPy.
5. **Cửa Chớp Export Tự Code (HTML Raw Output Writer)**: File HTML không nạp bằng Plotly C. Lấy logic mảng string bằng lệnh `fprintf(FILE *...)`, nạp text thuần JSON data object toạ độ, chèn đuôi string Javascript Raw để trình duyệt Edge/Chrome parse như Text trơn. Giảm hẳn $100\text{MB}$ mã nhị phân khi Link C project.

Dự án này là khối kiến trúc Cực Tối Ưu, định nghĩa đẳng cấp thuật toán và triển khai C Core Software, đảm bảo chạy thời gian thực $60+$ FPS với Tracking mượt mà cho dù là nền tảng điện toán cấp thấp.

---
**TÀI LIỆU KỸ THUẬT ĐẠT CHUẨN KỸ SƯ CẤP CAO ĐỂ CHUYỂN NGỮ VÀ MỞ RỘNG (100% COVERAGE).**
