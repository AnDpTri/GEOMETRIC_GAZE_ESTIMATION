"""
YOLOv8 Face Detection - Batch Image Processing
================================================
Xử lý tất cả ảnh trong thư mục input và lưu kết quả vào thư mục output.

Cách dùng:
  python batch_detect.py                          # mặc định model nano
  python batch_detect.py --model s                # dùng model Small
  python batch_detect.py --model m -i ./photos -o ./results
  python batch_detect.py --models                 # liệt kê tất cả model

Tham số:
  -i / --input   : Thư mục chứa ảnh gốc   (mặc định: ./input)
  -o / --output  : Thư mục lưu ảnh kết quả (mặc định: ./output)
  --model        : Kích thước model n/s/m/l/x  (mặc định: n)
  --models       : Liệt kê model khả dụng và thoát
  --conf         : Ngưỡng tin cậy 0.0-1.0  (mặc định: 0.50)
  --show         : Hiện preview mỗi ảnh (nhấn phím bất kỳ để tiếp tục)
  --no-label     : Không vẽ nhãn, chỉ vẽ bounding box
"""

import argparse
import sys
import io

# Đảm bảo in ra đúng định dạng UTF-8 trên Windows
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except Exception:
        pass

import time
from pathlib import Path

# Thêm đường dẫn gốc vào sys.path để import config
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import config
import cv2
from ultralytics import YOLO
from model_manager import get_model_path, list_models, get_model_info

# ─── Cấu hình ────────────────────────────────────────────────────────────────
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

COLOR_BOX   = (0, 220, 100)
COLOR_LABEL = (0, 0, 0)
COLOR_BG    = (0, 220, 100)


# ─── Vẽ detection ─────────────────────────────────────────────────────────────
def draw_detection(frame, box, conf, show_label=True, label_text="Object"):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)

    if show_label:
        label = f"{label_text} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), COLOR_BG, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_LABEL, 1, cv2.LINE_AA)


# ─── Xử lý 1 ảnh ─────────────────────────────────────────────────────────────
def process_image(model, img_path: Path, out_path: Path,
                   conf: float, show: bool, show_label: bool, label_text: str = "Object") -> dict:
    img = cv2.imread(str(img_path))
    if img is None:
        return {"file": img_path.name, "status": "ERROR (không đọc được ảnh)", "count": 0}

    results = model(img, conf=conf, verbose=False)[0]

    count = 0
    if results.boxes is not None:
        for box in results.boxes:
            draw_detection(img, box.xyxy[0].tolist(), float(box.conf[0]), 
                           show_label, label_text=label_text)
            count += 1

    # Vẽ thông tin góc trên trái
    h, w = img.shape[:2]
    info = f"{label_text}s: {count}  |  {img_path.name}"
    cv2.putText(img, info, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)

    if show:
        cv2.imshow(f"Preview - {img_path.name}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {"file": img_path.name, "status": "OK", "count": count}


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8 Face Detection - Batch Image Processing")
    parser.add_argument("-i", "--input",  default="input",
                        help="Thư mục ảnh đầu vào (mặc định: ./input)")
    parser.add_argument("-o", "--output", default="output",
                        help="Thư mục ảnh đầu ra (mặc định: ./output)")
    parser.add_argument("--model", default="n", choices=["n","s","m","l","x"],
                        metavar="SIZE",
                        help="Kích thước model: n(ano) s(mall) m(edium) l(arge) x(large) (mặc định: n)")
    parser.add_argument("--models", action="store_true",
                        help="Liệt kê tất cả model và thoát")
    parser.add_argument("--conf", type=float, default=0.50,
                        help="Ngưỡng tin cậy (mặc định: 0.50)")
    parser.add_argument("--show", action="store_true",
                        help="Hiện preview từng ảnh sau khi xử lý")
    parser.add_argument("--no-label", action="store_true",
                        help="Không vẽ nhãn, chỉ vẽ bounding box")
    args = parser.parse_args()

    if args.models:
        list_models()
        sys.exit(0)

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    show_label = not args.no_label

    # Kiểm tra thư mục input
    if not input_dir.exists():
        input_dir.mkdir(parents=True)
        print(f"[!] Đã tạo thư mục input: {input_dir.resolve()}")
        print(f"    Hãy đặt ảnh vào thư mục trên rồi chạy lại.")
        sys.exit(0)

    # Lấy danh sách ảnh
    images = sorted([
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    ])

    if not images:
        print(f"[!] Không tìm thấy ảnh nào trong: {input_dir.resolve()}")
        print(f"    Định dạng hỗ trợ: {', '.join(SUPPORTED_EXT)}")
        sys.exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Nạp model
    model_key = args.model if args.model != 'n' else config.YOLO_MODEL_SIZE
    model_info = get_model_info(model_key)
    model_path = get_model_path(model_key)
    label_type = model_info["type"]

    print(f"[*] Đang nạp model (Device: {config.get_device()}) ...")
    model = YOLO(str(model_path)).to(config.get_device())
    print(f"[✓] Model sẵn sàng!")
    print(f"\n{'─'*55}")
    print(f"  Model  : yolov8{args.model}-face")
    print(f"  Input  : {input_dir.resolve()}")
    print(f"  Output : {output_dir.resolve()}")
    print(f"  Ảnh    : {len(images)} file")
    print(f"  Conf   : {args.conf}")
    print(f"{'─'*55}\n")

    # Xử lý từng ảnh
    results_log = []
    t_start = time.time()
    total_count = 0

    for idx, img_path in enumerate(images, 1):
        out_path = output_dir / img_path.name
        result = process_image(model, img_path, out_path,
                               args.conf, args.show, show_label, label_text=label_type)
        results_log.append(result)
        total_count += result["count"]

        status_icon = "✓" if result["status"] == "OK" else "✗"
        print(f"  [{idx:>3}/{len(images)}] [{status_icon}] "
              f"{result['file']:<35}  {result['count']} {label_type.lower()}")

    elapsed = time.time() - t_start

    # Tổng kết
    ok_count  = sum(1 for r in results_log if r["status"] == "OK")
    err_count = len(results_log) - ok_count

    print(f"\n{'─'*55}")
    print(f"  ✅ Hoàn thành! ({elapsed:.1f}s)")
    print(f"  Đã xử lý : {ok_count}/{len(images)} ảnh")
    print(f"  Tổng {label_type.lower()} : {total_count} {label_type.lower()}")
    print(f"  Lưu tại  : {output_dir.resolve()}")
    print(f"{'─'*55}")


if __name__ == "__main__":
    main()
