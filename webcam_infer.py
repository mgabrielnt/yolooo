import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def draw_boxes(frame, results, class_names):
    for box, cls, score in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_names[int(cls)]}: {score:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def run_webcam(weights: Path, source: int = 0, conf: float = 0.35, imgsz: int = 640) -> None:
    """Menjalankan inferensi webcam real-time-ish."""

    model = YOLO(weights)
    class_names = model.model.names

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak bisa membuka webcam {source}")

    print("[INFO] Tekan 'q' untuk keluar.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, conf=conf, imgsz=imgsz, verbose=False)[0]
        frame = draw_boxes(frame, results, class_names)
        cv2.imshow("YOLO11 Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Webcam inference dengan YOLO11")
    parser.add_argument("--weights", type=Path, default=Path("models/yolov11/best_baseline.pt"), help="Path weight YOLO11")
    parser.add_argument("--source", type=int, default=0, help="Index webcam")
    parser.add_argument("--conf", type=float, default=0.35, help="Threshold confidence")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolusi inferensi")
    args = parser.parse_args()

    run_webcam(args.weights, args.source, args.conf, args.imgsz)


if __name__ == "__main__":
    main()
