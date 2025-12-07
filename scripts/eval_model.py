import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Evaluasi model YOLO11")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path ke file .pt (misal: experiments/runs/baseline_coco128/weights/best.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="coco128.yaml",
        help="Config dataset YOLO (contoh: coco128.yaml atau yaml custom)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Resolusi input (default 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device: '0' untuk GPU pertama, 'cpu' untuk CPU",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Split evaluasi: val/test",
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"File weights tidak ditemukan: {weights_path}")

    model = YOLO(str(weights_path))

    print(f"[INFO] Evaluasi model: {weights_path}")
    print(f"[INFO] Data: {args.data}, imgsz={args.imgsz}, split={args.split}, device={args.device}")

    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        split=args.split,
        device=args.device,
        verbose=True,
    )

    try:
        box = metrics.box
        print("\n===== RINGKASAN METRIK BOX =====")
        print(f"mAP50-95 : {box.map:.4f}")
        print(f"mAP50    : {box.map50:.4f}")
        print(f"mAP75    : {box.map75:.4f}")
        print(f"Precision: {box.p.mean():.4f}")
        print(f"Recall   : {box.r.mean():.4f}")
    except Exception:
        print("[WARN] Tidak bisa baca metrics.box detail, cek log Ultralytics di atas.")


if __name__ == "__main__":
    main()
