import argparse
from pathlib import Path

from src.eval_pipeline import evaluate_model
from src.config import CONFIG_DIR


def main():
    parser = argparse.ArgumentParser(description="Evaluasi model YOLO11")
    parser.add_argument("--weights", type=Path, default=Path("models/yolov11/best_advanced.pt"), help="Path weight")
    parser.add_argument("--data", type=Path, default=CONFIG_DIR / "data_coco_full.yaml", help="YAML dataset")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolusi evaluasi")
    parser.add_argument("--split", type=str, default="val", help="Split dataset (train/val/test)")
    parser.add_argument("--run-name", type=str, default=None, help="Nama run evaluasi")
    args = parser.parse_args()

    evaluate_model(str(args.weights), str(args.data), imgsz=args.imgsz, split=args.split, run_name=args.run_name)


if __name__ == "__main__":
    main()
