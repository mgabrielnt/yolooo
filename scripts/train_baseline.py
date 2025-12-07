import argparse
from pathlib import Path
import sys

# ============================================
# Pastikan root project (yolooo) ada di sys.path
# ============================================
ROOT_DIR = Path(__file__).resolve().parents[1]  # .../yolooo
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.train_pipeline import run_training
from src.config import CONFIG_DIR


def main() -> None:
    """
    Menjalankan training baseline YOLO11 (E1) dengan config train_baseline.yaml.
    """
    parser = argparse.ArgumentParser(
        description="Training baseline YOLO11 pada coco128 / COCO subset"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DIR / "train_baseline.yaml",
        help="Path config training (YAML Ultralytics)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="baseline",
        help="Nama run (folder di experiments/runs)",
    )

    args = parser.parse_args()
    run_training(str(args.config), run_name=args.run_name)


if __name__ == "__main__":
    main()
