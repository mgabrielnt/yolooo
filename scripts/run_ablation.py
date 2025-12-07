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


# Mapping varian ablation -> file config
ABLATION_MAP = {
    "nomosaic": CONFIG_DIR / "train_ablation_nomosaic.yaml",
    "img768": CONFIG_DIR / "train_ablation_img768.yaml",
}


def main() -> None:
    """
    Menjalankan eksperimen ablation YOLO11:
      - nomosaic: tanpa mosaic/mixup
      - img768 : resolusi lebih tinggi (768)
    """
    parser = argparse.ArgumentParser(
        description="Menjalankan eksperimen ablation YOLO11"
    )
    parser.add_argument(
        "--variant",
        choices=ABLATION_MAP.keys(),
        default="nomosaic",
        help="Pilih varian ablation (nomosaic / img768)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Nama run custom (default: ablation_<variant>)",
    )

    args = parser.parse_args()
    cfg = ABLATION_MAP[args.variant]

    run_name = args.run_name or f"ablation_{args.variant}"
    run_training(str(cfg), run_name=run_name)


if __name__ == "__main__":
    main()
