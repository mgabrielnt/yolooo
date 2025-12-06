import argparse
from pathlib import Path

from src.train_pipeline import run_training
from src.config import CONFIG_DIR


ABLATION_MAP = {
    "nomosaic": CONFIG_DIR / "train_ablation_nomosaic.yaml",
    "img768": CONFIG_DIR / "train_ablation_img768.yaml",
}


def main():
    parser = argparse.ArgumentParser(description="Menjalankan eksperimen ablation YOLO11")
    parser.add_argument("--variant", choices=ABLATION_MAP.keys(), default="nomosaic", help="Pilih varian ablation")
    parser.add_argument("--run-name", type=str, default=None, help="Nama run custom")
    args = parser.parse_args()

    cfg = ABLATION_MAP[args.variant]
    run_training(str(cfg), run_name=args.run_name or f"ablation_{args.variant}")


if __name__ == "__main__":
    main()
