import argparse
from pathlib import Path

from src.train_pipeline import run_training
from src.config import CONFIG_DIR


def main():
    parser = argparse.ArgumentParser(description="Training advanced YOLO11 dengan bag-of-freebies")
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "train_advanced.yaml", help="Path config training")
    parser.add_argument("--run-name", type=str, default="advanced", help="Nama run")
    args = parser.parse_args()

    run_training(str(args.config), run_name=args.run_name)


if __name__ == "__main__":
    main()
