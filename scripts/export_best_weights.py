import argparse
from pathlib import Path
import shutil
import sys

# ==============================
# Pastikan root project di sys.path
# ==============================
ROOT_DIR = Path(__file__).resolve().parents[1]  # .../yolooo
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import MODELS_DIR, RUNS_DIR


def export_best(run_dir: Path, output_name: str) -> Path:
    """Menyalin best.pt dari run Ultralytics ke folder models/yolov11."""

    best_path = run_dir / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"best.pt tidak ditemukan di {best_path}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / output_name
    shutil.copy2(best_path, dest)
    print(f"[INFO] best.pt disalin ke {dest}")
    return dest


def main():
    parser = argparse.ArgumentParser(description="Export best.pt ke models/yolov11")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=RUNS_DIR / "advanced",
        help="Folder run YOLO",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="best_advanced.pt",
        help="Nama file output",
    )
    args = parser.parse_args()

    export_best(args.run_dir, args.output_name)


if __name__ == "__main__":
    main()
