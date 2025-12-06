from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from .config import LOGS_DIR, RUNS_DIR, resolve_path


def evaluate_model(
    weights_path: str,
    data_path: str,
    imgsz: int = 640,
    split: str = "val",
    project_dir: Optional[Path] = None,
    run_name: Optional[str] = None,
) -> None:
    """Evaluasi model YOLO11 dan simpan metrik utama."""

    project_dir = project_dir or RUNS_DIR
    project_dir.mkdir(parents=True, exist_ok=True)
    run_name = run_name or f"eval_{Path(weights_path).stem}"
    log_path = LOGS_DIR / f"eval_{run_name}.md"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights_path)
    print(f"[INFO] Evaluasi {weights_path} pada {data_path} (split={split})")
    results = model.val(
        data=str(resolve_path(data_path)),
        imgsz=imgsz,
        split=split,
        project=str(project_dir),
        name=run_name,
        save_json=True,
        save_hybrid=True,
    )

    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# Ringkasan Evaluasi {run_name}\n")
        f.write(f"Weights: {weights_path}\n")
        f.write(f"Data: {data_path}\n")
        f.write(f"imgsz: {imgsz}\n")
        if results and hasattr(results, "metrics"):
            f.write(f"mAP50-95: {results.metrics.box.map}\n")
            f.write(f"mAP50: {results.metrics.box.map50}\n")
            f.write(f"mAP75: {results.metrics.box.map75}\n")
            f.write(f"Precision: {results.metrics.box.mp}\n")
            f.write(f"Recall: {results.metrics.box.mr}\n")
    print(f"[INFO] Log evaluasi tersimpan di {log_path}")


__all__ = ["evaluate_model"]
