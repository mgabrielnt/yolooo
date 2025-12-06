from pathlib import Path
from typing import Any, Dict, Optional

from ultralytics import YOLO

from .config import LOGS_DIR, RUNS_DIR, load_yaml, resolve_path


def _prepare_train_args(cfg: Dict[str, Any], project_dir: Path, run_name: str) -> Dict[str, Any]:
    """Mempersiapkan argumen untuk YOLO.train dari config YAML."""

    args: Dict[str, Any] = {}
    for key, value in cfg.items():
        if key in {"model", "data", "description"}:
            continue
        args[key] = value

    args["project"] = str(project_dir)
    args["name"] = run_name
    args.setdefault("device", 0)
    return args


def run_training(cfg_path: str, project_dir: Optional[Path] = None, run_name: Optional[str] = None) -> None:
    """Menjalankan training YOLO11 berdasarkan file YAML Ultralytics.

    Args:
        cfg_path: Path ke YAML konfigurasi training.
        project_dir: Direktori output run (default experiments/runs).
        run_name: Nama run (default nama file config tanpa ekstensi).
    """

    cfg_file = Path(cfg_path)
    cfg = load_yaml(cfg_file)
    project_dir = project_dir or RUNS_DIR
    project_dir.mkdir(parents=True, exist_ok=True)
    run_name = run_name or cfg_file.stem

    model_path = cfg.get("model", "yolo11s.pt")
    data_path = resolve_path(str(cfg.get("data", "configs/data_coco_full.yaml")))

    train_args = _prepare_train_args(cfg, project_dir, run_name)
    train_args["data"] = str(data_path)

    model = YOLO(model_path)
    print(f"[INFO] Mulai training: model={model_path}, data={data_path}")
    print(f"[INFO] project={project_dir}, name={run_name}")
    results = model.train(**train_args)

    log_path = LOGS_DIR / f"train_{run_name}.md"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# Ringkasan Training {run_name}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Data: {data_path}\n")
        f.write(f"Args: {train_args}\n")
        if results and hasattr(results, "metrics"):
            f.write(f"mAP50-95: {results.metrics.box.map}\n")
            f.write(f"mAP50: {results.metrics.box.map50}\n")
    print(f"[INFO] Selesai. Log: {log_path}")


__all__ = ["run_training"]
