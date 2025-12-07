from pathlib import Path
from typing import Any, Dict, Optional

from ultralytics import YOLO

from .config import LOGS_DIR, RUNS_DIR, load_yaml, resolve_path


def _prepare_train_args(cfg: Dict[str, Any], project_dir: Path, run_name: str) -> Dict[str, Any]:
    """
    Mempersiapkan argumen untuk YOLO.train dari config YAML.
    - Mengabaikan key 'model' dan 'data' (ditangani terpisah).
    """
    args: Dict[str, Any] = {}

    for key, value in cfg.items():
        # 'model' dan 'data' di-handle manual di run_training()
        if key in {"model", "data", "description"}:
            continue
        args[key] = value

    # Setup output project & nama run
    args["project"] = str(project_dir)
    args["name"] = run_name

    # Default device = GPU 0 jika tidak ditentukan
    args.setdefault("device", 0)

    return args


def _resolve_data_arg(raw_data: Any) -> str:
    """
    Menghasilkan nilai argumen 'data' untuk YOLO.train.

    - Jika raw_data adalah salah satu nama dataset built-in
      seperti 'coco128.yaml', 'coco8.yaml', 'coco.yaml',
      maka DIKEMBALIKAN apa adanya (biar Ultralytics yang handle
      dan auto-download dataset).
    - Kalau bukan, diasumsikan path lokal dan di-resolve terhadap ROOT_DIR.
    """
    if isinstance(raw_data, str) and raw_data in {"coco128.yaml", "coco8.yaml", "coco.yaml"}:
        # Biarkan YOLO yang cari & download sendiri
        return raw_data

    # Selain itu: anggap ini path lokal (relatif/absolut)
    path = resolve_path(str(raw_data))
    return str(path)


def run_training(
    cfg_path: str,
    project_dir: Optional[Path] = None,
    run_name: Optional[str] = None,
) -> None:
    """
    Menjalankan training YOLO11 berdasarkan file YAML Ultralytics.

    Args:
        cfg_path: Path ke YAML konfigurasi training.
        project_dir: Direktori output run (default: experiments/runs).
        run_name: Nama run (default: nama file config tanpa ekstensi).
    """
    cfg_file = Path(cfg_path)
    cfg = load_yaml(cfg_file)

    # Direktori output (project) untuk semua run
    project_dir = project_dir or RUNS_DIR
    project_dir.mkdir(parents=True, exist_ok=True)

    # Nama run default = nama file YAML
    run_name = run_name or cfg_file.stem

    # Model & data dari config
    model_path = cfg.get("model", "yolo11s.pt")
    raw_data = cfg.get("data", "configs/data_coco_full.yaml")
    data_arg = _resolve_data_arg(raw_data)

    # Siapkan argumen training (tanpa 'model' & 'data')
    train_args = _prepare_train_args(cfg, project_dir, run_name)
    train_args["data"] = data_arg

    # Load model YOLO
    model = YOLO(model_path)

    print(f"[INFO] Mulai training: model={model_path}, data={data_arg}")
    print(f"[INFO] project={project_dir}, name={run_name}")

    # Jalankan training
    results = model.train(**train_args)

    # Tulis ringkasan log sederhana
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"train_{run_name}.md"

    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# Ringkasan Training {run_name}\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Data: {data_arg}\n\n")
        f.write(f"Args:\n{train_args}\n\n")

        # Kalau metrics tersedia, simpan mAP ringkas
        try:
            if results is not None and hasattr(results, "metrics") and results.metrics is not None:
                box_metrics = results.metrics.box
                f.write(f"mAP50-95: {box_metrics.map}\n")
                f.write(f"mAP50: {box_metrics.map50}\n")
        except Exception:
            # Jangan sampai error log menghentikan training
            pass

    print(f"[INFO] Selesai. Log: {log_path}")


__all__ = ["run_training"]
