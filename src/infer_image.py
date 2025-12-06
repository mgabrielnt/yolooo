from pathlib import Path
from typing import List, Optional, Union

from ultralytics import YOLO

from .config import RUNS_DIR, resolve_path


PathLike = Union[str, Path]


def infer_images(
    weights: PathLike,
    sources: List[PathLike],
    conf: float = 0.25,
    imgsz: int = 640,
    project_dir: Optional[Path] = None,
    run_name: Optional[str] = None,
) -> None:
    """Inference untuk satu atau banyak gambar."""

    project_dir = project_dir or RUNS_DIR
    project_dir.mkdir(parents=True, exist_ok=True)
    run_name = run_name or f"infer_{Path(weights).stem}"

    model = YOLO(weights)
    resolved_sources = [str(resolve_path(str(s))) for s in sources]
    print(f"[INFO] Inference pada {resolved_sources}")
    model.predict(
        source=resolved_sources,
        conf=conf,
        imgsz=imgsz,
        project=str(project_dir),
        name=run_name,
        save=True,
    )
    print(f"[INFO] Hasil tersimpan di {project_dir/run_name}")


__all__ = ["infer_images"]
