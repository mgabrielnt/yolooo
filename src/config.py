from pathlib import Path
from typing import Any, Dict

import yaml

# Direktori akar proyek
ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / "configs"
EXPERIMENT_DIR = ROOT_DIR / "experiments"
RUNS_DIR = EXPERIMENT_DIR / "runs"
LOGS_DIR = EXPERIMENT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models" / "yolov11"
DOCS_DIR = ROOT_DIR / "docs"


def load_yaml(path: Path) -> Dict[str, Any]:
    """Membaca file YAML dan mengembalikan dictionary.

    Args:
        path: Path ke file YAML.
    Returns:
        Dictionary hasil parsing YAML.
    """

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_str: str) -> Path:
    """Konversi string path relatif/absolut menjadi Path yang absolut."""

    path = Path(path_str)
    return path if path.is_absolute() else ROOT_DIR / path


__all__ = [
    "ROOT_DIR",
    "CONFIG_DIR",
    "EXPERIMENT_DIR",
    "RUNS_DIR",
    "LOGS_DIR",
    "MODELS_DIR",
    "DOCS_DIR",
    "load_yaml",
    "resolve_path",
]
