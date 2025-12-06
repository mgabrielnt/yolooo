from pathlib import Path
from typing import Dict, Iterable, Optional

import yaml

from .config import ROOT_DIR


def _count_images(path: Path) -> int:
    """Menghitung jumlah file gambar di folder secara rekursif."""

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sum(1 for p in path.rglob("*") if p.suffix.lower() in exts)


def validate_coco_path(dataset_path: Path) -> Dict[str, Path]:
    """Memvalidasi struktur folder COCO dan mengembalikan path penting.

    Args:
        dataset_path: Path dasar COCO (berisi train2017/val2017/test2017).
    Returns:
        Dictionary path untuk train/val/test image directory.
    Raises:
        FileNotFoundError: Jika folder penting tidak ditemukan.
    """

    train_dir = dataset_path / "train2017" / "images"
    val_dir = dataset_path / "val2017" / "images"
    test_dir = dataset_path / "test2017" / "images"

    for p in [train_dir, val_dir]:
        if not p.exists():
            raise FileNotFoundError(f"Folder tidak ditemukan: {p}")

    return {"train": train_dir, "val": val_dir, "test": test_dir}


def summarize_split(image_dir: Path) -> str:
    """Memberikan ringkasan jumlah gambar pada satu split."""

    count = _count_images(image_dir)
    return f"{image_dir.name}: {count} gambar"


def summarize_dataset(dataset_path: Path) -> str:
    """Ringkasan singkat dataset (jumlah gambar train/val/test)."""

    paths = validate_coco_path(dataset_path)
    summary = [f"Dataset: {dataset_path}"]
    for split_name, split_path in paths.items():
        if split_path.exists():
            summary.append(summarize_split(split_path))
    return " | ".join(summary)


def load_dataset_yaml(yaml_path: Path) -> Dict:
    """Membaca YAML dataset dan memastikan path absolut."""

    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    base_path = Path(data.get("path", ROOT_DIR))
    data["path"] = base_path
    return data


def filter_classes(names: Iterable[str], keep: Optional[Iterable[str]] = None) -> Dict[int, str]:
    """Membuat mapping class index ke nama berdasarkan subset."""

    if keep is None:
        return {i: n for i, n in enumerate(names)}

    keep_set = {k.lower() for k in keep}
    filtered = {i: n for i, n in enumerate(names) if n.lower() in keep_set}
    return filtered


__all__ = [
    "validate_coco_path",
    "summarize_dataset",
    "load_dataset_yaml",
    "filter_classes",
]
