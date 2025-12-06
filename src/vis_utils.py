from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.switch_backend("Agg")


def plot_training_curves(results_csv: Path, output_path: Path) -> None:
    """Plot loss dan mAP dari results.csv Ultralytics."""

    df = pd.read_csv(results_csv)
    metrics = {
        "train/box_loss": "Box Loss",
        "train/cls_loss": "Cls Loss",
        "metrics/mAP50": "mAP50",
        "metrics/mAP50-95": "mAP50-95",
    }
    plt.figure(figsize=(10, 6))
    for key, label in metrics.items():
        if key in df.columns:
            plt.plot(df.index, df[key], label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Nilai")
    plt.title("Kurva Training YOLO11")
    plt.legend()
    plt.grid(True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(matrix: np.ndarray, class_names: Iterable[str], output_path: Path) -> None:
    """Plot confusion matrix menggunakan seaborn heatmap."""

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Prediksi")
    plt.ylabel("Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_pr_curve(precisions: List[float], recalls: List[float], output_path: Path) -> None:
    """Membuat Precision-Recall curve sederhana."""

    plt.figure(figsize=(6, 6))
    plt.plot(recalls, precisions, marker="o")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def visualize_detections(
    image_path: Path,
    boxes: List[Tuple[float, float, float, float]],
    scores: List[float],
    class_ids: List[int],
    class_names: Iterable[str],
    output_path: Path,
) -> None:
    """Overlay deteksi ke gambar dan simpan output."""

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Gagal membuka {image_path}")
    for box, score, cls in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{list(class_names)[cls]} {score:.2f}"
        cv2.putText(img, label, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)


__all__ = [
    "plot_training_curves",
    "plot_confusion_matrix",
    "plot_pr_curve",
    "visualize_detections",
]
