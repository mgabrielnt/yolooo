# YOLO11 COCO Pipeline (RTX 3050 Friendly)

Proyek ini menyiapkan pipeline training, evaluasi, dan inferensi YOLO11 (Ultralytics) pada COCO 2017 atau subset populer, dengan fokus reproducibility dan eksperimen bag-of-freebies.

## Setup Lingkungan (Windows + VS Code)

```ps1
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> Instal PyTorch dengan CUDA sesuai GPU Anda mengikuti panduan resmi: https://pytorch.org/get-started/locally/

## Panduan PowerShell: dari Awal sampai Akhir
Urutan eksekusi lengkap di PowerShell (buka dari root repo `yolooo`):

1) **Kloning atau buka repo** (jika belum):
   ```ps1
   git clone https://github.com/mgabrielnt/yolooo.git
   cd yolooo
   ```

2) **Buat dan aktifkan virtual environment + install dependency**:
   ```ps1
   py -3.10 -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3) **Atur path COCO di YAML**: edit `configs/data_coco_full.yaml` dan `configs/data_coco_subset.yaml` pada kunci `path:` ke lokasi COCO Anda, contoh `D:/datasets/coco2017`.

4) **Validasi struktur dataset** (opsional tapi disarankan):
   ```ps1
   python - <<'PY'
   from pathlib import Path
   from src.data_utils import summarize_dataset

   print(summarize_dataset(Path('D:/datasets/coco2017')))
   PY
   ```

5) **Jalankan training**:
   - Baseline (E1)
     ```ps1
     python scripts/train_baseline.py --run-name baseline
     ```
   - Advanced bag-of-freebies (E2)
     ```ps1
     python scripts/train_advanced.py --run-name advanced
     ```
   - Ablation
     ```ps1
     python scripts/run_ablation.py --variant nomosaic
     python scripts/run_ablation.py --variant img768
     ```

6) **Pantau hasil**: output tersimpan di `experiments/runs/<nama_run>`. File `results.csv` dan gambar default Ultralytics ada di sana.

7) **Evaluasi model** (contoh memakai best advanced):
   ```ps1
   python scripts/eval_model.py `
     --weights models/yolov11/best_advanced.pt `
     --data configs/data_coco_full.yaml `
     --split val
   ```

8) **Inferensi batch gambar**:
   ```ps1
   python - <<'PY'
   from src.infer_image import infer_images

   images = ['test_images/img1.jpg', 'test_images/img2.jpg']
   infer_images('models/yolov11/best_advanced.pt', images, conf=0.25)
   PY
   ```

9) **Inferensi webcam**:
   ```ps1
   python webcam_infer.py --weights models/yolov11/best_baseline.pt --source 0
   ```

10) **Ekspor weight terbaik dari sebuah run**:
    ```ps1
    python scripts/export_best_weights.py `
      --run-dir experiments/runs/advanced `
      --output-name best_advanced.pt
    ```

Tips: jika VRAM/RAM terbatas, kecilkan `batch`, gunakan `imgsz` lebih rendah, atau nonaktifkan mosaic/mixup via YAML ablation.

## Struktur Folder

```
yolooo/
  src/                # modul inti (training, evaluasi, inferensi)
  configs/            # YAML data + hyperparameter
  scripts/            # CLI untuk training/evaluasi/export
  experiments/        # output runs & log ringkas
  models/yolov11/     # checkpoint hasil terbaik
  docs/               # catatan eksperimen & aset laporan
  webcam_infer.py     # inferensi webcam
```

## Dataset COCO 2017
1. Unduh manual (train/val/test + annotations) ke folder lokal, contoh `D:/datasets/coco2017`.
2. Edit `configs/data_coco_full.yaml` dan `configs/data_coco_subset.yaml` pada kolom `path` agar menunjuk ke folder tersebut.
3. Jalankan validasi cepat:
   ```bash
   python - <<"PY"
   from pathlib import Path
   from src.data_utils import summarize_dataset
   print(summarize_dataset(Path('/absolute/path/to/coco2017')))
   PY
   ```

## Menjalankan Training
- **Baseline (E1)**
  ```bash
  python scripts/train_baseline.py --run-name baseline
  ```
- **Advanced bag-of-freebies (E2)**
  ```bash
  python scripts/train_advanced.py --run-name advanced
  ```
- **Ablation**
  ```bash
  python scripts/run_ablation.py --variant nomosaic
  python scripts/run_ablation.py --variant img768
  ```

Output training mengikuti struktur Ultralytics di `experiments/runs/<run_name>`.

## Evaluasi & Visualisasi
```bash
python scripts/eval_model.py --weights models/yolov11/best_advanced.pt --data configs/data_coco_full.yaml
```
- Kurva training dapat dibuat dari `results.csv` menggunakan `src.vis_utils.plot_training_curves`.
- Confusion matrix & PR curve dapat dihasilkan menggunakan fungsi di `src/vis_utils.py` atau dari output Ultralytics.

## Inferensi Gambar/Video
```bash
python - <<"PY"
from src.infer_image import infer_images
infer_images('models/yolov11/best_advanced.pt', ['sample.jpg'], conf=0.25)
PY
```

## Inferensi Webcam
```bash
python webcam_infer.py --weights models/yolov11/best_baseline.pt --source 0
```

## Export best.pt
Salin weight terbaik dari run tertentu ke `models/yolov11/`:
```bash
python scripts/export_best_weights.py --run-dir experiments/runs/advanced --output-name best_advanced.pt
```

## Tips Performa (RTX 3050)
- Gunakan batch 16 untuk imgsz 640; turunkan batch jika VRAM sempit.
- Aktifkan `amp=True` (default pada config) untuk mixed precision.
- Untuk debugging cepat gunakan dataset mini (coco8/coco128) dengan mengganti `data` di YAML.
- Aktifkan `cache=ram` di config advanced saat storage lambat.

## Catatan Eksperimen
Ringkasan eksperimen disimpan di `docs/experiment_log.md` dan `docs/ablation_results.md`. Log otomatis dari pipeline berada di `experiments/logs/`.
