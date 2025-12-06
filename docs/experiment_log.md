# Log Eksperimen YOLO11

Catatan ringkas untuk setiap eksperimen:

- **E1 Baseline**: yolo11s.pt, dataset COCO subset 10 kelas, augmentasi default Ultralytics dengan mosaic ringan.
- **E2 Advanced**: yolo11m.pt, augmentasi kuat (mosaic penuh, mixup, RandAugment, cosine LR, EMA).
- **E3a Ablation Tanpa Mosaic/Mixup**: mengevaluasi kontribusi mosaic/mixup.
- **E3b Ablation img768**: menguji resolusi lebih tinggi 768.

Gunakan `experiments/logs/` untuk menyimpan ringkasan otomatis dari pipeline.
