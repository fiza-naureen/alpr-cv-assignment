# ALPR System — Assignment #02 | Computer Vision | BSCS 07

**Automatic License Plate Recognition (ALPR) using YOLOv8 + Tesseract OCR**

| | |
|---|---|
| **Course** | Computer Vision — BSCS Semester 07 |
| **Institution** | KICSIT |
| **Submitted To** | Sir Ata Mustafa |
| **Team** | Fiza Naureen (232202012) · Syed Rayyan Hussain (232202021) · Rehan (232202030) |
| **Due Date** | 09 April 2026 |

---

## System Overview

A complete end-to-end ALPR pipeline:

```
Input Image
    │
    ▼
[Stage 1] Resize to 640×640 + letterbox + normalise
    │
    ▼
[Stage 2] YOLOv8n — License Plate Detection
    │
    ▼
[Stage 3] Confidence filter (≥0.5) + NMS (IoU=0.45)
    │
    ▼
[Stage 4] Crop plate region from original frame
    │
    ▼
[Stage 5] Greyscale → CLAHE → Adaptive Binarisation
    │
    ▼
[Stage 6] Tesseract OCR (--psm 8)
    │
    ▼
[Stage 7] Regex post-process + min-length filter
    │
    ▼
Output: plate text + bounding box + confidence
```

---

## Results Summary

| Metric | Value |
|---|---|
| **mAP@0.5 (detection)** | **0.953** |
| mAP@0.5:0.95 | 0.689 |
| Precision | 0.941 |
| Recall | 0.927 |
| F1 Score | 0.934 |
| **Full Plate Accuracy (OCR)** | **71.2 %** |
| Character Recognition Accuracy | 87.4 % |
| Character Error Rate | 0.126 |
| Inference speed | 8.3 ms/frame (~120 FPS) |
| End-to-End F1 | 0.846 |

---

## Repository Structure

```
alpr-cv-assignment/
├── dataset/
│   ├── split_dataset.py        ← 70/15/15 stratified split (seed=42)
│   └── ufpr_alpr.yaml          ← YOLOv8 dataset config
│
├── preprocessing/
│   ├── label_conversion.py     ← UFPR native format → YOLO .txt
│   └── verify_labels.py        ← bbox bounds check + integrity audit
│
├── training/
│   └── train.py                ← YOLOv8n fine-tuning script
│
├── inference/
│   └── inference.py            ← End-to-end inference pipeline
│
├── evaluation/
│   └── evaluate.py             ← Detection + OCR metrics + plot generation
│
├── results/
│   ├── PR_curve.png            ← Precision–Recall curve (auto-generated)
│   ├── loss_curves.png         ← Training/val loss curves
│   └── sample_outputs/         ← Annotated output images (Figures 1–3)
│
├── requirements.txt
└── README.md
```

---

## Dataset

**UFPR-ALPR Dataset**
- 4,500 annotated vehicle images at 1920×1080
- Real-world conditions: varying illumination, speeds, angles, occlusion
- Download: https://web.inf.ufpr.br/vri/databases/ufpr-alpr/
- Licence: Academic use only. Not redistributed in this repository.

### Dataset Split

| Split | Images | Proportion | Purpose |
|---|---|---|---|
| Train | 3,150 | 70 % | YOLOv8 fine-tuning |
| Validation | 675 | 15 % | Hyperparameter selection |
| Test | 675 | 15 % | Final unbiased evaluation |

Split performed at vehicle level with `seed=42` to prevent data leakage.

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/[team-username]/alpr-cv-assignment.git
cd alpr-cv-assignment
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Tesseract binary

**Windows:** Download installer from https://github.com/UB-Mannheim/tesseract/wiki

**Linux/macOS:**
```bash
sudo apt install tesseract-ocr   # Ubuntu/Debian
brew install tesseract            # macOS
```

### 4. Download the dataset

Download UFPR-ALPR from https://web.inf.ufpr.br/vri/databases/ufpr-alpr/ and place it in `raw_dataset/`.

---

## Reproducing the Results

Run scripts in this order:

### Step 1 — Convert labels

```bash
python preprocessing/label_conversion.py --src raw_dataset --dst dataset/labels/all
```

### Step 2 — Verify labels

```bash
python preprocessing/verify_labels.py --images dataset/images/train --labels dataset/labels/train
```

### Step 3 — Split dataset

```bash
python dataset/split_dataset.py --src raw_dataset --dst dataset --seed 42
```

### Step 4 — Train

```bash
python training/train.py
```

Training runs for up to 50 epochs with early stopping (patience=10). Best checkpoint is saved to `runs/detect/alpr_v1/weights/best.pt`.

### Step 5 — Evaluate

```bash
python evaluation/evaluate.py \
    --weights runs/detect/alpr_v1/weights/best.pt \
    --data    dataset/ufpr_alpr.yaml \
    --run     runs/detect/alpr_v1
```

Outputs metrics to console and saves plots to `results/`.

### Step 6 — Run inference

```bash
# Single image
python inference/inference.py \
    --source  dataset/images/test/your_image.jpg \
    --weights runs/detect/alpr_v1/weights/best.pt

# Entire test folder
python inference/inference.py \
    --source  dataset/images/test \
    --weights runs/detect/alpr_v1/weights/best.pt
```

Annotated images and a CSV results log are saved to `results/inference_output/`.

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Base weights | yolov8n.pt (COCO pre-trained) |
| Epochs | 50 (early stopping, patience=10) |
| Batch size | 16 |
| Image size | 640×640 |
| Optimiser | AdamW |
| Learning rate | 0.001 → cosine anneal to 0.0001 |
| Weight decay | 0.0005 |
| Warmup epochs | 3 |
| Confidence threshold | 0.5 |
| NMS IoU threshold | 0.45 |

---

## Key Findings

- YOLOv8n achieves **mAP@0.5 = 0.953** — near state-of-the-art for single-class plate detection
- Main bottleneck is **Tesseract OCR** — full plate accuracy drops to 42–47 % under dirty/distant conditions
- Recommended next step: replace Tesseract with a **CRNN** trained on license plate data (+15–20 pp accuracy)
- Pipeline runs at **~48 FPS** end-to-end, suitable for live surveillance

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| ultralytics | ≥ 8.0.0 | YOLOv8 training & inference |
| opencv-python | ≥ 4.8.0 | Image I/O, CLAHE, binarisation |
| pytesseract | ≥ 0.3.10 | OCR wrapper |
| Tesseract binary | 5.x | OCR engine |
| matplotlib | ≥ 3.7.0 | Plot generation |
| pandas | ≥ 2.0.0 | Results CSV + loss curve parsing |
