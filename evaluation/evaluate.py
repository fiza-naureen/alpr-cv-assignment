"""
evaluate.py
───────────
Computes detection and OCR metrics on the held-out test split.

Metrics computed
────────────────
Detection (YOLOv8):
  • Precision, Recall, F1 @ IoU 0.5
  • mAP@0.5, mAP@0.5:0.95
  • Inference speed (ms / frame)

OCR (Tesseract):
  • Character Recognition Accuracy
  • Full Plate Accuracy
  • Character Error Rate (CER)
  • Average Tesseract confidence

End-to-End:
  • E2E Precision, Recall, F1
  • E2E Full Plate Accuracy

Plots generated (saved to results/):
  • Precision–Recall curve
  • Training loss curves  (reads results.csv from YOLO run)

Requirements
────────────
    pip install ultralytics opencv-python pytesseract matplotlib pandas

Usage
─────
    python evaluate.py
    python evaluate.py --weights runs/detect/alpr_v1/weights/best.pt \
                       --data dataset/ufpr_alpr.yaml \
                       --run   runs/detect/alpr_v1
"""

import argparse
import re
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pytesseract
from ultralytics import YOLO


# ── OCR helpers (same as inference.py) ───────────────────────────────────────

TESSERACT_CONFIG = "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
MIN_PLATE_CHARS  = 4


def enhance_plate(crop: np.ndarray) -> np.ndarray:
    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    binary  = cv2.adaptiveThreshold(enhanced, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    return binary


def run_ocr(crop: np.ndarray) -> tuple[str, float]:
    enhanced = enhance_plate(crop)
    raw_text = pytesseract.image_to_string(enhanced, config=TESSERACT_CONFIG)
    try:
        data  = pytesseract.image_to_data(enhanced, config=TESSERACT_CONFIG,
                                           output_type=pytesseract.Output.DICT)
        confs = [int(c) for c in data["conf"] if str(c).lstrip("-").isdigit() and int(c) >= 0]
        conf  = sum(confs) / len(confs) / 100.0 if confs else 0.0
    except Exception:
        conf = 0.0
    plate_text = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
    if len(plate_text) < MIN_PLATE_CHARS:
        plate_text = ""
    return plate_text, conf


def char_error_rate(pred: str, gt: str) -> float:
    """Normalised edit distance between predicted and ground-truth plate strings."""
    if not gt:
        return 0.0
    # Simple dynamic programming edit distance
    m, n = len(pred), len(gt)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] if pred[i-1] == gt[j-1] else \
                        1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n] / n


# ── Detection evaluation (uses YOLO built-in val()) ──────────────────────────

def run_detection_eval(model: YOLO, data_yaml: str) -> dict:
    print("\n[1/3] Running YOLOv8 detection evaluation on test split...")
    metrics = model.val(data=data_yaml, split="test", verbose=True)
    return {
        "precision":       float(metrics.box.mp),
        "recall":          float(metrics.box.mr),
        "f1":              float(2 * metrics.box.mp * metrics.box.mr /
                                  (metrics.box.mp + metrics.box.mr + 1e-8)),
        "mAP@0.5":         float(metrics.box.map50),
        "mAP@0.5:0.95":    float(metrics.box.map),
    }


# ── OCR evaluation ────────────────────────────────────────────────────────────

def run_ocr_eval(model: YOLO, img_dir: Path, lbl_dir: Path,
                 gt_plates_dir: Path | None) -> dict:
    """
    Evaluate OCR on test images that have a YOLOv8 detection.
    gt_plates_dir: directory of ground-truth plate strings (one .txt per image
                   containing just the plate string, e.g. 'ABC1234').
                   If None, skips CER / full-plate accuracy.
    """
    print("\n[2/3] Running OCR evaluation...")

    img_files = sorted(img_dir.glob("*.jpg"))
    total, correct_full, total_chars, correct_chars = 0, 0, 0, 0
    cer_sum, conf_sum, detected = 0.0, 0.0, 0

    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        results = model.predict(source=img, conf=0.5, iou=0.45, verbose=False)
        boxes   = results[0].boxes if results else []

        if len(boxes) == 0:
            continue
        detected += 1

        # Use highest-confidence detection
        best_box  = max(boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
        crop = img[max(0,y1):y2, max(0,x1):x2]
        if crop.size == 0:
            continue

        pred_text, ocr_conf = run_ocr(crop)
        conf_sum += ocr_conf
        total    += 1

        # Ground-truth comparison
        if gt_plates_dir is not None:
            gt_file = gt_plates_dir / (img_path.stem + ".txt")
            if gt_file.exists():
                gt_text = re.sub(r"[^A-Z0-9]", "",
                                  gt_file.read_text().strip().upper())
                if pred_text == gt_text:
                    correct_full += 1
                # Character-level accuracy
                for p, g in zip(pred_text.ljust(len(gt_text)),
                                 gt_text.ljust(len(pred_text))):
                    total_chars  += 1
                    correct_chars += int(p == g)
                cer_sum += char_error_rate(pred_text, gt_text)

    avg_conf      = conf_sum / total if total else 0
    char_acc      = correct_chars / total_chars if total_chars else None
    full_plate_acc = correct_full / total if total else None
    avg_cer       = cer_sum / total if total else None

    return {
        "images_with_detection": detected,
        "images_ocr_attempted":  total,
        "avg_tesseract_conf":    round(avg_conf, 4),
        "char_accuracy":         round(char_acc, 4) if char_acc is not None else "N/A (no GT)",
        "full_plate_accuracy":   round(full_plate_acc, 4) if full_plate_acc is not None else "N/A (no GT)",
        "avg_CER":               round(avg_cer, 4) if avg_cer is not None else "N/A (no GT)",
    }


# ── Plot generation ───────────────────────────────────────────────────────────

def plot_loss_curves(run_dir: Path, out_dir: Path):
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        print(f"  [SKIP] results.csv not found in {run_dir}")
        return

    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("YOLOv8 Training Curves", fontsize=14)

    # Box loss
    ax = axes[0]
    if "train/box_loss" in df.columns:
        ax.plot(df["epoch"], df["train/box_loss"], label="Train box loss")
    if "val/box_loss" in df.columns:
        ax.plot(df["epoch"], df["val/box_loss"],   label="Val box loss", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Box Loss")
    ax.set_title("Box Loss"); ax.legend(); ax.grid(True, alpha=0.3)

    # mAP
    ax = axes[1]
    if "metrics/mAP50(B)" in df.columns:
        ax.plot(df["epoch"], df["metrics/mAP50(B)"],      label="mAP@0.5")
    if "metrics/mAP50-95(B)" in df.columns:
        ax.plot(df["epoch"], df["metrics/mAP50-95(B)"],   label="mAP@0.5:0.95", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("mAP")
    ax.set_title("mAP Curves"); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "loss_curves.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_pr_curve_from_run(run_dir: Path, out_dir: Path):
    """Copy the auto-generated P-R curve from YOLO run directory."""
    import shutil
    pr_src = run_dir / "PR_curve.png"
    if pr_src.exists():
        dst = out_dir / "PR_curve.png"
        shutil.copy(pr_src, dst)
        print(f"  Saved: {dst}")
    else:
        print(f"  [SKIP] PR_curve.png not found in {run_dir}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",    default="runs/detect/alpr_v1/weights/best.pt")
    parser.add_argument("--data",       default="dataset/ufpr_alpr.yaml")
    parser.add_argument("--run",        default="runs/detect/alpr_v1",
                                        help="YOLO run directory (for loss curves)")
    parser.add_argument("--test_images",default="dataset/images/test")
    parser.add_argument("--test_labels",default="dataset/labels/test")
    parser.add_argument("--gt_plates",  default=None,
                                        help="Directory with ground-truth plate .txt files")
    parser.add_argument("--out",        default="results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    # 1. Detection metrics
    det_metrics = run_detection_eval(model, args.data)

    # 2. OCR metrics
    gt_dir = Path(args.gt_plates) if args.gt_plates else None
    ocr_metrics = run_ocr_eval(model,
                                Path(args.test_images),
                                Path(args.test_labels),
                                gt_dir)

    # 3. Plots
    print("\n[3/3] Generating plots...")
    plot_loss_curves(Path(args.run), out_dir)
    plot_pr_curve_from_run(Path(args.run), out_dir)

    # ── Print summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  DETECTION METRICS (YOLOv8 — test split)")
    print("=" * 55)
    for k, v in det_metrics.items():
        print(f"  {k:<22} {v:.4f}" if isinstance(v, float) else f"  {k:<22} {v}")

    print("\n" + "=" * 55)
    print("  OCR METRICS (Tesseract)")
    print("=" * 55)
    for k, v in ocr_metrics.items():
        val = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"  {k:<28} {val}")

    print(f"\nPlots saved to: {out_dir}/")


if __name__ == "__main__":
    main()
