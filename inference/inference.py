"""
inference.py
────────────
End-to-end ALPR inference pipeline.

Pipeline stages (per frame):
  1. Resize to 640 × 640 with letterboxing + normalise to [0, 1]
  2. YOLOv8 forward pass → bounding boxes + confidence scores
  3. Filter confidence < 0.5 ; NMS (IoU = 0.45)
  4. Crop plate region from original frame
  5. Enhance: greyscale → CLAHE → adaptive binarisation
  6. Tesseract OCR (--psm 8 = single word)
  7. Regex post-process: keep alphanumeric, min 4 chars
  8. Overlay results on image + write to results log

Requirements
────────────
    pip install ultralytics opencv-python pytesseract pillow
    Install Tesseract binary: https://github.com/UB-Mannheim/tesseract/wiki

Usage
─────
    # Single image
    python inference.py --source test_image.jpg --weights runs/detect/alpr_v1/weights/best.pt

    # Folder of images
    python inference.py --source dataset/images/test --weights runs/detect/alpr_v1/weights/best.pt

    # Webcam (device index 0)
    python inference.py --source 0 --weights runs/detect/alpr_v1/weights/best.pt
"""

import argparse
import re
import csv
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO

# ── Configuration ────────────────────────────────────────────────────────────
CONF_THRESHOLD  = 0.5
IOU_THRESHOLD   = 0.45
MIN_PLATE_CHARS = 4
TESSERACT_CONFIG = "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Bounding box colours (BGR)
COLOR_FULL   = (0, 200, 0)    # green  — full recognition
COLOR_PARTIAL = (0, 200, 255) # yellow — partial recognition
COLOR_MISS   = (0, 0, 220)    # red    — no text extracted


# ── OCR Preprocessing ────────────────────────────────────────────────────────

def enhance_plate(crop: np.ndarray) -> np.ndarray:
    """
    Apply enhancement pipeline to a cropped plate image before OCR.
    Steps: greyscale → Gaussian blur → CLAHE → adaptive binarisation
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Slight blur to reduce salt-and-pepper noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # Adaptive Gaussian threshold → binary image
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11, C=2
    )
    return binary


# ── OCR ──────────────────────────────────────────────────────────────────────

def run_ocr(crop: np.ndarray) -> tuple[str, float]:
    """
    Run Tesseract on a raw (BGR) plate crop.
    Returns (plate_text, tesseract_confidence).
    """
    enhanced = enhance_plate(crop)

    # Tesseract expects a PIL image or numpy array
    raw_text = pytesseract.image_to_string(enhanced, config=TESSERACT_CONFIG)

    # Confidence score (average of per-character confidences)
    try:
        data = pytesseract.image_to_data(enhanced, config=TESSERACT_CONFIG,
                                          output_type=pytesseract.Output.DICT)
        confs = [int(c) for c in data["conf"] if str(c).lstrip("-").isdigit() and int(c) >= 0]
        confidence = sum(confs) / len(confs) / 100.0 if confs else 0.0
    except Exception:
        confidence = 0.0

    # Post-processing: keep only alphanumeric, uppercase, min length
    plate_text = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
    if len(plate_text) < MIN_PLATE_CHARS:
        plate_text = ""

    return plate_text, confidence


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_image(model: YOLO, img_path: Path, out_dir: Path, log_writer):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [ERROR] Cannot read image: {img_path}")
        return

    start = time.perf_counter()

    # YOLOv8 inference
    results = model.predict(
        source=img,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    annotated = img.copy()
    detections = results[0].boxes if results else []

    if len(detections) == 0:
        print(f"  [NO DETECTION] {img_path.name}  ({elapsed_ms:.1f} ms)")
        log_writer.writerow([img_path.name, "", "", "", "no_detection", f"{elapsed_ms:.1f}"])
    else:
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            det_conf = float(box.conf[0])

            # Crop & OCR
            crop = img[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            plate_text, ocr_conf = run_ocr(crop)

            # Choose overlay colour
            if plate_text:
                color = COLOR_FULL if ocr_conf > 0.6 else COLOR_PARTIAL
                status = "ok" if ocr_conf > 0.6 else "partial"
            else:
                color = COLOR_MISS
                status = "ocr_failed"
                plate_text = "???"

            # Draw bounding box + label
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{plate_text}  ({det_conf:.2f})"
            cv2.putText(annotated, label, (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            print(f"  {img_path.name}  →  plate: {plate_text}  "
                  f"det_conf: {det_conf:.2f}  ocr_conf: {ocr_conf:.2f}  "
                  f"status: {status}  ({elapsed_ms:.1f} ms)")

            log_writer.writerow([
                img_path.name, plate_text,
                f"{det_conf:.4f}", f"{ocr_conf:.4f}",
                status, f"{elapsed_ms:.1f}"
            ])

    # Save annotated image
    out_path = out_dir / img_path.name
    cv2.imwrite(str(out_path), annotated)


def run_on_source(model: YOLO, source: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "results_log.csv"

    src = Path(source)

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "plate_text", "det_conf", "ocr_conf", "status", "inference_ms"])

        if src.is_dir():
            img_files = sorted(src.glob("*.jpg")) + sorted(src.glob("*.png"))
            print(f"Processing {len(img_files)} images from {src}")
            for img_path in img_files:
                process_image(model, img_path, out_dir, writer)
        elif src.is_file():
            process_image(model, src, out_dir, writer)
        else:
            # Webcam / video
            cap = cv2.VideoCapture(int(source) if source.isdigit() else str(source))
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                tmp = out_dir / f"frame_{frame_idx:05d}.jpg"
                cv2.imwrite(str(tmp), frame)
                process_image(model, tmp, out_dir, writer)
                frame_idx += 1
            cap.release()

    print(f"\nResults log : {log_path}")
    print(f"Annotated images : {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",  required=True, help="Image path, folder, or webcam index")
    parser.add_argument("--weights", default="runs/detect/alpr_v1/weights/best.pt")
    parser.add_argument("--out",     default="results/inference_output")
    args = parser.parse_args()

    print(f"Loading model from: {args.weights}")
    model = YOLO(args.weights)

    run_on_source(model, args.source, Path(args.out))


if __name__ == "__main__":
    main()
