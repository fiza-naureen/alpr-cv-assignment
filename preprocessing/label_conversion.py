"""
label_conversion.py
───────────────────
Converts UFPR-ALPR native annotation files to YOLO .txt format.

UFPR native format (per frame):
    plate: ABC1234
    position_vehicle: x1 y1 x2 y2
    position_plate: x1 y1 x2 y2    ← we use this
    ...

YOLO format (one line per object):
    <class_index> <x_centre> <y_centre> <width> <height>
    All values normalised to [0, 1] relative to image dimensions.

For UFPR-ALPR the single class is: license_plate → index 0
Image resolution: 1920 × 1080 (used for normalisation)

Usage
─────
python label_conversion.py --src raw_dataset --dst dataset/labels/all
"""

import os
import re
import argparse
from pathlib import Path

# UFPR-ALPR native image resolution
IMG_W = 1920
IMG_H = 1080


def xyxy_to_yolo(x1, y1, x2, y2, img_w=IMG_W, img_h=IMG_H):
    """Convert absolute [x1,y1,x2,y2] to normalised YOLO [cx,cy,w,h]."""
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    # Clamp to [0, 1]
    cx, cy = max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))
    w,  h  = max(0.0, min(1.0, w)),  max(0.0, min(1.0, h))
    return cx, cy, w, h


def parse_ufpr_annotation(ann_path: Path):
    """
    Parse a UFPR native annotation file.
    Returns (x1, y1, x2, y2) of the plate bounding box, or None on failure.
    """
    text = ann_path.read_text(errors="ignore")
    match = re.search(r"position_plate:\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", text)
    if not match:
        return None
    x1, y1, x2, y2 = map(int, match.groups())
    return x1, y1, x2, y2


def convert_file(ann_path: Path, dst_dir: Path, img_w=IMG_W, img_h=IMG_H):
    coords = parse_ufpr_annotation(ann_path)
    if coords is None:
        print(f"  [SKIP] No plate annotation found: {ann_path.name}")
        return False

    x1, y1, x2, y2 = coords
    cx, cy, w, h = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)

    # Basic validity check
    if not (0 < w <= 1 and 0 < h <= 1):
        print(f"  [WARN] Out-of-bounds bbox corrected: {ann_path.name}")

    out_name = ann_path.stem + ".txt"
    out_path = dst_dir / out_name
    out_path.write_text(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="raw_dataset",      help="Path to raw UFPR-ALPR dataset")
    parser.add_argument("--dst", default="dataset/labels/all", help="Output directory for YOLO .txt files")
    parser.add_argument("--img_w", type=int, default=1920)
    parser.add_argument("--img_h", type=int, default=1080)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(src.rglob("*.txt"))
    print(f"Found {len(ann_files)} annotation files in {src}")

    ok, skip = 0, 0
    for ann_path in ann_files:
        # Skip already-converted YOLO files (single-line, starts with digit)
        first_line = ann_path.read_text(errors="ignore").strip().splitlines()[0] if ann_path.stat().st_size else ""
        if re.match(r"^\d[\d.e\- ]+$", first_line):
            continue  # already YOLO format
        if convert_file(ann_path, dst, args.img_w, args.img_h):
            ok += 1
        else:
            skip += 1

    print(f"\nDone. Converted: {ok} | Skipped: {skip}")
    print(f"YOLO labels saved to: {dst}")


if __name__ == "__main__":
    main()
