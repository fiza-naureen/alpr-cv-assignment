"""
verify_labels.py
────────────────
Runs integrity checks on YOLO-format label files before training.

Checks performed:
  1. All bbox coordinates are within [0, 1] (normalised space).
  2. No image file is missing its corresponding label file.
  3. Prints a random sample of 10 labels for manual spot-check.

Usage
─────
python verify_labels.py --images dataset/images/train --labels dataset/labels/train
"""

import argparse
import random
from pathlib import Path


def check_bbox_bounds(label_path: Path):
    """Return list of out-of-bounds lines (if any)."""
    bad = []
    for i, line in enumerate(label_path.read_text().strip().splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            bad.append((i, line, "wrong number of fields"))
            continue
        _, cx, cy, w, h = parts
        vals = list(map(float, [cx, cy, w, h]))
        if not all(0.0 <= v <= 1.0 for v in vals):
            bad.append((i, line, f"value out of [0,1]: {vals}"))
    return bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="dataset/images/train")
    parser.add_argument("--labels", default="dataset/labels/train")
    args = parser.parse_args()

    img_dir = Path(args.images)
    lbl_dir = Path(args.labels)

    img_files = sorted(img_dir.glob("*.jpg"))
    lbl_files = {p.stem for p in lbl_dir.glob("*.txt")}

    print(f"Images  : {len(img_files)}")
    print(f"Labels  : {len(lbl_files)}")

    missing, oob_count = 0, 0

    for img_path in img_files:
        if img_path.stem not in lbl_files:
            print(f"  [MISSING LABEL] {img_path.name}")
            missing += 1
            continue

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        bad_lines = check_bbox_bounds(lbl_path)
        for line_no, line, reason in bad_lines:
            print(f"  [OOB] {lbl_path.name} line {line_no}: {reason}")
            oob_count += 1

    print(f"\nSummary")
    print(f"  Missing label files : {missing}")
    print(f"  Out-of-bounds boxes : {oob_count}")

    # Random spot-check sample
    sample = random.sample(list(lbl_files), min(10, len(lbl_files)))
    print(f"\nRandom spot-check (10 labels):")
    for stem in sample:
        lbl_path = lbl_dir / (stem + ".txt")
        content = lbl_path.read_text().strip()
        print(f"  {stem}.txt → {content}")


if __name__ == "__main__":
    main()
