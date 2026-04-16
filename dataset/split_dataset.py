"""
split_dataset_kaggle.py
───────────────────────
Splits the Kaggle "Car Plate Detection" dataset (Andrew MVD) into
train / val / test subsets and converts XML annotations to YOLO .txt format.

Dataset expected layout (source):
    <src>/
        images/          ← all .jpg images
        annotations/     ← corresponding .xml files (PASCAL VOC)

Output layout (destination):
    <dst>/
        images/
            train/  val/  test/
        labels/
            train/  val/  test/

Usage:
    python split_dataset_kaggle.py --src /path/to/kaggle_car_plate --dst ./dataset --seed 42
"""

import os
import shutil
import random
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple


def collect_image_annotation_pairs(images_dir: Path, anns_dir: Path) -> List[Tuple[Path, Path]]:
    """Return list of (image_path, annotation_path) pairs for files that have both .jpg and .xml."""
    pairs = []
    for img_path in images_dir.glob("*.png"):
        ann_path = anns_dir / f"{img_path.stem}.xml"
        if ann_path.exists():
            pairs.append((img_path, ann_path))
        else:
            print(f"Warning: No annotation for {img_path.name}, skipping.")
    return pairs


def split_pairs(pairs: List[Tuple[Path, Path]], train_ratio=0.70, val_ratio=0.15, seed=42) -> Tuple[List, List, List]:
    """Randomly split the list of (image, annotation) pairs."""
    random.seed(seed)
    shuffled = pairs[:]
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train = shuffled[:n_train]
    val   = shuffled[n_train:n_train + n_val]
    test  = shuffled[n_train + n_val:]
    return train, val, test


def convert_voc_to_yolo(ann_path, img_width, img_height) -> List[str]:
    """Parse VOC XML and return lines in YOLO format."""
    tree = ET.parse(ann_path)
    root = tree.getroot()
    
    yolo_lines = []
    # Look for object tags (ignoring XML namespaces)
    for obj in root.findall(".//object"):
        name_elem = obj.find(".//name")
        if name_elem is None:
            continue
        name = name_elem.text.lower()
        # Accept common variations of plate names
        if name not in ["licence plate", "license plate", "plate", "car plate", "licence"]:
            continue
        class_id = 0
        
        bndbox = obj.find(".//bndbox")
        if bndbox is None:
            continue
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        
        # Normalize to YOLO format
        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        width    = (xmax - xmin) / img_width
        height   = (ymax - ymin) / img_height
        
        # Clamp to [0,1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width    = max(0.0, min(1.0, width))
        height   = max(0.0, min(1.0, height))
        
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    if not yolo_lines:
        print(f"Warning: No valid bounding boxes found in {ann_path}")
    return yolo_lines


def copy_and_convert(pairs: List[Tuple[Path, Path]], dst_images: Path, dst_labels: Path):
    """Copy images and write converted YOLO label files."""
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    for img_path, ann_path in pairs:
        # Copy image
        dest_img = dst_images / img_path.name
        shutil.copy(img_path, dest_img)

        # Get image dimensions (required for normalization)
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Cannot read {img_path}, skipping label conversion.")
            continue
        h, w = img.shape[:2]

        # Convert annotation
        yolo_lines = convert_voc_to_yolo(ann_path, w, h)
        label_filename = img_path.stem + ".txt"
        label_path = dst_labels / label_filename
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="car_plate_detection", help="Path to Kaggle dataset root (contains images/ and annotations/)")
    parser.add_argument("--dst", default="dataset", help="Output directory for YOLO-style dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    images_dir = src / "images"
    anns_dir   = src / "annotations"

    if not images_dir.exists() or not anns_dir.exists():
        print(f"Error: Expected subdirectories 'images' and 'annotations' under {src}")
        return

    pairs = collect_image_annotation_pairs(images_dir, anns_dir)
    print(f"Found {len(pairs)} valid image-annotation pairs.")

    train_pairs, val_pairs, test_pairs = split_pairs(pairs, seed=args.seed)
    print(f"Train: {len(train_pairs)} images")
    print(f"Val  : {len(val_pairs)} images")
    print(f"Test : {len(test_pairs)} images")

    for subset, subset_pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        copy_and_convert(
            subset_pairs,
            dst / "images" / subset,
            dst / "labels" / subset,
        )
        print(f"  Copied {len(subset_pairs)} images → {dst}/images/{subset}/")

    print("\nSplit and conversion complete.")


if __name__ == "__main__":
    main()