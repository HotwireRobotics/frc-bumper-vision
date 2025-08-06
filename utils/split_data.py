import os
import random
import shutil
from pathlib import Path
from collections import defaultdict

# Configuration
PROCESSED_DIR = Path("data/processed")
LABELS_FLAT_DIR = Path("data/labels")
SPLIT_DIR = Path("data/split")
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

random.seed(42)

def get_matching_label(img_path):
    return LABELS_FLAT_DIR / (img_path.stem + ".txt")

def copy_pair(img_path, label_path, out_img_dir, out_label_dir):
    dest_img = out_img_dir / img_path.name
    dest_lbl = out_label_dir / label_path.name
    if dest_img.resolve() != img_path.resolve():
        shutil.copy2(img_path, dest_img)
    if dest_lbl.resolve() != label_path.resolve():
        shutil.copy2(label_path, dest_lbl)

def clear_split_folders():
    for split in ["train", "val", "test"]:
        img_dir = SPLIT_DIR / split / "images"
        lbl_dir = SPLIT_DIR / split / "labels"
        for f in img_dir.glob("*"): f.unlink()
        for f in lbl_dir.glob("*"): f.unlink()

def create_split_folders():
    for split in ["train", "val", "test"]:
        (SPLIT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (SPLIT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

def main():
    create_split_folders()

    images = sorted([p for p in PROCESSED_DIR.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    paired_images = []

    for img_path in images:
        label_path = get_matching_label(img_path)
        if label_path.exists():
            paired_images.append((img_path, label_path))
        else:
            print(f"⚠️ Skipping image with no label: {img_path.name}")

    if not paired_images:
        print("⚠️ No image-label pairs found.")
        return

    total = len(paired_images)
    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)
    test_count = total - train_count - val_count  # Ensure all are assigned

    random.shuffle(paired_images)
    train_split = paired_images[:train_count]
    val_split = paired_images[train_count:train_count + val_count]
    test_split = paired_images[train_count + val_count:]

    splits = {
        "train": train_split,
        "val": val_split,
        "test": test_split
    }

    # Clear previous split contents
    clear_split_folders()

    # Copy new split data
    for split, pairs in splits.items():
        img_out = SPLIT_DIR / split / "images"
        lbl_out = SPLIT_DIR / split / "labels"
        for img_path, label_path in pairs:
            copy_pair(img_path, label_path, img_out, lbl_out)

    print(f"✅ Balanced split complete: {len(train_split)} train, {len(val_split)} val, {len(test_split)} test.")

if __name__ == "__main__":
    main()
