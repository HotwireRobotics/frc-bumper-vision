import os
import random
import shutil
from pathlib import Path

# Configuration
PROCESSED_DIR = Path("data/processed")
LABELS_FLAT_DIR = Path("data/labels")  # All .txt files in here, flat
SPLIT_DIR = Path("data/split")
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

# Define split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# Set random seed for reproducibility
random.seed(42)

def get_matching_label(img_path):
    """Returns the label path matching the image name."""
    return LABELS_FLAT_DIR / (img_path.stem + ".txt")

def copy_pair(img_path, label_path, out_img_dir, out_label_dir):
    shutil.copy2(img_path, out_img_dir / img_path.name)
    shutil.copy2(label_path, out_label_dir / label_path.name)

def create_split_folders():
    for split in ["train", "val", "test"]:
        (SPLIT_DIR / split).mkdir(exist_ok=True)
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

    random.shuffle(paired_images)
    total = len(paired_images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": paired_images[:train_end],
        "val": paired_images[train_end:val_end],
        "test": paired_images[val_end:]
    }

    for split, pairs in splits.items():
        img_out = SPLIT_DIR / split / "images"
        lbl_out = SPLIT_DIR / split / "labels"

        for img_path, label_path in pairs:
            copy_pair(img_path, label_path, img_out, lbl_out)

    print(f"✅ Done! Split {total} image-label pairs into train/val/test.")

if __name__ == "__main__":
    main()
