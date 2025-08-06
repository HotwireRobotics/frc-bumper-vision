import cv2
from pathlib import Path

image_dir = Path("data/split/train/images")
bad = 0

for img_path in image_dir.glob("*.*"):
    img = cv2.imread(str(img_path))
    if img is None or img.size == 0:
        print(f"⚠️ Removing corrupt image: {img_path}")
        img_path.unlink()
        bad += 1

print(f"🧹 Removed {bad} corrupt images.")
