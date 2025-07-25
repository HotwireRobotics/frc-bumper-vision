import os
import cv2
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def is_grayscale(image):
    """Check if an image is already grayscale."""
    if len(image.shape) == 2:
        return True
    if len(image.shape) == 3 and image.shape[2] == 1:
        return True
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Check if all channels are equal (common greyscale-in-RGB case)
        b, g, r = cv2.split(image)
        return (b == g).all() and (b == r).all()
    return False

def convert_images():
    images = list(RAW_DIR.glob("*.jpg")) + list(RAW_DIR.glob("*.png")) + list(RAW_DIR.glob("*.jpeg"))

    print(f"Found {len(images)} images in raw folder.")
    
    for img_path in tqdm(images, desc="Converting to grayscale"):
        dest_path = PROCESSED_DIR / img_path.name
        if dest_path.exists():
            continue  # Already converted

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ Skipping unreadable file: {img_path.name}")
            continue

        if is_grayscale(img):
            cv2.imwrite(str(dest_path), img)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(dest_path), gray)

if __name__ == "__main__":
    convert_images()
    print("✅ Conversion complete!")
