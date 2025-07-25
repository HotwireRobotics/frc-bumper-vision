import os
import cv2
import shutil
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
TEMP_DIR = Path(r"C:\Users\Zanea\Downloads\image_temp")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

def is_grayscale(image):
    """Check if an image is already grayscale."""
    if len(image.shape) == 2:
        return True
    if len(image.shape) == 3 and image.shape[2] == 1:
        return True
    if len(image.shape) == 3 and image.shape[2] == 3:
        b, g, r = cv2.split(image)
        return (b == g).all() and (b == r).all()
    return False

def convert_images():
    images = list(RAW_DIR.glob("*.jpg")) + list(RAW_DIR.glob("*.png")) + list(RAW_DIR.glob("*.jpeg"))

    print(f"Found {len(images)} images in raw folder.")

    # Process images and save to SSD temp folder only
    for img_path in tqdm(images, desc="Converting to grayscale"):
        temp_save_path = TEMP_DIR / img_path.name
        final_save_path = PROCESSED_DIR / img_path.name

        # Skip if already processed (check final HDD folder)
        if final_save_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ Skipping unreadable file: {img_path.name}")
            continue

        if is_grayscale(img):
            cv2.imwrite(str(temp_save_path), img)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(str(temp_save_path), gray)

    print("✅ Conversion done, now moving files to processed folder...")

    # Batch move all processed images from SSD temp to HDD processed folder
    for temp_img_path in tqdm(list(TEMP_DIR.glob("*")), desc="Moving to processed folder"):
        final_path = PROCESSED_DIR / temp_img_path.name
        if not final_path.exists():
            shutil.move(str(temp_img_path), str(final_path))

    print("✅ All images moved to processed folder!")

if __name__ == "__main__":
    convert_images()
