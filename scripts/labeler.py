import os
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# --- CONFIG ---
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
LABEL_DIR = REPO_ROOT / "data" / "labels"
LABEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
MODEL_PATH = REPO_ROOT / "models" / "frc_bumper_run" / "weights" / "best.pt"

# Load model
model = YOLO(MODEL_PATH)

def list_unlabeled_images():
    all_images = sorted([p for p in RAW_DIR.glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    labeled_basenames = {p.stem for p in LABEL_DIR.glob("*.txt")}
    return [img for img in all_images if img.stem not in labeled_basenames]

def write_yolo_labels(predictions, image_paths):
    for pred, path in zip(predictions, image_paths):
        txt_path = LABEL_DIR / (path.stem + ".txt")
        lines = []
        if pred.boxes is not None:
            for box in pred.boxes.data.cpu().numpy():
                cls, x1, y1, x2, y2, conf = int(box[5]), *box[0:4], box[4]
                # Convert to YOLO format: class cx cy w h (normalized)
                w, h = path.stat().st_size, path.stat().st_size
                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        with open(txt_path, "w") as f:
            f.write("\n".join(lines))

def main():
    images = list_unlabeled_images()
    print(f"🖼️ Found {len(images)} images needing labels.")

    for i in tqdm(range(0, len(images), BATCH_SIZE), desc="Labeling"):
        batch = images[i:i + BATCH_SIZE]
        imgs = [cv2.imread(str(p)) for p in batch]
        results = model.predict(imgs, verbose=False)
        write_yolo_labels(results, batch)

    print("✅ Labeling complete!")

if __name__ == "__main__":
    main()
