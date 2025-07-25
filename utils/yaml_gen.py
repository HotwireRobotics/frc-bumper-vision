from pathlib import Path
import yaml

BASE_DIR = Path("data/split")
TRAIN_LABELS = BASE_DIR / "train" / "labels"
OUTPUT_YAML = Path("config/data.yaml")
OUTPUT_YAML.parent.mkdir(parents=True, exist_ok=True)

def detect_classes(label_dir: Path):
    class_ids = set()
    for txt_file in label_dir.glob("*.txt"):
        with txt_file.open("r") as f:
            for line in f:
                if not line.strip():
                    continue
                class_id = line.strip().split()[0]
                if class_id.isdigit():
                    class_ids.add(int(class_id))
    class_list = sorted(class_ids)
    return class_list

def generate_data_yaml():
    class_ids = detect_classes(TRAIN_LABELS)
    class_names = [f"class_{i}" for i in class_ids]  # You can rename these manually after generation

    data = {
        "train": str((BASE_DIR / "train" / "images").resolve()),
        "val": str((BASE_DIR / "val" / "images").resolve()),
        "test": str((BASE_DIR / "test" / "images").resolve()),
        "nc": len(class_ids),
        "names": class_names
    }

    with open(OUTPUT_YAML, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"✅ YOLOv8 data.yaml generated at: {OUTPUT_YAML}")
    print(f"🔍 Detected classes: {class_names}")

if __name__ == "__main__":
    generate_data_yaml()
