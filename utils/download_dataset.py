"""
Real Floating Debris Dataset Downloader
========================================
Downloads the open-source Floating Debris dataset from Roboflow Universe.
Dataset: https://universe.roboflow.com/garbage-detection-1/floating-debris

This replaces the synthetic dataset generator with real labeled waterway images.
- 10,000+ labeled images of floating debris in rivers, canals, and coastal waterways
- 6 classes matching our taxonomy
- Pre-split into train/val/test in YOLO format

Usage:
    python utils/download_dataset.py

Or call download_real_dataset() from main.py instead of generate_dataset().
"""

import os
import sys
import yaml
import shutil
from pathlib import Path


# ── Class mapping ─────────────────────────────────────────────────────────────
# Our 6-class taxonomy (from the assignment):
OUR_CLASSES = [
    "plastic_bottle",   # 0
    "plastic_bag",      # 1
    "foam_styrofoam",   # 2
    "fishing_net",      # 3
    "other_debris",     # 4
    "micro_plastic",    # 5
]

# Roboflow "Floating Debris" dataset classes → map to our taxonomy
# (adjust if the downloaded dataset uses different names)
ROBOFLOW_CLASS_MAP = {
    "plastic_bottle": 0,
    "bottle":         0,
    "plastic_bag":    1,
    "bag":            1,
    "polythene":      1,
    "foam":           2,
    "styrofoam":      2,
    "fishing_net":    3,
    "net":            3,
    "debris":         4,
    "other":          4,
    "trash":          4,
    "waste":          4,
    "micro_plastic":  5,
    "microplastic":   5,
}


def download_real_dataset(base_dir, api_key=None):
    """
    Download the Floating Debris dataset from Roboflow.

    Args:
        base_dir: Where to save the dataset (same as DATA_DIR in main.py)
        api_key:  Your Roboflow API key. Get one free at https://roboflow.com
                  Or set the ROBOFLOW_API_KEY environment variable.

    Returns:
        Path to dataset.yaml
    """
    from roboflow import Roboflow

    # Get API key
    if api_key is None:
        api_key = os.environ.get("ROBOFLOW_API_KEY")

    if not api_key:
        print("\n" + "=" * 60)
        print("ERROR: Roboflow API key required.")
        print("=" * 60)
        print("\nTo download the real Floating Debris dataset:")
        print("  1. Go to https://roboflow.com and create a free account")
        print("  2. Copy your API key from Settings → Workspace")
        print("  3. Run: python utils/download_dataset.py --key YOUR_KEY")
        print("     OR set: export ROBOFLOW_API_KEY=YOUR_KEY")
        print("\nFalling back to synthetic dataset generator...")
        return None

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    print("Connecting to Roboflow...")
    rf = Roboflow(api_key=api_key)

    # Primary: Floating Debris dataset on Roboflow Universe
    # https://universe.roboflow.com/garbage-detection-1/floating-debris
    try:
        print("Downloading 'Floating Debris' dataset from Roboflow Universe...")
        project = rf.workspace("garbage-detection-1").project("floating-debris")
        dataset = project.version(1).download("yolov8", location=str(base_dir))

    except Exception as e:
        print(f"Primary dataset unavailable ({e})")
        print("Trying alternative: 'Aquatic Garbage' dataset...")

        try:
            project = rf.workspace("").project("aquatic-garbage-detection")
            dataset = project.version(1).download("yolov8", location=str(base_dir))
        except Exception as e2:
            print(f"Alternative also unavailable ({e2})")
            print("\nManual download instructions:")
            print("  1. Go to: https://universe.roboflow.com")
            print("  2. Search: 'floating debris waterway'")
            print("  3. Download in YOLOv8 format to:")
            print(f"     {base_dir}")
            return None

    # Locate the downloaded dataset.yaml and fix paths
    yaml_path = _find_and_fix_yaml(base_dir)
    print(f"\nReal dataset ready at: {yaml_path}")
    return yaml_path


def _find_and_fix_yaml(base_dir):
    """Find dataset.yaml in downloaded folder and ensure paths are correct."""
    base_dir = Path(base_dir)

    # Roboflow downloads to a subfolder — find dataset.yaml
    yaml_files = list(base_dir.rglob("data.yaml")) + list(base_dir.rglob("dataset.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No dataset YAML found under {base_dir}")

    src_yaml = yaml_files[0]
    with open(src_yaml) as f:
        cfg = yaml.safe_load(f)

    # Roboflow uses absolute paths — rewrite to relative
    dataset_root = src_yaml.parent

    cfg["path"] = str(dataset_root.resolve())
    cfg["train"] = "train/images"
    cfg["val"] = "valid/images" if (dataset_root / "valid").exists() else "val/images"
    cfg["test"] = "test/images"

    # Remap classes to our taxonomy if needed
    roboflow_names = cfg.get("names", {})
    if isinstance(roboflow_names, list):
        roboflow_names = {i: n for i, n in enumerate(roboflow_names)}

    print(f"\nDownloaded dataset classes: {list(roboflow_names.values())}")

    # Check if classes match — if different, remap labels
    our_names_set = set(OUR_CLASSES)
    downloaded_names_set = set(roboflow_names.values())

    if not our_names_set.issubset(downloaded_names_set):
        print("Class names differ from our taxonomy — remapping labels...")
        _remap_labels(dataset_root, roboflow_names)

    # Write final dataset.yaml
    final_yaml = dataset_root / "dataset.yaml"
    cfg["names"] = {i: name for i, name in enumerate(OUR_CLASSES)}
    cfg["nc"] = 6
    with open(final_yaml, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    return str(final_yaml)


def _remap_labels(dataset_root, roboflow_class_names):
    """
    Remap downloaded YOLO label files to our 6-class taxonomy.
    Unmapped classes are merged into 'other_debris' (class 4).
    """
    dataset_root = Path(dataset_root)

    # Build old_id → new_id mapping
    id_map = {}
    for old_id, name in roboflow_class_names.items():
        name_lower = name.lower().replace(" ", "_")
        new_id = ROBOFLOW_CLASS_MAP.get(name_lower, 4)  # default: other_debris
        id_map[int(old_id)] = new_id
        print(f"  {old_id}:{name} → {new_id}:{OUR_CLASSES[new_id]}")

    # Remap all label files
    for split in ["train", "valid", "val", "test"]:
        label_dir = dataset_root / split / "labels"
        if not label_dir.exists():
            continue
        for label_file in label_dir.glob("*.txt"):
            lines = label_file.read_text().strip().split("\n")
            new_lines = []
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split()
                old_id = int(parts[0])
                new_id = id_map.get(old_id, 4)
                new_lines.append(f"{new_id} " + " ".join(parts[1:]))
            label_file.write_text("\n".join(new_lines))

    print(f"  Label remapping complete.")


def check_dataset_exists(base_dir):
    """Check if a real (non-synthetic) dataset already exists."""
    base_dir = Path(base_dir)
    yaml_path = base_dir / "dataset.yaml"
    if not yaml_path.exists():
        return False

    # Check it has enough images to be real (>200 in train)
    train_images = list((base_dir / "train" / "images").glob("*.jpg"))
    train_images += list((base_dir / "train" / "images").glob("*.png"))

    if len(train_images) > 200:
        return True
    return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Floating Debris dataset")
    parser.add_argument("--key", type=str, default=None,
                        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)")
    parser.add_argument("--dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                             "data", "floating_debris"),
                        help="Directory to save dataset")
    args = parser.parse_args()

    yaml_path = download_real_dataset(args.dir, api_key=args.key)
    if yaml_path:
        print(f"\nSuccess! Dataset YAML: {yaml_path}")
        print("Now run: python main.py")
    else:
        print("\nDownload failed. Check your API key and try again.")
