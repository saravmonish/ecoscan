"""
IWHR Floating Debris Dataset Converter
=======================================
Converts the IWHR_AI_Label_Floater_V1 dataset (Pascal VOC XML format)
into YOLO format compatible with our EcoScan pipeline.

Dataset: https://doi.org/10.6084/m9.figshare.27376851.v1
Paper:   https://doi.org/10.1038/s41597-025-04594-9
Images:  3,000 real waterway images from Beijing Grand Canal
Source:  Shore-based surveillance cameras + mobile devices

The IWHR dataset uses LabelImg XML (Pascal VOC format).
We map all IWHR debris classes to our 6-class taxonomy.
"""

import os
import sys
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# Our 6-class taxonomy
OUR_CLASSES = [
    "plastic_bottle",   # 0
    "plastic_bag",      # 1
    "foam_styrofoam",   # 2
    "fishing_net",      # 3
    "other_debris",     # 4
    "micro_plastic",    # 5
]

# Map IWHR class names → our class IDs
# IWHR uses LabelImg with Chinese / English labels — common variants:
IWHR_CLASS_MAP = {
    # English variants
    "floater":             4,  # generic → other_debris
    "floating":            4,
    "debris":              4,
    "trash":               4,
    "waste":               4,
    "bottle":              0,  # → plastic_bottle
    "plastic bottle":      0,
    "plastic_bottle":      0,
    "bag":                 1,  # → plastic_bag
    "plastic bag":         1,
    "plastic_bag":         1,
    "foam":                2,  # → foam_styrofoam
    "foam board":          2,
    "styrofoam":           2,
    "foam_board":          2,
    "net":                 3,  # → fishing_net
    "fishing net":         3,
    "fishing_net":         3,
    "plant":               4,  # water plants → other_debris
    "algae":               4,
    "water plant":         4,
    "leaf":                4,
    "branch":              4,
    "wood":                4,
    "grass":               4,
    # Chinese transliterations (common in LabelImg exports)
    "漂浮物":              4,
    "塑料瓶":              0,
    "塑料袋":              1,
    "泡沫":                2,
    "渔网":                3,
    "水草":                4,
    "树枝":                4,
}


def parse_voc_xml(xml_path):
    """Parse a Pascal VOC XML annotation file. Returns (width, height, list of (class_id, xmin, ymin, xmax, ymax))."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()

        # Map to our class taxonomy
        class_id = None
        for key, cid in IWHR_CLASS_MAP.items():
            if key in name or name in key:
                class_id = cid
                break
        if class_id is None:
            class_id = 4  # default: other_debris

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # Clamp to image bounds
        xmin = max(0.0, min(xmin, width - 1))
        ymin = max(0.0, min(ymin, height - 1))
        xmax = max(0.0, min(xmax, width))
        ymax = max(0.0, min(ymax, height))

        if xmax > xmin and ymax > ymin:
            objects.append((class_id, xmin, ymin, xmax, ymax))

    return width, height, objects


def voc_to_yolo(class_id, xmin, ymin, xmax, ymax, width, height):
    """Convert VOC bbox to YOLO format (normalized cx, cy, w, h)."""
    cx = (xmin + xmax) / 2.0 / width
    cy = (ymin + ymax) / 2.0 / height
    w  = (xmax - xmin) / width
    h  = (ymax - ymin) / height
    return cx, cy, w, h


def convert_iwhr_to_yolo(raw_dir, output_dir,
                          train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                          seed=42):
    """
    Convert IWHR dataset from VOC XML → YOLO format.
    Splits into train/val/test.

    Args:
        raw_dir:    Path where IWHR zip was extracted (contains images + XML)
        output_dir: Where to write the YOLO dataset (our DATA_DIR)
        train_ratio, val_ratio, test_ratio: split percentages
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    # Find all XML annotation files
    xml_files = list(raw_dir.rglob("*.xml"))
    print(f"Found {len(xml_files)} XML annotation files in {raw_dir}")

    if not xml_files:
        print("ERROR: No XML files found. Check extraction path.")
        return None

    # Pair each XML with its image
    pairs = []
    for xml_path in xml_files:
        stem = xml_path.stem
        # Try same folder, sibling 'JPEGImages', sibling 'images' folders
        search_dirs = [
            xml_path.parent,
            xml_path.parent.parent / "JPEGImages",
            xml_path.parent.parent / "images",
            xml_path.parent.parent / "Images",
        ]
        img_path = None
        for search_dir in search_dirs:
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                candidate = search_dir / (stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path:
                break
        if img_path:
            pairs.append((img_path, xml_path))
        else:
            print(f"  Warning: no image found for {xml_path.name}")

    print(f"Paired {len(pairs)} image-annotation pairs")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    splits = {
        "train": pairs[:n_train],
        "val":   pairs[n_train:n_train + n_val],
        "test":  pairs[n_train + n_val:],
    }

    class_counts = {i: 0 for i in range(6)}

    for split, split_pairs in splits.items():
        img_out = output_dir / split / "images"
        lbl_out = output_dir / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        print(f"\nConverting {split} set ({len(split_pairs)} images)...")

        for img_path, xml_path in split_pairs:
            try:
                width, height, objects = parse_voc_xml(xml_path)
            except Exception as e:
                print(f"  Skipping {xml_path.name}: {e}")
                continue

            if not objects:
                continue

            # Copy image
            dst_img = img_out / img_path.name
            shutil.copy2(img_path, dst_img)

            # Write YOLO label
            dst_lbl = lbl_out / (img_path.stem + ".txt")
            lines = []
            for class_id, xmin, ymin, xmax, ymax in objects:
                cx, cy, w, h = voc_to_yolo(class_id, xmin, ymin, xmax, ymax, width, height)
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                class_counts[class_id] += 1
            dst_lbl.write_text("\n".join(lines))

    print(f"\nClass distribution: { {OUR_CLASSES[i]: v for i, v in class_counts.items()} }")
    return class_counts


def build_dataset_yaml(output_dir):
    """Write dataset.yaml for the converted YOLO dataset."""
    import yaml
    output_dir = Path(output_dir)
    cfg = {
        "path":  str(output_dir.resolve()),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc":    6,
        "names": {i: name for i, name in enumerate(OUR_CLASSES)},
    }
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"\ndataset.yaml written → {yaml_path}")
    return str(yaml_path)


def run_conversion(raw_zip_dir, output_dir):
    """Full pipeline: find extracted IWHR folder → convert → write YAML."""
    raw_zip_dir = Path(raw_zip_dir)
    output_dir  = Path(output_dir)

    # Find the extracted IWHR folder
    iwhr_dirs = list(raw_zip_dir.glob("IWHR*")) + list(raw_zip_dir.glob("iwhr*"))
    if not iwhr_dirs:
        # Maybe extracted directly into raw_zip_dir
        iwhr_dirs = [raw_zip_dir]

    raw_dir = iwhr_dirs[0]
    print(f"Using raw dataset directory: {raw_dir}")

    class_counts = convert_iwhr_to_yolo(raw_dir, output_dir)
    if class_counts is None:
        return None

    yaml_path = build_dataset_yaml(output_dir)
    return yaml_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert IWHR dataset to YOLO format")
    parser.add_argument("--raw",    required=True, help="Path to extracted IWHR dataset")
    parser.add_argument("--output", required=True, help="Output directory (YOLO format)")
    args = parser.parse_args()
    run_conversion(args.raw, args.output)
