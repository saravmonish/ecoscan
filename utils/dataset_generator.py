"""
Synthetic Floating Debris Dataset Generator
Generates images mimicking waterway debris with 6 classes:
  0: plastic_bottle, 1: plastic_bag, 2: foam_styrofoam,
  3: fishing_net, 4: other_debris, 5: micro_plastic
"""

import os
import cv2
import numpy as np
import random
import yaml
from pathlib import Path


CLASS_NAMES = [
    "plastic_bottle", "plastic_bag", "foam_styrofoam",
    "fishing_net", "other_debris", "micro_plastic"
]

# ── Exact class distribution from the assignment PDF ─────────────────────────
# Total 10,000 images: 7,000 train | 1,500 val | 1,500 test
# Each number = images where that class is the PRIMARY object
PDF_CLASS_COUNTS = [3420, 2180, 1650, 1280, 980, 490]   # sums to 10,000

# Weights (used only when generating a random split, not exact counts)
CLASS_WEIGHTS = [c / sum(PDF_CLASS_COUNTS) for c in PDF_CLASS_COUNTS]
# → [0.342, 0.218, 0.165, 0.128, 0.098, 0.049]


def generate_water_background(size=640):
    """Generate realistic water-like background with varying conditions."""
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Base water color (blue-green variations)
    base_colors = [
        (120, 80, 40),   # murky river
        (160, 120, 60),  # canal
        (180, 140, 80),  # clear water
        (100, 70, 50),   # turbid
        (140, 100, 55),  # coastal
    ]
    base = random.choice(base_colors)
    img[:] = base

    # Add Perlin-like noise for water texture
    for scale in [30, 60, 120]:
        noise = np.random.randint(-15, 15, (size // scale + 1, size // scale + 1, 3), dtype=np.int16)
        noise_resized = cv2.resize(noise.astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
        img = np.clip(img.astype(np.int16) + noise_resized.astype(np.int16), 0, 255).astype(np.uint8)

    # Random ripple/wave patterns
    for _ in range(random.randint(3, 8)):
        y = random.randint(0, size - 1)
        amplitude = random.randint(1, 3)
        freq = random.uniform(0.01, 0.05)
        for x in range(size):
            offset = int(amplitude * np.sin(freq * x))
            y_new = min(max(y + offset, 0), size - 1)
            color_shift = random.randint(-10, 10)
            img[y_new, x] = np.clip(img[y_new, x].astype(np.int16) + color_shift, 0, 255).astype(np.uint8)

    # Random glare/reflection spots
    if random.random() < 0.3:
        cx, cy = random.randint(50, size-50), random.randint(50, size-50)
        radius = random.randint(20, 80)
        cv2.circle(img, (cx, cy), radius, (200, 200, 180), -1)
        img = cv2.GaussianBlur(img, (21, 21), 10)

    return img


def draw_plastic_bottle(img, bbox):
    """Draw a simplified plastic bottle shape."""
    x, y, w, h = bbox
    color = (random.randint(180, 255), random.randint(180, 255), random.randint(180, 255))

    # Bottle body (rectangle with rounded ends)
    cx, cy = x + w // 2, y + h // 2
    angle = random.randint(-30, 30)

    pts = np.array([
        [x + w // 4, y],
        [x + 3 * w // 4, y],
        [x + 3 * w // 4, y + h // 5],
        [x + w, y + h // 4],
        [x + w, y + 3 * h // 4],
        [x + 3 * w // 4, y + h],
        [x + w // 4, y + h],
        [x, y + 3 * h // 4],
        [x, y + h // 4],
        [x + w // 4, y + h // 5],
    ], dtype=np.int32)

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    pts_float = pts.astype(np.float32).reshape(-1, 1, 2)
    pts_rot = cv2.transform(pts_float, M).astype(np.int32)

    cv2.fillPoly(img, [pts_rot], color)
    cv2.polylines(img, [pts_rot], True, (100, 100, 100), 1)

    # Add label/cap detail
    cap_color = (random.randint(0, 100), random.randint(0, 100), random.randint(200, 255))
    cap_pts = pts_rot[:3].reshape(-1, 2)
    if len(cap_pts) >= 3:
        cv2.fillPoly(img, [pts_rot[:3]], cap_color)

    return img


def draw_plastic_bag(img, bbox):
    """Draw an irregular plastic bag shape."""
    x, y, w, h = bbox
    alpha = random.uniform(0.3, 0.7)  # Semi-transparent

    # Irregular polygon for bag shape
    n_points = random.randint(6, 10)
    pts = []
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        r = random.uniform(0.3, 0.5) * min(w, h)
        px = int(x + w / 2 + r * np.cos(angle))
        py = int(y + h / 2 + r * np.sin(angle))
        pts.append([px, py])
    pts = np.array(pts, dtype=np.int32)

    overlay = img.copy()
    color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Wrinkle lines
    for _ in range(3):
        p1 = pts[random.randint(0, len(pts) - 1)]
        p2 = pts[random.randint(0, len(pts) - 1)]
        cv2.line(img, tuple(p1), tuple(p2), (180, 180, 180), 1)

    return img


def draw_foam(img, bbox):
    """Draw foam/styrofoam pieces."""
    x, y, w, h = bbox
    color = (random.randint(230, 255), random.randint(230, 255), random.randint(220, 240))

    # Irregular chunky shape
    pts = np.array([
        [x + random.randint(0, w // 4), y + random.randint(0, h // 4)],
        [x + w - random.randint(0, w // 4), y + random.randint(0, h // 4)],
        [x + w - random.randint(0, w // 3), y + h - random.randint(0, h // 4)],
        [x + random.randint(0, w // 3), y + h - random.randint(0, h // 4)],
    ], dtype=np.int32)

    cv2.fillPoly(img, [pts], color)

    # Texture dots (styrofoam texture)
    for _ in range(random.randint(5, 15)):
        dx, dy = random.randint(x, x + w), random.randint(y, y + h)
        cv2.circle(img, (dx, dy), 1, (200, 200, 200), -1)

    return img


def draw_fishing_net(img, bbox):
    """Draw fishing net pattern."""
    x, y, w, h = bbox
    color = (random.randint(80, 140), random.randint(100, 160), random.randint(80, 120))

    # Grid/net pattern
    spacing = max(w, h) // random.randint(4, 8)
    if spacing < 3:
        spacing = 3
    for i in range(0, w, spacing):
        cv2.line(img, (x + i, y), (x + i + random.randint(-5, 5), y + h), color, 1)
    for j in range(0, h, spacing):
        cv2.line(img, (x, y + j), (x + w, y + j + random.randint(-5, 5)), color, 1)

    return img


def draw_other_debris(img, bbox):
    """Draw miscellaneous debris."""
    x, y, w, h = bbox
    color = (random.randint(50, 180), random.randint(50, 150), random.randint(30, 120))

    shape = random.choice(["ellipse", "rect", "irregular"])
    if shape == "ellipse":
        cv2.ellipse(img, (x + w // 2, y + h // 2), (w // 2, h // 2),
                    random.randint(-45, 45), 0, 360, color, -1)
    elif shape == "rect":
        angle = random.randint(-20, 20)
        rect_pts = np.array([
            [x, y], [x + w, y], [x + w, y + h], [x, y + h]
        ], dtype=np.float32).reshape(-1, 1, 2)
        M = cv2.getRotationMatrix2D((x + w / 2, y + h / 2), angle, 1.0)
        rect_pts = cv2.transform(rect_pts, M).astype(np.int32)
        cv2.fillPoly(img, [rect_pts], color)
    else:
        n = random.randint(4, 7)
        pts = []
        for i in range(n):
            angle = 2 * np.pi * i / n
            r = random.uniform(0.3, 0.5) * min(w, h)
            pts.append([int(x + w / 2 + r * np.cos(angle)),
                        int(y + h / 2 + r * np.sin(angle))])
        cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], color)

    return img


def draw_micro_plastic(img, bbox):
    """Draw translucent micro-plastic particle (small, hard to see)."""
    x, y, w, h = bbox
    overlay = img.copy()
    alpha = random.uniform(0.15, 0.4)  # Very translucent

    color = (random.randint(180, 240), random.randint(180, 240), random.randint(180, 240))

    shape = random.choice(["circle", "fragment", "fiber"])
    if shape == "circle":
        cv2.circle(overlay, (x + w // 2, y + h // 2), max(w, h) // 2, color, -1)
    elif shape == "fragment":
        pts = np.array([
            [x + random.randint(0, w // 3), y + random.randint(0, h // 3)],
            [x + w - random.randint(0, w // 3), y + random.randint(0, h // 3)],
            [x + w - random.randint(0, w // 3), y + h - random.randint(0, h // 3)],
            [x + random.randint(0, w // 3), y + h - random.randint(0, h // 3)],
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], color)
    else:  # fiber
        cv2.line(overlay, (x, y + h // 2), (x + w, y + h // 2 + random.randint(-h // 4, h // 4)),
                 color, max(1, min(w, h) // 4))

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img


DRAW_FUNCTIONS = [
    draw_plastic_bottle, draw_plastic_bag, draw_foam,
    draw_fishing_net, draw_other_debris, draw_micro_plastic
]


def generate_bbox(img_size, class_id):
    """Generate bounding box sized appropriately for class."""
    if class_id == 5:  # micro-plastic: small (median 32x32)
        w = random.randint(16, 48)
        h = random.randint(16, 48)
    elif class_id in [3, 4]:  # nets & other: medium
        w = random.randint(40, 150)
        h = random.randint(40, 150)
    else:  # macro-plastics: larger (median 128x96)
        w = random.randint(60, 200)
        h = random.randint(60, 200)

    margin = 10
    x = random.randint(margin, max(margin + 1, img_size - w - margin))
    y = random.randint(margin, max(margin + 1, img_size - h - margin))

    # Clamp
    x = min(x, img_size - w - 1)
    y = min(y, img_size - h - 1)
    x = max(0, x)
    y = max(0, y)

    return x, y, w, h


def generate_single_image(img_size=640):
    """Generate one image with random debris objects."""
    img = generate_water_background(img_size)
    annotations = []

    n_objects = random.randint(1, 6)
    for _ in range(n_objects):
        class_id = random.choices(range(6), weights=CLASS_WEIGHTS, k=1)[0]
        bbox = generate_bbox(img_size, class_id)
        x, y, w, h = bbox

        # Draw object
        DRAW_FUNCTIONS[class_id](img, bbox)

        # YOLO format: class_id x_center y_center width height (normalized)
        x_center = (x + w / 2) / img_size
        y_center = (y + h / 2) / img_size
        w_norm = w / img_size
        h_norm = h / img_size

        annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    return img, annotations


def _split_class_counts(total_counts, n_train, n_val, n_test):
    """
    Distribute per-class image counts across train/val/test splits
    so that:
      - train gets floor(count * train_ratio) images per class
      - val   gets floor(count * val_ratio)
      - test  gets the remainder (so total is exact)
    Returns dict: {"train": [list of 6], "val": [...], "test": [...]}
    """
    total = sum(total_counts)
    train_r = n_train / total
    val_r   = n_val   / total

    split_counts = {"train": [], "val": [], "test": []}
    for c in total_counts:
        tr = int(c * train_r)
        vl = int(c * val_r)
        te = c - tr - vl          # remainder goes to test (keeps total exact)
        split_counts["train"].append(tr)
        split_counts["val"].append(vl)
        split_counts["test"].append(te)

    return split_counts


def generate_image_with_primary_class(primary_class_id, img_size=640):
    """
    Generate one image that is guaranteed to contain at least one
    object of `primary_class_id` plus 0–3 random secondary objects.
    Returns (img, annotations).
    """
    img = generate_water_background(img_size)
    annotations = []

    # Primary object (guaranteed)
    bbox = generate_bbox(img_size, primary_class_id)
    DRAW_FUNCTIONS[primary_class_id](img, bbox)
    x, y, w, h = bbox
    annotations.append(
        f"{primary_class_id} {(x + w / 2) / img_size:.6f} "
        f"{(y + h / 2) / img_size:.6f} "
        f"{w / img_size:.6f} {h / img_size:.6f}"
    )

    # 0–3 random secondary objects
    n_extra = random.randint(0, 3)
    for _ in range(n_extra):
        class_id = random.choices(range(6), weights=CLASS_WEIGHTS, k=1)[0]
        bbox2 = generate_bbox(img_size, class_id)
        DRAW_FUNCTIONS[class_id](img, bbox2)
        x2, y2, w2, h2 = bbox2
        annotations.append(
            f"{class_id} {(x2 + w2 / 2) / img_size:.6f} "
            f"{(y2 + h2 / 2) / img_size:.6f} "
            f"{w2 / img_size:.6f} {h2 / img_size:.6f}"
        )

    return img, annotations


def generate_dataset(base_dir, n_train=7000, n_val=1500, n_test=1500, img_size=640):
    """
    Generate Floating Debris dataset in YOLO format with EXACT class distribution
    matching the assignment PDF:
        plastic_bottle  : 3,420 images
        plastic_bag     : 2,180 images
        foam_styrofoam  : 1,650 images
        fishing_net     : 1,280 images
        other_debris    :   980 images
        micro_plastic   :   490 images
        ─────────────────────────────
        Total           : 10,000 images  (7,000 train | 1,500 val | 1,500 test)

    Each image has one guaranteed primary-class object plus 0–3 secondary objects.
    """
    base_dir = Path(base_dir)

    total = n_train + n_val + n_test   # should be 10,000
    # Scale PDF counts if a different total is requested
    scale = total / sum(PDF_CLASS_COUNTS)
    scaled_counts = [max(1, round(c * scale)) for c in PDF_CLASS_COUNTS]
    # Adjust last class so total stays exact
    diff = total - sum(scaled_counts)
    scaled_counts[-1] += diff

    split_counts = _split_class_counts(scaled_counts, n_train, n_val, n_test)

    print(f"\nTarget class distribution (total {total} images):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:18s}: {scaled_counts[i]:5d}  "
              f"(train={split_counts['train'][i]}, "
              f"val={split_counts['val'][i]}, "
              f"test={split_counts['test'][i]})")

    for split in ["train", "val", "test"]:
        img_dir = base_dir / split / "images"
        lbl_dir = base_dir / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        # Build the ordered list of primary classes for this split
        primary_classes = []
        for class_id, count in enumerate(split_counts[split]):
            primary_classes.extend([class_id] * count)
        random.shuffle(primary_classes)   # random order within the split

        annotation_counts = {i: 0 for i in range(6)}
        n_images = len(primary_classes)
        print(f"\nGenerating {split} set ({n_images} images)...")

        for i, primary_class in enumerate(primary_classes):
            img, annotations = generate_image_with_primary_class(primary_class, img_size)

            # PDF requirement: oversample micro-plastics in training until ~12% of annotations
            # Add an extra micro-plastic object to ~15% of training images (any class)
            if split == "train" and random.random() < 0.15:
                bbox = generate_bbox(img_size, 5)   # class 5 = micro_plastic
                draw_micro_plastic(img, bbox)
                x, y, w, h = bbox
                annotations.append(
                    f"5 {(x + w / 2) / img_size:.6f} {(y + h / 2) / img_size:.6f} "
                    f"{w / img_size:.6f} {h / img_size:.6f}"
                )

            fname = f"{split}_{i:05d}"
            cv2.imwrite(str(img_dir / f"{fname}.jpg"), img)
            with open(lbl_dir / f"{fname}.txt", "w") as f:
                f.write("\n".join(annotations))

            for ann in annotations:
                cid = int(ann.split()[0])
                annotation_counts[cid] += 1

            if (i + 1) % 500 == 0:
                print(f"  [{i + 1}/{n_images}] images generated...")

        print(f"  Done. Primary-class image counts: "
              f"{ {CLASS_NAMES[k]: split_counts[split][k] for k in range(6)} }")

    # Create YOLO dataset YAML
    yaml_content = {
        "path": str(base_dir.resolve()),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
        "nc": 6,
    }

    yaml_path = base_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"\nDataset YAML saved to: {yaml_path}")
    return str(yaml_path)


if __name__ == "__main__":
    generate_dataset("../data/floating_debris")
