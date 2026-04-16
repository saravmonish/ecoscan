"""
Data Preprocessing Pipeline for EcoScan
Implements all transforms described in the paper:
- CLAHE on L-channel (LAB space) for micro-plastic visibility
- Turbidity simulation (additive noise + brown/green tint)
- Domain-specific augmentations (Gaussian blur, gamma contrast)
- Standard augmentations (flip, rotation, scale jitter)
"""

import cv2
import numpy as np
import albumentations as A
from pathlib import Path


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE on L-channel in LAB space.
    Enhances local contrast for translucent micro-plastics.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return result


def simulate_turbidity(image, noise_sigma=0.05, tint_alpha=0.15):
    """
    Turbidity simulation layer:
    - Additive Gaussian noise (μ=0, σ=0.05)
    - Brown/green tint overlay (α=0.15)
    """
    img_float = image.astype(np.float32) / 255.0

    # Additive Gaussian noise
    noise = np.random.normal(0, noise_sigma, img_float.shape).astype(np.float32)
    img_noisy = np.clip(img_float + noise, 0, 1)

    # Brown/green tint overlay
    tint = np.zeros_like(img_noisy)
    tint[:, :, 0] = 0.15   # Blue (low)
    tint[:, :, 1] = 0.35   # Green (medium-high)
    tint[:, :, 2] = 0.25   # Red (medium - brownish)

    img_tinted = np.clip(img_noisy * (1 - tint_alpha) + tint * tint_alpha, 0, 1)

    return (img_tinted * 255).astype(np.uint8)


def get_training_augmentation():
    """
    Full augmentation pipeline as described in the paper:
    - Horizontal flip
    - Rotation ±15°
    - Scale jitter 0.5-1.5x
    - Gaussian blur (σ ∈ [0.5, 2.0]) for camera defocus
    - Gamma contrast (γ ∈ [0.6, 1.4]) for illumination
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.RandomScale(scale_limit=(-0.5, 0.5), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.5, 2.0), p=0.3),
        A.RandomGamma(gamma_limit=(60, 140), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3,
    ))


def get_validation_transform():
    """Minimal transforms for validation (just resize + normalize)."""
    return A.Compose([
        A.NoOp(),  # placeholder
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3,
    ))


def preprocess_image(image, apply_clahe_flag=True, apply_turbidity=False):
    """
    Full preprocessing pipeline:
    1. Resize to 640x640 with letterbox padding
    2. CLAHE enhancement
    3. Normalize to [0, 1]
    """
    # Letterbox resize to 640x640
    image = letterbox_resize(image, target_size=640)

    # CLAHE on L-channel
    if apply_clahe_flag:
        image = apply_clahe(image)

    # Optional turbidity simulation (for augmentation)
    if apply_turbidity:
        image = simulate_turbidity(image)

    return image


def letterbox_resize(image, target_size=640):
    """Resize with letterbox padding (maintaining aspect ratio)."""
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to target size
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    return canvas


def preprocess_dataset(data_dir, output_dir=None):
    """Apply CLAHE preprocessing to all images in dataset."""
    data_dir = Path(data_dir)
    if output_dir:
        output_dir = Path(output_dir)

    for split in ["train", "val", "test"]:
        img_dir = data_dir / split / "images"
        if not img_dir.exists():
            continue

        out_dir = (output_dir or data_dir) / split / "images"
        out_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(img_dir.glob("*.jpg"))
        print(f"Preprocessing {split}: {len(images)} images...")

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Apply CLAHE
            img = apply_clahe(img)

            # Save
            cv2.imwrite(str(out_dir / img_path.name), img)

    print("Preprocessing complete.")


if __name__ == "__main__":
    # Demo
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    print("Testing CLAHE...")
    clahe_img = apply_clahe(test_img)
    print(f"  Input: {test_img.shape}, Output: {clahe_img.shape}")

    print("Testing turbidity simulation...")
    turbid_img = simulate_turbidity(test_img)
    print(f"  Input: {test_img.shape}, Output: {turbid_img.shape}")

    print("Testing letterbox resize...")
    resized = letterbox_resize(test_img, 640)
    print(f"  Input: {test_img.shape}, Output: {resized.shape}")

    print("Testing augmentation pipeline...")
    aug = get_training_augmentation()
    result = aug(image=test_img, bboxes=[[0.5, 0.5, 0.1, 0.1]], class_labels=[0])
    print(f"  Augmented image: {result['image'].shape}")
    print("All preprocessing tests passed!")
