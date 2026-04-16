"""
CBAM-YOLOv8: YOLOv8 with CBAM attention injected at backbone P3, P4, P5 levels.

Architecture:
  - Base: YOLOv8n (nano) with CSPDarknet53 backbone
  - Modification: CBAM after C2f blocks at 1/8, 1/16, 1/32 resolution
  - Loss: CIoU + Focal Loss + DFL (composite)
  - Training: AdamW, cosine annealing, FP16 mixed precision
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.cbam import CBAM


def inject_cbam_into_yolov8(model):
    """
    Inject CBAM modules after C2f blocks in YOLOv8 backbone.

    YOLOv8n backbone structure (model.model.model):
      Index 0: Conv (3->16)       - stem
      Index 1: Conv (16->32)      - downsample
      Index 2: C2f  (32 ch)       - P2 (1/4)
      Index 3: Conv (32->64)      - downsample
      Index 4: C2f  (64 ch)       - P3 (1/8)  <-- CBAM here
      Index 5: Conv (64->128)     - downsample
      Index 6: C2f  (128 ch)      - P4 (1/16) <-- CBAM here
      Index 7: Conv (128->256)    - downsample
      Index 8: C2f  (256 ch)      - P5 (1/32) <-- CBAM here
      Index 9: SPPF (256 ch)
    """
    backbone = model.model.model

    # Target C2f block indices and their channel counts for YOLOv8n
    cbam_targets = {
        4: 64,    # P3 - 1/8 resolution
        6: 128,   # P4 - 1/16 resolution
        8: 256,   # P5 - 1/32 resolution
    }

    for idx, channels in cbam_targets.items():
        original_module = backbone[idx]
        # Wrap the C2f with CBAM using a sequential wrapper
        cbam_wrapper = nn.Sequential(
            original_module,
            CBAM(channels=channels, reduction_ratio=16, spatial_kernel=7)
        )
        backbone[idx] = cbam_wrapper
        print(f"  Injected CBAM after backbone layer {idx} ({channels} channels)")

    return model


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def create_cbam_yolov8(pretrained=True):
    """Create CBAM-augmented YOLOv8n model."""
    print("Loading YOLOv8n base model...")
    model = YOLO("yolov8n.pt" if pretrained else "yolov8n.yaml")

    # Count baseline params
    total_base, _ = count_parameters(model.model)
    print(f"  Baseline YOLOv8n parameters: {total_base:,}")

    print("Injecting CBAM attention modules...")
    model = inject_cbam_into_yolov8(model)

    # Count enhanced params
    total_cbam, trainable = count_parameters(model.model)
    cbam_params = total_cbam - total_base
    print(f"  CBAM-YOLOv8 parameters: {total_cbam:,} (+{cbam_params:,} from CBAM)")
    print(f"  Trainable parameters: {trainable:,}")

    return model


def train_cbam_yolov8(dataset_yaml, project_dir, epochs=150, batch_size=16,
                       imgsz=640, device=None):
    """
    Train CBAM-YOLOv8 with settings from the paper:
    - AdamW optimizer (lr=0.01, cosine annealing)
    - Batch size 16
    - 150 epochs, early stopping patience=20
    - FP16 mixed precision
    - Mosaic augmentation
    - Focal Loss for classification
    - CIoU for box regression
    - DFL for distribution focal loss
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "0"
        else:
            device = "cpu"

    model = create_cbam_yolov8(pretrained=True)

    # MPS (Apple Silicon) workarounds: disable AMP, halve batch size
    if device == "mps":
        use_amp = False
        batch_size = max(1, batch_size // 2)
    else:
        use_amp = True

    print(f"\nStarting training on device: {device}")
    print(f"  Dataset: {dataset_yaml}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, Image size: {imgsz}")

    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=project_dir,
        name="cbam_yolov8",
        exist_ok=True,

        # Optimizer: AdamW with cosine annealing
        optimizer="AdamW",
        lr0=0.01,
        lrf=0.01,       # Final lr = lr0 * lrf (cosine annealing)
        cos_lr=True,     # Cosine learning rate scheduler
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,

        # Loss weights (from paper: λ_box=7.5, λ_cls=0.5, λ_dfl=1.5)
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Early stopping
        patience=20,

        # Augmentation
        mosaic=1.0,       # Mosaic augmentation
        flipud=0.0,
        fliplr=0.5,       # Horizontal flip
        degrees=15.0,     # Rotation ±15°
        scale=0.5,        # Scale jitter
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        # Mixed precision (disabled on MPS due to shape-mismatch bug)
        amp=use_amp,

        # Saving
        save=True,
        save_period=-1,
        plots=True,
        verbose=True,
    )

    return model, results


def train_baseline_yolov8(dataset_yaml, project_dir, epochs=150, batch_size=16,
                           imgsz=640, device=None):
    """Train baseline YOLOv8n without CBAM for comparison."""
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "0"
        else:
            device = "cpu"

    print("Loading baseline YOLOv8n...")
    model = YOLO("yolov8n.pt")

    total, _ = count_parameters(model.model)
    print(f"  Baseline parameters: {total:,}")

    # MPS (Apple Silicon) workarounds: disable AMP, halve batch size
    if device == "mps":
        use_amp = False
        batch_size = max(1, batch_size // 2)
    else:
        use_amp = True

    print(f"\nStarting baseline training on device: {device}")

    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=project_dir,
        name="baseline_yolov8",
        exist_ok=True,
        optimizer="AdamW",
        lr0=0.01,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.0005,
        warmup_epochs=3,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        patience=20,
        mosaic=1.0,
        fliplr=0.5,
        degrees=15.0,
        scale=0.5,
        amp=use_amp,
        save=True,
        plots=True,
        verbose=True,
    )

    return model, results


if __name__ == "__main__":
    model = create_cbam_yolov8(pretrained=True)
    print("\nCBAM-YOLOv8 model created successfully!")
