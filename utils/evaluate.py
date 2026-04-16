"""
Evaluation and Visualization for EcoScan
Implements:
- mAP@0.5 and mAP@0.5:0.95 computation
- Per-class Precision, Recall, F1
- Class distribution plots
- Training curves
- Baseline vs CBAM comparison charts
- Attention map visualization
- Sample detection visualization
"""

import os
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter


CLASS_NAMES = [
    "plastic_bottle", "plastic_bag", "foam_styrofoam",
    "fishing_net", "other_debris", "micro_plastic"
]

SHORT_NAMES = ["Bottle", "Bag", "Foam", "Net", "Other", "Micro"]


def plot_class_distribution(data_dir, output_dir):
    """Plot class distribution (Figure 1 in paper)."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_counts = Counter()
    for split in ["train", "val", "test"]:
        lbl_dir = data_dir / split / "labels"
        if not lbl_dir.exists():
            continue
        for lbl_file in lbl_dir.glob("*.txt"):
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_counts[int(parts[0])] += 1

    total = sum(class_counts.values())
    classes = list(range(6))
    counts = [class_counts.get(c, 0) for c in classes]
    percentages = [c / total * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("viridis", 6)
    bars = ax.bar(SHORT_NAMES, counts, color=colors, edgecolor="black", linewidth=0.5)

    for bar, pct in zip(bars, percentages):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xlabel("Class", fontsize=13)
    ax.set_ylabel("Number of Annotations", fontsize=13)
    ax.set_title("Class Distribution — Floating Debris Dataset", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'class_distribution.png'}")


def plot_augmentation_samples(data_dir, output_dir):
    """Plot augmentation pipeline samples (Figure 2 in paper)."""
    from utils.preprocessing import apply_clahe, simulate_turbidity

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_dir = data_dir / "train" / "images"
    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        print("No training images found for augmentation demo.")
        return

    img = cv2.imread(str(images[0]))
    if img is None:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original", fontsize=12, fontweight="bold")

    # CLAHE
    clahe_img = apply_clahe(img)
    axes[0, 1].imshow(cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("CLAHE Enhanced", fontsize=12, fontweight="bold")

    # Turbidity
    turbid_img = simulate_turbidity(img)
    axes[0, 2].imshow(cv2.cvtColor(turbid_img, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("Turbidity Simulation", fontsize=12, fontweight="bold")

    # Gaussian blur
    blurred = cv2.GaussianBlur(img, (7, 7), 1.5)
    axes[1, 0].imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Gaussian Blur (σ=1.5)", fontsize=12, fontweight="bold")

    # Gamma contrast
    gamma = 0.7
    gamma_img = np.clip(np.power(img / 255.0, gamma) * 255, 0, 255).astype(np.uint8)
    axes[1, 1].imshow(cv2.cvtColor(gamma_img, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f"Gamma Contrast (γ={gamma})", fontsize=12, fontweight="bold")

    # Horizontal flip
    flipped = cv2.flip(img, 1)
    axes[1, 2].imshow(cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("Horizontal Flip", fontsize=12, fontweight="bold")

    for ax in axes.flat:
        ax.axis("off")

    plt.suptitle("Data Augmentation Pipeline — Environmental Challenges",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "augmentation_pipeline.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'augmentation_pipeline.png'}")


def plot_training_comparison(baseline_csv, cbam_csv, output_dir):
    """Plot training curves: baseline vs CBAM (Figure 4 style)."""
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ("metrics/mAP50(B)", "mAP@0.5"),
        ("metrics/mAP50-95(B)", "mAP@0.5:0.95"),
        ("train/box_loss", "Box Loss (CIoU)"),
        ("train/cls_loss", "Classification Loss (Focal)"),
    ]

    for ax, (col, title) in zip(axes.flat, metrics):
        for csv_path, label, color in [
            (baseline_csv, "Baseline YOLOv8", "#2196F3"),
            (cbam_csv, "CBAM-YOLOv8 (Ours)", "#FF5722"),
        ]:
            if csv_path and Path(csv_path).exists():
                df = pd.read_csv(csv_path)
                # Strip whitespace from column names
                df.columns = [c.strip() for c in df.columns]
                if col in df.columns:
                    ax.plot(df["epoch"], df[col], label=label, color=color, linewidth=2)

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    plt.suptitle("Training Comparison: Baseline YOLOv8 vs CBAM-YOLOv8",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "training_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'training_comparison.png'}")


def plot_per_class_metrics(results_dict, output_dir):
    """Plot per-class P, R, F1 comparison bar chart."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    x = np.arange(6)
    width = 0.35

    for ax, metric_name in zip(axes, ["Precision", "Recall", "F1"]):
        for i, (model_name, metrics) in enumerate(results_dict.items()):
            vals = metrics.get(metric_name, [0] * 6)
            offset = -width / 2 + i * width
            color = "#2196F3" if "Baseline" in model_name else "#FF5722"
            bars = ax.bar(x + offset, vals, width, label=model_name, color=color, alpha=0.85)

            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_xlabel("Class", fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f"Per-Class {metric_name}", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(SHORT_NAMES, rotation=30)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Per-Class Metrics: Baseline vs CBAM-YOLOv8",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "per_class_metrics.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'per_class_metrics.png'}")


def plot_map_comparison(baseline_results, cbam_results, output_dir):
    """Bar chart comparing mAP scores."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["mAP@0.5", "mAP@0.5:0.95"]
    baseline_vals = [baseline_results.get(m, 0) for m in metrics]
    cbam_vals = [cbam_results.get(m, 0) for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width / 2, baseline_vals, width, label="Baseline YOLOv8",
                   color="#2196F3", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, cbam_vals, width, label="CBAM-YOLOv8 (Ours)",
                   color="#FF5722", edgecolor="black", linewidth=0.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom",
                    fontsize=12, fontweight="bold")

    ax.set_ylabel("mAP Score", fontsize=13)
    ax.set_title("Detection Performance: Baseline vs CBAM-YOLOv8", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "map_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'map_comparison.png'}")


def visualize_attention_maps(model, sample_image_path, output_dir):
    """Visualize CBAM attention maps at P3, P4, P5 levels."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(sample_image_path))
    if img is None:
        print(f"Could not read {sample_image_path}")
        return

    img_resized = cv2.resize(img, (640, 640))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    # Hook to capture attention maps
    attention_maps = {}

    def get_hook(name):
        def hook(module, input, output):
            if hasattr(module, 'spatial_attention'):
                # Get spatial attention map
                x = input[0] if isinstance(input, tuple) else input
                avg_pool = x.mean(dim=1, keepdim=True)
                max_pool = x.amax(dim=1, keepdim=True)
                combined = torch.cat([avg_pool, max_pool], dim=1)
                spatial_att = module.spatial_attention.sigmoid(module.spatial_attention.conv(combined))
                attention_maps[name] = spatial_att.detach().cpu().numpy()[0, 0]
        return hook

    # Register hooks on CBAM modules
    backbone = model.model.model
    cbam_layers = {4: "P3 (1/8)", 6: "P4 (1/16)", 8: "P5 (1/32)"}
    hooks = []
    for idx, name in cbam_layers.items():
        module = backbone[idx]
        if isinstance(module, torch.nn.Sequential) and len(module) > 1:
            cbam_module = module[1]  # CBAM is second in Sequential
            hooks.append(cbam_module.register_forward_hook(get_hook(name)))

    # Forward pass
    device = next(model.model.parameters()).device
    with torch.no_grad():
        model.model(img_tensor.to(device))

    # Remove hooks
    for h in hooks:
        h.remove()

    if not attention_maps:
        print("No attention maps captured (model may not have CBAM layers).")
        return

    fig, axes = plt.subplots(1, len(attention_maps) + 1, figsize=(5 * (len(attention_maps) + 1), 5))

    axes[0].imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    for i, (name, att_map) in enumerate(attention_maps.items()):
        att_resized = cv2.resize(att_map, (640, 640))
        axes[i + 1].imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
        axes[i + 1].imshow(att_resized, cmap="jet", alpha=0.5)
        axes[i + 1].set_title(f"CBAM Attention — {name}", fontsize=12, fontweight="bold")
        axes[i + 1].axis("off")

    plt.suptitle("CBAM Spatial Attention Maps at Multi-Scale Backbone Levels",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "attention_maps.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'attention_maps.png'}")


def visualize_detections(model, data_dir, output_dir, n_samples=6):
    """Run inference and visualize detection results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_dir = Path(data_dir) / "test" / "images"
    images = sorted(test_dir.glob("*.jpg"))[:n_samples]

    if not images:
        print("No test images found.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    for ax, img_path in zip(axes.flat, images):
        results = model.predict(str(img_path), conf=0.25, verbose=False)
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls[0].cpu())
                conf = float(box.conf[0].cpu())
                color = colors[cls % len(colors)]

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f"{SHORT_NAMES[cls]} {conf:.2f}"
                cv2.putText(img, label, (x1, max(y1 - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        ax.imshow(img)
        ax.set_title(img_path.stem, fontsize=10)
        ax.axis("off")

    plt.suptitle("CBAM-YOLOv8 Detection Results on Test Set",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "detection_samples.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'detection_samples.png'}")


def generate_results_summary(baseline_metrics, cbam_metrics, output_dir):
    """Generate a text summary of results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    summary.append("=" * 70)
    summary.append("EcoScan: CBAM-YOLOv8 Results Summary")
    summary.append("=" * 70)
    summary.append("")
    summary.append(f"{'Metric':<25} {'Baseline YOLOv8':>18} {'CBAM-YOLOv8':>18} {'Δ':>10}")
    summary.append("-" * 70)

    for metric in ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1"]:
        b = baseline_metrics.get(metric, 0)
        c = cbam_metrics.get(metric, 0)
        delta = c - b
        sign = "+" if delta >= 0 else ""
        summary.append(f"{metric:<25} {b:>18.4f} {c:>18.4f} {sign}{delta:>9.4f}")

    summary.append("-" * 70)
    summary.append("")
    summary.append("Per-Class mAP@0.5:")
    summary.append(f"{'Class':<20} {'Baseline':>12} {'CBAM':>12} {'Δ':>10}")
    summary.append("-" * 55)

    for i, name in enumerate(CLASS_NAMES):
        b = baseline_metrics.get(f"AP_{name}", 0)
        c = cbam_metrics.get(f"AP_{name}", 0)
        delta = c - b
        sign = "+" if delta >= 0 else ""
        summary.append(f"{name:<20} {b:>12.4f} {c:>12.4f} {sign}{delta:>9.4f}")

    summary.append("=" * 70)

    summary_text = "\n".join(summary)
    print(summary_text)

    with open(output_dir / "results_summary.txt", "w") as f:
        f.write(summary_text)
    print(f"\nSaved: {output_dir / 'results_summary.txt'}")

    return summary_text
