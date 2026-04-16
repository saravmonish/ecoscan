"""
EcoScan: Micro-Plastic Detection in Waterways using Attention-Augmented CNNs
=============================================================================
Main pipeline script — runs end-to-end:
  1. Generate synthetic Floating Debris dataset
  2. Apply CLAHE preprocessing
  3. Train baseline YOLOv8n
  4. Train CBAM-YOLOv8 (proposed architecture)
  5. Evaluate both on test set
  6. Generate comparison visualizations

Author: Monish | a1994640 | Deep Learning Applications
"""

import os
import sys
import time
import torch
import numpy as np
import random

# Ensure project root is on path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from pathlib import Path
from utils.dataset_generator import generate_dataset
from utils.preprocessing import preprocess_dataset, apply_clahe
from models.cbam_yolov8 import (
    create_cbam_yolov8, train_cbam_yolov8,
    train_baseline_yolov8, count_parameters
)
from utils.evaluate import (
    plot_class_distribution, plot_augmentation_samples,
    plot_training_comparison, plot_per_class_metrics,
    plot_map_comparison, visualize_detections,
    visualize_attention_maps, generate_results_summary,
    CLASS_NAMES
)


# ─── Configuration ───────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_DIR, "data", "floating_debris")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
RUNS_DIR = os.path.join(PROJECT_DIR, "runs")

# Training config (reduced for demo — paper uses 150 epochs, 10k+ images)
DEMO_MODE = True  # Set False for full training
if DEMO_MODE:
    N_TRAIN = 500
    N_VAL = 100
    N_TEST = 100
    EPOCHS = 15
    BATCH_SIZE = 16
    IMG_SIZE = 640
else:
    N_TRAIN = 7000
    N_VAL = 1500
    N_TEST = 1500
    EPOCHS = 150
    BATCH_SIZE = 16
    IMG_SIZE = 640

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "0"
    return "cpu"


def step_1_generate_dataset():
    """Step 1: Generate synthetic Floating Debris dataset."""
    print("\n" + "=" * 70)
    print("STEP 1: Generating Synthetic Floating Debris Dataset")
    print("=" * 70)

    yaml_path = generate_dataset(
        DATA_DIR,
        n_train=N_TRAIN,
        n_val=N_VAL,
        n_test=N_TEST,
        img_size=IMG_SIZE,
    )
    return yaml_path


def step_2_preprocess():
    """Step 2: Apply CLAHE preprocessing to enhance micro-plastic visibility."""
    print("\n" + "=" * 70)
    print("STEP 2: Applying CLAHE Preprocessing")
    print("=" * 70)

    preprocess_dataset(DATA_DIR)


def step_3_visualize_data():
    """Step 3: Generate dataset visualizations."""
    print("\n" + "=" * 70)
    print("STEP 3: Generating Dataset Visualizations")
    print("=" * 70)

    plot_class_distribution(DATA_DIR, OUTPUT_DIR)
    plot_augmentation_samples(DATA_DIR, OUTPUT_DIR)


def step_4_train_baseline(yaml_path):
    """Step 4: Train baseline YOLOv8n."""
    print("\n" + "=" * 70)
    print("STEP 4: Training Baseline YOLOv8n")
    print("=" * 70)

    device = get_device()
    print(f"Device: {device}")

    model, results = train_baseline_yolov8(
        dataset_yaml=yaml_path,
        project_dir=RUNS_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=device,
    )
    return model, results


def step_5_train_cbam(yaml_path):
    """Step 5: Train CBAM-YOLOv8 (proposed architecture)."""
    print("\n" + "=" * 70)
    print("STEP 5: Training CBAM-YOLOv8 (Proposed Architecture)")
    print("=" * 70)

    device = get_device()
    print(f"Device: {device}")

    model, results = train_cbam_yolov8(
        dataset_yaml=yaml_path,
        project_dir=RUNS_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=device,
    )
    return model, results


def step_6_evaluate(baseline_model, cbam_model, yaml_path):
    """Step 6: Evaluate both models and generate comparison plots."""
    print("\n" + "=" * 70)
    print("STEP 6: Evaluation & Visualization")
    print("=" * 70)

    device = get_device()

    # Evaluate baseline
    print("\nEvaluating Baseline YOLOv8...")
    baseline_val = baseline_model.val(data=yaml_path, split="test", device=device, verbose=False)

    # Evaluate CBAM model
    print("Evaluating CBAM-YOLOv8...")
    cbam_val = cbam_model.val(data=yaml_path, split="test", device=device, verbose=False)

    # Extract metrics
    def extract_metrics(val_results):
        metrics = {}
        metrics["mAP@0.5"] = float(val_results.box.map50)
        metrics["mAP@0.5:0.95"] = float(val_results.box.map)
        metrics["Precision"] = float(val_results.box.mp)
        metrics["Recall"] = float(val_results.box.mr)

        p = metrics["Precision"]
        r = metrics["Recall"]
        metrics["F1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0

        # Per-class AP
        if hasattr(val_results.box, 'ap50') and val_results.box.ap50 is not None:
            ap50 = val_results.box.ap50
            for i, name in enumerate(CLASS_NAMES):
                if i < len(ap50):
                    metrics[f"AP_{name}"] = float(ap50[i])

        # Per-class P, R, F1
        if hasattr(val_results.box, 'p') and val_results.box.p is not None:
            per_p = val_results.box.p
            per_r = val_results.box.r
            p_list, r_list, f1_list = [], [], []
            for i in range(min(6, len(per_p))):
                pi, ri = float(per_p[i]), float(per_r[i])
                fi = 2 * pi * ri / (pi + ri) if (pi + ri) > 0 else 0
                p_list.append(pi)
                r_list.append(ri)
                f1_list.append(fi)
            metrics["per_class_P"] = p_list
            metrics["per_class_R"] = r_list
            metrics["per_class_F1"] = f1_list

        return metrics

    baseline_metrics = extract_metrics(baseline_val)
    cbam_metrics = extract_metrics(cbam_val)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    generate_results_summary(baseline_metrics, cbam_metrics, OUTPUT_DIR)

    # Generate comparison plots
    print("\nGenerating comparison visualizations...")

    # mAP comparison bar chart
    plot_map_comparison(baseline_metrics, cbam_metrics, OUTPUT_DIR)

    # Per-class metrics
    if "per_class_P" in baseline_metrics and "per_class_P" in cbam_metrics:
        results_dict = {
            "Baseline YOLOv8": {
                "Precision": baseline_metrics["per_class_P"],
                "Recall": baseline_metrics["per_class_R"],
                "F1": baseline_metrics["per_class_F1"],
            },
            "CBAM-YOLOv8 (Ours)": {
                "Precision": cbam_metrics["per_class_P"],
                "Recall": cbam_metrics["per_class_R"],
                "F1": cbam_metrics["per_class_F1"],
            },
        }
        plot_per_class_metrics(results_dict, OUTPUT_DIR)

    # Training curve comparison
    baseline_csv = os.path.join(RUNS_DIR, "baseline_yolov8", "results.csv")
    cbam_csv = os.path.join(RUNS_DIR, "cbam_yolov8", "results.csv")
    if os.path.exists(baseline_csv) and os.path.exists(cbam_csv):
        plot_training_comparison(baseline_csv, cbam_csv, OUTPUT_DIR)

    # Detection samples
    print("Generating detection visualizations...")
    try:
        visualize_detections(cbam_model, DATA_DIR, OUTPUT_DIR)
    except Exception as e:
        print(f"Detection visualization skipped (inference mode conflict): {e}")

    # Attention maps
    test_images = sorted(Path(DATA_DIR, "test", "images").glob("*.jpg"))
    if test_images:
        try:
            visualize_attention_maps(cbam_model, test_images[0], OUTPUT_DIR)
        except Exception as e:
            print(f"Attention map visualization skipped: {e}")

    return baseline_metrics, cbam_metrics


def main():
    start_time = time.time()

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  EcoScan: Micro-Plastic Detection using Attention-Augmented CNNs   ║")
    print("║  Assignment 2: Data and Method Implementation                       ║")
    print("║  Monish | a1994640 | Deep Learning Applications                     ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"\nDevice: {get_device()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Mode: {'DEMO (reduced dataset/epochs)' if DEMO_MODE else 'FULL'}")

    # Step 1: Generate dataset
    yaml_path = step_1_generate_dataset()

    # Step 2: Preprocess with CLAHE
    step_2_preprocess()

    # Step 3: Data visualizations
    step_3_visualize_data()

    # Step 4: Train baseline
    baseline_model, baseline_results = step_4_train_baseline(yaml_path)

    # Step 5: Train CBAM-YOLOv8
    cbam_model, cbam_results = step_5_train_cbam(yaml_path)

    # Step 6: Evaluate and compare
    baseline_metrics, cbam_metrics = step_6_evaluate(baseline_model, cbam_model, yaml_path)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Pipeline complete! Total time: {elapsed / 60:.1f} minutes")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print(f"Training runs saved to: {RUNS_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
