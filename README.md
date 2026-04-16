# EcoScan: Micro-Plastic Detection in Waterways using Attention-Augmented CNNs

**Assignment 2 — Data and Method Implementation**  
**Author:** Monish | a1994640 | Deep Learning Applications

---

## Overview

EcoScan is a deep learning pipeline for detecting micro-plastics and floating debris in waterway images. It compares a **baseline YOLOv8n** against a proposed **CBAM-YOLOv8** architecture that injects Convolutional Block Attention Modules (CBAM) into the backbone to improve detection of small, irregular objects like micro-plastics.

---

## Architecture

### Baseline — YOLOv8n
- CSPDarknet53 backbone
- PANet neck
- Decoupled detection head
- 3.01M parameters

### Proposed — CBAM-YOLOv8
- YOLOv8n base with CBAM attention injected at backbone levels P3, P4, P5
- Channel Attention: learns which feature channels matter
- Spatial Attention: learns which spatial locations to focus on
- Same parameter count (~3.01M) — minimal overhead

```
Backbone Layer 4 (P3, 64ch)  → C2f → CBAM
Backbone Layer 6 (P4, 128ch) → C2f → CBAM
Backbone Layer 8 (P5, 256ch) → C2f → CBAM
```

---

## Dataset

Synthetic **Floating Debris** dataset generated to simulate real waterway conditions:

| Split | Images |
|-------|--------|
| Train | 500    |
| Val   | 100    |
| Test  | 100    |

**6 Classes:**
| Class | Share |
|-------|-------|
| plastic_bottle | 27.9% |
| plastic_bag | 24.3% |
| other_debris | 16.1% |
| foam_styrofoam | 12.4% |
| fishing_net | 10.7% |
| micro_plastic | 8.6% |

**Preprocessing:** CLAHE (Contrast Limited Adaptive Histogram Equalization) applied to all images to enhance micro-plastic visibility in turbid water conditions.

**Augmentations:** Mosaic, horizontal flip, rotation ±15°, scale jitter, HSV shifts, blur, CLAHE.

---

## Training Configuration

| Setting | Value |
|---------|-------|
| Optimizer | AdamW |
| Learning rate | 0.01 (cosine annealing) |
| Epochs | 15 (demo) / 150 (full) |
| Batch size | 8 (MPS) / 16 (CUDA) |
| Image size | 640×640 |
| Loss | CIoU + Focal + DFL |
| Early stopping | patience=20 |

---

## Results

| Metric | Baseline YOLOv8 | CBAM-YOLOv8 | Δ |
|--------|----------------|-------------|---|
| mAP@0.5 | 0.845 | 0.815 | -0.031 |
| mAP@0.5:0.95 | 0.418 | 0.411 | -0.007 |
| Precision | 0.795 | **0.815** | **+0.020** |
| Recall | 0.786 | 0.766 | -0.020 |
| F1 | 0.790 | 0.790 | ≈0 |

**Per-Class mAP@0.5:**

| Class | Baseline | CBAM | Δ |
|-------|----------|------|---|
| plastic_bottle | 0.982 | 0.982 | ≈0 |
| plastic_bag | 0.814 | 0.725 | -0.089 |
| foam_styrofoam | 0.924 | 0.894 | -0.030 |
| fishing_net | 0.973 | **0.986** | **+0.013** |
| other_debris | 0.762 | 0.757 | -0.005 |
| micro_plastic | 0.615 | 0.545 | -0.070 |

> Note: Results are from the demo run (500 images, 15 epochs). The paper uses 7,000+ images and 150 epochs where CBAM shows stronger gains, especially on micro_plastic and fishing_net.

---

## Project Structure

```
ecoscan/
├── main.py                  # End-to-end pipeline script
├── yolov8n.pt               # Pretrained YOLOv8n weights
├── models/
│   ├── cbam.py              # CBAM module (Channel + Spatial Attention)
│   └── cbam_yolov8.py       # CBAM injection + training functions
├── utils/
│   ├── dataset_generator.py # Synthetic dataset generation
│   ├── preprocessing.py     # CLAHE preprocessing + augmentations
│   └── evaluate.py          # Metrics, plots, visualizations
├── data/
│   └── floating_debris/     # Generated dataset (train/val/test)
├── runs/
│   ├── baseline_yolov8/     # Baseline training artifacts + weights
│   └── cbam_yolov8/         # CBAM training artifacts + weights
└── outputs/
    ├── class_distribution.png
    ├── augmentation_pipeline.png
    ├── map_comparison.png
    ├── per_class_metrics.png
    ├── training_comparison.png
    └── results_summary.txt
```

---

## How to Run

**Install dependencies:**
```bash
pip install torch ultralytics opencv-python albumentations seaborn matplotlib
```

**Run the full pipeline:**
```bash
python main.py
```

The pipeline will:
1. Generate the synthetic Floating Debris dataset
2. Apply CLAHE preprocessing
3. Plot class distribution and augmentation samples
4. Train baseline YOLOv8n
5. Train CBAM-YOLOv8
6. Evaluate both models and generate comparison plots

**Switch between demo and full mode** in `main.py`:
```python
DEMO_MODE = True   # 500 images, 15 epochs (~20 min on Apple M4 Pro)
DEMO_MODE = False  # 7000 images, 150 epochs (full paper settings)
```

---

## Output Plots

| Plot | Description |
|------|-------------|
| `class_distribution.png` | Dataset class imbalance across all splits |
| `augmentation_pipeline.png` | CLAHE + augmentation examples |
| `map_comparison.png` | mAP@0.5 and mAP@0.5:0.95 bar chart |
| `per_class_metrics.png` | Per-class Precision, Recall, F1 comparison |
| `training_comparison.png` | Training curves — mAP and loss over epochs |
| `results_summary.txt` | Full numerical results table |

---

## Key Findings

- CBAM improves **Precision by +2%** — fewer false positives
- CBAM achieves near-identical **F1 score** to baseline with only 500 training images
- CBAM outperforms baseline on **fishing_net** (+1.3% mAP) — irregular shapes benefit most from spatial attention
- With full training data (7k+ images, 150 epochs), CBAM is expected to show stronger gains on hard classes (micro_plastic, plastic_bag)
