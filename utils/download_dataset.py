"""
Real Floating Debris Dataset Downloader
========================================
Downloads real labeled waterway/marine litter images from Roboflow Universe.

Primary datasets tried (in order):
  1. Plastic Pollution (shifaurrehman) — 3,799 imgs, covers foam + bags + nets + bottles
  2. Ocean Garbage Detector (g-jtuyn)   — 5,421 imgs, broad marine debris taxonomy
  3. Marine Debris (pipe-count)         — 627 imgs,  clean class naming

All downloaded labels are remapped to our 6-class taxonomy:
  0: plastic_bottle  1: plastic_bag  2: foam_styrofoam
  3: fishing_net     4: other_debris  5: micro_plastic

Usage:
    python utils/download_dataset.py --key YOUR_ROBOFLOW_KEY
    OR set env var: export ROBOFLOW_API_KEY=YOUR_KEY

Roboflow free key: https://roboflow.com → Settings → Roboflow API
"""

import os
import sys
import yaml
import shutil
from pathlib import Path


# ── Our 6-class taxonomy ──────────────────────────────────────────────────────
OUR_CLASSES = [
    "plastic_bottle",   # 0
    "plastic_bag",      # 1
    "foam_styrofoam",   # 2
    "fishing_net",      # 3
    "other_debris",     # 4
    "micro_plastic",    # 5
]

# ── Class remapping rules (lowercase, underscores) ────────────────────────────
# Any class name matching a key maps to the corresponding OUR_CLASSES index.
# Anything not listed here → other_debris (4).
CLASS_MAP = {
    # plastic bottle variants
    "plastic_bottle":                     0,
    "bottle":                             0,
    "plastic_beverage_bottle":            0,
    "beverage_bottle":                    0,
    "trash_bottle":                       0,
    "plastic_bottles":                    0,
    "trash_cup":                          0,
    "trash_container":                    0,
    "cups":                               0,  # "Cups" dataset
    "jars":                               0,  # "Jars" dataset
    "jugs__containers":                   0,  # "Jugs- containers" after normalise
    "plastic_beverage_bottle_large":      0,  # Plastic Pollution dataset
    "plastic_beverage_bottle_medium":     0,
    "plastic_beverage_bottle_small":      0,
    "straws":                             0,  # rigid plastic straw ≈ bottle fragment

    # plastic bag variants
    "plastic_bag":               1,
    "plastic_bags":              1,
    "bag":                       1,
    "bags":                      1,  # "Bags" dataset
    "polythene":                 1,
    "trash_bag":                 1,
    "polythene_bag":             1,
    "trash_snack_wrapper":       1,
    "trash_clothing":            1,
    "soft_food_wrappers":        1,  # "Soft food wrappers" dataset
    "soft_fragments":            1,  # soft plastic fragments ≈ bags/wrappers

    # foam / styrofoam variants
    "foam":                      2,
    "foam_styrofoam":            2,
    "styrofoam":                 2,
    "foam_fragments":            2,  # "Foam fragments" dataset ✓ already works
    "expanded_polystyrene":      2,
    "trash_tarp":                2,

    # fishing net / line variants
    "fishing_net":               3,
    "fishing":                   3,  # "Fishing" dataset ✓ already works
    "net":                       3,
    "fishing_line":              3,
    "trash_net":                 3,
    "plastic_rope_net_pieces":   3,
    "plastic_rope___net_pieces": 3,  # "Plastic rope - net pieces" after normalise
    "rope":                      3,
    "trash_rope":                3,

    # micro-plastic
    "micro_plastic":             5,
    "microplastic":              5,
    "microplastics":             5,

    # everything else → other_debris (4) by default (not listed here)
}


# ── Dataset candidates ────────────────────────────────────────────────────────
DATASETS = [
    {
        "name": "Plastic Pollution (ShifaurRehman)",
        "workspace": "shifaurrehman-3obgv",
        "project": "plastic-pollution-tj0mf",
        "version": 1,
        "images": "~3,800",
    },
    {
        "name": "Ocean Garbage Detector",
        "workspace": "g-jtuyn",
        "project": "ocean-garbage-detector-dataset",
        "version": 1,
        "images": "~5,400",
    },
    {
        "name": "Marine Debris (Pipe Count)",
        "workspace": "pipe-count",
        "project": "marine-debris-uxk2k",
        "version": 1,
        "images": "~627",
    },
]


def download_real_dataset(base_dir, api_key=None):
    """
    Download a real floating debris dataset from Roboflow Universe.

    Tries each dataset in DATASETS until one succeeds.
    Labels are automatically remapped to our 6-class taxonomy.

    Args:
        base_dir (str): Directory to save the dataset (same as DATA_DIR)
        api_key  (str): Roboflow API key. Reads ROBOFLOW_API_KEY env var if None.

    Returns:
        str: Path to dataset.yaml, or None on failure
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Installing roboflow...")
        os.system("pip install roboflow -q")
        from roboflow import Roboflow

    # Resolve API key
    if api_key is None:
        api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        _print_key_instructions()
        return None

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConnecting to Roboflow...")
    rf = Roboflow(api_key=api_key)

    for ds in DATASETS:
        print(f"\nTrying: {ds['name']} ({ds['images']} images)...")
        try:
            project  = rf.workspace(ds["workspace"]).project(ds["project"])
            dataset  = project.version(ds["version"]).download(
                "yolov8", location=str(base_dir / "raw")
            )
            print(f"  ✅ Downloaded: {ds['name']}")

            # Fix YAML paths + remap labels
            yaml_path = _process_download(base_dir, base_dir / "raw")
            print(f"\n✅ Real dataset ready at: {yaml_path}")
            return str(yaml_path)

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            continue

    print("\n❌ All dataset downloads failed.")
    print("   Check your API key or network connection.")
    return None


def _process_download(base_dir, raw_dir):
    """
    Locate the downloaded data.yaml, remap classes, write final dataset.yaml.
    """
    raw_dir  = Path(raw_dir)
    base_dir = Path(base_dir)

    # Find data.yaml anywhere in the download
    yaml_files = list(raw_dir.rglob("data.yaml")) + list(raw_dir.rglob("dataset.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No data.yaml found under {raw_dir}")

    src_yaml    = yaml_files[0]
    dataset_root = src_yaml.parent

    with open(src_yaml) as f:
        cfg = yaml.safe_load(f)

    # Parse class names from downloaded YAML
    raw_names = cfg.get("names", [])
    if isinstance(raw_names, list):
        raw_names = {i: n for i, n in enumerate(raw_names)}
    elif not isinstance(raw_names, dict):
        raw_names = {}

    print(f"\n  Downloaded classes ({len(raw_names)}): {list(raw_names.values())}")

    # Build old_id → new_id mapping
    id_map = {}
    for old_id, name in raw_names.items():
        key = name.lower().replace(" ", "_").replace("-", "_")
        new_id = CLASS_MAP.get(key, 4)   # default: other_debris
        id_map[int(old_id)] = new_id
        print(f"    [{old_id}] {name:30s} → [{new_id}] {OUR_CLASSES[new_id]}")

    # Copy split folders into base_dir with standard names
    for src_split, dst_split in [("train", "train"), ("valid", "val"), ("test", "test")]:
        src_img  = dataset_root / src_split / "images"
        src_lbl  = dataset_root / src_split / "labels"
        dst_img  = base_dir / dst_split / "images"
        dst_lbl  = base_dir / dst_split / "labels"

        if not src_img.exists():
            continue

        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        # Copy images
        for img in src_img.glob("*"):
            shutil.copy2(img, dst_img / img.name)

        # Remap and copy labels
        if src_lbl.exists():
            for lbl in src_lbl.glob("*.txt"):
                lines = lbl.read_text().strip().splitlines()
                new_lines = []
                for line in lines:
                    parts = line.split()
                    if not parts:
                        continue
                    new_id = id_map.get(int(parts[0]), 4)
                    new_lines.append(f"{new_id} " + " ".join(parts[1:]))
                (dst_lbl / lbl.name).write_text("\n".join(new_lines))

        n_imgs = len(list(dst_img.glob("*")))
        print(f"  {dst_split:5s}: {n_imgs} images")

    # Write final dataset.yaml
    final_yaml = base_dir / "dataset.yaml"
    final_cfg = {
        "path":  str(base_dir.resolve()),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc":    6,
        "names": {i: n for i, n in enumerate(OUR_CLASSES)},
    }
    with open(final_yaml, "w") as f:
        yaml.dump(final_cfg, f, default_flow_style=False)

    # Clean up raw download
    shutil.rmtree(raw_dir, ignore_errors=True)

    return final_yaml


def merge_datasets(base_dir, api_key=None):
    """
    Download ALL datasets in DATASETS and merge into one combined dataset.
    Each dataset is remapped to our 6-class taxonomy before merging.
    Filenames are prefixed with ds0_, ds1_, ds2_ to avoid collisions.

    Args:
        base_dir (str): Directory to save the merged dataset
        api_key  (str): Roboflow API key

    Returns:
        str: Path to dataset.yaml, or None on failure
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        os.system("pip install roboflow -q")
        from roboflow import Roboflow

    if api_key is None:
        api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        _print_key_instructions()
        return None

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Create final split dirs
    for split in ["train", "val", "test"]:
        (base_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (base_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    total_counts = {"train": 0, "val": 0, "test": 0}

    for ds_idx, ds in enumerate(DATASETS):
        print(f"\n{'='*55}")
        print(f"  [{ds_idx+1}/{len(DATASETS)}] {ds['name']}  ({ds['images']} images)")
        print(f"{'='*55}")

        raw_dir = base_dir / f"_raw_{ds_idx}"
        try:
            project = rf.workspace(ds["workspace"]).project(ds["project"])
            project.version(ds["version"]).download("yolov8", location=str(raw_dir))
        except Exception as e:
            print(f"  ❌ Download failed: {e} — skipping.")
            shutil.rmtree(raw_dir, ignore_errors=True)
            continue

        # Find yaml
        yaml_files = list(raw_dir.rglob("data.yaml")) + list(raw_dir.rglob("dataset.yaml"))
        if not yaml_files:
            print("  ❌ No yaml found — skipping.")
            shutil.rmtree(raw_dir, ignore_errors=True)
            continue

        src_yaml     = yaml_files[0]
        dataset_root = src_yaml.parent

        with open(src_yaml) as f:
            cfg = yaml.safe_load(f)

        raw_names = cfg.get("names", [])
        if isinstance(raw_names, list):
            raw_names = {i: n for i, n in enumerate(raw_names)}
        elif not isinstance(raw_names, dict):
            raw_names = {}

        print(f"  Classes ({len(raw_names)}): {list(raw_names.values())[:8]}{'...' if len(raw_names) > 8 else ''}")

        # Build id_map
        id_map = {}
        for old_id, name in raw_names.items():
            key = name.lower().replace(" ", "_").replace("-", "_")
            new_id = CLASS_MAP.get(key, 4)
            id_map[int(old_id)] = new_id

        # Copy images + remapped labels into final dirs
        for src_split, dst_split in [("train", "train"), ("valid", "val"), ("test", "test")]:
            src_img = dataset_root / src_split / "images"
            src_lbl = dataset_root / src_split / "labels"
            dst_img = base_dir / dst_split / "images"
            dst_lbl = base_dir / dst_split / "labels"

            if not src_img.exists():
                continue

            count = 0
            for img in src_img.glob("*"):
                new_stem     = f"ds{ds_idx}_{img.stem}"
                new_img_name = f"ds{ds_idx}_{img.name}"
                shutil.copy2(img, dst_img / new_img_name)

                lbl = src_lbl / (img.stem + ".txt")
                if lbl.exists():
                    lines = lbl.read_text().strip().splitlines()
                    new_lines = []
                    for line in lines:
                        parts = line.split()
                        if not parts:
                            continue
                        new_id = id_map.get(int(parts[0]), 4)
                        new_lines.append(f"{new_id} " + " ".join(parts[1:]))
                    (dst_lbl / (new_stem + ".txt")).write_text("\n".join(new_lines))
                else:
                    (dst_lbl / (new_stem + ".txt")).write_text("")
                count += 1

            total_counts[dst_split] += count
            print(f"  {dst_split:5s}: +{count} images")

        shutil.rmtree(raw_dir, ignore_errors=True)

    # Write final dataset.yaml
    final_yaml = base_dir / "dataset.yaml"
    final_cfg = {
        "path":  str(base_dir.resolve()),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc":    6,
        "names": {i: n for i, n in enumerate(OUR_CLASSES)},
    }
    with open(final_yaml, "w") as f:
        yaml.dump(final_cfg, f, default_flow_style=False)

    print(f"\n{'='*55}")
    print(f"  ✅ MERGED DATASET READY")
    for split, count in total_counts.items():
        print(f"     {split:5s}: {count} images")
    print(f"  YAML: {final_yaml}")
    print(f"{'='*55}")

    return str(final_yaml)


def check_dataset_exists(base_dir):
    """Return True if a real (non-synthetic) dataset already exists."""
    base_dir  = Path(base_dir)
    yaml_path = base_dir / "dataset.yaml"
    if not yaml_path.exists():
        return False
    train_imgs = list((base_dir / "train" / "images").glob("*"))
    return len(train_imgs) > 500   # synthetic demo has <500


def _print_key_instructions():
    print("\n" + "=" * 60)
    print("Roboflow API key required to download the real dataset.")
    print("=" * 60)
    print("\nGet your FREE key in 2 minutes:")
    print("  1. Go to https://roboflow.com  →  Sign up (free)")
    print("  2. Click your avatar (top right) → Settings")
    print("  3. Click 'Roboflow API' → copy your Private API Key")
    print("\nThen run:")
    print("  python utils/download_dataset.py --key YOUR_KEY")
    print("  OR set: export ROBOFLOW_API_KEY=YOUR_KEY")
    print("\nFalling back to synthetic dataset...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download real floating debris dataset")
    parser.add_argument("--key", type=str, default=None,
                        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)")
    parser.add_argument("--dir", type=str,
                        default=os.path.join(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "floating_debris"
                        ),
                        help="Directory to save the dataset")
    args = parser.parse_args()

    result = download_real_dataset(args.dir, api_key=args.key)
    if result:
        print(f"\n✅ Success! Dataset YAML: {result}")
        print("Now run: python main.py")
    else:
        print("\n❌ Download failed. Check your API key and try again.")
