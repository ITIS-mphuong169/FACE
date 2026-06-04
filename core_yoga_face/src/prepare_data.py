"""
prepare_data.py
---------------
Utility script to help prepare train/test CSV splits from a raw MoveNet
skeleton CSV.

Usage:
  python src/prepare_data.py \\
      --input_csv data/raw_skeleton.csv \\
      --output_dir data/ \\
      --test_size 0.2 \\
      --label_col label

Expects a CSV with:
  - image_path column (relative to FACE repo root recommended)
  - label column
  - 51 keypoint columns: kp{i}_* OR nose_x/y/score, left_wrist_x, … (MoveNet names)
  - Optional: split, pose_name, correctness
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from load_dataset import KEYPOINT_NAMES, KP_COLS, _KP_NAME_COLS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def check_required_columns(df: pd.DataFrame, label_col: str):
    """Verify required columns exist (kp{i}_* or nose_x, left_wrist_x, …)."""
    missing = []
    if "image_path" not in df.columns:
        missing.append("image_path")
    if label_col not in df.columns:
        missing.append(label_col)
    has_kp = all(c in df.columns for c in KP_COLS)
    has_named = all(c in df.columns for c in _KP_NAME_COLS)
    if not has_kp and not has_named:
        missing.append(
            "51 keypoint columns (kp{i}_x/y/c or "
            + ", ".join(f"{KEYPOINT_NAMES[0]}_x, {KEYPOINT_NAMES[9]}_x, …)")
        )
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def generate_sample_csv(output_path: str, n: int = 100, n_classes: int = 5):
    """
    Generate a synthetic sample CSV for testing the pipeline.
    Labels 1..n_classes.
    """
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        label = rng.integers(1, n_classes + 1)
        kp = rng.random(51).tolist()
        # confidence channels: random 0.5..1.0
        for k in range(17):
            kp[k * 3 + 2] = float(rng.uniform(0.5, 1.0))
        row = {
            "image_path": f"data/images/sample_{i:04d}.jpg",
            "label": int(label),
            "pose_name": f"pose_{label}",
            "correctness": rng.choice(["correct", "incorrect"]),
        }
        for j, col in enumerate(KP_COLS):
            row[col] = kp[j]
        rows.append(row)
    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Sample CSV written → {output_path}  ({n} rows, {n_classes} classes)")
    return df


def split_and_save(
    input_csv : str,
    output_dir: str,
    test_size : float = 0.2,
    label_col : str   = "label",
    seed      : int   = 42,
):
    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df)} rows from {input_csv}")
    check_required_columns(df, label_col)

    # Use pre-existing split column if available
    if "split" in df.columns:
        train_df = df[df["split"] == "train"].copy()
        test_df  = df[df["split"] == "test"].copy()
        logger.info(f"Using 'split' column: train={len(train_df)}, test={len(test_df)}")
    else:
        train_df, test_df = train_test_split(
            df, test_size=test_size, stratify=df[label_col], random_state=seed
        )
        logger.info(f"Split: train={len(train_df)}, test={len(test_df)}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / "train.csv"
    test_path  = out / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info(f"Saved → {train_path}, {test_path}")

    # Stats
    print(f"\nLabel distribution:")
    print(f"  Train:\n{train_df[label_col].value_counts().sort_index().to_string()}")
    print(f"\n  Test:\n{test_df[label_col].value_counts().sort_index().to_string()}")

    return train_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare train/test CSV splits")
    parser.add_argument("--input_csv",   default=None,
                        help="Path to raw skeleton CSV. If not provided, generates synthetic data.")
    parser.add_argument("--output_dir",  default="data")
    parser.add_argument("--test_size",   type=float, default=0.2)
    parser.add_argument("--label_col",   default="label")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--generate_sample", action="store_true",
                        help="Generate synthetic sample CSVs for testing")
    parser.add_argument("--n_samples",   type=int, default=300)
    parser.add_argument("--n_classes",   type=int, default=5)
    args = parser.parse_args()

    if args.generate_sample or args.input_csv is None:
        logger.info("Generating synthetic sample CSV ...")
        raw_path = Path(args.output_dir) / "raw_skeleton_sample.csv"
        generate_sample_csv(str(raw_path), n=args.n_samples, n_classes=args.n_classes)
        args.input_csv = str(raw_path)

    split_and_save(
        input_csv =args.input_csv,
        output_dir=args.output_dir,
        test_size =args.test_size,
        label_col =args.label_col,
        seed      =args.seed,
    )
