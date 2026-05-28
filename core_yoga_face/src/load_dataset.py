"""
load_dataset.py
---------------
Load and normalize skeleton CSV data extracted by MoveNet.
Skeleton: 17 keypoints × (x, y, confidence) = 51 features.
"""

import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# MoveNet 17 keypoint names
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


# Generate 51 feature column names: kp0_x, kp0_y, kp0_c, ...
KP_COLS: List[str] = []
for _ki in range(17):
    KP_COLS += [f"kp{_ki}_x", f"kp{_ki}_y", f"kp{_ki}_c"]

# Hip indices for centering (left_hip=11, right_hip=12)
LEFT_HIP_IDX       = 11
RIGHT_HIP_IDX      = 12
# Shoulder indices for torso length (left_shoulder=5, right_shoulder=6)
LEFT_SHOULDER_IDX  = 5
RIGHT_SHOULDER_IDX = 6

# Alternative column names used by YogaSinglePose-PTIT dataset
_KP_NAME_COLS: List[str] = []
for _name in KEYPOINT_NAMES:
    _KP_NAME_COLS += [f"{_name}_x", f"{_name}_y", f"{_name}_score"]

# Map named → kp-indexed
_NAME_TO_KP: dict = {}
for _i, _name in enumerate(KEYPOINT_NAMES):
    _NAME_TO_KP[f"{_name}_x"]     = f"kp{_i}_x"
    _NAME_TO_KP[f"{_name}_y"]     = f"kp{_i}_y"
    _NAME_TO_KP[f"{_name}_score"] = f"kp{_i}_c"


def _build_kp_columns(df: pd.DataFrame) -> List[str]:
    """
    Return the 51 keypoint columns that exist in df.
    
    Supports two naming conventions:
      1. kp{i}_x, kp{i}_y, kp{i}_c   (standard)
      2. nose_x, nose_y, nose_score, left_eye_x, ... (YogaSinglePose-PTIT dataset)
    
    If convention 2 detected, renames columns in-place to convention 1.
    """
    # Check if standard kp cols exist
    present_kp = [c for c in KP_COLS if c in df.columns]
    if len(present_kp) == 51:
        return KP_COLS

    # Try named convention
    present_named = [c for c in _KP_NAME_COLS if c in df.columns]
    if len(present_named) == 51:
        logger.info("Detected named keypoint columns (nose_x, ...). Remapping to kp{i}_x format.")
        df.rename(columns=_NAME_TO_KP, inplace=True)
        return KP_COLS

    # Mixed or partial — try to rename whatever matches
    rename_map = {k: v for k, v in _NAME_TO_KP.items() if k in df.columns}
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    present_kp = [c for c in KP_COLS if c in df.columns]
    if len(present_kp) < 51:
        raise ValueError(
            f"Expected 51 keypoint columns but found {len(present_kp)}. "
            f"Detected columns (first 10): {list(df.columns[:10])}. "
            "Ensure CSV has either kp{{i}}_x/y/c or nose_x/y/score, ... format."
        )
    return KP_COLS



def load_csv(csv_path: str | Path) -> pd.DataFrame:
    """Read a CSV file and return a DataFrame."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")
    return df


def handle_nan_keypoints(
    features: np.ndarray,
    strategy: str = "zero",
) -> np.ndarray:
    """
    Replace NaN / infinite values in keypoint feature matrix.

    Parameters
    ----------
    features : (N, 51) float array
    strategy : 'zero' | 'mean' | 'drop'
        'drop'  → raises error; caller must handle row-level dropping.
    """
    if strategy == "zero":
        features = np.where(np.isfinite(features), features, 0.0)
    elif strategy == "mean":
        col_means = np.nanmean(features, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        inds = np.where(~np.isfinite(features))
        features[inds] = np.take(col_means, inds[1])
    elif strategy == "drop":
        mask = np.all(np.isfinite(features), axis=1)
        if not mask.all():
            logger.warning(
                f"'drop' strategy would remove {(~mask).sum()} rows. "
                "Returning boolean mask as second element."
            )
        return features, mask
    else:
        raise ValueError(f"Unknown NaN strategy: {strategy}")
    return features


def center_skeleton(features: np.ndarray) -> np.ndarray:
    """
    Translate each skeleton so that the mid-hip point is at the origin.
    features : (N, 51)  layout [..., kp_x, kp_y, kp_c, ...]
    """
    features = features.copy()
    lhip_x = features[:, LEFT_HIP_IDX * 3]
    lhip_y = features[:, LEFT_HIP_IDX * 3 + 1]
    rhip_x = features[:, RIGHT_HIP_IDX * 3]
    rhip_y = features[:, RIGHT_HIP_IDX * 3 + 1]

    cx = (lhip_x + rhip_x) / 2.0
    cy = (lhip_y + rhip_y) / 2.0

    for i in range(17):
        features[:, i * 3]     -= cx
        features[:, i * 3 + 1] -= cy

    return features


def scale_skeleton(features: np.ndarray, mode: str = "torso") -> np.ndarray:
    """
    Scale skeleton by torso length or bbox diagonal.

    mode : 'torso' | 'bbox'
    """
    features = features.copy()
    eps = 1e-8

    if mode == "torso":
        # mid-hip → mid-shoulder distance
        lhip_y  = features[:, LEFT_HIP_IDX * 3 + 1]
        rhip_y  = features[:, RIGHT_HIP_IDX * 3 + 1]
        lsho_y  = features[:, LEFT_SHOULDER_IDX * 3 + 1]
        rsho_y  = features[:, RIGHT_SHOULDER_IDX * 3 + 1]
        lhip_x  = features[:, LEFT_HIP_IDX * 3]
        rhip_x  = features[:, RIGHT_HIP_IDX * 3]
        lsho_x  = features[:, LEFT_SHOULDER_IDX * 3]
        rsho_x  = features[:, RIGHT_SHOULDER_IDX * 3]

        mid_hip_x  = (lhip_x + rhip_x) / 2.0
        mid_hip_y  = (lhip_y + rhip_y) / 2.0
        mid_sho_x  = (lsho_x + rsho_x) / 2.0
        mid_sho_y  = (lsho_y + rsho_y) / 2.0

        scale = np.sqrt(
            (mid_sho_x - mid_hip_x) ** 2 +
            (mid_sho_y - mid_hip_y) ** 2
        ) + eps

    elif mode == "bbox":
        x_cols = [i * 3 for i in range(17)]
        y_cols = [i * 3 + 1 for i in range(17)]
        x_vals = features[:, x_cols]
        y_vals = features[:, y_cols]
        w = x_vals.max(axis=1) - x_vals.min(axis=1) + eps
        h = y_vals.max(axis=1) - y_vals.min(axis=1) + eps
        scale = np.sqrt(w ** 2 + h ** 2)

    else:
        raise ValueError(f"Unknown scale mode: {mode}")

    # Scale only x, y (not confidence)
    for i in range(17):
        features[:, i * 3]     /= scale
        features[:, i * 3 + 1] /= scale

    return features


def normalize_skeleton(
    features: np.ndarray,
    center: bool = True,
    scale_mode: Optional[str] = "torso",
) -> np.ndarray:
    """Full normalization pipeline: center → scale."""
    if center:
        features = center_skeleton(features)
    if scale_mode:
        features = scale_skeleton(features, mode=scale_mode)
    return features


def load_dataset(
    csv_path: str | Path,
    use_confidence: bool = True,
    center: bool = True,
    scale_mode: Optional[str] = "torso",
    nan_strategy: str = "zero",
    label_col: str = "label",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Full dataset loading pipeline.

    Returns
    -------
    features : (N, 51) or (N, 34) float array  (34 if use_confidence=False)
    labels   : (N,) int/str array
    df       : full DataFrame (includes image_path, label, optional columns)
    """
    df = load_csv(csv_path)
    kp_cols = _build_kp_columns(df)

    # Extract raw features
    features = df[kp_cols].values.astype(np.float32)

    # Handle NaN
    result = handle_nan_keypoints(features, strategy=nan_strategy)
    if isinstance(result, tuple):
        features, mask = result
        df = df[mask].reset_index(drop=True)
        features = features[mask]
    else:
        features = result

    # Normalize
    features = normalize_skeleton(features, center=center, scale_mode=scale_mode)

    # Drop confidence columns if not needed
    if not use_confidence:
        keep = []
        for i in range(17):
            keep += [i * 3, i * 3 + 1]
        features = features[:, keep]
        logger.info("Dropped confidence channels → %d features", features.shape[1])
    else:
        logger.info("Using all 51 features (x, y, confidence per keypoint)")

    labels = df[label_col].values
    logger.info(
        f"Dataset: {features.shape[0]} samples, "
        f"{features.shape[1]} features, "
        f"{len(np.unique(labels))} classes"
    )
    return features, labels, df


# ---------------------------------------------------------------------------
# CLI for quick inspection
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and inspect skeleton dataset")
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--no_confidence", action="store_true")
    parser.add_argument("--scale_mode", default="torso", choices=["torso", "bbox", "none"])
    parser.add_argument("--no_center", action="store_true")
    args = parser.parse_args()

    scale = None if args.scale_mode == "none" else args.scale_mode
    X, y, df = load_dataset(
        args.csv,
        use_confidence=not args.no_confidence,
        center=not args.no_center,
        scale_mode=scale,
        label_col=args.label_col,
    )
    print(f"\nFeature matrix : {X.shape}")
    print(f"Labels         : {np.unique(y, return_counts=True)}")
    print(f"\nSample row 0:\n{X[0]}")
    print(f"\nDF columns: {list(df.columns)}")
