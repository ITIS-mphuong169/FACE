"""
normalize_skeleton_csv.py
-------------------------
Đổi tên cột kp{i}_* → tên keypoint MoveNet (nose_x, left_wrist_x, …)
và chuyển image_path tuyệt đối → tương đối repo FACE (portable).

Usage:
  python src/normalize_skeleton_csv.py \\
      --input data/movenet_with_paths.csv \\
      --output data/movenet_with_paths.csv \\
      --inplace
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from load_dataset import KEYPOINT_NAMES, canonical_dataset_image_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def kp_index_columns() -> list[str]:
    cols: list[str] = []
    for i in range(17):
        cols += [f"kp{i}_x", f"kp{i}_y", f"kp{i}_c"]
    return cols


def named_columns() -> list[str]:
    cols: list[str] = []
    for name in KEYPOINT_NAMES:
        cols += [f"{name}_x", f"{name}_y", f"{name}_score"]
    return cols


def kp_to_named_rename_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for i, name in enumerate(KEYPOINT_NAMES):
        mapping[f"kp{i}_x"] = f"{name}_x"
        mapping[f"kp{i}_y"] = f"{name}_y"
        mapping[f"kp{i}_c"] = f"{name}_score"
    return mapping


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def to_relative_image_path(path: str, repo_root: Path) -> str:
    p = Path(path)
    s = p.as_posix()

    try:
        rel = p.resolve().relative_to(repo_root.resolve()).as_posix()
        return canonical_dataset_image_path(rel)
    except ValueError:
        pass

    for part in ("Documents/FACE/", "/FACE/"):
        idx = s.find(part)
        if idx >= 0:
            return canonical_dataset_image_path(s[idx + len(part) :].lstrip("/"))

    return canonical_dataset_image_path(s)


def normalize_dataframe(df: pd.DataFrame, repo_root: Path) -> pd.DataFrame:
    out = df.copy()

    kp_cols = kp_index_columns()
    if all(c in out.columns for c in kp_cols):
        out = out.rename(columns=kp_to_named_rename_map())
        logger.info("Renamed kp0..kp16 columns → MoveNet keypoint names (…_score).")
    elif all(c in out.columns for c in named_columns()):
        logger.info("Keypoint columns already use named format.")
    else:
        missing = [c for c in kp_cols if c not in out.columns]
        raise ValueError(
            f"Expected 51 kp* or named keypoint columns; missing {len(missing)} (e.g. {missing[:3]})."
        )

    if "image_path" not in out.columns:
        raise ValueError("CSV must contain image_path column.")

    out["image_path"] = [
        to_relative_image_path(str(v), repo_root) for v in out["image_path"]
    ]
    logger.info("Converted image_path → dataset/<label>/<file> (short form).")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize skeleton CSV columns and image paths.")
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output", default=None, help="Output CSV (default: --input).")
    parser.add_argument(
        "--repo-root",
        default=None,
        help="FACE repository root for relative paths (default: auto).",
    )
    parser.add_argument("--inplace", action="store_true", help="Overwrite --input.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if args.inplace:
        output_path = input_path
    else:
        output_path = Path(args.output or args.input).expanduser().resolve()

    repo_root = Path(args.repo_root).expanduser().resolve() if args.repo_root else default_repo_root()

    df = pd.read_csv(input_path)
    logger.info("Loaded %d rows from %s", len(df), input_path)
    normalized = normalize_dataframe(df, repo_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output_path, index=False)
    logger.info("Saved → %s", output_path)
    print("\nSample columns:", list(normalized.columns[:6]), "…", list(normalized.columns[-3:]))
    print("Sample image_path:", normalized["image_path"].iloc[0])


if __name__ == "__main__":
    main()
