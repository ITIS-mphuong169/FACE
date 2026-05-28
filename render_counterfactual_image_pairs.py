#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
from pathlib import Path

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from movenet_wrapper import extract_pose_vector as movenet_extract_pose_vector
except Exception:
    movenet_extract_pose_vector = None

# COCO 17 edges (indices) — same skeleton as MoveNet BlazePose/COCO mapping used in CSV.
COCO17_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render wrong/suggested image pairs from existing dataset images.")
    parser.add_argument("--counterfactual-csv", required=True, help="Counterfactual CSV path.")
    parser.add_argument("--output-dir", default="visualizations", help="Output directory.")
    parser.add_argument("--max-pairs", type=int, default=6, help="Maximum pairs to render.")
    parser.add_argument(
        "--features-csv",
        nargs="*",
        default=None,
        help="One or more pose CSV files with columns image_path, f0..f50 for skeleton overlay.",
    )
    parser.add_argument(
        "--skeleton-min-confidence",
        type=float,
        default=0.15,
        help="Do not draw joint/edge if MoveNet confidence is below this value.",
    )
    parser.add_argument("--no-skeleton", action="store_true", help="Disable skeleton overlay even if features are available.")
    parser.add_argument(
        "--skeleton-source",
        choices=["auto", "movenet", "csv"],
        default="auto",
        help="auto: prefer MoveNet from image then fallback CSV; movenet: only MoveNet; csv: only CSV lookup.",
    )
    return parser.parse_args()


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if str(c).startswith("f")]


def load_pose_lookup(paths: list[Path]) -> dict[str, np.ndarray]:
    """Map relative image_path -> (51,) float pose vector."""
    if not paths:
        return {}
    frames = []
    for p in paths:
        frames.append(pd.read_csv(p.expanduser().resolve()))
    df = pd.concat(frames, ignore_index=True)
    cols = _feature_columns(df)
    if not cols:
        raise ValueError("No f0..f50 columns found in features CSV.")
    cols = sorted(cols, key=lambda c: int(c[1:]) if c[1:].isdigit() else c)
    if "image_path" not in df.columns:
        raise ValueError("Features CSV must contain image_path column.")
    vectors = df[cols].to_numpy(dtype=np.float32)
    keys = df["image_path"].astype(str)
    lookup: dict[str, np.ndarray] = {}
    for k, vec in zip(keys, vectors, strict=False):
        lookup[str(k)] = vec
    return lookup


def lookup_pose_vec(lookup: dict[str, np.ndarray], image_key: str) -> np.ndarray | None:
    if not lookup:
        return None
    if image_key in lookup:
        return lookup[image_key]
    base = Path(image_key).name
    candidates = [v for k, v in lookup.items() if Path(k).name == base]
    if len(candidates) == 1:
        return candidates[0]
    return None


def flatten51_to_xy_c(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """51-D flattened (x, y, c) per MoveNet extractor in movenet_wrapper.py."""
    if vec.size != 51:
        raise ValueError(f"Expected 51 features, got {vec.size}")
    v = vec.reshape(17, 3)
    xn, yn, conf = v[:, 0], v[:, 1], v[:, 2]
    return np.stack([xn, yn], axis=1), conf


def draw_skeleton_bgr(img_bgr: np.ndarray, vec: np.ndarray, min_conf: float, line_color=(0, 255, 255), joint_color=(0, 0, 255)) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    xy, conf = flatten51_to_xy_c(vec.astype(np.float32))
    px = np.clip(xy[:, 0] * w, 0, w - 1).astype(np.int32)
    py = np.clip(xy[:, 1] * h, 0, h - 1).astype(np.int32)
    out = img_bgr.copy()
    for a, b in COCO17_EDGES:
        if conf[a] < min_conf or conf[b] < min_conf:
            continue
        cv2.line(out, (px[a], py[a]), (px[b], py[b]), line_color, 2, lineType=cv2.LINE_AA)
    for i in range(17):
        if conf[i] < min_conf:
            continue
        cv2.circle(out, (px[i], py[i]), 4, joint_color, -1, lineType=cv2.LINE_AA)
    return out


def load_image_bgr(image_path: Path) -> np.ndarray | None:
    raw = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    if raw.ndim == 2:
        return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    if raw.shape[2] == 3:
        return raw
    if raw.shape[2] == 4:
        # Composite transparent PNGs on white to avoid black-background artifacts.
        bgr = raw[:, :, :3].astype(np.float32)
        alpha = (raw[:, :, 3:4].astype(np.float32) / 255.0)
        white = np.full_like(bgr, 255.0)
        blended = bgr * alpha + white * (1.0 - alpha)
        return blended.astype(np.uint8)
    return None


def overlay_skeleton_on_file(image_path: Path, vec: np.ndarray | None, min_conf: float) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Returns (rgb_no_overlay|None, rgb_with_overlay|None)."""
    bgr = load_image_bgr(image_path)
    if bgr is None:
        return None, None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if vec is None:
        return rgb, rgb
    sk_bgr = draw_skeleton_bgr(bgr, vec, min_conf=min_conf)
    return rgb, cv2.cvtColor(sk_bgr, cv2.COLOR_BGR2RGB)


def extract_pose_from_image(image_path: Path) -> np.ndarray | None:
    if movenet_extract_pose_vector is None:
        return None
    bgr = load_image_bgr(image_path)
    if bgr is None:
        return None
    try:
        vec = movenet_extract_pose_vector(bgr)
    except Exception:
        return None
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size < 51:
        return None
    return arr[:51]


def _safe_read_image(path: Path):
    try:
        return mpimg.imread(path)
    except Exception:
        return None


def _rgb_to_float_display(rgb_uint8: np.ndarray) -> np.ndarray:
    if rgb_uint8.dtype == np.uint8:
        return (rgb_uint8.astype(np.float32) / 255.0).clip(0.0, 1.0)
    return rgb_uint8


def rgb_to_bgr_save(rgb: np.ndarray) -> np.ndarray:
    arr = rgb
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _file_url(path: Path) -> str:
    return path.resolve().as_uri()


def _write_html_report(rows: pd.DataFrame, overlay_dir: Path, output_path: Path) -> None:
    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Counterfactual Image Pairs</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;background:#fafafa;color:#222}",
        ".pair{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:28px;padding:18px;background:white;border:1px solid #ddd;border-radius:10px}",
        ".card img{max-width:100%;max-height:320px;display:block;margin:0 auto 12px auto;border-radius:8px}",
        ".title{font-weight:700;margin-bottom:10px}",
        ".meta{font-size:14px;line-height:1.5;word-break:break-all}",
        "a{color:#0b57d0;text-decoration:none}",
        "a:hover{text-decoration:underline}",
        "</style></head><body>",
        "<h1>Counterfactual Image Pairs</h1>",
        "<p>Click the image path links below to open the original files directly.</p>",
    ]

    for idx, (_, row) in enumerate(rows.iterrows()):
        wrong_path = Path(str(row["wrong_image_abspath"]))
        suggested_path = Path(str(row["suggested_image_abspath"]))
        wrong_url = _file_url(wrong_path)
        suggested_url = _file_url(suggested_path)
        w_sk = overlay_dir / f"pair_{idx:02d}_wrong_sk.png"
        s_sk = overlay_dir / f"pair_{idx:02d}_suggested_sk.png"
        wrong_sk_url = _file_url(w_sk) if w_sk.is_file() else wrong_url
        sugg_sk_url = _file_url(s_sk) if s_sk.is_file() else suggested_url
        dist = float(row["distance"])
        dist_txt = f"{dist:.4f}" if abs(dist) < 1e5 else f"{dist:.3e}"
        parts.extend(
            [
                "<div class='pair'>",
                "<div class='card'>",
                "<div class='title'>Wrong input (with skeleton)</div>",
                f"<a href='{html.escape(wrong_sk_url)}'><img src='{html.escape(wrong_sk_url)}' alt='wrong with skeleton'></a>",
                "<div class='meta'>"
                f"prob_before={float(row['prob_before']):.3f}<br>"
                f"<a href='{html.escape(wrong_url)}'>{html.escape(str(wrong_path))}</a>"
                "</div>",
                "</div>",
                "<div class='card'>",
                "<div class='title'>Suggested existing image (with skeleton)</div>",
                f"<a href='{html.escape(sugg_sk_url)}'><img src='{html.escape(sugg_sk_url)}' alt='suggested with skeleton'></a>",
                "<div class='meta'>"
                f"prob_after={float(row['prob_after']):.3f}<br>"
                f"distance={dist_txt}<br>"
                f"<a href='{html.escape(suggested_url)}'>{html.escape(str(suggested_path))}</a>"
                "</div>",
                "</div>",
                "</div>",
            ]
        )

    parts.append("</body></html>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cf_df = pd.read_csv(Path(args.counterfactual_csv).expanduser().resolve())
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "skeleton_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    pose_lookup: dict[str, np.ndarray] = {}
    if args.features_csv and not args.no_skeleton:
        pose_lookup = load_pose_lookup([Path(p) for p in args.features_csv])

    if cf_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No counterfactual pairs available", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / "counterfactual_image_pairs.png", dpi=200)
        plt.close(fig)
        _write_html_report(cf_df, overlay_dir, output_dir / "counterfactual_image_pairs.html")
        print(f"Saved empty image-pair panel: {output_dir / 'counterfactual_image_pairs.png'}")
        print(f"Saved clickable HTML report: {output_dir / 'counterfactual_image_pairs.html'}")
        return

    rows = cf_df.head(args.max_pairs).copy()
    fig, axes = plt.subplots(len(rows), 2, figsize=(10, 4 * len(rows)))
    if len(rows) == 1:
        axes = [axes]

    for idx, (_, row) in enumerate(rows.iterrows()):
        left_ax, right_ax = axes[idx]
        wrong_path = Path(str(row["wrong_image_abspath"]))
        suggested_path = Path(str(row["suggested_image_abspath"]))
        wrong_key = str(row["wrong_image_path"]) if "wrong_image_path" in row else Path(row["wrong_image_abspath"]).name
        sugg_key = (
            str(row["suggested_image_path"]) if "suggested_image_path" in row else Path(row["suggested_image_abspath"]).name
        )
        vw = None
        vs = None
        if args.skeleton_source in {"auto", "movenet"}:
            vw = extract_pose_from_image(wrong_path)
            vs = extract_pose_from_image(suggested_path)
        if args.skeleton_source in {"auto", "csv"}:
            if vw is None and pose_lookup:
                vw = lookup_pose_vec(pose_lookup, wrong_key)
            if vs is None and pose_lookup:
                vs = lookup_pose_vec(pose_lookup, sugg_key)
        rgb_w_raw, rgb_w_sk = overlay_skeleton_on_file(wrong_path, vw, args.skeleton_min_confidence)
        rgb_s_raw, rgb_s_sk = overlay_skeleton_on_file(suggested_path, vs, args.skeleton_min_confidence)

        if rgb_w_sk is not None:
            left_ax.imshow(_rgb_to_float_display(rgb_w_sk))
            sk_w_png = overlay_dir / f"pair_{idx:02d}_wrong_sk.png"
            cv2.imwrite(str(sk_w_png), rgb_to_bgr_save(rgb_w_sk))
        else:
            wrong_img = rgb_w_raw if rgb_w_raw is not None else _safe_read_image(wrong_path)
            if wrong_img is not None:
                left_ax.imshow(wrong_img)
            else:
                left_ax.text(0.5, 0.5, f"Missing image\n{wrong_path.name}", ha="center", va="center")

        if rgb_s_sk is not None:
            right_ax.imshow(_rgb_to_float_display(rgb_s_sk))
            sk_s_png = overlay_dir / f"pair_{idx:02d}_suggested_sk.png"
            cv2.imwrite(str(sk_s_png), rgb_to_bgr_save(rgb_s_sk))
        else:
            suggested_img = rgb_s_raw if rgb_s_raw is not None else _safe_read_image(suggested_path)
            if suggested_img is not None:
                right_ax.imshow(suggested_img)
            else:
                right_ax.text(0.5, 0.5, f"Missing image\n{suggested_path.name}", ha="center", va="center")

        left_ax.set_title(
            f"Wrong input\n{row['wrong_image_path']}\nprob_before={row['prob_before']:.3f}",
            fontsize=10,
        )
        right_ax.set_title(
            f"Suggested existing image\n{row['suggested_image_path']}\nprob_after={row['prob_after']:.3f}",
            fontsize=10,
        )
        left_ax.axis("off")
        right_ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "counterfactual_image_pairs.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    _write_html_report(rows, overlay_dir, output_dir / "counterfactual_image_pairs.html")
    print(f"Saved image-pair panel: {output_dir / 'counterfactual_image_pairs.png'}")
    print(f"Saved clickable HTML report: {output_dir / 'counterfactual_image_pairs.html'}")
    if pose_lookup:
        print(f"Saved skeleton overlay PNGs under: {overlay_dir}")


if __name__ == "__main__":
    main()
