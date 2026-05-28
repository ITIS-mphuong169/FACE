"""
plot_pca_face.py  (v4 — paper faithful)
-----------------------------------------
PCA visualization giong Figure 2/3 trong paper FACE:

  - Background: contourf P(target class) — do cao, xanh thap (RdBu_r)
  - Scatter: do=target class, xanh=other classes
  - FACE path: duong den noi input -> intermediate nodes -> CF
  - Input: filled black circle (starting point)
  - CF: open circle vien den (target point, nhat)
  - Intermediate nodes: filled gray circles (path nodes)
  - PCA fit chi tren train set (khong include outlier test point)
  - Axis zoom vao vung chua tat ca path nodes de thay ro duong di

Save: outputs/pca_case{1|2}.png
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def plot_pca_face(
    train_features   : np.ndarray,          # (N, D) scaled
    train_labels_raw : np.ndarray,          # (N,) original labels
    x_input          : np.ndarray,          # (D,) scaled — input sai
    x_counterfactual : np.ndarray,          # (D,) scaled — CF dung
    face_path_indices: Optional[List[int]], # train indices tren path (khong gom query node)
    model,
    le,
    target_label,                           # original target label
    case_id          : int,
    output_dir       : str = "outputs",
    input_label_str  : str = "Input",
    cf_label_str     : str = "CF",
    method           : str = "FACE shortest path",
    input_image_path : Optional[str] = None,   # ignored, backward compat
    cf_image_path    : Optional[str] = None,   # ignored, backward compat
    title_suffix     : str = "",
) -> str:
    """
    Ve PCA FACE plot giong Figure 2/3 trong paper.
    PCA duoc fit CHI tren train set.
    Axis duoc zoom vao vung chua path.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(out_dir / f"pca_case{case_id}.png")

    N = len(train_features)
    target_enc = int(le.transform([target_label])[0])

    # ── PCA: fit CHI tren train, project input/CF sau ─────────────────────
    pca = PCA(n_components=2, random_state=42)
    train_2d = pca.fit_transform(train_features)

    input_2d = pca.transform(x_input.reshape(1, -1))[0]
    cf_2d    = pca.transform(x_counterfactual.reshape(1, -1))[0]

    # ── Path nodes 2D ─────────────────────────────────────────────────────
    valid_path_idxs = []
    if face_path_indices is not None:
        valid_path_idxs = [i for i in face_path_indices if 0 <= i < N]

    path_2d = train_2d[valid_path_idxs] if valid_path_idxs else None

    # ── Full path array: input -> path nodes -> CF ─────────────────────────
    if path_2d is not None and len(path_2d) > 0:
        full_path_2d = np.vstack([
            input_2d.reshape(1, -1),
            path_2d,
            cf_2d.reshape(1, -1),
        ])
    else:
        full_path_2d = np.vstack([
            input_2d.reshape(1, -1),
            cf_2d.reshape(1, -1),
        ])

    # ── Axis range: zoom vao vung chua path + padding + 1/3 train spread ──
    # Lay bbox cua tat ca path points de xac dinh vung zoom
    all_path_pts = full_path_2d
    px_min, px_max = all_path_pts[:, 0].min(), all_path_pts[:, 0].max()
    py_min, py_max = all_path_pts[:, 1].min(), all_path_pts[:, 1].max()

    # Mo rong them vung train xung quanh path (context)
    spread_x = (train_2d[:, 0].max() - train_2d[:, 0].min()) * 0.35
    spread_y = (train_2d[:, 1].max() - train_2d[:, 1].min()) * 0.35
    pad_x = max(spread_x, (px_max - px_min) * 0.6 + 1.0)
    pad_y = max(spread_y, (py_max - py_min) * 0.6 + 1.0)

    x_lo = max(train_2d[:, 0].min() - 0.5, px_min - pad_x)
    x_hi = min(train_2d[:, 0].max() + 0.5, px_max + pad_x)
    y_lo = max(train_2d[:, 1].min() - 0.5, py_min - pad_y)
    y_hi = min(train_2d[:, 1].max() + 0.5, py_max + pad_y)

    # ── Figure ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ── Decision boundary background ───────────────────────────────────────
    n_grid = 150
    xx = np.linspace(x_lo, x_hi, n_grid)
    yy = np.linspace(y_lo, y_hi, n_grid)
    XX, YY = np.meshgrid(xx, yy)
    grid_2d   = np.c_[XX.ravel(), YY.ravel()]
    grid_feat = pca.inverse_transform(grid_2d)

    try:
        probs_grid = model.predict_proba(grid_feat)[:, target_enc]
        ZZ = probs_grid.reshape(XX.shape)

        # RdBu_r: xanh = P thap (vung class khac), do = P cao (vung target class)
        cmap = matplotlib.colormaps.get_cmap("RdBu_r")
        ax.contourf(XX, YY, ZZ,
                    levels=np.linspace(0, 1, 101),
                    cmap=cmap, alpha=0.85, zorder=0)

        # Decision boundary line (P = 0.5)
        ax.contour(XX, YY, ZZ,
                   levels=[0.5],
                   colors=["#333333"], linewidths=1.0,
                   linestyles="-", zorder=1, alpha=0.7)
    except Exception as e:
        logger.warning(f"Decision background failed: {e}")

    # ── Scatter: chi hien cac diem TRONG vung axis zoom ───────────────────
    in_view = (
        (train_2d[:, 0] >= x_lo) & (train_2d[:, 0] <= x_hi) &
        (train_2d[:, 1] >= y_lo) & (train_2d[:, 1] <= y_hi)
    )

    # 12 professional, vibrant colors
    colors_12 = [
        "#E53935", # Red
        "#1E88E5", # Blue
        "#43A047", # Green
        "#8E24AA", # Purple
        "#FB8C00", # Orange
        "#00ACC1", # Cyan
        "#D81B60", # Pink
        "#7CB342", # Light Green
        "#FFB300", # Amber
        "#5E35B1", # Deep Purple
        "#00897B", # Teal
        "#F4511E"  # Deep Orange
    ]

    unique_labels = sorted(list(set(train_labels_raw)))
    label_to_color = {lbl: colors_12[i % len(colors_12)] for i, lbl in enumerate(unique_labels)}

    # Plot each class separately so they get unique colors and show up in the legend
    for lbl in unique_labels:
        lbl_mask = (train_labels_raw == lbl) & in_view
        if lbl_mask.any():
            is_target = (str(lbl) == str(target_label))
            edge_w = 0.75 if is_target else 0.25
            alpha_val = 0.85 if is_target else 0.55
            marker_size = 35 if is_target else 18
            
            label_name = f"{lbl} (Target)" if is_target else str(lbl)
            
            ax.scatter(
                train_2d[lbl_mask, 0], train_2d[lbl_mask, 1],
                c=label_to_color[lbl], s=marker_size, alpha=alpha_val,
                edgecolors="black" if is_target else "white", linewidths=edge_w,
                zorder=3 if is_target else 2, rasterized=True,
                label=label_name,
            )

    # ── FACE PATH: duong den qua cac node ─────────────────────────────────
    ax.plot(
        full_path_2d[:, 0], full_path_2d[:, 1],
        color="black", linewidth=2.5,
        linestyle="-", marker="o", markersize=6,
        markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.2,
        zorder=6, alpha=0.95,
        label=f"FACE path ({input_label_str} -> {cf_label_str})",
    )

    # ── Axis, grid, styling ────────────────────────────────────────────────
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("PCA Component 1", fontsize=11)
    ax.set_ylabel("PCA Component 2", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.grid(True, color="#cccccc", linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("#aaaaaa")
        spine.set_linewidth(0.8)

    # ── Legend ────────────────────────────────────────────────────────────
    ax.legend(loc="upper right", ncol=2, fontsize=7,
              framealpha=0.88, edgecolor="#cccccc", facecolor="white")

    # ── Title ──────────────────────────────────────────────────────────────
    var = pca.explained_variance_ratio_
    path_len = len(full_path_2d) - 1   # so buoc
    n_interm = len(valid_path_idxs) - 1 if valid_path_idxs else 0
    method_tag = "KDE graph" if "FACE" in method else "fallback nearest"

    ax.set_title(
        f"FACE shortest path: {input_label_str} -> {cf_label_str}"
        f"  ({path_len} hop{'s' if path_len > 1 else ''}"
        f", {n_interm} intermediate node{'s' if n_interm != 1 else ''})\n"
        f"Case {case_id} | {method_tag} | $t_p \\geq 0.90$",
        fontsize=9, pad=10, linespacing=1.5,
    )
    ax.text(0.01, 0.01,
            f"PCA variance: {var[0]*100:.1f}% + {var[1]*100:.1f}%",
            transform=ax.transAxes, fontsize=7,
            color="#666666", va="bottom")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    logger.info(f"PCA plot saved -> {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, joblib

    parser = argparse.ArgumentParser(description="Plot PCA FACE (paper style)")
    parser.add_argument("--result_json",  required=True)
    parser.add_argument("--model_dir",    required=True)
    parser.add_argument("--train_csv",    required=True)
    parser.add_argument("--case",         type=int, choices=[1, 2], required=True)
    parser.add_argument("--output_dir",   default="outputs")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from load_dataset import load_dataset

    model_dir = Path(args.model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    model  = joblib.load(model_dir / "model.pkl")
    scaler = joblib.load(model_dir / "scaler.pkl")
    le     = joblib.load(model_dir / "label_encoder.pkl")

    X_train, y_train, df_train = load_dataset(
        args.train_csv,
        use_confidence=config["use_confidence"],
        scale_mode=config["scale_mode"],
        center=config["center"],
        label_col=config["label_col"],
    )
    X_train_s = scaler.transform(X_train)

    result = json.loads(Path(args.result_json).read_text())
    x_input = np.array(result["x_input_scaled"])
    x_cf    = X_train_s[result["counterfactual_idx"]]

    raw_path = result.get("face_path")
    path_idxs = [i for i in (raw_path or []) if 0 <= i < len(X_train_s)]

    plot_pca_face(
        train_features   =X_train_s,
        train_labels_raw =y_train,
        x_input          =x_input,
        x_counterfactual =x_cf,
        face_path_indices=path_idxs,
        model=model, le=le,
        target_label=result["target_label"],
        case_id=args.case,
        output_dir=args.output_dir,
        input_label_str=str(result.get("predicted_label", "?")),
        cf_label_str=str(result.get("counterfactual_label", "?")),
        method=result.get("method", "FACE shortest path"),
    )
