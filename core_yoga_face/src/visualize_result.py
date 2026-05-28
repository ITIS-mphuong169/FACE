"""
visualize_skeleton.py  (v3 — images only, no skeleton)
--------------------------------------------------------
Hiển thị ảnh THẬT từ dataset, không có skeleton overlay.

Layout:
  [Ảnh Input Sai]  |  mũi tên  |  [Ảnh Counterfactual Đúng]
  
  + (nếu có FACE path dài > 1 bước) các ảnh intermediate bên dưới

Save: outputs/result_visualization.png
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as FancyArrowPatch
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

BG_COLOR   = "#0d1117"
AX_COLOR   = "#161b22"
GRID_COLOR = "#30363d"


def _load_image(path: Optional[str]) -> Optional[np.ndarray]:
    """Load image tu path, return RGB ndarray."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        logger.warning(f"Image not found: {p}")
        return None
    try:
        return np.array(Image.open(p).convert("RGB"))
    except Exception as e:
        logger.warning(f"Cannot load {p}: {e}")
        return None


def _draw_image_panel(
    ax,
    img_arr,
    title: str,
    title_color: str = "#ffffff",
    border_color: str = "#ffffff",
    subtitle: str = "",
):
    """Ve anh that vao ax, co border va title dep."""
    ax.set_facecolor(AX_COLOR)
    if img_arr is not None:
        ax.imshow(img_arr, aspect="auto")
    else:
        ax.text(
            0.5, 0.5,
            "Anh khong\ntim thay",
            ha="center", va="center",
            color="#8b949e", fontsize=11,
            transform=ax.transAxes,
        )
    full_title = title
    if subtitle:
        full_title += f"\n{subtitle}"
    ax.set_title(full_title, color=title_color, fontsize=10,
                 fontweight="bold", pad=7, linespacing=1.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(3.0)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_arrow_ax(ax, text: str = "FACE\npath"):
    """Ve mui ten tu trai sang phai trong ax."""
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.annotate(
        "",
        xy=(0.85, 0.5), xytext=(0.15, 0.5),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>",
            color="#00e676",
            lw=3.5,
            mutation_scale=28,
        ),
    )
    ax.text(0.5, 0.62, text, ha="center", va="bottom",
            color="#00e676", fontsize=9, fontweight="bold",
            transform=ax.transAxes)
    ax.axis("off")


def visualize_result(
    input_img_path  : Optional[str],
    cf_img_path     : Optional[str],
    input_label     : str,
    cf_label        : str,
    input_prob      : float,
    cf_prob         : float,
    case_id         : int,
    output_dir      : str = "outputs",
    method          : str = "FACE shortest path",
    face_path_images: Optional[List[str]] = None,
    gt_label        : str = "",
) -> str:
    """
    Tao visualization chi voi anh that, khong co skeleton.

    Layout chinh:
      [Input sai]  [->]  [CF dung]

    Neu co path images (intermediate nodes):
      [Input] [->] [node1] [->] [node2] ... [->] [CF]
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(out_dir / "result_visualization.png")

    img_input = _load_image(input_img_path)
    img_cf    = _load_image(cf_img_path)

    # Intermediate path images (toi da 3)
    path_imgs  = []
    path_paths = face_path_images or []
    for p in path_paths[:3]:
        arr = _load_image(p)
        path_imgs.append((p, arr))

    has_path = len(path_imgs) > 0

    # ── Tinh so cot ────────────────────────────────────────────────────────
    # [input] [arrow] [node1] [arrow] ... [arrow] [CF]
    img_items  = [("input", input_img_path, img_input)]
    for i, (pp, arr) in enumerate(path_imgs):
        img_items.append((f"node{i+1}", pp, arr))
    img_items.append(("cf", cf_img_path, img_cf))

    n_imgs   = len(img_items)
    n_arrows = n_imgs - 1
    n_cols   = n_imgs + n_arrows  # alt: img, arrow, img, arrow, ..., img

    # Width ratios: images are wider than arrows
    w_ratios = []
    for i in range(n_cols):
        w_ratios.append(0.18 if (i % 2 == 1) else 1.0)

    fig_w = 5 * n_imgs + 1.5 * n_arrows
    fig_w = max(fig_w, 12)

    fig = plt.figure(figsize=(fig_w, 6.5), facecolor=BG_COLOR)
    gs  = gridspec.GridSpec(1, n_cols, figure=fig,
                            width_ratios=w_ratios,
                            wspace=0.04)

    # ── Ve tung item ────────────────────────────────────────────────────────
    col = 0
    for item_idx, (key, img_path, img_arr) in enumerate(img_items):
        ax = fig.add_subplot(gs[0, col])
        col += 1

        if key == "input":
            border = "#ff1744"
            tc     = "#ff6b6b"
            tag    = "[X] INPUT SAI"
            sub    = f"GT: {gt_label}  |  Pred: {input_label}\nP(target) = {input_prob:.2%}"
        elif key == "cf":
            border = "#00e676"
            tc     = "#69f0ae"
            tag    = "[OK] COUNTERFACTUAL"
            sub    = f"Label: {cf_label}\nP(target) = {cf_prob:.2%}"
        else:
            num    = key.replace("node", "")
            border = "#b0bec5"
            tc     = "#b0bec5"
            tag    = f"Path node {num}"
            sub    = ""

        _draw_image_panel(ax, img_arr, title=tag,
                          title_color=tc, border_color=border,
                          subtitle=sub)

        # Ve mui ten sau moi anh (tru cai cuoi)
        if item_idx < len(img_items) - 1:
            ax_arrow = fig.add_subplot(gs[0, col])
            col += 1
            path_len_str = f"FACE\npath" if item_idx == 0 else ""
            _draw_arrow_ax(ax_arrow, text=path_len_str)

    # ── Tieu de toan cuc ────────────────────────────────────────────────────
    method_tag = "FACE shortest path" if "FACE" in method else "fallback nearest"
    path_note  = f" ({len(path_imgs)} intermediate node(s))" if has_path else ""
    fig.suptitle(
        f"CORE-Yoga FACE  |  Case {case_id}  |  {method_tag}{path_note}\n"
        f"Anh that tu dataset  --  Tu the sai -> Counterfactual dung",
        color="#c9d1d9", fontsize=11, fontweight="bold",
        y=1.02,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR)
    plt.close(fig)
    logger.info(f"Visualization saved -> {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Backward-compat alias (giu nguyen ten ham cu de run_core_yoga_demo goi duoc)
# ---------------------------------------------------------------------------
def visualize_result_compare(
    input_kp         = None,
    cf_kp            = None,
    input_label      : str = "",
    cf_label         : str = "",
    input_prob       : float = 0.0,
    cf_prob          : float = 0.0,
    case_id          : int = 1,
    output_dir       : str = "outputs",
    input_img_path   : Optional[str] = None,
    cf_img_path      : Optional[str] = None,
    method           : str = "FACE shortest path",
    conf_threshold   : float = 0.15,
    face_path_images : Optional[List[str]] = None,
    gt_label         : str = "",
) -> str:
    """Backward-compatible wrapper — ignores skeleton params."""
    return visualize_result(
        input_img_path   = input_img_path,
        cf_img_path      = cf_img_path,
        input_label      = input_label,
        cf_label         = cf_label,
        input_prob       = input_prob,
        cf_prob          = cf_prob,
        case_id          = case_id,
        output_dir       = output_dir,
        method           = method,
        face_path_images = face_path_images,
        gt_label         = gt_label,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize real images (no skeleton)")
    parser.add_argument("--result_json",    required=True)
    parser.add_argument("--train_csv",      required=True)
    parser.add_argument("--model_dir",      required=True)
    parser.add_argument("--case",           type=int, choices=[1, 2], required=True)
    parser.add_argument("--output_dir",     default="outputs")
    args = parser.parse_args()

    import joblib
    sys.path.insert(0, str(Path(__file__).parent))
    from load_dataset import load_dataset

    model_dir = Path(args.model_dir)
    config    = json.loads((model_dir / "config.json").read_text())

    X_train, y_train, df_train = load_dataset(
        args.train_csv,
        use_confidence=config["use_confidence"],
        scale_mode=config["scale_mode"],
        center=config["center"],
        label_col=config["label_col"],
    )

    result = json.loads(Path(args.result_json).read_text())

    cf_idx = result["counterfactual_idx"]
    raw_path = result.get("face_path")
    N = len(X_train)
    path_train_idxs  = [i for i in (raw_path or []) if 0 <= i < N]
    intermediate_idxs = path_train_idxs[:-1] if path_train_idxs else []
    path_images = []
    if "image_path" in df_train.columns:
        path_images = [df_train["image_path"].iloc[i]
                       for i in intermediate_idxs if 0 <= i < len(df_train)]

    visualize_result(
        input_img_path   = result.get("input_image_path"),
        cf_img_path      = result.get("counterfactual_path"),
        input_label      = str(result.get("predicted_label", "")),
        cf_label         = str(result.get("counterfactual_label", "")),
        input_prob       = result.get("input_prob_target", 0.0),
        cf_prob          = result.get("counterfactual_prob", 0.0),
        case_id          = args.case,
        output_dir       = args.output_dir,
        method           = result.get("method", "FACE shortest path"),
        face_path_images = path_images,
        gt_label         = str(result.get("ground_truth_label", "")),
    )
