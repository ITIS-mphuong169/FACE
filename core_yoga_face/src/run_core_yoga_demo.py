"""
run_core_yoga_demo.py
---------------------
End-to-end CORE-Yoga FACE demo.

Usage:
  python src/run_core_yoga_demo.py \\
      --train_csv data/train.csv \\
      --test_csv  data/test.csv  \\
      --model_dir outputs/model  \\
      --case 1 \\
      --threshold 0.90 \\
      --output_dir outputs/case1

Case 1: Correct action, wrong posture → fix posture (same label)
Case 2: Wrong action → target label 5
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

import joblib
import pandas as pd

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from load_dataset import load_dataset
from face_graph import build_face_graph, find_counterfactual
from plot_pca_face import plot_pca_face
from visualize_result import visualize_result_compare

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Case 2 mapping: target pose among remaining 11 (e.g. plank) ─────────────
CASE2_TARGET = "plank"

# ── Case 1 mapping: label → same label (identity) ───────────────────────────
# (posture correction within same class)
CASE1_MAPPING = {1: 1, 2: 2, 3: 3, 4: 4}


def load_model_artifacts(model_dir: str):
    """Load model, scaler, label_encoder, config from model_dir."""
    model_dir = Path(model_dir)
    config    = json.loads((model_dir / "config.json").read_text())

    # Support both .pkl and .pt
    if (model_dir / "model.pkl").exists():
        model = joblib.load(model_dir / "model.pkl")
    elif (model_dir / "model.pt").exists():
        arch  = json.loads((model_dir / "model_arch.json").read_text())
        from train_classifier import TorchMLP
        model = TorchMLP.load(
            model_dir / "model.pt",
            input_dim=arch["input_dim"],
            hidden=tuple(arch["hidden"]),
            n_classes=arch["n_classes"],
        )
    else:
        raise FileNotFoundError(f"No model file in {model_dir}")

    scaler = joblib.load(model_dir / "scaler.pkl")
    le     = joblib.load(model_dir / "label_encoder.pkl")

    return model, scaler, le, config


def _select_input(
    df_test    : pd.DataFrame,
    X_test_raw : np.ndarray,
    input_index: Optional[int],
    input_image_path: Optional[str],
    label_col  : str = "label",
) -> tuple:
    """Return (idx_in_test, x_raw, ground_truth_label, image_path, row)."""
    if input_index is not None:
        idx = int(input_index)
    elif input_image_path is not None:
        matches = df_test.index[df_test["image_path"] == input_image_path].tolist()
        if not matches:
            raise ValueError(f"image_path not found in test CSV: {input_image_path}")
        idx = matches[0]
    else:
        raise ValueError("Provide --input_index or --input_image_path")

    row = df_test.iloc[idx]
    x_raw = X_test_raw[idx]
    gt_label = row[label_col]
    img_path = row.get("image_path", "N/A")
    return idx, x_raw, gt_label, img_path, row


def determine_case_target(
    case        : int,
    gt_label,
    pred_label,
    pred_probs  : np.ndarray,
    le,
    target_label_override: Optional[str],
    threshold   : float,
) -> tuple:
    """
    Returns (selected_case, target_label, trigger_reason).

    Case 1 logic:
      - GT label is k
      - Model predicts k but confidence < threshold, OR sample marked incorrect
      - Target = k (same label)

    Case 2 logic:
      - Input belongs to label 1/2/3/4 or model predicts wrong
      - Target = 5 (or override)
    """
    gt_str   = str(gt_label)
    pred_str = str(pred_label)

    # Get confidence of gt label
    try:
        gt_enc      = le.transform([gt_label])[0]
        gt_conf     = float(pred_probs[gt_enc])
    except Exception:
        gt_conf = 0.0

    if case == 1:
        target = gt_label  # same label
        reason = f"Case 1: same-label posture correction (GT={gt_str}, conf={gt_conf:.3f}<{threshold})"
        return 1, target, reason

    elif case == 2:
        if target_label_override is not None:
            try:
                tgt = int(target_label_override)
                tgt = le.classes_[tgt]
            except ValueError:
                tgt = target_label_override
        else:
            tgt = CASE2_TARGET
            if isinstance(tgt, int) and len(le.classes_) > tgt:
                tgt = le.classes_[tgt]

        # Ensure that the target class is not lowPlank (select one of the other 11 classes)
        if str(tgt).lower() == "lowplank":
            for c in le.classes_:
                if c.lower() != "lowplank":
                    tgt = c
                    break

        reason = f"Case 2: wrong action → target label {tgt}"
        return 2, tgt, reason

    else:
        raise ValueError(f"Unknown case: {case}")


def print_banner(text: str):
    print(f"\n{'━'*60}")
    print(f"  {text}")
    print(f"{'━'*60}")


def run_demo(
    train_csv        : str,
    test_csv         : str,
    model_dir        : str,
    case             : int,
    output_dir       : str,
    threshold        : float = 0.90,
    input_index      : Optional[int] = None,
    input_image_path : Optional[str] = None,
    target_label_override: Optional[str] = None,
    k_graph          : int = 7,
    k_density        : int = 5,
    top_k_candidates : int = 50,
    skip_pca         : bool = False,
    skip_skeleton    : bool = False,
    run_all_test     : bool = False,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model artifacts ───────────────────────────────────────────────
    model, scaler, le, config = load_model_artifacts(model_dir)
    label_col     = config["label_col"]
    use_confidence = config["use_confidence"]
    scale_mode    = config.get("scale_mode", "torso")
    center        = config.get("center", True)

    # ── Load datasets ──────────────────────────────────────────────────────
    X_train_raw, y_train_raw, df_train = load_dataset(
        train_csv,
        use_confidence=use_confidence,
        center=center,
        scale_mode=scale_mode,
        label_col=label_col,
    )
    X_test_raw, y_test_raw, df_test = load_dataset(
        test_csv,
        use_confidence=use_confidence,
        center=center,
        scale_mode=scale_mode,
        label_col=label_col,
    )

    # ── Scale ──────────────────────────────────────────────────────────────
    X_train_s = scaler.transform(X_train_raw)
    X_test_s  = scaler.transform(X_test_raw)

    train_paths = (
        df_train["image_path"].tolist()
        if "image_path" in df_train.columns
        else [f"train_{i}" for i in range(len(df_train))]
    )

    # ── Pre-build FACE graph (shared across samples) ───────────────────────
    logger.info("Pre-building FACE graph on train set ...")
    try:
        prebuilt = build_face_graph(
            X_train_s, k=k_graph, density_k=k_density
        )
    except Exception as e:
        logger.warning(f"Could not pre-build graph: {e}")
        prebuilt = None

    # ── Select test sample(s) ─────────────────────────────────────────────
    if run_all_test:
        indices = list(range(len(df_test)))
    else:
        if input_index is not None:
            indices = [int(input_index)]
        elif input_image_path is not None:
            matches = df_test.index[
                df_test["image_path"] == input_image_path
            ].tolist()
            if not matches:
                raise ValueError(f"image_path not in test CSV: {input_image_path}")
            indices = [matches[0]]
        else:
            # Default: first test sample where model confidence < threshold
            probs_all = model.predict_proba(X_test_s)
            pred_all  = np.argmax(probs_all, axis=1)
            conf_all  = probs_all[np.arange(len(pred_all)), pred_all]
            gt_enc_all = le.transform(y_test_raw)

            if case == 1:
                mask = (pred_all == gt_enc_all) & (conf_all < threshold)
            elif case == 2:
                try:
                    lowplank_enc = le.transform(["lowPlank"])[0]
                except Exception:
                    lowplank_enc = [i for i, c in enumerate(le.classes_) if c.lower() == "lowplank"][0]
                mask = (gt_enc_all == lowplank_enc) & (pred_all == lowplank_enc)
            else:
                mask = pred_all != gt_enc_all

            cands = np.where(mask)[0]
            if len(cands) == 0:
                logger.warning(
                    "No test sample matches the trigger condition. "
                    "Using index 0."
                )
                indices = [0]
            else:
                indices = [int(cands[0])]
                logger.info(
                    f"Auto-selected test index {indices[0]} "
                    f"(conf={conf_all[indices[0]]:.3f})"
                )

    all_results = []

    for idx in indices:
        print_banner(f"Test sample index: {idx}")
        row       = df_test.iloc[idx]
        x_raw     = X_test_raw[idx]
        x_scaled  = X_test_s[idx]
        gt_label  = y_test_raw[idx]
        img_path  = row.get("image_path", "N/A") if "image_path" in row.index else "N/A"

        # ── Predict ───────────────────────────────────────────────────────
        probs     = model.predict_proba(x_scaled.reshape(1, -1))[0]
        pred_enc  = int(np.argmax(probs))
        pred_label = le.inverse_transform([pred_enc])[0]
        pred_conf  = float(probs[pred_enc])

        # GT confidence
        try:
            gt_enc  = int(le.transform([gt_label])[0])
            gt_conf = float(probs[gt_enc])
        except Exception:
            gt_enc  = pred_enc
            gt_conf = pred_conf

        # ── Determine case & target label ─────────────────────────────────
        sel_case, target_label, trigger_reason = determine_case_target(
            case, gt_label, pred_label, probs, le,
            target_label_override, threshold,
        )

        # Check trigger condition
        if case == 1:
            trigger = (str(pred_label) == str(gt_label)) and (gt_conf < threshold)
        elif case == 2:
            trigger = (str(gt_label).lower() == "lowplank") and (str(pred_label).lower() == "lowplank")
        else:
            trigger = str(pred_label) != str(gt_label)

        # ── Print input info ──────────────────────────────────────────────
        print(f"  Input image path    : {img_path}")
        print(f"  Ground-truth label  : {gt_label}")
        print(f"  Predicted label     : {pred_label}  (conf={pred_conf:.4f})")
        print(f"  GT label confidence : {gt_conf:.4f}")
        all_prob_str = "  |  ".join(
            [f"cls{le.classes_[i]}={probs[i]:.3f}" for i in range(len(probs))]
        )
        print(f"  All probabilities   : {all_prob_str}")
        print(f"  Trigger reason      : {trigger_reason}")
        print(f"  Target label        : {target_label}")
        print(f"  Trigger condition   : {'MET ✓' if trigger else 'NOT MET (running anyway)'}")

        # ── FACE ─────────────────────────────────────────────────────────
        print(f"\n  → Running FACE (threshold={threshold}) ...")
        cf_result = find_counterfactual(
            x_input=x_scaled,
            target_label=target_label,
            model=model,
            train_features=X_train_s,
            train_labels_raw=y_train_raw,
            train_paths=train_paths,
            le=le,
            threshold=threshold,
            k_graph=k_graph,
            k_density=k_density,
            top_k_candidates=top_k_candidates,
            prebuilt_graph=prebuilt,
            exclude_input_idx=None,
        )

        # ── Print FACE result ─────────────────────────────────────────────
        print(f"\n  ╔═══ FACE Result ═══╗")
        print(f"  Method              : {cf_result['method']}")
        print(f"  Counterfactual path : {cf_result['counterfactual_path']}")
        print(f"  Counterfactual label: {cf_result['counterfactual_label']}")
        print(f"  CF P(target)        : {cf_result['counterfactual_prob']:.4f}")
        if cf_result["face_path"] is not None:
            print(f"  FACE path length    : {len(cf_result['face_path'])} nodes")
            print(f"  FACE path cost      : {cf_result['face_path_cost']:.4f}")
        else:
            print(f"  FACE path           : N/A (fallback used)")
        print(f"  ╚═══════════════════╝")

        # ── Build result dict ─────────────────────────────────────────────
        result_dict = {
            "test_index"           : int(idx),
            "input_image_path"     : str(img_path),
            "ground_truth_label"   : str(gt_label),
            "predicted_label"      : str(pred_label),
            "predicted_conf"       : float(pred_conf),
            "gt_conf"              : float(gt_conf),
            "probabilities"        : {str(le.classes_[i]): float(probs[i]) for i in range(len(probs))},
            "selected_case"        : int(sel_case),
            "target_label"         : str(target_label),
            "trigger_met"          : bool(trigger),
            "threshold"            : float(threshold),
            "counterfactual_path"  : str(cf_result["counterfactual_path"]),
            "counterfactual_label" : str(cf_result["counterfactual_label"]),
            "counterfactual_prob"  : float(cf_result["counterfactual_prob"]),
            "face_path"            : cf_result["face_path"],
            "face_path_cost"       : cf_result["face_path_cost"],
            "face_path_length"     : len(cf_result["face_path"]) if cf_result["face_path"] else None,
            "method"               : cf_result["method"],
            "counterfactual_idx"   : int(cf_result["counterfactual_idx"]),
            # For visualization scripts
            "x_input_scaled"       : x_scaled.tolist(),
            "x_input_raw"          : x_raw.tolist(),
            "input_prob_target"    : float(probs[int(le.transform([target_label])[0])]) if str(target_label) in [str(c) for c in le.classes_] else 0.0,
        }

        # ── Save JSON ─────────────────────────────────────────────────────
        json_path = out_dir / "result.json"
        json_path.write_text(json.dumps(result_dict, indent=2, cls=_NumpyEncoder))
        print(f"\n  JSON saved → {json_path}")

        # ── PCA plot ──────────────────────────────────────────────────────
        if not skip_pca:
            print("  → Generating PCA plot ...")
            try:
                x_cf_scaled = X_train_s[cf_result["counterfactual_idx"]]
                # Lọc face_path chỉ lấy train indices (bỏ query node = N)
                raw_path = cf_result.get("face_path")
                N_train  = len(X_train_s)
                path_train_idxs = [i for i in (raw_path or []) if 0 <= i < N_train]
                plot_pca_face(
                    train_features    = X_train_s,
                    train_labels_raw  = y_train_raw,
                    x_input           = x_scaled,
                    x_counterfactual  = x_cf_scaled,
                    face_path_indices = path_train_idxs,
                    model             = model,
                    le                = le,
                    target_label      = target_label,
                    case_id           = sel_case,
                    output_dir        = str(out_dir),
                    input_label_str   = str(gt_label),
                    cf_label_str      = str(cf_result["counterfactual_label"]),
                    method            = cf_result["method"],
                    input_image_path  = str(img_path) if img_path != "N/A" else None,
                    cf_image_path     = str(cf_result["counterfactual_path"]),
                )
                print(f"  PCA plot saved → {out_dir}/pca_case{sel_case}.png")
            except Exception as e:
                import traceback
                logger.error(f"PCA plot failed: {e}\n{traceback.format_exc()}")

        # ── Visualization (ảnh thật + skeleton) ──────────────────────────
        if not skip_skeleton:
            print("  → Generating result visualization ...")
            try:
                cf_idx_v   = cf_result["counterfactual_idx"]
                cf_kp_raw  = X_train_raw[cf_idx_v]

                # Lấy ảnh các intermediate nodes trên FACE path
                raw_path = cf_result.get("face_path")
                N_train  = len(X_train_raw)
                path_train_idxs = [i for i in (raw_path or []) if 0 <= i < N_train]
                # Bỏ node cuối (= CF) vì CF đã được show riêng
                intermediate_idxs = path_train_idxs[:-1] if path_train_idxs else []
                path_images = []
                if "image_path" in df_train.columns:
                    path_images = [
                        df_train["image_path"].iloc[i]
                        for i in intermediate_idxs
                        if 0 <= i < len(df_train)
                    ]

                visualize_result_compare(
                    input_kp         = x_raw,
                    cf_kp            = cf_kp_raw,
                    input_label      = str(pred_label),
                    cf_label         = str(cf_result["counterfactual_label"]),
                    input_prob       = float(result_dict["input_prob_target"]),
                    cf_prob          = float(cf_result["counterfactual_prob"]),
                    case_id          = sel_case,
                    output_dir       = str(out_dir),
                    input_img_path   = str(img_path) if img_path != "N/A" else None,
                    cf_img_path      = str(cf_result["counterfactual_path"]),
                    method           = cf_result["method"],
                    face_path_images = path_images,
                    gt_label         = str(gt_label),
                )
                print(f"  Visualization saved → {out_dir}/result_visualization.png")
            except Exception as e:
                import traceback
                logger.error(f"Visualization failed: {e}\n{traceback.format_exc()}")

        all_results.append(result_dict)

    # ── Summary ────────────────────────────────────────────────────────────
    print_banner("Done")
    print(f"  Output directory: {out_dir}")
    print(f"  Samples processed: {len(all_results)}")
    if all_results:
        methods = [r["method"] for r in all_results]
        face_n    = methods.count("FACE shortest path")
        fallback_n = len(methods) - face_n
        print(f"  FACE paths found   : {face_n}")
        print(f"  Fallback used      : {fallback_n}")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CORE-Yoga FACE demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train_csv",        required=True)
    parser.add_argument("--test_csv",         required=True)
    parser.add_argument("--model_dir",        required=True)
    parser.add_argument("--case",             type=int, choices=[1, 2], required=True)
    parser.add_argument("--output_dir",       default="outputs")
    parser.add_argument("--threshold",        type=float, default=0.90)
    parser.add_argument("--input_index",      type=int,   default=None)
    parser.add_argument("--input_image_path", type=str,   default=None)
    parser.add_argument("--target_label",     type=str,   default=None,
                        help="Override target label (Case 2 default=5)")
    parser.add_argument("--k_graph",          type=int,   default=7)
    parser.add_argument("--k_density",        type=int,   default=5)
    parser.add_argument("--top_k_candidates", type=int,   default=50)
    parser.add_argument("--skip_pca",         action="store_true")
    parser.add_argument("--skip_skeleton",    action="store_true")
    parser.add_argument("--run_all_test",     action="store_true",
                        help="Run FACE for every test sample")
    args = parser.parse_args()

    run_demo(
        train_csv            =args.train_csv,
        test_csv             =args.test_csv,
        model_dir            =args.model_dir,
        case                 =args.case,
        output_dir           =args.output_dir,
        threshold            =args.threshold,
        input_index          =args.input_index,
        input_image_path     =args.input_image_path,
        target_label_override=args.target_label,
        k_graph              =args.k_graph,
        k_density            =args.k_density,
        top_k_candidates     =args.top_k_candidates,
        skip_pca             =args.skip_pca,
        skip_skeleton        =args.skip_skeleton,
        run_all_test         =args.run_all_test,
    )
