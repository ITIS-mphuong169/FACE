#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.pipeline import Pipeline


COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the two yoga label-target tests: same-label corrections 1-1..4-4 "
            "and source-to-target corrections 1-5..4-5 using raw, non-normalized pose features."
        )
    )
    parser.add_argument("--input-csv", default="poses_yoga_multiclass.csv", help="Old/full multiclass pose CSV.")
    parser.add_argument("--output-dir", default="outputs_label5_test_plan", help="Output directory.")
    parser.add_argument(
        "--target-label",
        default=None,
        help="Target label name or id. Default: last label by label_id/class order.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Stratified test ratio.")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--max-source-labels", type=int, default=4, help="How many non-target labels to test.")
    parser.add_argument("--face-k-neighbors", type=int, default=20, help="kNN edges per node in the FACE graph.")
    parser.add_argument("--kde-bandwidth", type=float, default=0.25, help="KDE bandwidth for density-aware FACE edge costs.")
    parser.add_argument("--density-weight", type=float, default=0.35, help="How strongly low-density edges are penalized.")
    parser.add_argument(
        "--dataset-root",
        default=".",
        help="Root used to resolve relative image_path values for the image-pair renderer.",
    )
    parser.add_argument("--skip-render", action="store_true", help="Only write CSV/PCA outputs.")
    parser.add_argument(
        "--all-targets",
        action="store_true",
        help=(
            "Chạy lần lượt mỗi nhãn làm target (4 nhãn còn lại → gợi ý sang target), "
            "cùng một tách train/test và cùng một model multiclass (không chuẩn hóa). "
            "Kết quả mỗi target trong thư mục con target_<tên>/; bảng tổng hợp ở gốc output-dir."
        ),
    )
    return parser.parse_args()


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if str(c).startswith("f") and str(c)[1:].isdigit()]
    return sorted(cols, key=lambda c: int(c[1:]))


def ordered_labels(df: pd.DataFrame) -> list[str]:
    names = df["label_name"].astype(str).str.lower()
    if "label_id" in df.columns:
        tmp = pd.DataFrame({"label_name": names, "label_id": df["label_id"]})
        tmp = tmp.drop_duplicates("label_name").sort_values(["label_id", "label_name"])
        return tmp["label_name"].tolist()
    return sorted(names.unique().tolist())


def label_ids(df: pd.DataFrame, labels: list[str]) -> dict[str, int]:
    if "label_id" not in df.columns:
        return {label: idx for idx, label in enumerate(labels)}
    tmp = df[["label_name", "label_id"]].copy()
    tmp["label_name"] = tmp["label_name"].astype(str).str.lower()
    return {
        str(row["label_name"]): int(row["label_id"])
        for _, row in tmp.drop_duplicates("label_name").iterrows()
    }


def resolve_label_arg(value: str | None, labels: list[str], id_by_label: dict[str, int], default_label: str) -> str:
    if value is None:
        return default_label
    needle = str(value).strip().lower()
    if needle in labels:
        return needle
    if needle.lstrip("-").isdigit():
        numeric = int(needle)
        for label, label_id in id_by_label.items():
            if label_id == numeric:
                return label
        one_based_idx = numeric - 1
        if 0 <= one_based_idx < len(labels):
            return labels[one_based_idx]
    raise ValueError(f"Label '{value}' not found. Available: {labels}; ids: {id_by_label}")


def resolve_image_path(image_path: str, dataset_root: Path) -> str:
    p = Path(image_path)
    if p.is_absolute():
        return str(p)
    return str((dataset_root / p).resolve())


def predict_frame(
    df: pd.DataFrame,
    probs: np.ndarray,
    classes: list[str],
    threshold: float,
) -> pd.DataFrame:
    out = df.copy()
    y_true = out["label_name"].astype(str).str.lower().to_numpy()
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    true_idx = np.array([class_to_idx[x] for x in y_true])
    pred_idx = probs.argmax(axis=1)
    out["y_true_label"] = y_true
    out["y_pred_label"] = np.array(classes, dtype=object)[pred_idx]
    out["prob_true_label"] = probs[np.arange(len(out)), true_idx]
    out["prob_pred_label"] = probs.max(axis=1)
    out["is_top1_correct"] = out["y_true_label"] == out["y_pred_label"]
    out["is_correct"] = out["prob_true_label"] >= threshold
    out["probabilities"] = [json.dumps({cls: float(row[i]) for i, cls in enumerate(classes)}) for row in probs]
    return out


def class_prob(row: pd.Series, label: str) -> float:
    return float(json.loads(str(row["probabilities"]))[label])


def nearest_raw_l2(
    x: np.ndarray,
    pool_df: pd.DataFrame,
    pool_X: np.ndarray,
) -> tuple[pd.Series, float]:
    distances = np.linalg.norm(pool_X - x[None, :], axis=1)
    idx = int(np.argmin(distances))
    return pool_df.iloc[idx], float(distances[idx])


def face_edge_cost(
    a: np.ndarray,
    b: np.ndarray,
    kde: KernelDensity,
    density_reference: float,
    density_weight: float,
) -> tuple[float, float, float]:
    l2 = float(np.linalg.norm(a - b))
    mid = ((a + b) / 2.0).reshape(1, -1)
    mid_log_density = float(kde.score_samples(mid)[0])
    penalty = float(np.exp(np.clip(density_weight * (density_reference - mid_log_density), -4.0, 4.0)))
    return l2 * penalty, l2, mid_log_density


def build_face_adjacency(
    train_X: np.ndarray,
    x_source: np.ndarray,
    kde: KernelDensity,
    density_reference: float,
    density_weight: float,
    k_neighbors: int,
) -> tuple[list[list[tuple[int, float]]], int]:
    graph_X = np.vstack([train_X, x_source.reshape(1, -1)])
    source_idx = len(graph_X) - 1
    n_neighbors = min(max(2, k_neighbors + 1), len(graph_X))
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    neighbors.fit(graph_X)
    _, indices = neighbors.kneighbors(graph_X)

    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(len(graph_X))]
    seen_edges: set[tuple[int, int]] = set()
    for i, neighs in enumerate(indices):
        for j_raw in neighs:
            j = int(j_raw)
            if i == j:
                continue
            a, b = sorted((i, j))
            if (a, b) in seen_edges:
                continue
            seen_edges.add((a, b))
            cost, _, _ = face_edge_cost(graph_X[a], graph_X[b], kde, density_reference, density_weight)
            adjacency[a].append((b, cost))
            adjacency[b].append((a, cost))
    return adjacency, source_idx


def shortest_path_to_candidates(
    adjacency: list[list[tuple[int, float]]],
    source_idx: int,
    candidate_indices: set[int],
) -> tuple[int | None, float, list[int]]:
    distances = [float("inf")] * len(adjacency)
    previous: list[int | None] = [None] * len(adjacency)
    distances[source_idx] = 0.0
    heap: list[tuple[float, int]] = [(0.0, source_idx)]

    while heap:
        current_dist, node = heapq.heappop(heap)
        if current_dist > distances[node]:
            continue
        if node in candidate_indices:
            path: list[int] = []
            cursor: int | None = node
            while cursor is not None:
                path.append(cursor)
                cursor = previous[cursor]
            path.reverse()
            return node, current_dist, path
        for neighbor, edge_cost in adjacency[node]:
            new_dist = current_dist + edge_cost
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous[neighbor] = node
                heapq.heappush(heap, (new_dist, neighbor))
    return None, float("inf"), []


def select_face_counterfactual(
    wrong: pd.Series,
    candidate_df: pd.DataFrame,
    train_pred_df: pd.DataFrame,
    train_X: np.ndarray,
    feat_cols: list[str],
    median: np.ndarray,
    kde: KernelDensity,
    density_reference: float,
    density_weight: float,
    k_neighbors: int,
) -> dict[str, Any]:
    if candidate_df.empty:
        raise RuntimeError("FACE candidate pool is empty.")

    x_wrong = wrong[feat_cols].to_numpy(dtype=np.float32)
    x_wrong = np.where(np.isnan(x_wrong), median, x_wrong)
    candidate_indices = {int(i) for i in candidate_df.index.to_numpy()}
    adjacency, source_idx = build_face_adjacency(
        train_X=train_X,
        x_source=x_wrong,
        kde=kde,
        density_reference=density_reference,
        density_weight=density_weight,
        k_neighbors=k_neighbors,
    )
    target_idx, path_cost, graph_path = shortest_path_to_candidates(adjacency, source_idx, candidate_indices)
    used_fallback = False

    if target_idx is None:
        candidate_order = sorted(candidate_indices)
        candidate_X = train_X[candidate_order]
        candidate_df_ordered = train_pred_df.iloc[candidate_order]
        suggested, direct_l2 = nearest_raw_l2(x_wrong, candidate_df_ordered, candidate_X)
        target_idx = int(suggested.name)
        graph_path = [source_idx, target_idx]
        path_cost = direct_l2
        used_fallback = True

    suggested = train_pred_df.iloc[int(target_idx)]
    direct_cost, direct_l2, direct_mid_log_density = face_edge_cost(
        x_wrong,
        train_X[int(target_idx)],
        kde,
        density_reference,
        density_weight,
    )
    path_train_indices = [int(i) for i in graph_path if int(i) < len(train_pred_df)]
    path_image_paths = [str(wrong["image_path"])] + [
        str(train_pred_df.iloc[i]["image_path"]) for i in path_train_indices
    ]
    path_labels = [str(wrong["y_true_label"])] + [
        str(train_pred_df.iloc[i]["y_true_label"]) for i in path_train_indices
    ]
    return {
        "suggested": suggested,
        "face_path_cost": float(path_cost),
        "distance_l2": float(direct_l2),
        "direct_face_cost": float(direct_cost),
        "kde_mid_log_density": float(direct_mid_log_density),
        "face_path_train_indices": path_train_indices,
        "face_path_image_paths": path_image_paths,
        "face_path_labels": path_labels,
        "face_path_length": max(0, len(graph_path) - 1),
        "face_graph_k": int(k_neighbors),
        "face_used_fallback": bool(used_fallback),
    }


def choose_wrong_sample(pred_df: pd.DataFrame, source_label: str, threshold: float) -> pd.Series | None:
    src = pred_df[pred_df["y_true_label"].astype(str).str.lower() == source_label].copy()
    if src.empty:
        return None
    low_conf = src[src["prob_true_label"].astype(float) < threshold].copy()
    if not low_conf.empty:
        return low_conf.sort_values("prob_true_label").iloc[0]
    return src.sort_values("prob_true_label").iloc[0]


def candidate_pool(
    train_pred_df: pd.DataFrame,
    label: str,
    threshold: float,
) -> pd.DataFrame:
    pool = train_pred_df[train_pred_df["y_true_label"].astype(str).str.lower() == label].copy()
    strong = pool[
        (pool["y_pred_label"].astype(str).str.lower() == label)
        & (pool["prob_true_label"].astype(float) >= threshold)
    ].copy()
    if not strong.empty:
        return strong
    weak = pool[pool["y_pred_label"].astype(str).str.lower() == label].copy()
    if not weak.empty:
        return weak
    return pool


def build_test_rows(
    source_labels: list[str],
    target_label: str,
    train_pred_df: pd.DataFrame,
    test_pred_df: pd.DataFrame,
    train_X: np.ndarray,
    feat_cols: list[str],
    median: np.ndarray,
    dataset_root: Path,
    threshold: float,
    kde: KernelDensity,
    density_reference: float,
    density_weight: float,
    k_neighbors: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    same_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []

    target_pool = candidate_pool(train_pred_df, target_label, threshold)
    if target_pool.empty:
        raise RuntimeError(f"No candidate train samples found for target label: {target_label}")

    for one_based_idx, source_label in enumerate(source_labels, start=1):
        wrong = choose_wrong_sample(test_pred_df, source_label, threshold)
        if wrong is None:
            continue

        same_pool = candidate_pool(train_pred_df, source_label, threshold)
        if not same_pool.empty:
            same_result = select_face_counterfactual(
                wrong=wrong,
                candidate_df=same_pool,
                train_pred_df=train_pred_df,
                train_X=train_X,
                feat_cols=feat_cols,
                median=median,
                kde=kde,
                density_reference=density_reference,
                density_weight=density_weight,
                k_neighbors=k_neighbors,
            )
            suggested = same_result["suggested"]
            same_rows.append(
                make_row(
                    display_label=f"{one_based_idx}-{one_based_idx}",
                    wrong=wrong,
                    suggested=suggested,
                    source_label=source_label,
                    target_label=source_label,
                    face_result=same_result,
                    prob_before=float(wrong["prob_true_label"]),
                    prob_after=class_prob(suggested, source_label),
                    dataset_root=dataset_root,
                    case_type="same_label_correction",
                    threshold=threshold,
                )
            )

        cross_result = select_face_counterfactual(
            wrong=wrong,
            candidate_df=target_pool,
            train_pred_df=train_pred_df,
            train_X=train_X,
            feat_cols=feat_cols,
            median=median,
            kde=kde,
            density_reference=density_reference,
            density_weight=density_weight,
            k_neighbors=k_neighbors,
        )
        suggested = cross_result["suggested"]
        cross_rows.append(
            make_row(
                display_label=f"{one_based_idx}-{len(source_labels) + 1}",
                wrong=wrong,
                suggested=suggested,
                source_label=source_label,
                target_label=target_label,
                face_result=cross_result,
                prob_before=class_prob(wrong, target_label),
                prob_after=class_prob(suggested, target_label),
                dataset_root=dataset_root,
                case_type="source_to_target_correction",
                threshold=threshold,
            )
        )

    return pd.DataFrame(same_rows), pd.DataFrame(cross_rows)


def make_row(
    display_label: str,
    wrong: pd.Series,
    suggested: pd.Series,
    source_label: str,
    target_label: str,
    face_result: dict[str, Any],
    prob_before: float,
    prob_after: float,
    dataset_root: Path,
    case_type: str,
    threshold: float,
) -> dict[str, Any]:
    return {
        "label_name": display_label,
        "case_type": case_type,
        "source_label": source_label,
        "target_label": target_label,
        "wrong_image_path": wrong["image_path"],
        "wrong_image_abspath": resolve_image_path(str(wrong["image_path"]), dataset_root),
        "wrong_true_label": wrong["y_true_label"],
        "wrong_pred_label": wrong["y_pred_label"],
        "suggested_image_path": suggested["image_path"],
        "suggested_image_abspath": resolve_image_path(str(suggested["image_path"]), dataset_root),
        "suggested_label": target_label,
        "distance": face_result["face_path_cost"],
        "distance_l2": face_result["distance_l2"],
        "direct_face_cost": face_result["direct_face_cost"],
        "kde_mid_log_density": face_result["kde_mid_log_density"],
        "face_path_cost": face_result["face_path_cost"],
        "face_path_length": face_result["face_path_length"],
        "face_graph_k": face_result["face_graph_k"],
        "face_used_fallback": face_result["face_used_fallback"],
        "face_path_train_indices": json.dumps(face_result["face_path_train_indices"]),
        "face_path_image_paths": json.dumps(face_result["face_path_image_paths"]),
        "face_path_labels": json.dumps(face_result["face_path_labels"]),
        "prob_before": prob_before,
        "prob_after": prob_after,
        "is_wrong_by_threshold": bool(float(wrong["prob_true_label"]) < threshold),
    }


def make_raw_pca_plot(
    train_df: pd.DataFrame,
    test_pred_df: pd.DataFrame,
    same_df: pd.DataFrame,
    cross_df: pd.DataFrame,
    feat_cols: list[str],
    median: np.ndarray,
    labels: list[str],
    target_label: str,
    output_path: Path,
) -> None:
    X_train = train_df[feat_cols].to_numpy(dtype=np.float32)
    X_train = np.where(np.isnan(X_train), median, X_train)
    pca = PCA(n_components=2, random_state=42)
    train2 = pca.fit_transform(X_train)

    lookup_df = pd.concat([train_df, test_pred_df], ignore_index=True).drop_duplicates("image_path")
    lookup_X = lookup_df[feat_cols].to_numpy(dtype=np.float32)
    lookup_X = np.where(np.isnan(lookup_X), median, lookup_X)
    lookup2 = pca.transform(lookup_X)
    xy_by_path = {path: xy for path, xy in zip(lookup_df["image_path"].astype(str), lookup2, strict=False)}

    fig, ax = plt.subplots(figsize=(11, 8))
    for idx, label in enumerate(labels):
        mask = train_df["label_name"].astype(str).str.lower() == label
        is_target = label == target_label
        ax.scatter(
            train2[mask.to_numpy(), 0],
            train2[mask.to_numpy(), 1],
            s=24 if is_target else 16,
            c=COLORS[idx % len(COLORS)],
            alpha=0.72 if is_target else 0.45,
            edgecolors="none",
            label=f"{idx + 1}: {label}",
        )

    for rows, style, prefix in [(same_df, "-", "H1"), (cross_df, "--", "H2")]:
        for _, row in rows.iterrows():
            wrong_xy = xy_by_path.get(str(row["wrong_image_path"]))
            suggested_xy = xy_by_path.get(str(row["suggested_image_path"]))
            if wrong_xy is None or suggested_xy is None:
                continue
            path_points: list[np.ndarray] = []
            if "face_path_image_paths" in row and pd.notna(row["face_path_image_paths"]):
                try:
                    for image_path in json.loads(str(row["face_path_image_paths"])):
                        xy = xy_by_path.get(str(image_path))
                        if xy is not None:
                            path_points.append(xy)
                except json.JSONDecodeError:
                    path_points = []
            if len(path_points) < 2:
                path_points = [wrong_xy, suggested_xy]
            path_arr = np.asarray(path_points, dtype=float)
            ax.plot(path_arr[:, 0], path_arr[:, 1], color="black", lw=1.4, linestyle=style, alpha=0.78, zorder=4)
            ax.scatter(wrong_xy[0], wrong_xy[1], s=130, facecolors="#fff2a8", edgecolors="black", zorder=5)
            ax.scatter(suggested_xy[0], suggested_xy[1], s=130, facecolors="#c7ff9b", edgecolors="black", zorder=5)
            ax.annotate(
                "",
                xy=(path_arr[-1, 0], path_arr[-1, 1]),
                xytext=(path_arr[-2, 0], path_arr[-2, 1]),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.6, linestyle=style),
            )
            ax.text(wrong_xy[0], wrong_xy[1], f"{prefix} {row['label_name']}", fontsize=8, weight="bold")

    ax.set_title("PCA raw features: same-label and source-to-target corrections")
    ax.set_xlabel("PCA component 1 (raw features, no scaling)")
    ax.set_ylabel("PCA component 2 (raw features, no scaling)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def render_pairs(cf_csv: Path, output_dir: Path, train_csv: Path, test_csv: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "render_counterfactual_image_pairs.py",
            "--counterfactual-csv",
            str(cf_csv),
            "--output-dir",
            str(output_dir),
            "--features-csv",
            str(train_csv),
            str(test_csv),
            "--max-pairs",
            "4",
            "--no-skeleton",
            "--skeleton-source",
            "csv",
        ],
        check=True,
    )


def run_one_target_plan(
    *,
    target_label: str,
    output_dir: Path,
    labels: list[str],
    id_by_label: dict[str, int],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_csv: Path,
    test_csv: Path,
    train_pred_df: pd.DataFrame,
    test_pred_df: pd.DataFrame,
    train_X_filled: np.ndarray,
    feat_cols: list[str],
    median: np.ndarray,
    kde: KernelDensity,
    density_reference: float,
    dataset_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_labels = [label for label in labels if label != target_label][: args.max_source_labels]
    if not source_labels:
        raise ValueError("At least one source label is required.")

    same_df, cross_df = build_test_rows(
        source_labels=source_labels,
        target_label=target_label,
        train_pred_df=train_pred_df,
        test_pred_df=test_pred_df,
        train_X=train_X_filled,
        feat_cols=feat_cols,
        median=median,
        dataset_root=dataset_root,
        threshold=args.confidence_threshold,
        kde=kde,
        density_reference=density_reference,
        density_weight=args.density_weight,
        k_neighbors=args.face_k_neighbors,
    )
    same_csv = output_dir / "same_label_1-1_to_4-4.csv"
    cross_csv = output_dir / "source_to_target_1-5_to_4-5.csv"
    same_df.to_csv(same_csv, index=False)
    cross_df.to_csv(cross_csv, index=False)

    make_raw_pca_plot(
        train_df=train_df,
        test_pred_df=test_pred_df,
        same_df=same_df,
        cross_df=cross_df,
        feat_cols=feat_cols,
        median=median,
        labels=labels,
        target_label=target_label,
        output_path=output_dir / "pca_raw_no_normalize.png",
    )

    if not args.skip_render:
        render_pairs(same_csv, output_dir / "ht1_same_label_images", train_csv, test_csv)
        render_pairs(cross_csv, output_dir / "ht2_source_to_target_images", train_csv, test_csv)

    return {
        "target_label": target_label,
        "target_label_id": id_by_label[target_label],
        "source_labels": ",".join(source_labels),
        "n_same_rows": int(len(same_df)),
        "n_cross_rows": int(len(cross_df)),
        "same_csv": str(same_csv),
        "cross_csv": str(cross_csv),
        "pca_png": str(output_dir / "pca_raw_no_normalize.png"),
    }


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    feat_cols = feature_columns(df)
    labels = ordered_labels(df)
    id_by_label = label_ids(df, labels)

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=df["label_name"].astype(str).str.lower(),
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    if args.all_targets:
        train_csv = output_dir / "shared_label_target_train.csv"
        test_csv = output_dir / "shared_label_target_test.csv"
    else:
        train_csv = output_dir / "label_target_train.csv"
        test_csv = output_dir / "label_target_test.csv"
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    X_train = train_df[feat_cols].to_numpy(dtype=np.float32)
    X_test = test_df[feat_cols].to_numpy(dtype=np.float32)
    classes = labels
    y_train = train_df["label_name"].astype(str).str.lower().map({c: i for i, c in enumerate(classes)}).to_numpy()
    y_test = test_df["label_name"].astype(str).str.lower().map({c: i for i, c in enumerate(classes)}).to_numpy()

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(n_estimators=600, class_weight="balanced", random_state=args.random_seed, n_jobs=-1)),
        ]
    )
    model.fit(X_train, y_train)
    train_probs = model.predict_proba(X_train)
    test_probs = model.predict_proba(X_test)
    pred = test_probs.argmax(axis=1)
    true_probs = test_probs[np.arange(len(y_test)), y_test]

    metrics = {
        "model_name": "extra_trees_no_scaler",
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "threshold_success_rate": float((true_probs >= args.confidence_threshold).mean()),
        "all_targets": bool(args.all_targets),
        "normalized": False,
        "face_k_neighbors": args.face_k_neighbors,
        "kde_bandwidth": args.kde_bandwidth,
        "density_weight": args.density_weight,
    }
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics_no_normalize.csv", index=False)

    train_pred_df = predict_frame(train_df, train_probs, classes, args.confidence_threshold)
    test_pred_df = predict_frame(test_df, test_probs, classes, args.confidence_threshold)
    train_pred_df.to_csv(output_dir / "train_predictions_no_normalize.csv", index=False)
    test_pred_df.to_csv(output_dir / "test_predictions_no_normalize.csv", index=False)

    median = np.nanmedian(X_train, axis=0)
    train_X_filled = np.where(np.isnan(X_train), median, X_train)
    kde = KernelDensity(kernel="gaussian", bandwidth=args.kde_bandwidth)
    kde.fit(train_X_filled)
    density_reference = float(np.median(kde.score_samples(train_X_filled)))

    if args.all_targets:
        summary_rows: list[dict[str, Any]] = []
        for target_label in labels:
            sub = output_dir / f"target_{target_label}"
            row = run_one_target_plan(
                target_label=target_label,
                output_dir=sub,
                labels=labels,
                id_by_label=id_by_label,
                train_df=train_df,
                test_df=test_df,
                train_csv=train_csv,
                test_csv=test_csv,
                train_pred_df=train_pred_df,
                test_pred_df=test_pred_df,
                train_X_filled=train_X_filled,
                feat_cols=feat_cols,
                median=median,
                kde=kde,
                density_reference=density_reference,
                dataset_root=dataset_root,
                args=args,
            )
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(output_dir / "summary_all_targets.csv", index=False)
        print(f"Saved output dir: {output_dir}")
        print(f"Labels: {', '.join(f'{idx + 1}={label}' for idx, label in enumerate(labels))}")
        print(f"Ran {len(labels)} target folders: {[str(output_dir / f'target_{t}') for t in labels]}")
        print(f"Summary: {output_dir / 'summary_all_targets.csv'}")
        print(f"Shared split: {train_csv} | {test_csv}")
        return

    target_label = resolve_label_arg(args.target_label, labels, id_by_label, labels[-1])
    source_labels = [label for label in labels if label != target_label][: args.max_source_labels]
    metrics_one = {
        **metrics,
        "source_labels": ",".join(source_labels),
        "source_label_ids": ",".join(str(id_by_label[label]) for label in source_labels),
        "target_label": target_label,
        "target_label_id": id_by_label[target_label],
    }
    pd.DataFrame([metrics_one]).to_csv(output_dir / "metrics_no_normalize.csv", index=False)

    run_one_target_plan(
        target_label=target_label,
        output_dir=output_dir,
        labels=labels,
        id_by_label=id_by_label,
        train_df=train_df,
        test_df=test_df,
        train_csv=train_csv,
        test_csv=test_csv,
        train_pred_df=train_pred_df,
        test_pred_df=test_pred_df,
        train_X_filled=train_X_filled,
        feat_cols=feat_cols,
        median=median,
        kde=kde,
        density_reference=density_reference,
        dataset_root=dataset_root,
        args=args,
    )

    same_csv = output_dir / "same_label_1-1_to_4-4.csv"
    cross_csv = output_dir / "source_to_target_1-5_to_4-5.csv"
    print(f"Saved output dir: {output_dir}")
    print(f"Labels: {', '.join(f'{idx + 1}={label}' for idx, label in enumerate(labels))}")
    print(f"Target label: {len(source_labels) + 1}={target_label}")
    print(f"Same-label CSV: {same_csv}")
    print(f"Cross-label CSV: {cross_csv}")
    print(f"PCA: {output_dir / 'pca_raw_no_normalize.png'}")


if __name__ == "__main__":
    main()
