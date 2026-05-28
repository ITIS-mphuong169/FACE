"""
face_graph.py
-------------
FACE (Feasible and Actionable Counterfactual Explanations) graph implementation.

Key design:
  - Nodes  = train samples (real data points, preserving image_path)
  - Edges  = kNN weighted by  dist(u,v) / (density(u,v) + eps)
  - Density via KernelDensity (fallback: inverse mean kNN distance)
  - Shortest path via networkx Dijkstra
  - Always returns a real sample → image_path guaranteed

Public API:
    build_face_graph(...)
    find_counterfactual(...)
"""

import logging
import warnings
from typing import Optional, Tuple, List, Dict, Any

import networkx as nx
import numpy as np
from sklearn.neighbors import KernelDensity, NearestNeighbors

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Density estimation
# ---------------------------------------------------------------------------
def _estimate_density_kde(
    features: np.ndarray,
    bandwidth: float = 0.5,
) -> np.ndarray:
    """Fit KDE on train features, return log-density at each point."""
    try:
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        kde.fit(features)
        log_dens = kde.score_samples(features)
        dens = np.exp(log_dens)
        return dens
    except Exception as e:
        logger.warning(f"KDE failed ({e}). Falling back to kNN density.")
        return None


def _estimate_density_knn(
    features: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """kNN density: 1 / mean_distance_to_k_nearest_neighbours."""
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree")
    nn.fit(features)
    dists, _ = nn.kneighbors(features)
    mean_dist = dists[:, 1:].mean(axis=1)  # exclude self (index 0)
    eps = 1e-8
    return 1.0 / (mean_dist + eps)


def estimate_density(
    features: np.ndarray,
    k: int = 5,
    bandwidth: float = 0.5,
) -> np.ndarray:
    dens = _estimate_density_kde(features, bandwidth=bandwidth)
    if dens is None:
        dens = _estimate_density_knn(features, k=k)
    return dens


# ---------------------------------------------------------------------------
# Build kNN graph
# ---------------------------------------------------------------------------
def build_face_graph(
    features: np.ndarray,
    k: int = 7,
    density_bandwidth: float = 0.5,
    density_k: int = 5,
    epsilon: float = 1e-8,
) -> Tuple[nx.Graph, np.ndarray]:
    """
    Build weighted kNN graph for FACE.

    Edge weight = dist(u,v) / (density_avg(u,v) + epsilon)

    Returns
    -------
    G     : networkx.Graph  (nodes 0..N-1)
    dens  : (N,) density array
    """
    N = features.shape[0]
    logger.info(f"Building FACE graph: {N} nodes, k={k} ...")

    dens = estimate_density(features, k=density_k, bandwidth=density_bandwidth)

    # kNN
    k_eff = min(k + 1, N)
    nn = NearestNeighbors(n_neighbors=k_eff, algorithm="ball_tree")
    nn.fit(features)
    dists, indices = nn.kneighbors(features)

    G = nx.Graph()
    G.add_nodes_from(range(N))

    for u in range(N):
        for j in range(1, k_eff):  # skip self at index 0
            v = indices[u, j]
            dist_uv = dists[u, j]
            dens_avg = (dens[u] + dens[v]) / 2.0
            weight = dist_uv / (dens_avg + epsilon)
            if G.has_edge(u, v):
                # keep minimum weight edge
                if weight < G[u][v]["weight"]:
                    G[u][v]["weight"] = weight
            else:
                G.add_edge(u, v, weight=weight)

    logger.info(
        f"Graph built: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )
    return G, dens


# ---------------------------------------------------------------------------
# Find valid counterfactual candidates
# ---------------------------------------------------------------------------
def _get_candidates(
    train_labels: np.ndarray,
    train_probs: np.ndarray,
    target_label_enc: int,
    threshold: float,
    exclude_idx: Optional[int] = None,
) -> np.ndarray:
    """
    Return indices of train samples that are valid counterfactual candidates:
      - label == target_label_enc (encoded integer)
      - model probability of target_label >= threshold
    """
    label_mask = train_labels == target_label_enc
    prob_mask  = train_probs[:, target_label_enc] >= threshold
    mask = label_mask & prob_mask

    if exclude_idx is not None:
        mask[exclude_idx] = False

    candidates = np.where(mask)[0]
    return candidates


# ---------------------------------------------------------------------------
# Shortest FACE path via networkx
# ---------------------------------------------------------------------------
def _face_shortest_path(
    G: nx.Graph,
    source_node: int,
    candidate_nodes: np.ndarray,
) -> Tuple[Optional[List[int]], float]:
    """
    Find the shortest-cost path from source_node to any node in candidate_nodes.

    Returns (path, cost) or (None, inf) if not reachable.
    """
    best_path = None
    best_cost = float("inf")

    for target_node in candidate_nodes:
        try:
            path = nx.shortest_path(
                G, source=source_node, target=target_node, weight="weight"
            )
            cost = nx.path_weight(G, path, weight="weight")
            if cost < best_cost:
                best_cost = cost
                best_path = path
        except nx.NetworkXNoPath:
            continue
        except nx.NodeNotFound:
            continue

    return best_path, best_cost


# ---------------------------------------------------------------------------
# Add query point to graph temporarily
# ---------------------------------------------------------------------------
def _add_query_node(
    G: nx.Graph,
    features: np.ndarray,
    x_query: np.ndarray,
    dens: np.ndarray,
    k: int = 7,
    epsilon: float = 1e-8,
    candidate_nodes: Optional[np.ndarray] = None,
) -> int:
    """
    Add x_query as a temporary node to G (index = len(features)).

    Connect to k*2 nearest train neighbours via proper kNN edges.
    Dijkstra will find the true shortest feasible path through the graph.

    If no path found through graph, we then add direct edges to top candidates
    as a last resort (fallback is handled in find_counterfactual).

    Returns the new node index.
    """
    N = len(features)
    query_node = N

    dists = np.linalg.norm(features - x_query, axis=1)

    # Estimate density at query point via kNN
    k_dens = min(k + 1, N)
    nn_idxs_dens = np.argsort(dists)[:k_dens]
    mean_d = dists[nn_idxs_dens[1:]].mean() if len(nn_idxs_dens) > 1 else dists[nn_idxs_dens[0]]
    query_dens = 1.0 / (mean_d + epsilon)

    # Connect to k*2 nearest neighbors — enough for connectivity without shortcuts
    k_connect = min(k * 2, N)
    nn_idxs = np.argsort(dists)[:k_connect]

    G.add_node(query_node)
    for v in nn_idxs:
        dist_uv  = float(dists[v])
        dens_avg = (query_dens + dens[v]) / 2.0
        weight   = dist_uv / (dens_avg + epsilon)
        if not G.has_edge(query_node, v):
            G.add_edge(query_node, v, weight=weight)

    return query_node


# ---------------------------------------------------------------------------
# Fallback: nearest feasible counterfactual
# ---------------------------------------------------------------------------
def _fallback_nearest(
    x_query: np.ndarray,
    train_features: np.ndarray,
    candidates: np.ndarray,
    train_paths: List[str],
    train_labels_enc: np.ndarray,
    train_probs: np.ndarray,
    target_label_enc: int,
    le,
) -> Dict[str, Any]:
    dists = np.linalg.norm(train_features[candidates] - x_query, axis=1)
    best  = candidates[np.argmin(dists)]

    return {
        "counterfactual_idx"    : int(best),
        "counterfactual_feature": train_features[best],
        "counterfactual_path"   : train_paths[best],
        "counterfactual_prob"   : float(train_probs[best, target_label_enc]),
        "counterfactual_label"  : le.inverse_transform([train_labels_enc[best]])[0],
        "face_path"             : None,
        "face_path_cost"        : None,
        "method"                : "fallback nearest feasible",
    }


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------
def find_counterfactual(
    x_input: np.ndarray,
    target_label,                      # original label (before encoding)
    model,                             # trained classifier with predict_proba
    train_features: np.ndarray,
    train_labels_raw: np.ndarray,      # original labels (before encoding)
    train_paths: List[str],
    le,                                # LabelEncoder
    threshold: float = 0.90,
    k_graph: int = 7,
    k_density: int = 5,
    density_bandwidth: float = 0.5,
    top_k_candidates: int = 50,
    prebuilt_graph: Optional[Tuple[nx.Graph, np.ndarray]] = None,
    exclude_input_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Find the most feasible counterfactual explanation using FACE.

    Parameters
    ----------
    x_input          : (n_features,) query feature vector (scaled)
    target_label     : desired class label (original, not encoded)
    model            : classifier with .predict_proba()
    train_features   : (N, n_features) scaled train features
    train_labels_raw : (N,) original labels
    train_paths      : (N,) image paths
    le               : fitted LabelEncoder
    threshold        : minimum P(target_label) for counterfactual
    prebuilt_graph   : (G, dens) if already computed, to avoid rebuilding
    exclude_input_idx: index in train set if x_input is a train sample

    Returns
    -------
    dict with keys:
        counterfactual_idx, counterfactual_feature, counterfactual_path,
        counterfactual_prob, counterfactual_label,
        face_path (list of node indices or None),
        face_path_cost, method
    """
    # ── Encode target label ────────────────────────────────────────────────
    try:
        target_label_enc = int(le.transform([target_label])[0])
    except Exception:
        raise ValueError(f"target_label '{target_label}' not in LabelEncoder classes: {le.classes_}")

    # ── Get model probabilities on train set ───────────────────────────────
    logger.info("Computing model probabilities on train set ...")
    train_probs = model.predict_proba(train_features)   # (N, n_classes)
    train_labels_enc = le.transform(train_labels_raw)

    # ── Find valid candidates ──────────────────────────────────────────────
    candidates = _get_candidates(
        train_labels_enc,
        train_probs,
        target_label_enc,
        threshold,
        exclude_idx=exclude_input_idx,
    )
    logger.info(f"Valid counterfactual candidates: {len(candidates)}")

    if len(candidates) == 0:
        logger.warning(
            f"No candidates with label={target_label} and prob>={threshold}. "
            "Lowering threshold to 0.5 for fallback."
        )
        candidates = _get_candidates(
            train_labels_enc,
            train_probs,
            target_label_enc,
            0.5,
            exclude_idx=exclude_input_idx,
        )

    if len(candidates) == 0:
        # Last resort: any sample with correct label, ignore probability
        logger.warning(
            f"No candidates with label={target_label} and prob>=0.5. "
            "Using label-only match (no probability filter)."
        )
        label_only = np.where(train_labels_enc == target_label_enc)[0]
        if exclude_input_idx is not None:
            label_only = label_only[label_only != exclude_input_idx]
        if len(label_only) == 0:
            raise RuntimeError(
                f"No samples found with label={target_label} in training set!"
            )
        candidates = label_only

    # Limit candidates by distance for performance
    if len(candidates) > top_k_candidates:
        dists = np.linalg.norm(train_features[candidates] - x_input, axis=1)
        sorted_c = candidates[np.argsort(dists)]
        candidates = sorted_c[:top_k_candidates]

    # ── Build / reuse FACE graph ───────────────────────────────────────────
    try:
        if prebuilt_graph is not None:
            G_base, dens = prebuilt_graph
            G = G_base.copy()
        else:
            G, dens = build_face_graph(
                train_features,
                k=k_graph,
                density_bandwidth=density_bandwidth,
                density_k=k_density,
            )

        # Add query node — connect via true kNN (Dijkstra will route through graph)
        query_node = _add_query_node(
            G, train_features, x_input, dens,
            k=k_graph,
        )

        # Try shortest path through graph (true FACE algorithm)
        path, cost = _face_shortest_path(G, query_node, candidates)

        if path is None:
            # No path via kNN graph — add direct edges to nearest candidates as last resort
            logger.warning("No path via kNN graph. Adding direct edges to nearest candidates.")
            dists_all = np.linalg.norm(train_features - x_input, axis=1)
            cand_dists = dists_all[candidates]
            top_cands  = candidates[np.argsort(cand_dists)[:10]]
            _eps = 1e-8
            for v in top_cands:
                d = float(dists_all[v])
                dens_avg = (1.0 / (d + _eps) + float(dens[v])) / 2.0
                w = d / (dens_avg + _eps)
                if not G.has_edge(query_node, v):
                    G.add_edge(query_node, v, weight=w)
            path, cost = _face_shortest_path(G, query_node, candidates)

        if path is None:
            raise RuntimeError("No path found in FACE graph.")

        # face_path = full list of node indices (query_node first, CF last)
        path_train = path[1:]   # train indices only (drop query_node)
        cf_idx     = path_train[-1]

        result = {
            "counterfactual_idx"    : int(cf_idx),
            "counterfactual_feature": train_features[cf_idx],
            "counterfactual_path"   : train_paths[cf_idx],
            "counterfactual_prob"   : float(train_probs[cf_idx, target_label_enc]),
            "counterfactual_label"  : le.inverse_transform([train_labels_enc[cf_idx]])[0],
            "face_path"             : path,
            "face_path_cost"        : float(cost),
            "method"                : "FACE shortest path",
        }
        logger.info(
            f"FACE path found: length={len(path)}, cost={cost:.4f}, "
            f"CF={result['counterfactual_path']}"
        )

    except Exception as e:
        logger.warning(f"FACE graph failed: {e}. Using fallback nearest feasible.")
        result = _fallback_nearest(
            x_input,
            train_features,
            candidates,
            list(train_paths),
            train_labels_enc,
            train_probs,
            target_label_enc,
            le,
        )

    return result


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FACE graph smoke test")
    parser.add_argument("--n", type=int, default=200, help="Synthetic N points")
    parser.add_argument("--k", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    X   = rng.standard_normal((args.n, 51))
    G, dens = build_face_graph(X, k=args.k)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Density min={dens.min():.4f} max={dens.max():.4f}")
