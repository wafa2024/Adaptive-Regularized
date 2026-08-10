from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _has_structural_relation(row_i: pd.Series, row_j: pd.Series) -> bool:
    """Return True when two flows share at least one available relation.

    The function is intentionally conservative. If no metadata columns are
    available, graph construction falls back to temporal and feature similarity.
    """

    if row_i.empty or row_j.empty:
        return False

    for col in row_i.index:
        if pd.notna(row_i[col]) and pd.notna(row_j[col]) and str(row_i[col]) == str(row_j[col]):
            return True
    return False


def build_temporal_graph(
    features: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    metadata: Optional[pd.DataFrame] = None,
    temporal_window: float = 30.0,
    similarity_threshold: float = 0.65,
    max_neighbors: int = 30,
    alpha: float = 0.5,
    beta: float = 0.1,
) -> Data:
    """Construct a sparse temporal flow graph.

    Each flow record becomes a node. Edges connect records that are close in time
    and either share a structural relation or have sufficient feature similarity.
    Initial edge weights combine temporal proximity and feature similarity.
    """

    if len(features) != len(labels) or len(features) != len(timestamps):
        raise ValueError("features, labels, and timestamps must have the same length.")

    n_nodes = features.shape[0]
    order = np.argsort(timestamps)
    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(n_nodes)

    x_sorted = features[order]
    y_sorted = labels[order]
    t_sorted = timestamps[order]
    meta_sorted = metadata.iloc[order].reset_index(drop=True) if metadata is not None else pd.DataFrame(index=range(n_nodes))

    edge_sources: list[int] = []
    edge_targets: list[int] = []
    edge_weights: list[float] = []
    temporal_flags: list[int] = []

    right = 0
    for left in range(n_nodes):
        while right < n_nodes and t_sorted[right] - t_sorted[left] <= temporal_window:
            right += 1

        candidate_indices = list(range(left + 1, right))[:max_neighbors]
        for j in candidate_indices:
            time_gap = abs(float(t_sorted[left] - t_sorted[j]))
            sim = _cosine_similarity(x_sorted[left], x_sorted[j])
            related = _has_structural_relation(meta_sorted.iloc[left], meta_sorted.iloc[j])

            if not related and sim < similarity_threshold:
                continue

            temporal_part = np.exp(-beta * time_gap)
            weight = alpha * temporal_part + (1.0 - alpha) * max(sim, 0.0)
            weight = float(max(weight, 1e-8))

            # Add both directions so message passing works on an undirected graph.
            edge_sources.extend([left, j])
            edge_targets.extend([j, left])
            edge_weights.extend([weight, weight])
            temporal_flags.extend([1, 1])

    if not edge_sources:
        # A small fallback prevents empty-graph crashes during development.
        # It should not be reached with normal temporal windows.
        for i in range(n_nodes - 1):
            edge_sources.extend([i, i + 1])
            edge_targets.extend([i + 1, i])
            edge_weights.extend([1.0, 1.0])
            temporal_flags.extend([1, 1])

    data = Data(
        x=torch.tensor(x_sorted, dtype=torch.float32),
        y=torch.tensor(y_sorted, dtype=torch.long),
        edge_index=torch.tensor([edge_sources, edge_targets], dtype=torch.long),
        edge_weight=torch.tensor(edge_weights, dtype=torch.float32),
        temporal_edge_mask=torch.tensor(temporal_flags, dtype=torch.bool),
        timestamp=torch.tensor(t_sorted, dtype=torch.float32),
    )
    data.original_order = torch.tensor(order, dtype=torch.long)
    data.inverse_order = torch.tensor(inverse_order, dtype=torch.long)
    return data
