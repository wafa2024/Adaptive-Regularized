"""Classification metrics and the four diagnostics (manuscript Sections 6.7-6.8).

TCS  = (1/|E_t|) sum_{(i,j) in E_t} ||z_i - z_j||^2                 unweighted, temporal edges
GSI  = (1/N) Tr(Z^T L0 Z) = (1/N) sum_{(i,j) in E} w0_ij ||z_i - z_j||^2
       L0 is the Laplacian of the INITIAL weights W0. Using the same reference
       Laplacian for every model makes GSI comparable across variants. (With the
       adaptive weights w~ = a*w0, a in (0,1), GSI would fall mechanically whenever
       the coefficients shrink, even with identical embeddings.) The adaptive-weight
       energy is also reported as ``gsi_adaptive``.
SR   = fraction of |theta| < tau over Theta_s = {phi, classifier weights}
Delta_eps = (F1_clean - F1_perturbed) / F1_clean, computed per seed and then averaged.

TCS and GSI use different edge sets and weights, so neither can be derived from
the other.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from losses import pairwise_energy


def classification_metrics(y_true, y_pred, binary: bool = True) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {"accuracy": float(accuracy_score(y_true, y_pred))}
    if binary:
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
        out.update({"precision": float(p), "recall": float(r), "f1": float(f)})
    else:
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        out.update({"precision": float(p), "recall": float(r), "f1": float(f),
                    "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0))})
    return out


@torch.no_grad()
def diagnostics(model, g, z: torch.Tensor, kind: str, sparsity_threshold: float, chunk: int = 200_000) -> dict[str, float]:
    n = max(g.num_nodes, 1)
    m_t = g.tedge_index.size(1)
    tcs = float(pairwise_energy(z, g.tedge_index, None) / max(m_t, 1)) if m_t else float("nan")
    gsi = float(pairwise_energy(z, g.edge_index, g.w0) / n)
    out = {"tcs": tcs, "gsi": gsi, "num_temporal_edges": int(m_t), "num_relational_edges": int(g.edge_index.size(1))}

    if kind == "artgl":
        energy = 0.0
        for s in range(0, g.edge_index.size(1), chunk):
            ei = g.edge_index[:, s:s + chunk]
            a = model.edge_coefficients(g.x, ei)
            energy += float(pairwise_energy(z, ei, a * g.w0[s:s + chunk]))
        out["gsi_adaptive"] = energy / n
        theta = model.sparsity_parameters().detach()
        out["sr"] = float((theta.abs() < sparsity_threshold).float().mean())
    return out


def robustness_degradation(clean_f1: float, perturbed_f1: float) -> float:
    return float((clean_f1 - perturbed_f1) / clean_f1) if clean_f1 > 0 else float("nan")
