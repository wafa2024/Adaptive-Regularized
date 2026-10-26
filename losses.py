"""Composite objective (manuscript Section 4.6). For a training subgraph with N nodes:

  L_cls    = (1/N) sum_i CE(logits_i, y_i)
  L_temp   = (1/N) sum_{(i,j) in E_t} omega_ij ||z_i - z_j||^2        (temporal edges, fixed weights)
  L_graph  = (1/N) Tr(Z^T L~ Z) = (1/N) sum_{(i,j) in E} w~_ij ||z_i - z_j||^2   (relational edges, adaptive)
  L_sparse = (1/|Theta_s|) ||Theta_s||_1
  L_rob    = (1/N) sum_i || p(x_i) - p(x_i + delta_i) ||^2

Sums over E and E_t run over undirected edges, each counted once, so the Laplacian
identity Tr(Z^T L Z) = sum_{i<j} w_ij ||z_i - z_j||^2 holds exactly. p(.) is the
softmax output with the clean prediction detached. delta_i has L2 norm epsilon on
the continuous features only, and is then clipped to the valid feature range
(perturbation.py).
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from perturbation import random_l2_perturbation


def pairwise_energy(z: torch.Tensor, edge_index: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    """sum over undirected edges of weight * ||z_i - z_j||^2 (unnormalised)."""

    if edge_index.size(1) == 0:
        return z.sum() * 0.0
    diff = z[edge_index[0]] - z[edge_index[1]]
    sq = diff.pow(2).sum(dim=1)
    return (sq if weight is None else weight * sq).sum()


def composite_loss(model, g, lambdas: dict[str, float], epsilon: float,
                   continuous_mask: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor,
                   kind: str = "artgl"):
    """g is a LocalGraph (a contiguous temporal training subgraph)."""

    n = max(g.num_nodes, 1)
    logits, z, w_eff = model(g.x, g.edge_index, g.w0)
    l_cls = F.cross_entropy(logits, g.y)
    logs = {"loss_cls": float(l_cls.detach())}
    if kind != "artgl":
        logs["loss_total"] = logs["loss_cls"]
        return l_cls, logs

    total = l_cls
    zero = l_cls.new_zeros(())
    l_temp = pairwise_energy(z, g.tedge_index, g.tw) / n if lambdas["temp"] > 0 else zero
    l_graph = pairwise_energy(z, g.edge_index, w_eff) / n if lambdas["graph"] > 0 else zero
    theta = model.sparsity_parameters()
    l_sparse = theta.abs().mean() if lambdas["sparse"] > 0 else zero
    if lambdas["robust"] > 0:
        delta = random_l2_perturbation(g.x, epsilon, continuous_mask, lower, upper)
        pert_logits, _, _ = model(g.x + delta, g.edge_index, g.w0)
        p_clean = F.softmax(logits.detach(), dim=1)
        p_pert = F.softmax(pert_logits, dim=1)
        l_rob = (p_clean - p_pert).pow(2).sum(dim=1).mean()
    else:
        l_rob = zero

    total = (l_cls + lambdas["temp"] * l_temp + lambdas["graph"] * l_graph
             + lambdas["sparse"] * l_sparse + lambdas["robust"] * l_rob)
    logs.update({
        "loss_total": float(total.detach()),
        "loss_temp": float(l_temp.detach()),
        "loss_graph": float(l_graph.detach()),
        "loss_sparse": float(l_sparse.detach()),
        "loss_robust": float(l_rob.detach()),
    })
    return total, logs
