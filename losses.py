from __future__ import annotations

import torch
from torch.nn import functional as F


def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def temporal_consistency_loss(
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    temporal_edge_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    src, dst = edge_index
    if temporal_edge_mask is not None:
        src = src[temporal_edge_mask]
        dst = dst[temporal_edge_mask]
        edge_weight = edge_weight[temporal_edge_mask]

    diff = embeddings[src] - embeddings[dst]
    squared_distance = diff.pow(2).sum(dim=1)
    return (edge_weight * squared_distance).mean()


def graph_smoothness_loss(
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> torch.Tensor:
    src, dst = edge_index
    diff = embeddings[src] - embeddings[dst]
    squared_distance = diff.pow(2).sum(dim=1)

    # The 0.5 factor matches the standard undirected Laplacian identity.
    return 0.5 * (edge_weight * squared_distance).mean()


def sparsity_loss(parameter_vector: torch.Tensor) -> torch.Tensor:
    return parameter_vector.abs().mean()


def make_l2_noise(x: torch.Tensor, epsilon: float) -> torch.Tensor:
    noise = torch.randn_like(x)
    noise_norm = noise.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    return epsilon * noise / noise_norm


def robustness_loss(
    model,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    clean_logits: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    delta = make_l2_noise(x, epsilon)
    perturbed_logits, _, _ = model(x + delta, edge_index, edge_weight)

    clean_prob = F.softmax(clean_logits.detach(), dim=1)
    perturbed_prob = F.softmax(perturbed_logits, dim=1)
    return (clean_prob - perturbed_prob).pow(2).sum(dim=1).mean()


def composite_loss(
    model,
    data,
    lambdas: dict[str, float],
    epsilon: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits, embeddings, adaptive_edge_weight = model(data.x, data.edge_index, data.edge_weight)

    l_cls = classification_loss(logits, data.y)
    l_temp = temporal_consistency_loss(
        embeddings,
        data.edge_index,
        adaptive_edge_weight,
        getattr(data, "temporal_edge_mask", None),
    )
    l_graph = graph_smoothness_loss(embeddings, data.edge_index, adaptive_edge_weight)
    l_sparse = sparsity_loss(model.sparsity_parameters())
    l_rob = robustness_loss(model, data.x, data.edge_index, data.edge_weight, logits, epsilon)

    total = (
        l_cls
        + lambdas["temp"] * l_temp
        + lambdas["graph"] * l_graph
        + lambdas["sparse"] * l_sparse
        + lambdas["robust"] * l_rob
    )

    logs = {
        "loss_total": float(total.detach().cpu()),
        "loss_cls": float(l_cls.detach().cpu()),
        "loss_temp": float(l_temp.detach().cpu()),
        "loss_graph": float(l_graph.detach().cpu()),
        "loss_sparse": float(l_sparse.detach().cpu()),
        "loss_robust": float(l_rob.detach().cpu()),
    }
    return total, logs
