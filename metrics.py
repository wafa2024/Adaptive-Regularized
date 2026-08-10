from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from losses import graph_smoothness_loss, temporal_consistency_loss


@torch.no_grad()
def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    y_true = labels.detach().cpu().numpy()
    y_pred = logits.argmax(dim=1).detach().cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary" if len(np.unique(y_true)) == 2 else "weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


@torch.no_grad()
def diagnostic_metrics(
    model,
    data,
    sparsity_threshold: float = 1e-3,
) -> dict[str, float]:
    logits, embeddings, adaptive_edge_weight = model(data.x, data.edge_index, data.edge_weight)

    tcs = temporal_consistency_loss(
        embeddings,
        data.edge_index,
        adaptive_edge_weight,
        getattr(data, "temporal_edge_mask", None),
    )
    gsi = graph_smoothness_loss(embeddings, data.edge_index, adaptive_edge_weight) / max(data.x.size(0), 1)

    sparse_params = model.sparsity_parameters().detach()
    sparsity_ratio = (sparse_params.abs() < sparsity_threshold).float().mean()

    out = classification_metrics(logits, data.y)
    out.update(
        {
            "tcs": float(tcs.detach().cpu()),
            "gsi": float(gsi.detach().cpu()),
            "sr": float(sparsity_ratio.detach().cpu()),
        }
    )
    return out


def robustness_degradation(clean_f1: float, perturbed_f1: float) -> float:
    if clean_f1 <= 0:
        return 0.0
    return float((clean_f1 - perturbed_f1) / clean_f1)
