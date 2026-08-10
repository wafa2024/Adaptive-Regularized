from __future__ import annotations

import copy
from typing import Any

import torch
from tqdm import trange

from losses import composite_loss
from metrics import classification_metrics, diagnostic_metrics


def move_data_to_device(data, device: torch.device):
    data = copy.copy(data)
    for key, value in data:
        if torch.is_tensor(value):
            setattr(data, key, value.to(device))
    return data


def evaluate_model(model, data, sparsity_threshold: float) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(data.x, data.edge_index, data.edge_weight)
        metrics = classification_metrics(logits, data.y)
        metrics.update(diagnostic_metrics(model, data, sparsity_threshold=sparsity_threshold))
    return metrics


def train_model(
    model,
    train_data,
    val_data,
    learning_rate: float,
    weight_decay: float,
    lambdas: dict[str, float],
    epsilon: float,
    max_epochs: int,
    patience: int,
    sparsity_threshold: float,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[Any, list[dict[str, float]]]:
    device = torch.device(device)
    model = model.to(device)
    train_data = move_data_to_device(train_data, device)
    val_data = move_data_to_device(val_data, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0
    wait = 0
    history: list[dict[str, float]] = []

    for epoch in trange(1, max_epochs + 1, desc="Training", leave=False):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, loss_logs = composite_loss(model, train_data, lambdas=lambdas, epsilon=epsilon)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        val_metrics = evaluate_model(model, val_data, sparsity_threshold=sparsity_threshold)
        row = {"epoch": epoch, **loss_logs, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            break

    model.load_state_dict(best_state)
    return model, history
