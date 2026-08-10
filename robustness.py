from __future__ import annotations

import torch
from torch.nn import functional as F

from losses import make_l2_noise
from metrics import classification_metrics, robustness_degradation


@torch.no_grad()
def evaluate_under_l2_noise(model, data, epsilons: list[float]) -> list[dict[str, float]]:
    """Evaluate F1 degradation under bounded random L2 perturbations."""

    model.eval()
    clean_logits, _, _ = model(data.x, data.edge_index, data.edge_weight)
    clean_f1 = classification_metrics(clean_logits, data.y)["f1"]

    rows = []
    for eps in epsilons:
        delta = make_l2_noise(data.x, eps)
        perturbed_logits, _, _ = model(data.x + delta, data.edge_index, data.edge_weight)
        perturbed_f1 = classification_metrics(perturbed_logits, data.y)["f1"]
        rows.append(
            {
                "epsilon": eps,
                "clean_f1": clean_f1,
                "perturbed_f1": perturbed_f1,
                "delta_epsilon": robustness_degradation(clean_f1, perturbed_f1),
            }
        )
    return rows


def fgsm_attack(model, data, epsilon: float) -> torch.Tensor:
    x_adv = data.x.detach().clone().requires_grad_(True)
    logits, _, _ = model(x_adv, data.edge_index, data.edge_weight)
    loss = F.cross_entropy(logits, data.y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    return torch.clamp(x_adv + epsilon * grad.sign(), min=-5.0, max=5.0).detach()


def pgd_attack(model, data, epsilon: float, step_size: float, steps: int) -> torch.Tensor:
    x_clean = data.x.detach()
    x_adv = x_clean + torch.empty_like(x_clean).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, min=-5.0, max=5.0).detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits, _, _ = model(x_adv, data.edge_index, data.edge_weight)
        loss = F.cross_entropy(logits, data.y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv + step_size * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x_clean + epsilon), x_clean - epsilon)
        x_adv = torch.clamp(x_adv, min=-5.0, max=5.0).detach()

    return x_adv


def evaluate_adversarial(model, data, epsilon: float = 0.05, pgd_steps: int = 10, pgd_step_size: float = 0.01) -> dict[str, float]:
    model.eval()

    clean_logits, _, _ = model(data.x, data.edge_index, data.edge_weight)
    clean_f1 = classification_metrics(clean_logits, data.y)["f1"]

    x_fgsm = fgsm_attack(model, data, epsilon=epsilon)
    fgsm_logits, _, _ = model(x_fgsm, data.edge_index, data.edge_weight)
    fgsm_f1 = classification_metrics(fgsm_logits, data.y)["f1"]

    x_pgd = pgd_attack(model, data, epsilon=epsilon, step_size=pgd_step_size, steps=pgd_steps)
    pgd_logits, _, _ = model(x_pgd, data.edge_index, data.edge_weight)
    pgd_f1 = classification_metrics(pgd_logits, data.y)["f1"]

    return {
        "clean_f1": clean_f1,
        "fgsm_f1": fgsm_f1,
        "fgsm_drop": robustness_degradation(clean_f1, fgsm_f1),
        "pgd_f1": pgd_f1,
        "pgd_drop": robustness_degradation(clean_f1, pgd_f1),
    }
