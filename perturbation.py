"""Perturbations used in training (L_rob) and in the robustness evaluation.

Rules (manuscript Section 6.10):
  * only continuous flow features are perturbed; one-hot categorical columns
    (protocol, state, service), ports and binary flags are left unchanged;
  * random noise is drawn on the continuous coordinates and rescaled so that its
    L2 norm is exactly epsilon (L2 sphere);
  * after perturbation each coordinate is clipped to the valid range observed in
    the training split (standardised units). The range is widened per sample to
    include the clean value, so clipping only moves a coordinate back towards its
    clean value: |delta'_k| <= |delta_k| for every k. Clipping can therefore only
    shrink a perturbation, and the L2 and L-infinity budgets always hold. The
    effective norms after clipping are reported alongside the results.
"""

from __future__ import annotations

import torch


def clip_to_valid(x_clean: torch.Tensor, x_pert: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    lo = torch.minimum(lower.unsqueeze(0), x_clean)
    hi = torch.maximum(upper.unsqueeze(0), x_clean)
    return torch.maximum(torch.minimum(x_pert, hi), lo)


def random_l2_perturbation(x: torch.Tensor, epsilon: float, continuous_mask: torch.Tensor,
                           lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Return delta (already clipped) with ||delta_i||_2 <= epsilon, zero on non-continuous columns."""

    mask = continuous_mask.to(x.dtype).unsqueeze(0)
    noise = torch.randn_like(x) * mask
    noise = epsilon * noise / noise.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    x_pert = clip_to_valid(x, x + noise, lower, upper)
    return (x_pert - x).detach()


def effective_norms(delta: torch.Tensor) -> dict[str, float]:
    l2 = delta.norm(p=2, dim=1)
    linf = delta.abs().amax(dim=1)
    return {
        "eff_l2_mean": float(l2.mean()),
        "eff_l2_max": float(l2.max()),
        "eff_linf_mean": float(linf.mean()),
        "eff_linf_max": float(linf.max()),
    }
