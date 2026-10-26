"""Robustness evaluation (manuscript Section 6.10).

Bounded random noise: L2 sphere of radius epsilon on the continuous features,
clipped to the valid range; ``noise_draws`` independent draws per epsilon, with F1
averaged over the draws.

FGSM / PGD: L-infinity budget epsilon on the continuous features only. PGD uses a
random start, step size alpha, projection onto the epsilon-ball after every step,
and clipping to the valid range. The graph (E, W0) is built from the clean data
and held fixed; the perturbation changes node features and therefore also the
adaptive coefficients a_ij. All test nodes are perturbed simultaneously, and the
gradient is the exact full-graph gradient (batching.loss_gradient).

Every row reports the effective perturbation norms after clipping.
"""

from __future__ import annotations

import torch

from batching import infer, loss_gradient
from metrics import classification_metrics, robustness_degradation
from perturbation import clip_to_valid, effective_norms, random_l2_perturbation


def _f1(model, g, cfg, x, binary):
    logits, _ = infer(model, g, cfg.num_layers, cfg.eval_chunk_size, x=x)
    return classification_metrics(g.y.cpu().numpy(), logits.argmax(1).cpu().numpy(), binary)["f1"]


@torch.no_grad()
def evaluate_noise(model, g, cfg, clean_f1: float, binary: bool, seed: int) -> list[dict]:
    gen_state = torch.random.get_rng_state()
    torch.manual_seed(10_000 + seed)
    rows = []
    for eps in cfg.noise_epsilons:
        f1s, norms = [], []
        for _ in range(cfg.noise_draws):
            delta = random_l2_perturbation(g.x, eps, g.continuous_mask, g.lower, g.upper)
            f1s.append(_f1(model, g, cfg, g.x + delta, binary))
            norms.append(effective_norms(delta))
        pert = sum(f1s) / len(f1s)
        rows.append({
            "epsilon": eps,
            "clean_f1": clean_f1,
            "perturbed_f1": pert,
            "perturbed_f1_draws": f1s,
            "delta_epsilon": robustness_degradation(clean_f1, pert),
            **{k: max(n[k] for n in norms) if k.endswith("max") else sum(n[k] for n in norms) / len(norms)
               for k in norms[0]},
        })
    torch.random.set_rng_state(gen_state)
    return rows


def fgsm(model, g, cfg, epsilon: float) -> torch.Tensor:
    mask = g.continuous_mask.to(g.x.dtype).unsqueeze(0)
    grad = loss_gradient(model, g, cfg.num_layers, cfg.eval_chunk_size, g.x)
    x_adv = g.x + epsilon * grad.sign() * mask
    return clip_to_valid(g.x, x_adv, g.lower, g.upper).detach()


def pgd(model, g, cfg, epsilon: float, step_size: float, steps: int) -> torch.Tensor:
    mask = g.continuous_mask.to(g.x.dtype).unsqueeze(0)
    x0 = g.x
    x_adv = x0 + torch.empty_like(x0).uniform_(-epsilon, epsilon) * mask
    x_adv = clip_to_valid(x0, x_adv, g.lower, g.upper)
    for _ in range(steps):
        grad = loss_gradient(model, g, cfg.num_layers, cfg.eval_chunk_size, x_adv)
        x_adv = x_adv + step_size * grad.sign() * mask
        x_adv = torch.maximum(torch.minimum(x_adv, x0 + epsilon), x0 - epsilon)
        x_adv = clip_to_valid(x0, x_adv, g.lower, g.upper).detach()
    return x_adv


def evaluate_adversarial(model, g, cfg, clean_f1: float, binary: bool, seed: int) -> dict:
    torch.manual_seed(20_000 + seed)
    x_f = fgsm(model, g, cfg, cfg.adv_epsilon)
    x_p = pgd(model, g, cfg, cfg.adv_epsilon, cfg.pgd_step_size, cfg.pgd_steps)
    f_f = _f1(model, g, cfg, x_f, binary)
    f_p = _f1(model, g, cfg, x_p, binary)
    nf = effective_norms(x_f - g.x)
    np_ = effective_norms(x_p - g.x)
    return {
        "clean_f1": clean_f1,
        "fgsm_f1": f_f,
        "fgsm_drop": robustness_degradation(clean_f1, f_f),
        "pgd_f1": f_p,
        "pgd_drop": robustness_degradation(clean_f1, f_p),
        "fgsm_eff_linf_max": nf["eff_linf_max"],
        "pgd_eff_linf_max": np_["eff_linf_max"],
        "fgsm_eff_l2_mean": nf["eff_l2_mean"],
        "pgd_eff_l2_mean": np_["eff_l2_mean"],
    }
