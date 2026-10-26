"""Mini-batching and exact chunked inference (manuscript Section 4.7).

Training: the time-sorted training graph is cut into contiguous node ranges of
``batch_size`` flows. Each range defines an induced temporal subgraph (edges
crossing the range boundary are dropped for that step). The order of the ranges
is shuffled every epoch.

Inference: every relational edge joins flows at most K positions apart in time
order (graph_building.py), so a node's L-layer receptive field lies within L*K
positions. Chunks are evaluated on [start - H, end + H) with H = (L + 1) * K
(the extra K also covers degree normalisation in GCN), and only the core
[start, end) outputs are kept. The result equals full-graph inference exactly,
in eval mode.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def training_ranges(n: int, batch_size: int, generator: torch.Generator) -> list[tuple[int, int]]:
    ranges = [(s, min(n, s + batch_size)) for s in range(0, n, batch_size)]
    order = torch.randperm(len(ranges), generator=generator).tolist()
    return [ranges[i] for i in order]


def halo_size(g, num_layers: int) -> int:
    return (num_layers + 1) * g.K


@torch.no_grad()
def infer(model, g, num_layers: int, chunk: int, x: torch.Tensor | None = None):
    """Exact full-graph logits and embeddings, computed chunk by chunk."""

    model.eval()
    n = g.num_nodes
    H = halo_size(g, num_layers)
    logits_out, z_out = [], []
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        lo, hi = max(0, start - H), min(n, end + H)
        local = g.node_range(lo, hi, x=x)
        logits, z, _ = model(local.x, local.edge_index, local.w0)
        logits_out.append(logits[start - lo:end - lo])
        z_out.append(z[start - lo:end - lo])
    return torch.cat(logits_out), torch.cat(z_out)


def loss_gradient(model, g, num_layers: int, chunk: int, x: torch.Tensor) -> torch.Tensor:
    """Exact gradient of the summed cross-entropy over all nodes with respect to x."""

    model.eval()
    n = g.num_nodes
    H = halo_size(g, num_layers)
    grad = torch.zeros_like(x)
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        lo, hi = max(0, start - H), min(n, end + H)
        x_ext = x[lo:hi].detach().clone().requires_grad_(True)
        local = g.node_range(lo, hi)
        logits, _, _ = model(x_ext, local.edge_index, local.w0)
        core = slice(start - lo, end - lo)
        loss = F.cross_entropy(logits[core], g.y[start:end], reduction="sum")
        (g_ext,) = torch.autograd.grad(loss, x_ext)
        grad[lo:hi] += g_ext
    return grad
