"""ARTGL model and the graph baselines (manuscript Sections 4.4-4.5 and 6.6).

ARTGL forward pass, with equation references to the revised manuscript:

  h_i^(0) = ReLU(W_in x_i + b_in)                                   (input projection)
  a_ij    = sigmoid(phi^T [x_i + x_j ; |x_i - x_j| ; x_i * x_j])     (Eq. 20, linear, raw inputs)
  w~_ij   = a_ij * w0_ij                                            (Eq. 19)
  hbar_i  = sum_j w~_ij h_j / sum_j w~_ij                           (normalised neighbourhood mean)
  m_i     = ReLU(W_s h_i + sum_j w~_ij W_n h_j / sum_j w~_ij)        (message)
  g_i     = sigmoid(W_g h_i + U_g hbar_i + b_g)                     (gate)
  h_i'    = g_i * m_i + (1 - g_i) * h_i                             (gated update)
  z_i     = h_i^(L),   logits_i = W_c z_i + b_c

The edge coefficients depend only on the two endpoint input vectors, so they are
computed once per forward pass and shared by all layers. They are symmetric by
construction (a_ij = a_ji). With ``adaptive_edges=False`` (fixed-edge variant) a_ij = 1.
Dropout is applied to the input of every graph layer and of the classifier.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _both_directions(edge_index: torch.Tensor, weight: torch.Tensor | None = None):
    ei = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    if weight is None:
        return ei, None
    return ei, torch.cat([weight, weight])


class ARTGL(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 2,
                 dropout: float = 0.30, adaptive_edges: bool = True) -> None:
        super().__init__()
        self.adaptive_edges = adaptive_edges
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.phi = nn.Linear(3 * input_dim, 1, bias=False)            # Eq. 20
        self.W_s = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.W_n = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)])
        self.W_g = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.U_g = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)])
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    def edge_coefficients(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if not self.adaptive_edges:
            return torch.ones(edge_index.size(1), device=x.device, dtype=x.dtype)
        xi, xj = x[edge_index[0]], x[edge_index[1]]
        pair = torch.cat([xi + xj, (xi - xj).abs(), xi * xj], dim=-1)
        return torch.sigmoid(self.phi(pair)).squeeze(-1)

    @staticmethod
    def _normalised_sum(values: torch.Tensor, ei: torch.Tensor, w: torch.Tensor, n: int) -> torch.Tensor:
        src, dst = ei
        out = torch.zeros(n, values.size(1), device=values.device, dtype=values.dtype)
        out.index_add_(0, dst, values[src] * w.unsqueeze(-1))
        deg = torch.zeros(n, device=values.device, dtype=values.dtype)
        deg.index_add_(0, dst, w)
        return out / deg.clamp_min(1e-8).unsqueeze(-1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, w0: torch.Tensor):
        """x: (n, d); edge_index: (2, m) undirected; w0: (m,).

        Returns logits (n, C), embeddings z (n, p), effective weights w~ (m,).
        """

        n = x.size(0)
        a = self.edge_coefficients(x, edge_index)
        w_eff = a * w0
        ei, w = _both_directions(edge_index, w_eff)

        h = F.relu(self.input_proj(self.dropout(x)))
        for layer in range(len(self.W_s)):
            h_in = self.dropout(h)
            hbar = self._normalised_sum(h_in, ei, w, n)
            neigh = self._normalised_sum(self.W_n[layer](h_in), ei, w, n)
            m = F.relu(self.W_s[layer](h_in) + neigh)
            g = torch.sigmoid(self.W_g[layer](h_in) + self.U_g[layer](hbar))
            h = g * m + (1.0 - g) * h
        z = h
        logits = self.classifier(self.dropout(z))
        return logits, z, w_eff

    def sparsity_parameters(self) -> torch.Tensor:
        """Theta_s: the edge-scoring vector phi and the classifier weights (no biases)."""

        return torch.cat([self.phi.weight.reshape(-1), self.classifier.weight.reshape(-1)])


# ---------------------------------------------------------------------------
# Graph baselines: identical graph, batching, inference and early stopping;
# trained with cross-entropy only.
# ---------------------------------------------------------------------------
class GCNBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 2, dropout: float = 0.30):
        super().__init__()
        from torch_geometric.nn import GCNConv

        dims = [input_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList([GCNConv(dims[i], dims[i + 1]) for i in range(num_layers)])
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, w0):
        ei, w = _both_directions(edge_index, w0)
        h = x
        for conv in self.convs:
            h = F.relu(conv(self.dropout(h), ei, w))
        return self.classifier(self.dropout(h)), h, w0


class GATBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 2,
                 dropout: float = 0.30, heads: int = 4):
        super().__init__()
        from torch_geometric.nn import GATConv

        dims = [input_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList(
            [GATConv(dims[i], hidden_dim // heads, heads=heads, dropout=dropout) for i in range(num_layers)]
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, w0):
        ei, _ = _both_directions(edge_index)
        h = x
        for conv in self.convs:
            h = F.elu(conv(self.dropout(h), ei))
        return self.classifier(self.dropout(h)), h, w0


def build_model(kind: str, input_dim: int, num_classes: int, cfg) -> nn.Module:
    if kind == "artgl":
        return ARTGL(input_dim, cfg.hidden_dim, num_classes, cfg.num_layers, cfg.dropout, cfg.adaptive_edges)
    if kind == "gcn":
        return GCNBaseline(input_dim, cfg.hidden_dim, num_classes, cfg.num_layers, cfg.dropout)
    if kind == "gat":
        return GATBaseline(input_dim, cfg.hidden_dim, num_classes, cfg.num_layers, cfg.dropout)
    raise ValueError(kind)
