from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AdaptiveTemporalGraphNet(nn.Module):
    """Adaptive regularized temporal graph learner.

    The implementation follows the manuscript equations closely:
    - edge coefficients are learned from pairwise node features,
    - weighted neighborhood messages are aggregated on the temporal graph,
    - a gate controls how much new neighborhood information updates each node,
    - the final embedding is passed to a classifier.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.self_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.neighbor_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.gate_layers = nn.ModuleList([nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(num_layers)])

        # Pairwise edge scorer uses [h_i + h_j, |h_i - h_j|, h_i * h_j].
        self.edge_scorer = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _adaptive_edge_weights(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        base_edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        h_src = h[src]
        h_dst = h[dst]
        pair_features = torch.cat([h_src + h_dst, torch.abs(h_src - h_dst), h_src * h_dst], dim=-1)
        edge_coefficients = self.edge_scorer(pair_features).squeeze(-1)
        return edge_coefficients * base_edge_weight

    def _weighted_aggregate(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        src, dst = edge_index
        messages = self.neighbor_layers[layer_id](h[src]) * edge_weight.view(-1, 1)

        out = torch.zeros_like(h)
        out.index_add_(0, dst, messages)

        degree = torch.zeros(h.size(0), device=h.device, dtype=h.dtype)
        degree.index_add_(0, dst, edge_weight)
        out = out / degree.clamp_min(1e-8).view(-1, 1)
        return out

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        base_edge_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = F.relu(self.input_proj(x))
        h = self.dropout(h)

        final_edge_weight = base_edge_weight
        for layer_id in range(len(self.self_layers)):
            final_edge_weight = self._adaptive_edge_weights(h, edge_index, base_edge_weight)
            neigh = self._weighted_aggregate(h, edge_index, final_edge_weight, layer_id)

            message = F.relu(self.self_layers[layer_id](h) + neigh)
            gate_input = torch.cat([h, neigh], dim=-1)
            gate = torch.sigmoid(self.gate_layers[layer_id](gate_input))
            h = gate * message + (1.0 - gate) * h
            h = self.dropout(h)

        embeddings = h
        logits = self.classifier(embeddings)
        return logits, embeddings, final_edge_weight

    def sparsity_parameters(self) -> torch.Tensor:
        """Return the parameter subset used in the L1 sparsity term."""

        params = []
        for module in [self.edge_scorer, self.classifier]:
            for param in module.parameters():
                params.append(param.view(-1))
        return torch.cat(params)
