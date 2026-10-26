"""Temporal flow-graph construction (manuscript Sections 3.2 and 4.3).

Two edge sets are built inside each split:

E   (relational edges, used for message passing and the Laplacian term L_graph)
    Flows are sorted by time. For each flow i, the candidates are the next K flows
    j = i+1..i+K with t_j - t_i <= Delta_t. A candidate becomes an edge when the two
    flows share an endpoint or protocol (R(i,j) = 1) or their cosine similarity is
    at least the threshold. Initial weight:
        w0_ij = alpha * exp(-beta |t_i - t_j|) + (1 - alpha) * max(cos(x_i, x_j), 0)
    Every edge in E joins flows at most K positions apart in time order; this
    locality is what makes chunked inference exact (see batching.py).

E_t (temporal edges, used only by L_temp and the TCS diagnostic)
    Each flow is linked to the most recent earlier flow of the same session key
    (source-destination pair by default) if that flow started within Delta_t.
    Weight: omega_ij = exp(-beta |t_i - t_j|). These weights are fixed; they are
    not adapted during training.

Because E_t and E are different edge sets with different weights, L_temp and
L_graph are different quadratic forms (the public v1 code used E for both).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass
class FlowGraph:
    x: torch.Tensor              # (n, d)
    y: torch.Tensor              # (n,)
    t: torch.Tensor              # (n,) float64
    edge_index: torch.Tensor     # (2, m) undirected, i < j, sorted by i
    w0: torch.Tensor             # (m,)
    tedge_index: torch.Tensor    # (2, m_t) undirected, i < j, sorted by j
    tw: torch.Tensor             # (m_t,)
    K: int                       # max index distance of any edge in E
    continuous_mask: torch.Tensor  # (d,) bool
    lower: torch.Tensor          # (d,) valid range, standardised units
    upper: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.x.size(0))

    # ------------------------------------------------------------------
    def node_range(self, start: int, end: int, x: torch.Tensor | None = None) -> "LocalGraph":
        """Induced subgraph on the contiguous node range [start, end)."""

        start, end = max(0, start), min(self.num_nodes, end)
        src = self.edge_index[0]
        lo, hi = _bounds(src, start, end)
        ei = self.edge_index[:, lo:hi]
        w0 = self.w0[lo:hi]
        keep = ei[1] < end
        ei, w0 = ei[:, keep] - start, w0[keep]

        tdst = self.tedge_index[1]
        lo, hi = _bounds(tdst, start, end)
        tei = self.tedge_index[:, lo:hi]
        tw = self.tw[lo:hi]
        keep = tei[0] >= start
        tei, tw = tei[:, keep] - start, tw[keep]

        xs = (self.x if x is None else x)[start:end]
        return LocalGraph(xs, self.y[start:end], ei, w0, tei, tw)

    def to(self, device) -> "FlowGraph":
        for name, value in vars(self).items():
            if torch.is_tensor(value):
                setattr(self, name, value.to(device))
        return self


def _bounds(sorted_keys: torch.Tensor, start: int, end: int) -> tuple[int, int]:
    """Positions of the first key >= start and the first key >= end (same device as the keys)."""

    probe = torch.tensor([start, end], dtype=sorted_keys.dtype, device=sorted_keys.device)
    lo, hi = torch.searchsorted(sorted_keys, probe).tolist()
    return lo, hi


@dataclass
class LocalGraph:
    x: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor     # undirected, local indices
    w0: torch.Tensor
    tedge_index: torch.Tensor
    tw: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.x.size(0))


# ---------------------------------------------------------------------------
def _codes(values: np.ndarray) -> np.ndarray:
    """Integer codes for metadata; missing values get -1 and never match."""

    s = pd.Series(values, dtype=object)
    missing = s.isna() | s.astype(str).str.lower().isin(["nan", "none", "", "missing"])
    codes, _ = pd.factorize(s.astype(str))
    codes = codes.astype(np.int64)
    codes[missing.to_numpy()] = -1
    return codes


def _relational_edges(x, t, meta_codes, K, window, threshold, alpha, beta, chunk=200_000):
    n = x.shape[0]
    norms = np.linalg.norm(x, axis=1)
    norms[norms == 0] = 1.0
    xn = (x / norms[:, None]).astype(np.float32)

    src_all, dst_all, w_all = [], [], []
    for k in range(1, K + 1):
        if k >= n:
            break
        for a in range(0, n - k, chunk):
            b = min(n - k, a + chunk)
            i = np.arange(a, b)
            j = i + k
            gap = t[j] - t[i]
            within = gap <= window
            if not within.any():
                continue
            related = np.zeros(b - a, dtype=bool)
            for codes in meta_codes:
                ci, cj = codes[i], codes[j]
                related |= (ci >= 0) & (ci == cj)
            sim = np.einsum("ij,ij->i", xn[i], xn[j])
            keep = within & (related | (sim >= threshold))
            if not keep.any():
                continue
            w = alpha * np.exp(-beta * gap[keep]) + (1.0 - alpha) * np.maximum(sim[keep], 0.0)
            src_all.append(i[keep])
            dst_all.append(j[keep])
            w_all.append(np.maximum(w, 1e-8))

    if not src_all:
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)
    src = np.concatenate(src_all)
    dst = np.concatenate(dst_all)
    w = np.concatenate(w_all).astype(np.float32)
    order = np.lexsort((dst, src))
    return np.stack([src[order], dst[order]]), w[order]


def _temporal_edges(t, src_codes, dst_codes, key, window, beta):
    """Session key s(i): (src, dst) pair; falls back to whichever endpoint exists
    (for CSE-CIC-IDS2018 files without addresses, __dst holds the destination port);
    key "none" or no endpoint information links each flow to its immediate predecessor."""

    n = len(t)
    if key == "pair" and src_codes is not None and dst_codes is not None:
        session = src_codes * (dst_codes.max() + 2) + dst_codes
        session[(src_codes < 0) | (dst_codes < 0)] = -1
    elif key == "pair" and dst_codes is not None:
        session = dst_codes.copy()
    elif key in ("pair", "src") and src_codes is not None:
        session = src_codes.copy()
    else:
        session = np.zeros(n, dtype=np.int64)

    frame = pd.DataFrame({"s": session, "pos": np.arange(n)})
    prev = frame.groupby("s", sort=False)["pos"].shift(1).to_numpy()
    j = np.arange(n)
    ok = ~np.isnan(prev) & (session >= 0)
    i = prev[ok].astype(np.int64)
    j = j[ok]
    gap = t[j] - t[i]
    within = gap <= window
    i, j, gap = i[within], j[within], gap[within]
    w = np.exp(-beta * gap).astype(np.float32)
    order = np.lexsort((i, j))
    return np.stack([i[order], j[order]]), w[order]


def build_graph(split, encoder, cfg) -> FlowGraph:
    t = split.t
    if len(t) > 1 and np.any(np.diff(t) < 0):
        raise ValueError("split must be sorted by time before graph construction")

    src_codes = _codes(split.src) if split.src is not None else None
    dst_codes = _codes(split.dst) if split.dst is not None else None
    proto_codes = _codes(split.proto) if split.proto is not None else None
    meta = [c for c in (src_codes, dst_codes, proto_codes) if c is not None and (c >= 0).any()]

    ei, w0 = _relational_edges(split.x, t, meta, cfg.max_neighbors, cfg.temporal_window,
                               cfg.similarity_threshold, cfg.alpha, cfg.beta)
    has_src = src_codes is not None and (src_codes >= 0).any()
    has_dst = dst_codes is not None and (dst_codes >= 0).any()
    tei, tw = _temporal_edges(t, src_codes if has_src else None, dst_codes if has_dst else None,
                              cfg.temporal_key, cfg.temporal_window, cfg.beta)

    return FlowGraph(
        x=torch.from_numpy(split.x),
        y=torch.from_numpy(split.y),
        t=torch.from_numpy(t),
        edge_index=torch.from_numpy(ei).long(),
        w0=torch.from_numpy(w0),
        tedge_index=torch.from_numpy(tei).long(),
        tw=torch.from_numpy(tw),
        K=int(cfg.max_neighbors),
        continuous_mask=torch.from_numpy(encoder.continuous_mask.copy()),
        lower=torch.from_numpy(encoder.lower),
        upper=torch.from_numpy(encoder.upper),
    )


def graph_stats(g: FlowGraph) -> dict:
    n = g.num_nodes
    return {
        "nodes": n,
        "relational_edges": int(g.edge_index.size(1)),
        "temporal_edges": int(g.tedge_index.size(1)),
        "mean_relational_degree": float(2 * g.edge_index.size(1) / max(n, 1)),
        "nodes_with_temporal_predecessor": float(g.tedge_index.size(1) / max(n, 1)),
        "overlap_E_and_Et": _overlap(g),
    }


def _overlap(g: FlowGraph) -> float:
    """Fraction of temporal edges that are also relational edges."""

    if g.tedge_index.size(1) == 0:
        return 0.0
    n = g.num_nodes
    e = (g.edge_index[0] * n + g.edge_index[1]).cpu().numpy()
    te = (g.tedge_index[0] * n + g.tedge_index[1]).cpu().numpy()
    return float(np.isin(te, e).mean())
