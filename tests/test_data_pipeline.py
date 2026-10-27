"""Data-pipeline checks that need no PyTorch: loaders, split, encoding, edge sets.

    python -m pytest -q tests/test_data_pipeline.py      (or: python tests/test_data_pipeline.py)
"""

from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

try:  # graph_building imports torch at module level; a stub is enough for the numpy helpers
    import torch  # noqa: F401
except ImportError:
    _stub = types.ModuleType("torch")
    _stub.Tensor = type("Tensor", (), {})
    sys.modules["torch"] = _stub

from config import ExperimentConfig  # noqa: E402
from datasets import load_dataset  # noqa: E402
from graph_building import _codes, _relational_edges, _temporal_edges  # noqa: E402
from preprocess import prepare  # noqa: E402
from synthetic import write_cicids2017, write_cse_cicids2018, write_unsw  # noqa: E402


def _cfg(name, paths, **kw):
    return ExperimentConfig(dataset=name, data_paths=paths, **kw)


def test_unsw_loader_and_split():
    with tempfile.TemporaryDirectory() as d:
        paths = write_unsw(Path(d))
        data = load_dataset(_cfg("unsw_nb15", paths))
        f = data.frame
        assert np.all(np.diff(f["__time"].to_numpy()) >= 0), "frame must be time-sorted"
        for leak in ["label", "attack_cat", "srcip", "dstip", "sport", "stime", "ltime"]:
            assert leak not in data.numeric_cols + data.discrete_cols + data.categorical_cols, leak
        assert set(f["__label"].unique()) <= {"benign", "fuzzers", "exploits", "backdoor"}
        assert any("duplicate" in n and not n.startswith("dropped 0 ") for n in data.notes)
        cfg = _cfg("unsw_nb15", paths)
        splits, enc, summary = prepare(data, cfg)
        n = sum(len(s.y) for s in splits.values())
        assert n == len(f)
        # chronological: every training flow starts no later than every test flow
        assert splits["train"].t.max() <= splits["val"].t.min() <= splits["test"].t.min()
        assert abs(len(splits["train"].y) / n - 0.7) < 0.01
        # continuous mask: one-hot columns and discrete columns are not perturbable
        names = np.array(enc.feature_names)
        assert not enc.continuous_mask[np.char.find(names.astype(str), "=") >= 0].any()
        assert not enc.continuous_mask[names == "dsport"].any()
        assert enc.continuous_mask[names == "dur"].all()
        # scaler fitted on train only
        assert np.allclose(splits["train"].x[:, enc.continuous_mask].mean(0), 0, atol=1e-4)
        assert summary["splits"]["test"]["rows"] == len(splits["test"].y)


def test_cicids2017_twelve_hour_clock():
    with tempfile.TemporaryDirectory() as d:
        paths = write_cicids2017(Path(d))
        data = load_dataset(_cfg("cicids2017", paths))
        span_h = data.frame["__time"].max() / 3600
        assert 2.5 < span_h < 3.1, (f"time span {span_h:.2f} h over {len(data.frame)} rows: "
                                    f"12-hour clock not corrected. Loader notes: {data.notes}")
        assert "Flow Bytes/s" in data.numeric_cols
        assert np.isfinite(data.frame["Flow Bytes/s"].fillna(0)).all()
        assert set(data.frame["__label"]) == {"benign", "dos hulk", "web attack - brute force"}
        assert "Destination Port" in data.discrete_cols and "SYN Flag Count" in data.discrete_cols


def test_cse_cicids2018_repeated_header_and_shared_features():
    with tempfile.TemporaryDirectory() as d:
        paths = write_cse_cicids2018(Path(d))
        data = load_dataset(_cfg("cse_cicids2018", paths))
        assert len(data.frame) == 3000
        cfg = _cfg("cse_cicids2018", paths)
        splits, _, _ = prepare(data, cfg)
        sp = splits["train"]
        src = _codes(sp.src)
        assert (src < 0).all(), "no source addresses in this file"
        dst = _codes(sp.dst)
        tei, _ = _temporal_edges(sp.t, None, dst, "pair", cfg.temporal_window, cfg.beta)
        assert tei.shape[1] > 0 and (dst[tei[0]] == dst[tei[1]]).all(), "temporal edges follow the destination port"
        shared = load_dataset(_cfg("cse_cicids2018", paths, feature_set="shared"))
        assert shared.numeric_cols == ["duration_s", "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes"]
        assert set(shared.frame["protocol"]) <= {"tcp", "udp", "icmp", "other"}
        u = load_dataset(_cfg("unsw_nb15", write_unsw(Path(d)), feature_set="shared"))
        # same units after alignment: durations in seconds on both sides
        assert abs(np.expm1(u.frame["duration_s"]).median() - np.expm1(shared.frame["duration_s"]).median()) < 2.0


def test_edge_sets_are_distinct_and_local():
    with tempfile.TemporaryDirectory() as d:
        data = load_dataset(_cfg("unsw_nb15", write_unsw(Path(d))))
    cfg = ExperimentConfig(dataset="unsw_nb15")
    splits, enc, _ = prepare(data, cfg)
    sp = splits["train"]
    src, dst, proto = _codes(sp.src), _codes(sp.dst), _codes(sp.proto)
    ei, w0 = _relational_edges(sp.x, sp.t, [src, dst, proto], cfg.max_neighbors, cfg.temporal_window,
                               cfg.similarity_threshold, cfg.alpha, cfg.beta, chunk=257)
    tei, tw = _temporal_edges(sp.t, src, dst, "pair", cfg.temporal_window, cfg.beta)

    assert ei.shape[1] > 0 and tei.shape[1] > 0
    assert (ei[0] < ei[1]).all() and (tei[0] < tei[1]).all()
    assert (ei[1] - ei[0]).max() <= cfg.max_neighbors          # locality used by exact chunked inference
    assert (sp.t[ei[1]] - sp.t[ei[0]]).max() <= cfg.temporal_window
    assert (sp.t[tei[1]] - sp.t[tei[0]]).max() <= cfg.temporal_window
    assert np.all(np.diff(ei[0]) >= 0) and np.all(np.diff(tei[1]) >= 0)   # sort orders used by node_range
    # temporal edges join the same (src, dst) pair, and each flow has at most one predecessor
    assert (src[tei[0]] == src[tei[1]]).all() and (dst[tei[0]] == dst[tei[1]]).all()
    assert len(np.unique(tei[1])) == tei.shape[1]
    assert np.allclose(tw, np.exp(-cfg.beta * (sp.t[tei[1]] - sp.t[tei[0]])))
    # E and E_t are genuinely different edge sets (v1 used one set for both)
    n = len(sp.t)
    overlap = np.isin(tei[0] * n + tei[1], ei[0] * n + ei[1]).mean()
    assert overlap < 1.0
    assert ei.shape[1] != tei.shape[1]

    # brute-force check of the vectorised relational edge rule on the first 400 flows
    m = 400
    xn = sp.x[:m] / np.maximum(np.linalg.norm(sp.x[:m], axis=1, keepdims=True), 1e-12)
    expect = set()
    for i in range(m):
        for j in range(i + 1, min(m, i + cfg.max_neighbors + 1)):
            if sp.t[j] - sp.t[i] > cfg.temporal_window:
                continue
            rel = any(c[i] >= 0 and c[i] == c[j] for c in (src, dst, proto))
            if rel or xn[i] @ xn[j] >= cfg.similarity_threshold:
                expect.add((i, j))
    got = {(int(a), int(b)) for a, b in ei.T if b < m}
    assert got == expect


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
