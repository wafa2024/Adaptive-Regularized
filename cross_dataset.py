"""Cross-dataset evaluation (revised manuscript Section 7, Tables 9-10).

    python cross_dataset.py --train configs/unsw_nb15.json --test configs/cicids2017.json
    python cross_dataset.py --train configs/unsw_nb15.json configs/cicids2017.json --test configs/cse_cicids2018.json

Protocol (identical to the text in the manuscript):
  * every dataset is loaded with feature_set="shared": the seven attributes in
    datasets.SHARED_* (log1p of duration in seconds, forward/backward packets,
    forward/backward bytes; destination port; protocol in {tcp, udp, icmp, other})
    and binary labels (benign = 0, malicious = 1);
  * each dataset is split chronologically 70/10/20. The model trains on the source
    training splits and early-stops on the source validation splits; with two
    sources, both are used and their graphs are kept separate (no edges between
    datasets);
  * the encoder and scaler are fitted on the source training data only and
    applied unchanged to the target;
  * graph-construction parameters (Delta_t, K, similarity threshold, alpha, beta)
    are taken from the first source config and are NOT re-tuned on the target;
  * evaluation uses the target's chronological test split (last 20%). No target
    labels are used for training, model selection or graph construction.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from batching import infer
from config import ExperimentConfig
from datasets import load_dataset
from graph_building import FlowGraph, build_graph, graph_stats
from metrics import classification_metrics
from model import build_model
from preprocess import Encoder, prepare, write_summary
from train import resolve_device, set_seed, train_graph_model


def _concat_graphs(graphs: list[FlowGraph]) -> FlowGraph:
    """Disjoint union of graphs with shifted indices. Time stays per dataset; there are no cross edges."""

    if len(graphs) == 1:
        return graphs[0]
    offset, ei, te = 0, [], []
    for g in graphs:
        ei.append(g.edge_index + offset)
        te.append(g.tedge_index + offset)
        offset += g.num_nodes
    tei = torch.cat(te, 1)
    order = torch.argsort(tei[1], stable=True)
    first = graphs[0]
    return FlowGraph(
        x=torch.cat([g.x for g in graphs]), y=torch.cat([g.y for g in graphs]),
        t=torch.cat([g.t for g in graphs]), edge_index=torch.cat(ei, 1), w0=torch.cat([g.w0 for g in graphs]),
        tedge_index=tei[:, order], tw=torch.cat([g.tw for g in graphs])[order], K=first.K,
        continuous_mask=first.continuous_mask, lower=first.lower, upper=first.upper,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--output_dir", default="results/cross_dataset")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    src_cfgs = [ExperimentConfig.from_json(p, feature_set="shared", label_mode="binary", split="chronological")
                for p in args.train]
    tgt_cfg = ExperimentConfig.from_json(args.test, feature_set="shared", label_mode="binary", split="chronological")
    graph_params = {k: getattr(src_cfgs[0], k) for k in
                    ["temporal_window", "max_neighbors", "similarity_threshold", "alpha", "beta", "temporal_key"]}
    seeds = args.seeds or src_cfgs[0].seeds
    name = "+".join(c.dataset for c in src_cfgs) + "__to__" + tgt_cfg.dataset
    out = Path(args.output_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    # source data: one encoder fitted on the union of the source training splits
    loaded = [load_dataset(c) for c in src_cfgs]
    train_frames = []
    from preprocess import split_indices
    for data, c in zip(loaded, src_cfgs):
        i_tr, _, _ = split_indices(data.frame, c)
        train_frames.append(data.frame.iloc[i_tr])
    enc = Encoder(loaded[0].numeric_cols, loaded[0].discrete_cols, loaded[0].categorical_cols, label_mode="binary")
    enc.class_names = ["benign", "malicious"]
    enc.categories = {"protocol": ["icmp", "other", "tcp", "udp"]}
    raw = enc._raw_matrix(pd.concat(train_frames, ignore_index=True))
    enc.mean = raw.mean(0, dtype=np.float64)
    std = raw.std(0, dtype=np.float64)
    enc.std = np.where(std < 1e-12, 1.0, std)
    enc.lower = ((raw.min(0) - enc.mean) / enc.std).astype(np.float32)
    enc.upper = ((raw.max(0) - enc.mean) / enc.std).astype(np.float32)

    src_tr, src_va, summaries = [], [], {}
    for data, c in zip(loaded, src_cfgs):
        cg = dataclasses.replace(c, **graph_params)
        splits, _, summ = prepare(data, cg, encoder=enc)
        src_tr.append(build_graph(splits["train"], enc, cg))
        src_va.append(build_graph(splits["val"], enc, cg))
        summaries[c.dataset] = summ
    tgt_data = load_dataset(tgt_cfg)
    tcfg = dataclasses.replace(tgt_cfg, **graph_params)
    tgt_splits, _, tsum = prepare(tgt_data, tcfg, encoder=enc)
    g_te = build_graph(tgt_splits["test"], enc, tcfg)
    summaries["target:" + tgt_cfg.dataset] = tsum
    summaries["target_test_graph"] = graph_stats(g_te)
    write_summary({"graph_params_from_source": graph_params, "features": enc.feature_names, **summaries},
                  out / "data_summary.json")

    g_tr = _concat_graphs(src_tr).to(device)
    g_va = _concat_graphs(src_va).to(device)
    g_te = g_te.to(device)

    rows = []
    for seed in seeds:
        cfg = dataclasses.replace(src_cfgs[0], **graph_params)
        set_seed(seed)
        model = build_model("artgl", g_tr.x.size(1), 2, cfg).to(device)
        model, _, info = train_graph_model(model, "artgl", g_tr, g_va, cfg, seed, True)
        logits, _ = infer(model, g_te, cfg.num_layers, cfg.eval_chunk_size)
        m = classification_metrics(g_te.y.cpu().numpy(), logits.argmax(1).cpu().numpy(), True)
        rows.append({"train": "+".join(c.dataset for c in src_cfgs), "test": tgt_cfg.dataset, "seed": seed,
                     **m, "source_val_f1": info["best_val_f1"]})
        print(rows[-1])
    pd.DataFrame(rows).to_csv(out / "cross_dataset_per_seed.csv", index=False)
    print(f"written {out / 'cross_dataset_per_seed.csv'}")


if __name__ == "__main__":
    main()
