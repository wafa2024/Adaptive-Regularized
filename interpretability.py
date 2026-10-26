"""Interpretability analysis of trained models (revised manuscript, Section "Interpretability").

Uses only models that experiments.py has already trained and saved under <output>/runs/;
nothing is retrained. For one dataset it writes, under <output>/interpretability/:

  tsne_points.csv / tsne.pdf
      t-SNE of a class-stratified sample of test flows: the standardized input features,
      and the learned embeddings z of GAT, the fixed-edge model and the proposed model
      (seed 0). Each point keeps its attack category.
  separability_per_seed.csv
      Quantitative counterpart of the t-SNE plot, computed in the original (not the
      t-SNE) space on the same sample, for every seed: silhouette score of the binary
      label (cosine distance) and the share of the 10 nearest neighbours with the same
      binary label.
  edge_coefficients_per_seed.csv / edge_coefficients.pdf
      Learned coefficients a_ij of the proposed model on the test graph, grouped by the
      labels of the two endpoints (benign-benign, attack-attack same category,
      attack-attack different category, benign-attack), and by whether the pair is
      also a temporal edge.
  phi_feature_weights.csv / phi_features.pdf
      |phi| of the edge scorer summed over its three blocks (x_i+x_j, |x_i-x_j|,
      x_i*x_j) per input feature (one-hot columns summed per attribute), per seed.
  noise_extended_per_seed.csv
      Bounded random noise at larger radii than in the main robustness table, for the
      proposed model, the model without L_rob, GCN and GAT.

    python interpretability.py --config configs/unsw_nb15.json [--output_dir results]
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
from experiments import TRAIN_FIELDS, _hash, load_or_build_data
from metrics import classification_metrics
from model import build_model
from robustness import evaluate_noise
from train import resolve_device, set_seed

EXTENDED_EPS = [0.5, 1.0, 2.0, 5.0]


def load_trained(cfg, bundle, out: Path, kind: str, overrides: dict, seed: int, device):
    """Load a model trained by experiments.py (same key as Runner.graph_run)."""

    cfg = dataclasses.replace(cfg, **overrides)
    if kind != "artgl":
        cfg = dataclasses.replace(cfg, lambda_temp=0.0, lambda_graph=0.0, lambda_sparse=0.0,
                                  lambda_robust=0.0, adaptive_edges=False)
    key = _hash({"kind": kind, "data": bundle["data_key"], "seed": seed,
                 **{k: getattr(cfg, k) for k in TRAIN_FIELDS}})
    path = out / "runs" / f"{key}.pt"
    if not path.exists():
        return None, cfg
    g = bundle["graphs"]["train"]
    set_seed(seed)
    model = build_model(kind, g.x.size(1), len(bundle["encoder"].class_names), cfg).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model, cfg


def stratified_sample(raw_labels: np.ndarray, y: np.ndarray, n_per_class: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        members = np.nonzero(y == c)[0]
        idx.append(rng.choice(members, size=min(n_per_class, len(members)), replace=False))
    return np.sort(np.concatenate(idx))


def separability(z: np.ndarray, y: np.ndarray, k: int = 10) -> dict:
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors

    zn = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    sil = float(silhouette_score(zn, y, metric="cosine"))
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(zn)
    _, nbr = nn.kneighbors(zn)
    knn = float((y[nbr[:, 1:]] == y[:, None]).mean())
    return {"silhouette": sil, "knn_label_agreement": knn}


@torch.no_grad()
def edge_coefficient_rows(model, g, raw_labels: np.ndarray, seed: int, chunk: int = 200_000) -> list[dict]:
    ei = g.edge_index
    a = torch.cat([model.edge_coefficients(g.x, ei[:, s:s + chunk]) for s in range(0, ei.size(1), chunk)]).cpu().numpy()
    i, j = ei[0].cpu().numpy(), ei[1].cpu().numpy()
    yi, yj = g.y.cpu().numpy()[i], g.y.cpu().numpy()[j]
    ci, cj = raw_labels[i], raw_labels[j]
    # pairs that are also temporal edges
    te = g.tedge_index.cpu().numpy()
    n = g.num_nodes
    tkeys = np.minimum(te[0], te[1]).astype(np.int64) * n + np.maximum(te[0], te[1])
    ekeys = np.minimum(i, j).astype(np.int64) * n + np.maximum(i, j)
    is_temporal = np.isin(ekeys, tkeys)
    groups = {
        "benign-benign": (yi == 0) & (yj == 0),
        "attack-attack, same category": (yi == 1) & (yj == 1) & (ci == cj),
        "attack-attack, different category": (yi == 1) & (yj == 1) & (ci != cj),
        "benign-attack": yi != yj,
        "also a temporal edge": is_temporal,
        "relational only": ~is_temporal,
        "all edges": np.ones_like(is_temporal),
    }
    w0 = g.w0.cpu().numpy()
    rows = []
    for name, m in groups.items():
        if m.sum() == 0:
            continue
        rows.append({"seed": seed, "group": name, "edges": int(m.sum()), "share_of_edges": float(m.mean()),
                     "mean_a": float(a[m].mean()), "median_a": float(np.median(a[m])),
                     "p10_a": float(np.quantile(a[m], 0.1)), "p90_a": float(np.quantile(a[m], 0.9)),
                     "share_of_effective_weight": float((a[m] * w0[m]).sum() / (a * w0).sum())})
    return rows


def phi_rows(model, feature_names: list[str], seed: int) -> list[dict]:
    w = model.phi.weight.detach().cpu().numpy().reshape(3, -1)       # blocks: sum, |diff|, product
    imp = np.abs(w).sum(axis=0)
    attr = [f.split("=", 1)[0] for f in feature_names]
    df = pd.DataFrame({"attribute": attr, "importance": imp,
                       "sum_block": np.abs(w[0]), "absdiff_block": np.abs(w[1]), "product_block": np.abs(w[2])})
    agg = df.groupby("attribute", sort=False).sum().reset_index()
    agg["share"] = agg.importance / agg.importance.sum()
    agg["seed"] = seed
    return agg.to_dict("records")


def plot_tsne(points: pd.DataFrame, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
                         "mathtext.fontset": "stix", "font.size": 9})
    panels = list(dict.fromkeys(points.panel))
    cats = [c for c in points.category.value_counts().index if c != "benign"]
    cmap = plt.get_cmap("tab10")
    colors = {"benign": "#b0b0b0", **{c: cmap(k % 10) for k, c in enumerate(cats)}}
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.2))
    for ax, panel in zip(np.atleast_1d(axes), panels):
        p = points[points.panel == panel]
        for c in ["benign"] + cats:
            q = p[p.category == c]
            ax.scatter(q.x, q.y, s=1.5, c=[colors[c]], label=c, alpha=0.6 if c == "benign" else 0.8, linewidths=0)
        ax.set_title(panel)
        ax.set_xticks([]); ax.set_yticks([])
    handles = [plt.Line2D([], [], marker="o", ls="", color=colors[c], label=c) for c in ["benign"] + cats]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), frameon=False, markerscale=1.2)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.12 if len(cats) > 5 else 0.08, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def plot_edges(df: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
                         "mathtext.fontset": "stix", "font.size": 9})
    order = ["benign-benign", "attack-attack, same category", "attack-attack, different category", "benign-attack",
             "also a temporal edge", "relational only"]
    s = df[df.group.isin(order)].groupby("group").mean_a.agg(["mean", "std"]).reindex(order).dropna()
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.barh(range(len(s)), s["mean"], xerr=s["std"], color="#4c72b0")
    ax.set_yticks(range(len(s)), s.index)
    ax.invert_yaxis()
    ax.set_xlabel(r"mean learned coefficient $a_{ij}$ (test graph)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_phi(df: pd.DataFrame, path: Path, top: int = 15) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
                         "mathtext.fontset": "stix", "font.size": 9})
    s = df.groupby("attribute").share.agg(["mean", "std"]).sort_values("mean", ascending=False).head(top)
    fig, ax = plt.subplots(figsize=(4.6, 0.22 * len(s) + 0.8))
    ax.barh(range(len(s)), s["mean"], xerr=s["std"], color="#4c72b0")
    ax.set_yticks(range(len(s)), s.index)
    ax.invert_yaxis()
    ax.set_xlabel(r"share of $\|\phi\|_1$ (mean over seeds)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--per_class", type=int, default=3000, help="test flows per binary class in the t-SNE sample")
    ap.add_argument("--skip_noise", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = ExperimentConfig.from_json(args.config, output_dir=args.output_dir, device=args.device)
    out = Path(cfg.output_dir) / cfg.dataset
    dest = out / "interpretability"
    dest.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    bundle = load_or_build_data(cfg, out)
    g = bundle["graphs"]["test"].to(device)
    raw = np.asarray(bundle["splits"]["test"].raw_label).astype(str)
    raw = np.char.lower(np.char.strip(raw))
    y = g.y.cpu().numpy()
    binary = cfg.label_mode == "binary"
    names = bundle["encoder"].feature_names

    variants = {"GCN": ("gcn", {}), "GAT": ("gat", {}), "Fixed-edge model": ("artgl", {"adaptive_edges": False}),
                "Proposed model": ("artgl", {}), "Proposed w/o L_rob": ("artgl", {"lambda_robust": 0.0})}

    sample = stratified_sample(raw, y, args.per_class)
    pd.DataFrame({"index": sample, "label": y[sample], "category": raw[sample]}).to_csv(dest / "tsne_sample.csv", index=False)

    sep_rows, edge_rows, phi_rec, noise_rows, tsne_inputs = [], [], [], [], {}
    x_sample = g.x[torch.as_tensor(sample, device=device)].cpu().numpy()
    sep_rows.append({"model": "Input features", "seed": -1, **separability(x_sample, y[sample])})
    tsne_inputs["Input features"] = x_sample

    for seed in args.seeds:
        for label, (kind, ov) in variants.items():
            model, mcfg = load_trained(cfg, bundle, out, kind, ov, seed, device)
            if model is None:
                print(f"  {label} seed {seed}: no trained model found - skipped")
                continue
            logits, z = infer(model, g, mcfg.num_layers, mcfg.eval_chunk_size)
            f1 = classification_metrics(y, logits.argmax(1).cpu().numpy(), binary)["f1"]
            zs = z[torch.as_tensor(sample, device=device)].cpu().numpy()
            if label != "Proposed w/o L_rob":
                sep_rows.append({"model": label, "seed": seed, "test_f1": f1, **separability(zs, y[sample])})
                if seed == args.seeds[0] and label != "GCN":
                    tsne_inputs[label] = zs
            if label == "Proposed model":
                edge_rows += edge_coefficient_rows(model, g, raw, seed)
                phi_rec += phi_rows(model, names, seed)
            if not args.skip_noise:
                ncfg = dataclasses.replace(mcfg, noise_epsilons=EXTENDED_EPS)
                for row in evaluate_noise(model, g, ncfg, f1, binary, seed):
                    noise_rows.append({"model": label, "seed": seed,
                                       **{k: v for k, v in row.items() if k != "perturbed_f1_draws"}})
            print(f"  {label} seed {seed}: done (test F1 {f1:.4f})")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    pd.DataFrame(sep_rows).to_csv(dest / "separability_per_seed.csv", index=False)
    if edge_rows:
        pd.DataFrame(edge_rows).to_csv(dest / "edge_coefficients_per_seed.csv", index=False)
        plot_edges(pd.DataFrame(edge_rows), dest / "edge_coefficients.pdf")
    if phi_rec:
        pd.DataFrame(phi_rec).to_csv(dest / "phi_feature_weights.csv", index=False)
        plot_phi(pd.DataFrame(phi_rec), dest / "phi_features.pdf")
    if noise_rows:
        pd.DataFrame(noise_rows).to_csv(dest / "noise_extended_per_seed.csv", index=False)

    from sklearn.manifold import TSNE
    pts = []
    for panel, feats in tsne_inputs.items():
        emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(feats)
        pts.append(pd.DataFrame({"panel": panel, "x": emb[:, 0], "y": emb[:, 1], "label": y[sample], "category": raw[sample]}))
    pts = pd.concat(pts, ignore_index=True)
    pts.to_csv(dest / "tsne_points.csv", index=False)
    plot_tsne(pts, f"t-SNE of test flows ({cfg.dataset}, seed {args.seeds[0]})", dest / "tsne.pdf")
    (dest / "README.json").write_text(json.dumps({"seeds": args.seeds, "per_class": args.per_class,
                                                   "extended_noise_epsilons": EXTENDED_EPS}, indent=2))
    print(f"Interpretability results written to {dest}")


if __name__ == "__main__":
    main()
