"""Model-level checks and an end-to-end smoke run (needs PyTorch + PyTorch Geometric).

    python tests/test_model_torch.py

Checks the properties the reviewer's comments depend on:
  1. L_graph equals (1/N) Tr(Z^T L~ Z) computed with a dense Laplacian;
  2. L_temp and L_graph (and TCS and GSI) are different quantities;
  3. chunked inference with halos equals full-graph inference exactly;
  4. the chunked input gradient equals the full-graph gradient;
  5. perturbations touch only continuous features and respect the budget;
  6. adaptive edge coefficients are symmetric;
then runs every experiment suite, aggregation, plotting and cross-dataset
evaluation on tiny synthetic data.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from batching import infer, loss_gradient  # noqa: E402
from config import ExperimentConfig  # noqa: E402
from datasets import load_dataset  # noqa: E402
from graph_building import build_graph  # noqa: E402
from losses import pairwise_energy  # noqa: E402
from metrics import diagnostics  # noqa: E402
from model import build_model  # noqa: E402
from preprocess import prepare  # noqa: E402
from robustness import fgsm, pgd  # noqa: E402
from perturbation import random_l2_perturbation  # noqa: E402
from synthetic import write_cicids2017, write_cse_cicids2018, write_unsw  # noqa: E402

TINY = dict(max_epochs=3, patience=2, batch_size=256, eval_chunk_size=300, noise_draws=2, seeds=[0, 1],
            pgd_steps=3)


def _graphs(tmp: Path, **kw):
    cfg = ExperimentConfig(dataset="unsw_nb15", data_paths=write_unsw(tmp), **TINY, **kw)
    splits, enc, _ = prepare(load_dataset(cfg), cfg)
    return cfg, {k: build_graph(v, enc, cfg) for k, v in splits.items()}


def check_model_properties():
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as d:
        cfg, graphs = _graphs(Path(d))
    g = graphs["test"]
    d_in = g.x.size(1)

    for kind in ["artgl", "gcn", "gat"]:
        model = build_model(kind, d_in, 2, cfg).eval()
        with torch.no_grad():
            full_logits, full_z, _ = model(g.x, g.edge_index, g.w0)
        logits, z = infer(model, g, cfg.num_layers, chunk=137)
        assert torch.allclose(logits, full_logits, atol=1e-5), f"{kind}: chunked inference differs"
        assert torch.allclose(z, full_z, atol=1e-5)

        x = g.x.clone().requires_grad_(True)
        out, _, _ = model(x, g.edge_index, g.w0)
        (ref,) = torch.autograd.grad(torch.nn.functional.cross_entropy(out, g.y, reduction="sum"), x)
        assert torch.allclose(loss_gradient(model, g, cfg.num_layers, 137, g.x), ref, atol=1e-5), f"{kind}: gradient"
    print("PASS chunked inference and gradients are exact (ARTGL, GCN, GAT)")

    model = build_model("artgl", d_in, 2, cfg).eval()
    with torch.no_grad():
        logits, z, w_eff = model(g.x, g.edge_index, g.w0)
        n = g.num_nodes
        W = torch.zeros(n, n)
        W[g.edge_index[0], g.edge_index[1]] = w_eff
        W = W + W.T
        L = torch.diag(W.sum(1)) - W
        trace = torch.trace(z.T @ L @ z)
        assert torch.allclose(pairwise_energy(z, g.edge_index, w_eff), trace, rtol=1e-4)
        l_graph = trace / n
        l_temp = pairwise_energy(z, g.tedge_index, g.tw) / n
        assert not torch.allclose(l_temp, 2 * l_graph, rtol=1e-2), "L_temp must not be 2 * L_graph"
        diag = diagnostics(model, g, z, "artgl", cfg.sparsity_threshold)
        assert abs(diag["gsi"] - diag["tcs"] / (2 * n)) > 1e-6, "GSI must not equal TCS / (2N)"
        a_ij = model.edge_coefficients(g.x, g.edge_index)
        a_ji = model.edge_coefficients(g.x, g.edge_index.flip(0))
        assert torch.allclose(a_ij, a_ji, atol=1e-6)
    print("PASS L_graph = Tr(Z^T L Z)/N; L_temp and L_graph differ; GSI != TCS/(2N); a_ij symmetric")

    mask = g.continuous_mask
    delta = random_l2_perturbation(g.x, 0.2, mask, g.lower, g.upper)
    assert delta[:, ~mask].abs().max() == 0
    assert delta.norm(dim=1).max() <= 0.2 + 1e-5
    lin = dataclasses.replace(cfg, adv_epsilon=0.05)
    for x_adv in [fgsm(model, g, lin, 0.05), pgd(model, g, lin, 0.05, 0.01, 3)]:
        diff = x_adv - g.x
        assert diff[:, ~mask].abs().max() == 0
        assert diff.abs().max() <= 0.05 + 1e-6
        lo = torch.minimum(g.lower.unsqueeze(0), g.x)
        hi = torch.maximum(g.upper.unsqueeze(0), g.x)
        assert (x_adv >= lo - 1e-6).all() and (x_adv <= hi + 1e-6).all()
    print("PASS perturbations: continuous features only, budgets respected, valid range respected")


def check_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "unsw").mkdir()
        (d / "cic17").mkdir()
        (d / "cic18").mkdir()
        cfgs = {}
        for name, writer, folder, window in [("unsw_nb15", write_unsw, "unsw", 30.0),
                                             ("cicids2017", write_cicids2017, "cic17", 60.0),
                                             ("cse_cicids2018", write_cse_cicids2018, "cic18", 60.0)]:
            path = d / f"{name}.json"
            path.write_text(json.dumps({"dataset": name, "data_paths": writer(d / folder), "temporal_window": window,
                                        "output_dir": str(d / "results"), **TINY}))
            cfgs[name] = path

        def run(*args):
            subprocess.run([sys.executable, *args], cwd=ROOT, check=True)

        run("experiments.py", "--config", str(cfgs["unsw_nb15"]), "--seeds", "0", "1")
        run("aggregate.py", "--results", str(d / "results" / "unsw_nb15"))
        tables = d / "results" / "unsw_nb15" / "tables"
        for t in ["table3_main", "table4_adaptive_edges", "table5_ablation", "table6_noise",
                  "table7_adversarial", "table8_sensitivity", "table11_runtime"]:
            assert (tables / f"{t}.csv").exists() and (tables / f"{t}.tex").exists(), t
        run("plots.py", "--tables", str(tables), "--dataset", "UNSW-NB15")
        run("cross_dataset.py", "--train", str(cfgs["unsw_nb15"]), "--test", str(cfgs["cicids2017"]),
            "--seeds", "0", "--output_dir", str(d / "results" / "cross"))
        run("cross_dataset.py", "--train", str(cfgs["unsw_nb15"]), str(cfgs["cicids2017"]),
            "--test", str(cfgs["cse_cicids2018"]), "--seeds", "0", "--output_dir", str(d / "results" / "cross"))
        run("aggregate.py", "--results", str(d / "results" / "unsw_nb15"), "--cross", str(d / "results" / "cross"))
        assert (tables / "table9_cross_dataset.csv").exists()

        # the full model is the same cached run in Tables 2, 4 and 6
        import pandas as pd
        main = pd.read_csv(d / "results" / "unsw_nb15" / "main_per_seed.csv")
        abl = pd.read_csv(d / "results" / "unsw_nb15" / "ablation_per_seed.csv")
        adv = pd.read_csv(d / "results" / "unsw_nb15" / "adversarial_per_seed.csv")
        f_main = main[main.method == "Proposed Method"].sort_values("seed").f1.to_numpy()
        f_abl = abl[abl.variant == "Full model"].sort_values("seed").f1.to_numpy()
        f_adv = adv[adv.method == "Proposed Method"].sort_values("seed").clean_f1.to_numpy()
        assert (f_main == f_abl).all() and (f_main == f_adv).all()
        print("PASS end-to-end: all suites, aggregation, figures and cross-dataset runs complete;")
        print("     the full model contributes identical per-seed values to Tables 2, 4 and 6")
        print((tables / "table5_ablation.tex").read_text())


if __name__ == "__main__":
    check_model_properties()
    check_end_to_end()
