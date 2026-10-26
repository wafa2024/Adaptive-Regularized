"""Run every experiment reported in the manuscript and save per-seed results.

    python experiments.py --config configs/unsw_nb15.json --suites main ablation adversarial sensitivity
    python aggregate.py --results results/unsw_nb15

Suites -> tables of the revised manuscript
    main         Table 3 (classification) and Table 11 (runtime)
    ablation     Tables 4 and 5 (adaptive edges, regularisation ablation)
    noise        Table 6 (bounded random noise, full model)
    adversarial  Table 7 (FGSM/PGD: GCN, GAT, ARTGL, ARTGL without L_rob)
    sensitivity  Table 8 (one-at-a-time sweep of lambda_1..lambda_4)
(cross_dataset.py -> Table 9; data_summary.json -> Table 2)

Each (model, configuration, seed) is trained once. Its weights and every metric
computed from it are cached under <output>/runs/, so a model that appears in
several tables (e.g. the full ARTGL model) contributes identical numbers to all
of them.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch

from baselines import run_classical, run_neural
from batching import infer
from config import ExperimentConfig
from datasets import load_dataset
from graph_building import build_graph, graph_stats
from metrics import classification_metrics, diagnostics
from model import build_model
from preprocess import prepare, write_summary
from robustness import evaluate_adversarial, evaluate_noise
from train import resolve_device, set_seed, train_graph_model

DATA_FIELDS = ["dataset", "data_paths", "feature_set", "label_mode", "label_col", "time_col", "src_col",
               "dst_col", "protocol_col", "drop_cols", "discrete_cols", "benign_values", "split",
               "train_frac", "val_frac", "split_seed", "temporal_window", "max_neighbors",
               "similarity_threshold", "alpha", "beta", "temporal_key"]
TRAIN_FIELDS = ["hidden_dim", "num_layers", "dropout", "adaptive_edges", "learning_rate", "weight_decay",
                "max_epochs", "patience", "batch_size", "lambda_temp", "lambda_graph", "lambda_sparse",
                "lambda_robust", "train_epsilon"]

ABLATIONS = {
    "Full model": {},
    "w/o L_temp": {"lambda_temp": 0.0},
    "w/o L_graph": {"lambda_graph": 0.0},
    "w/o L_sparse": {"lambda_sparse": 0.0},
    "w/o L_rob": {"lambda_robust": 0.0},
    "Fixed-edge version": {"adaptive_edges": False},
}
LAMBDA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def _hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
def load_or_build_data(cfg: ExperimentConfig, out: Path, log=print):
    key = _hash({k: getattr(cfg, k) for k in DATA_FIELDS})
    cache_root = Path(os.environ.get("ARTGL_CACHE_DIR", out / "cache"))
    cache = cache_root / cfg.dataset / f"data_{key}.pt"
    if cache.exists():
        log(f"Loading cached data/graphs: {cache}")
        bundle = torch.load(cache, weights_only=False)
        for name, sp in bundle["splits"].items():       # x is stored once, inside the graph
            sp.x = bundle["graphs"][name].x.numpy()
        return bundle

    log("Loading dataset ...")
    data = load_dataset(cfg)
    splits, encoder, summary = prepare(data, cfg)
    log("Building graphs ...")
    graphs = {name: build_graph(sp, encoder, cfg) for name, sp in splits.items()}
    summary["graphs"] = {name: graph_stats(g) for name, g in graphs.items()}
    write_summary(summary, out / "data_summary.json")
    bundle = {"splits": splits, "encoder": encoder, "graphs": graphs, "summary": summary, "data_key": key}
    cache.parent.mkdir(parents=True, exist_ok=True)
    slim = dict(bundle, splits={n: dataclasses.replace(sp, x=None) for n, sp in splits.items()})
    torch.save(slim, cache)
    return bundle


# ---------------------------------------------------------------------------
class Runner:
    def __init__(self, cfg: ExperimentConfig, bundle: dict, out: Path, device, log=print):
        self.cfg, self.bundle, self.out, self.device, self.log = cfg, bundle, out, device, log
        self.binary = cfg.label_mode == "binary"
        self.graphs = {k: g.to(device) for k, g in bundle["graphs"].items()}
        (out / "runs").mkdir(parents=True, exist_ok=True)

    def graph_run(self, kind: str, overrides: dict, seed: int, need_adv: bool = False) -> dict:
        cfg = dataclasses.replace(self.cfg, **overrides)
        if kind != "artgl":  # baselines ignore the regularisation settings
            cfg = dataclasses.replace(cfg, lambda_temp=0.0, lambda_graph=0.0, lambda_sparse=0.0,
                                      lambda_robust=0.0, adaptive_edges=False)
        key = _hash({"kind": kind, "data": self.bundle["data_key"], "seed": seed,
                     **{k: getattr(cfg, k) for k in TRAIN_FIELDS}})
        res_path = self.out / "runs" / f"{key}.json"
        model_path = self.out / "runs" / f"{key}.pt"
        result = json.loads(res_path.read_text()) if res_path.exists() else None
        if result is not None and (not need_adv or "adversarial" in result):
            return result

        g_tr, g_va, g_te = self.graphs["train"], self.graphs["val"], self.graphs["test"]
        n_classes = len(self.bundle["encoder"].class_names)
        set_seed(seed)
        model = build_model(kind, g_tr.x.size(1), n_classes, cfg).to(self.device)

        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            self.log(f"  training {kind} {overrides or ''} seed={seed}")
            ckpt = self.out / "runs" / f"{key}_checkpoint.pt"
            model, history, info = train_graph_model(model, kind, g_tr, g_va, cfg, seed, self.binary, self.log,
                                                     checkpoint=ckpt)
            torch.save(model.state_dict(), model_path)
            pd.DataFrame(history).to_csv(self.out / "runs" / f"{key}_history.csv", index=False)
            result = {"kind": kind, "overrides": overrides, "seed": seed, "config": cfg.to_dict(), **info}

        if result is None:
            result = {"kind": kind, "overrides": overrides, "seed": seed, "config": cfg.to_dict()}

        if "test" not in result:
            t0 = time.perf_counter()
            logits, z = infer(model, g_te, cfg.num_layers, cfg.eval_chunk_size)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            result["inference_ms_per_sample"] = 1000 * (time.perf_counter() - t0) / g_te.num_nodes
            result["parameters"] = int(sum(p.numel() for p in model.parameters()))
            result["test"] = classification_metrics(g_te.y.cpu().numpy(), logits.argmax(1).cpu().numpy(), self.binary)
            result["diagnostics"] = diagnostics(model, g_te, z, kind, cfg.sparsity_threshold)
            result["noise"] = evaluate_noise(model, g_te, cfg, result["test"]["f1"], self.binary, seed)
        if need_adv and "adversarial" not in result:
            result["adversarial"] = evaluate_adversarial(model, g_te, cfg, result["test"]["f1"], self.binary, seed)

        res_path.write_text(json.dumps(result, indent=2, default=str))
        ckpt = self.out / "runs" / f"{key}_checkpoint.pt"
        if ckpt.exists():
            ckpt.unlink()
        return result

    def tabular_run(self, name: str, seed: int) -> dict:
        key = _hash({"tabular": name, "data": self.bundle["data_key"], "seed": seed,
                     **{k: getattr(self.cfg, k) for k in TRAIN_FIELDS}})
        path = self.out / "runs" / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())
        self.log(f"  training {name} seed={seed}")
        sp = self.bundle["splits"]
        set_seed(seed)
        if name in ("logreg", "rf", "xgb"):
            m = run_classical(name, sp["train"].x, sp["train"].y, sp["test"].x, sp["test"].y, seed, self.binary)
        else:
            m = run_neural(name, sp, self.cfg, seed, self.binary, self.device)
        result = {"kind": name, "seed": seed, "test": m}
        path.write_text(json.dumps(result, indent=2))
        return result


# ---------------------------------------------------------------------------
METHOD_NAMES = {"logreg": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost", "mlp": "MLP",
                "lstm": "LSTM", "gcn": "GCN", "gat": "GAT", "artgl": "Proposed Method"}


def _eps_row(result: dict, eps: float) -> dict:
    for row in result["noise"]:
        if abs(row["epsilon"] - eps) < 1e-12:
            return row
    raise KeyError(f"epsilon {eps} not in noise_epsilons")


def suite_main(r: Runner, seeds):
    rows, runtime = [], []
    for seed in seeds:
        for name in ["logreg", "rf", "xgb", "mlp", "lstm"]:
            res = r.tabular_run(name, seed)
            rows.append({"method": METHOD_NAMES[name], "seed": seed, **res["test"]})
        for kind in ["gcn", "gat", "artgl"]:
            res = r.graph_run(kind, {}, seed)
            rows.append({"method": METHOD_NAMES[kind], "seed": seed, **res["test"]})
            runtime.append({"method": METHOD_NAMES[kind], "seed": seed,
                            "train_time_per_epoch_s": res.get("train_time_per_epoch_s"),
                            "inference_ms_per_sample": res["inference_ms_per_sample"],
                            "parameters": res["parameters"], "epochs_run": res.get("epochs_run")})
    pd.DataFrame(rows).to_csv(r.out / "main_per_seed.csv", index=False)
    pd.DataFrame(runtime).to_csv(r.out / "runtime_per_seed.csv", index=False)


def suite_ablation(r: Runner, seeds):
    rows = []
    for seed in seeds:
        for variant, ov in ABLATIONS.items():
            res = r.graph_run("artgl", ov, seed)
            d = res["diagnostics"]
            rows.append({"variant": variant, "seed": seed, "f1": res["test"]["f1"], "tcs": d["tcs"],
                         "gsi": d["gsi"], "gsi_adaptive": d.get("gsi_adaptive"), "sr": d.get("sr"),
                         "delta_epsilon": _eps_row(res, r.cfg.ablation_epsilon)["delta_epsilon"]})
    pd.DataFrame(rows).to_csv(r.out / "ablation_per_seed.csv", index=False)


def suite_noise(r: Runner, seeds):
    rows = []
    for seed in seeds:
        res = r.graph_run("artgl", {}, seed)
        for row in res["noise"]:
            rows.append({"seed": seed, **{k: v for k, v in row.items() if k != "perturbed_f1_draws"}})
    pd.DataFrame(rows).to_csv(r.out / "noise_per_seed.csv", index=False)


def suite_adversarial(r: Runner, seeds):
    rows = []
    for seed in seeds:
        for label, kind, ov in [("GCN", "gcn", {}), ("GAT", "gat", {}),
                                ("Proposed Method", "artgl", {}), ("Proposed w/o L_rob", "artgl", {"lambda_robust": 0.0})]:
            res = r.graph_run(kind, ov, seed, need_adv=True)
            rows.append({"method": label, "seed": seed, **res["adversarial"]})
    pd.DataFrame(rows).to_csv(r.out / "adversarial_per_seed.csv", index=False)


def suite_sensitivity(r: Runner, seeds):
    rows = []
    for which in ["lambda_temp", "lambda_graph", "lambda_sparse", "lambda_robust"]:
        for value in LAMBDA_GRID:
            for seed in seeds:
                res = r.graph_run("artgl", {which: value}, seed)
                d = res["diagnostics"]
                rows.append({"parameter": which, "value": value, "seed": seed, "f1": res["test"]["f1"],
                             "tcs": d["tcs"], "gsi": d["gsi"], "sr": d.get("sr"),
                             "delta_epsilon": _eps_row(res, r.cfg.ablation_epsilon)["delta_epsilon"]})
    pd.DataFrame(rows).to_csv(r.out / "sensitivity_per_seed.csv", index=False)


SUITES = {"main": suite_main, "ablation": suite_ablation, "noise": suite_noise,
          "adversarial": suite_adversarial, "sensitivity": suite_sensitivity}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--suites", nargs="+", default=["main", "ablation", "noise", "adversarial", "sensitivity"])
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--data_paths", nargs="+", default=None, help="override data_paths in the config")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = ExperimentConfig.from_json(args.config, data_paths=args.data_paths, output_dir=args.output_dir,
                                     device=args.device)
    seeds = args.seeds or cfg.seeds
    out = Path(cfg.output_dir) / cfg.dataset
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "config_used.json")

    device = resolve_device(cfg.device)
    bundle = load_or_build_data(cfg, out)
    runner = Runner(cfg, bundle, out, device)
    for suite in args.suites:
        print(f"== suite: {suite}")
        SUITES[suite](runner, seeds)
    print(f"Per-seed results written to {out}. Next: python aggregate.py --results {out}")


if __name__ == "__main__":
    main()
