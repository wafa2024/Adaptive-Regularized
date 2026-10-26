"""Single ARTGL run (one seed) with every metric, for quick checks.

    python run_artgl.py --config configs/unsw_nb15.json --seed 0

The manuscript tables come from experiments.py and aggregate.py. This script
uses the same code path and the same run cache, so a run made here is reused by
experiments.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import ExperimentConfig
from experiments import Runner, load_or_build_data
from train import resolve_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_paths", nargs="+", default=None)
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()

    cfg = ExperimentConfig.from_json(args.config, data_paths=args.data_paths, output_dir=args.output_dir)
    out = Path(cfg.output_dir) / cfg.dataset
    out.mkdir(parents=True, exist_ok=True)
    bundle = load_or_build_data(cfg, out)
    runner = Runner(cfg, bundle, out, resolve_device(cfg.device))
    result = runner.graph_run("artgl", {}, args.seed, need_adv=True)
    print(json.dumps({k: result[k] for k in ["test", "diagnostics", "noise", "adversarial"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
