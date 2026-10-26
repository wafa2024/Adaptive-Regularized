"""Turn per-seed CSVs into the manuscript tables (mean +- sample std over seeds).

    python aggregate.py --results results/unsw_nb15 [--cross results/cross_dataset]

Writes tables/*.csv (mean, std, n_seeds) and tables/*.tex (LaTeX rows ready to
paste). Delta_eps and the relative drops are computed per seed from that seed's
unrounded F1 values and then averaged, which is the convention the manuscript
states. The per-seed CSVs are themselves the supplementary material the reviewer
asked for.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt(mean, std, digits=3):
    if pd.isna(mean):
        return "--"
    return f"${mean:.{digits}f} \\pm {std:.{digits}f}$"


def summarize(df: pd.DataFrame, by: list[str], cols: list[str]) -> pd.DataFrame:
    g = df.groupby(by, sort=False)[cols]
    mean, std, n = g.mean(), g.std(ddof=1), g.count()
    out = mean.add_suffix("_mean").join(std.add_suffix("_std")).join(n[cols[:1]].rename(columns={cols[0]: "n_seeds"}))
    return out.reset_index()


LATEX_NAMES = {
    "w/o L_temp": r"w/o $\mathcal{L}_{temp}$",
    "w/o L_graph": r"w/o $\mathcal{L}_{graph}$",
    "w/o L_sparse": r"w/o $\mathcal{L}_{sparse}$",
    "w/o L_rob": r"w/o $\mathcal{L}_{rob}$",
    "Proposed w/o L_rob": r"Proposed w/o $\mathcal{L}_{rob}$",
    "lambda_temp": r"$\lambda_1$",
    "lambda_graph": r"$\lambda_2$",
    "lambda_sparse": r"$\lambda_3$",
    "lambda_robust": r"$\lambda_4$",
    "unsw_nb15": "UNSW-NB15",
    "cicids2017": "CICIDS2017",
    "cse_cicids2018": "CSE-CIC-IDS2018",
}


def _key(value, column: str = "") -> str:
    if isinstance(value, float) and column != "value":
        return f"{value:.2f}"
    if isinstance(value, float):
        exp = round(float(np.log10(value)), 6) if value > 0 else None
        if exp is not None and exp == int(exp):
            return "$1$" if exp == 0 else f"$10^{{{int(exp)}}}$"
        return f"${value:g}$"
    text = str(value)
    if "+" in text and "_" in text:
        return " + ".join(LATEX_NAMES.get(t, t) for t in text.split("+"))
    return LATEX_NAMES.get(text, text)


def to_latex_rows(summary: pd.DataFrame, by: list[str], cols: list[str], digits=3) -> str:
    lines = []
    for _, row in summary.iterrows():
        keys = [_key(row[b], b) for b in by]
        vals = [_fmt(row[f"{c}_mean"], row[f"{c}_std"], digits) for c in cols]
        lines.append(" & ".join(keys + vals) + r" \\ \hline")
    return "\n".join(lines) + "\n"


def write(summary, by, cols, out: Path, name: str, digits=3):
    summary.to_csv(out / f"{name}.csv", index=False)
    (out / f"{name}.tex").write_text(to_latex_rows(summary, by, cols, digits))
    print(f"  {name}: {len(summary)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--cross", default=None)
    args = ap.parse_args()
    res = Path(args.results)
    out = res / "tables"
    out.mkdir(exist_ok=True)

    jobs = [
        ("main_per_seed.csv", ["method"], ["accuracy", "precision", "recall", "f1"], "table3_main", 3),
        ("ablation_per_seed.csv", ["variant"], ["f1", "tcs", "gsi", "sr", "delta_epsilon"], "table5_ablation", 3),
        ("noise_per_seed.csv", ["epsilon"], ["clean_f1", "perturbed_f1", "delta_epsilon", "eff_l2_max"], "table6_noise", 4),
        ("adversarial_per_seed.csv", ["method"], ["clean_f1", "fgsm_f1", "fgsm_drop", "pgd_f1", "pgd_drop"], "table7_adversarial", 3),
        ("sensitivity_per_seed.csv", ["parameter", "value"], ["f1", "tcs", "gsi", "sr", "delta_epsilon"], "table8_sensitivity", 3),
        ("runtime_per_seed.csv", ["method"], ["train_time_per_epoch_s", "inference_ms_per_sample", "parameters"], "table11_runtime", 3),
    ]
    for fname, by, cols, name, digits in jobs:
        path = res / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        write(summarize(df, by, cols), by, cols, out, name, digits)
        if name == "table5_ablation":
            sub = df[df["variant"].isin(["Fixed-edge version", "Full model"])]
            write(summarize(sub, ["variant"], ["f1", "tcs", "gsi"]), ["variant"], ["f1", "tcs", "gsi"], out, "table4_adaptive_edges")

    if args.cross:
        frames = [pd.read_csv(p) for p in Path(args.cross).glob("*/cross_dataset_per_seed.csv")]
        if frames:
            df = pd.concat(frames)
            write(summarize(df, ["train", "test"], ["f1"]), ["train", "test"], ["f1"], out, "table9_cross_dataset")


if __name__ == "__main__":
    main()
