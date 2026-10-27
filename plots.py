"""Manuscript figures, drawn only from the aggregated tables written by aggregate.py.

    python plots.py --tables results/unsw_nb15/tables --dataset UNSW-NB15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Times-style fonts for every figure (STIX ships with matplotlib, so this works without Times New Roman)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8,
                     "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "lines.markersize": 4})
# Figures are drawn at their printed size (full text width = 6.5 in), so LaTeX does not shrink the text.
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def plot_classification(tables: Path, dataset: str, out: Path) -> None:
    df = pd.read_csv(tables / "table3_main.csv")
    metrics = ["accuracy", "precision", "recall", "f1"]
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    for i, m in enumerate(metrics):
        ax.errorbar(df[f"{m}_mean"], y + (i - 1.5) * 0.16, xerr=df[f"{m}_std"], fmt="o", capsize=2, markersize=3.5,
                    label=m.capitalize() if m != "f1" else "F1-score")
    ax.set_yticks(y)
    ax.set_yticklabels(df["method"])
    ax.set_xlabel("Score")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(out / f"{dataset}_classification_results.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_ablation(tables: Path, out: Path) -> None:
    df = pd.read_csv(tables / "table5_ablation.csv")
    metrics = [("f1", "F1-score"), ("tcs", "TCS"), ("gsi", "GSI"), ("sr", "SR"), ("delta_epsilon", r"$\Delta_{\epsilon}$")]
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.2))
    axes = axes.flatten()
    for ax, (m, label) in zip(axes, metrics):
        pos = np.arange(len(df))
        ax.barh(pos, df[f"{m}_mean"], xerr=df[f"{m}_std"], capsize=2, color="#4c72b0", error_kw={"linewidth": 0.8})
        ax.set_yticks(pos)
        ax.set_yticklabels([v.replace("L_", "$\\mathcal{L}_{").replace("temp", "temp}$").replace("graph", "graph}$").replace("sparse", "sparse}$").replace("rob", "rob}$") if "L_" in v else v for v in df["variant"]], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.grid(True, axis="x", alpha=0.25)
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(out / "ablation_impact.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_robustness(tables: Path, out: Path) -> None:
    df = pd.read_csv(tables / "table6_noise.csv")
    eps = df["epsilon"]
    fig, axes = plt.subplots(2, 1, figsize=(3.3, 3.6), sharex=True)
    axes[0].errorbar(eps, df["clean_f1_mean"], yerr=df["clean_f1_std"], marker="o", capsize=3, label="Clean F1")
    axes[0].errorbar(eps, df["perturbed_f1_mean"], yerr=df["perturbed_f1_std"], marker="o", capsize=3, label="Perturbed F1")
    axes[0].set_ylabel("F1-score")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)
    axes[1].errorbar(eps, df["delta_epsilon_mean"], yerr=df["delta_epsilon_std"], marker="o", capsize=3)
    axes[1].set_xlabel(r"Perturbation radius $\epsilon$")
    axes[1].set_ylabel(r"$\Delta_{\epsilon}$")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "robustness_degradation.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    tables = Path(args.tables)
    out = tables.parent / "figures"
    out.mkdir(exist_ok=True)
    for fn, call in [("table3_main.csv", lambda: plot_classification(tables, args.dataset, out)),
                     ("table5_ablation.csv", lambda: plot_ablation(tables, out)),
                     ("table6_noise.csv", lambda: plot_robustness(tables, out))]:
        if (tables / fn).exists():
            call()
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
