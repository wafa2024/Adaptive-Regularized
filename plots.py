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
                     "mathtext.fontset": "stix", "font.size": 9})
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def plot_classification(tables: Path, dataset: str, out: Path) -> None:
    df = pd.read_csv(tables / "table3_main.csv")
    metrics = ["accuracy", "precision", "recall", "f1"]
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, m in enumerate(metrics):
        ax.errorbar(df[f"{m}_mean"], y + (i - 1.5) * 0.12, xerr=df[f"{m}_std"], fmt="o", capsize=3,
                    label=m.capitalize() if m != "f1" else "F1-score")
    ax.set_yticks(y)
    ax.set_yticklabels(df["method"])
    ax.set_xlabel("Score")
    ax.set_title(f"Classification performance on {dataset}")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / f"{dataset}_classification_results.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_ablation(tables: Path, out: Path) -> None:
    df = pd.read_csv(tables / "table5_ablation.csv")
    metrics = [("f1", "F1-score"), ("tcs", "TCS"), ("gsi", "GSI"), ("sr", "SR"), ("delta_epsilon", r"$\Delta_{\epsilon}$")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    axes = axes.flatten()
    for ax, (m, label) in zip(axes, metrics):
        pos = np.arange(len(df))
        ax.barh(pos, df[f"{m}_mean"], xerr=df[f"{m}_std"], capsize=3)
        ax.set_yticks(pos)
        ax.set_yticklabels(df["variant"], fontsize=8)
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
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=True)
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
