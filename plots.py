from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_classification_results(results: pd.DataFrame, dataset_name: str, output_path: str | Path) -> None:
    """Create a horizontal metric plot for the main classification comparison."""

    metrics = ["accuracy", "precision", "recall", "f1"]
    y = np.arange(len(results))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for metric in metrics:
        ax.plot(results[metric], y, marker="o", label=metric.capitalize())

    ax.set_yticks(y)
    ax.set_yticklabels(results["method"])
    ax.set_xlabel("Score")
    ax.set_title(f"Classification performance on {dataset_name}")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_impact(ablation_df: pd.DataFrame, output_path: str | Path) -> None:
    """Create the five-panel ablation figure used in the manuscript."""

    metrics = [
        ("f1", "F1-score"),
        ("tcs", "TCS"),
        ("gsi", "GSI"),
        ("sr", "SR"),
        ("delta_epsilon", r"$\Delta_{\epsilon}$"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    axes = axes.flatten()

    for ax, (metric, label) in zip(axes, metrics):
        y_pos = np.arange(len(ablation_df))
        values = ablation_df[metric].to_numpy()
        errors = ablation_df.get(f"{metric}_std", pd.Series(np.zeros(len(ablation_df)))).to_numpy()
        ax.barh(y_pos, values, xerr=errors, capsize=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ablation_df["variant"], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.grid(True, axis="x", alpha=0.25)

    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_robustness_curve(robustness_rows: Iterable[dict[str, float]], output_path: str | Path) -> None:
    rows = list(robustness_rows)
    eps = np.array([r["epsilon"] for r in rows])
    clean = np.array([r["clean_f1"] for r in rows])
    perturbed = np.array([r["perturbed_f1"] for r in rows])
    delta = np.array([r["delta_epsilon"] for r in rows])

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=True)

    axes[0].plot(eps, clean, marker="o", label="Clean F1")
    axes[0].plot(eps, perturbed, marker="o", label="Perturbed F1")
    axes[0].fill_between(eps, clean, perturbed, alpha=0.20)
    axes[0].set_ylabel("F1-score")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(eps, delta, marker="o")
    axes[1].set_xlabel(r"Perturbation radius $\epsilon$")
    axes[1].set_ylabel(r"$\Delta_{\epsilon}$")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
