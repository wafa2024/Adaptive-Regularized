"""Experiment configuration for ARTGL (release v2.0).

Every value that appears in the manuscript's "Implementation details" section is
defined here, so the paper and the code read from one place. Dataset-specific
values (temporal window, column layout) live in ``configs/<dataset>.json`` and
override these defaults.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional


@dataclass
class ExperimentConfig:
    # ------------------------------------------------------------------ data
    dataset: str = "generic"              # unsw_nb15 | cicids2017 | cse_cicids2018 | generic
    data_paths: list[str] = field(default_factory=list)
    feature_set: str = "full"             # full | shared (shared = cross-dataset feature set)
    label_mode: str = "binary"            # binary (benign=0, malicious=1) | multiclass

    # generic loader only (the named datasets define these in datasets.py)
    label_col: Optional[str] = None
    time_col: Optional[str] = None
    src_col: Optional[str] = None
    dst_col: Optional[str] = None
    protocol_col: Optional[str] = None
    drop_cols: list[str] = field(default_factory=list)
    discrete_cols: list[str] = field(default_factory=list)
    benign_values: list[str] = field(default_factory=lambda: ["benign", "normal", "0"])

    # ---------------------------------------------------------------- splits
    split: str = "chronological"          # chronological | stratified
    train_frac: float = 0.70
    val_frac: float = 0.10                 # test = 1 - train - val
    split_seed: int = 42                   # only used by the stratified split

    # ---------------------------------------------------------- graph (Sec. 4.3)
    temporal_window: float = 30.0          # Delta_t in seconds
    max_neighbors: int = 30                # K: candidates = next K flows in time order
    similarity_threshold: float = 0.65     # cosine threshold for feature-similarity edges
    alpha: float = 0.5
    beta: float = 0.1
    temporal_key: str = "pair"             # pair (src,dst) | src | none -> defines E_t

    # ---------------------------------------------------------- model (Sec. 4.5)
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.30
    adaptive_edges: bool = True            # False = fixed-edge variant (a_ij = 1)

    # ------------------------------------------------------- optimisation (4.7)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 200
    patience: int = 20
    batch_size: int = 1024                 # nodes per contiguous temporal subgraph

    # --------------------------------------------------------- objective (4.6)
    lambda_temp: float = 1e-2
    lambda_graph: float = 1e-2
    lambda_sparse: float = 1e-3
    lambda_robust: float = 1e-2
    train_epsilon: float = 0.05            # L2 radius of the training-time noise

    # ----------------------------------------------------------- evaluation
    sparsity_threshold: float = 1e-3       # tau in SR
    noise_epsilons: list[float] = field(default_factory=lambda: [0.01, 0.05, 0.10, 0.20])
    noise_draws: int = 5                   # random noise draws averaged per epsilon
    ablation_epsilon: float = 0.10         # epsilon used for Delta_eps in the ablation table
    adv_epsilon: float = 0.05
    pgd_steps: int = 10
    pgd_step_size: float = 0.01
    eval_chunk_size: int = 8192

    # ----------------------------------------------------------------- misc
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    output_dir: str = "results"
    device: str = "auto"

    # ------------------------------------------------------------- helpers
    @classmethod
    def from_json(cls, path: str | Path, **overrides) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as fp:
            values = json.load(fp)
        values.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        return cls(**values)

    def lambdas(self) -> dict[str, float]:
        return {
            "temp": self.lambda_temp,
            "graph": self.lambda_graph,
            "sparse": self.lambda_sparse,
            "robust": self.lambda_robust,
        }

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, indent=2)
