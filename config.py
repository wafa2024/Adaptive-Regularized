from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ExperimentConfig:
    """Configuration used across the ARTGL experiments."""

    csv_path: Path
    output_dir: Path
    label_col: str
    time_col: str

    src_col: Optional[str] = None
    dst_col: Optional[str] = None
    protocol_col: Optional[str] = None

    test_size: float = 0.20
    val_size: float = 0.10
    random_seed: int = 42

    temporal_window: float = 30.0
    similarity_threshold: float = 0.65
    max_neighbors: int = 30
    alpha: float = 0.5
    beta: float = 0.1

    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.30

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 200
    patience: int = 20

    lambda_temp: float = 1e-2
    lambda_graph: float = 1e-2
    lambda_sparse: float = 1e-3
    lambda_robust: float = 1e-2

    perturbation_epsilon: float = 0.05
    sparsity_threshold: float = 1e-3

    def make_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
