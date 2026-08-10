from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from config import ExperimentConfig
from graph_building import build_temporal_graph
from metrics import diagnostic_metrics
from model import AdaptiveTemporalGraphNet
from preprocess import load_and_prepare_splits
from robustness import evaluate_adversarial, evaluate_under_l2_noise
from train import evaluate_model, move_data_to_device, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ARTGL intrusion-detection experiment.")
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--label_col", required=True)
    parser.add_argument("--time_col", required=True)
    parser.add_argument("--output_dir", default="results_artgl")

    parser.add_argument("--src_col", default=None)
    parser.add_argument("--dst_col", default=None)
    parser.add_argument("--protocol_col", default=None)

    parser.add_argument("--temporal_window", type=float, default=30.0)
    parser.add_argument("--similarity_threshold", type=float, default=0.65)
    parser.add_argument("--max_neighbors", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
        csv_path=Path(args.csv_path),
        output_dir=Path(args.output_dir),
        label_col=args.label_col,
        time_col=args.time_col,
        src_col=args.src_col,
        dst_col=args.dst_col,
        protocol_col=args.protocol_col,
        temporal_window=args.temporal_window,
        similarity_threshold=args.similarity_threshold,
        max_neighbors=args.max_neighbors,
        random_seed=args.seed,
    )
    cfg.make_output_dir()

    torch.manual_seed(cfg.random_seed)

    meta_cols = [c for c in [cfg.src_col, cfg.dst_col, cfg.protocol_col] if c]
    splits = load_and_prepare_splits(
        csv_path=str(cfg.csv_path),
        label_col=cfg.label_col,
        time_col=cfg.time_col,
        meta_cols=meta_cols,
        test_size=cfg.test_size,
        val_size=cfg.val_size,
        random_seed=cfg.random_seed,
    )

    graph_kwargs = dict(
        temporal_window=cfg.temporal_window,
        similarity_threshold=cfg.similarity_threshold,
        max_neighbors=cfg.max_neighbors,
        alpha=cfg.alpha,
        beta=cfg.beta,
    )
    train_graph = build_temporal_graph(splits.x_train, splits.y_train, splits.t_train, splits.meta_train, **graph_kwargs)
    val_graph = build_temporal_graph(splits.x_val, splits.y_val, splits.t_val, splits.meta_val, **graph_kwargs)
    test_graph = build_temporal_graph(splits.x_test, splits.y_test, splits.t_test, splits.meta_test, **graph_kwargs)

    model = AdaptiveTemporalGraphNet(
        input_dim=splits.x_train.shape[1],
        hidden_dim=cfg.hidden_dim,
        num_classes=len(splits.class_names),
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )

    lambdas = {
        "temp": cfg.lambda_temp,
        "graph": cfg.lambda_graph,
        "sparse": cfg.lambda_sparse,
        "robust": cfg.lambda_robust,
    }

    model, history = train_model(
        model=model,
        train_data=train_graph,
        val_data=val_graph,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        lambdas=lambdas,
        epsilon=cfg.perturbation_epsilon,
        max_epochs=cfg.max_epochs,
        patience=cfg.patience,
        sparsity_threshold=cfg.sparsity_threshold,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_graph = move_data_to_device(test_graph, device)
    model = model.to(device)

    test_metrics = evaluate_model(model, test_graph, sparsity_threshold=cfg.sparsity_threshold)
    noise_rows = evaluate_under_l2_noise(model, test_graph, epsilons=[0.01, 0.05, 0.10, 0.20])
    adv_metrics = evaluate_adversarial(model, test_graph, epsilon=0.05, pgd_steps=10, pgd_step_size=0.01)

    pd.DataFrame(history).to_csv(cfg.output_dir / "training_history.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(cfg.output_dir / "test_metrics.csv", index=False)
    pd.DataFrame(noise_rows).to_csv(cfg.output_dir / "bounded_noise_robustness.csv", index=False)
    pd.DataFrame([adv_metrics]).to_csv(cfg.output_dir / "adversarial_robustness.csv", index=False)

    with open(cfg.output_dir / "run_config.json", "w", encoding="utf-8") as fp:
        json.dump(cfg.__dict__ | {"csv_path": str(cfg.csv_path), "output_dir": str(cfg.output_dir)}, fp, indent=2, default=str)

    print("Test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print(f"\nSaved outputs to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
