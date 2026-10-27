# Where each part of the manuscript is implemented (release v2.0)

Section, table and figure numbers refer to the revised manuscript.

## Method

| Manuscript | What | Code |
|---|---|---|
| §4.3.2 | Relational edges E: next K flows within Δt that share an address or protocol, or have cosine ≥ θ | `graph_building._relational_edges` |
| §4.4 | Initial weights w0 = α·exp(−β\|Δt\|) + (1−α)·max(cos, 0) | `graph_building._relational_edges` |
| §4.3.3 | Temporal edges E_t (same-session predecessor within Δt) and fixed weights ω = exp(−βΔt) | `graph_building._temporal_edges` |
| §4.4, Eq. 12 | Adaptive coefficients a_ij = σ(φᵀ[x_i+x_j; \|x_i−x_j\|; x_i⊙x_j]) (linear, on inputs, no bias), w~ = a·w0 | `model.ARTGL.edge_coefficients`, `ARTGL.forward` |
| §4.5 | Input projection h⁽⁰⁾, degree-normalised messages, gate with the neighbourhood mean h̄, gated update, classifier | `model.ARTGL.forward` |
| §4.6 | L_cls, L_temp (on E_t), L_graph (on E, adaptive Laplacian), L_sparse (φ + classifier weights), L_rob; all normalised per node | `losses.composite_loss`, `losses.pairwise_energy` |
| §4.6.5, Eq. 29 | Training perturbation: continuous features only, ℓ2 radius ε, valid-range clipping | `perturbation.random_l2_perturbation`, `perturbation.clip_to_valid` |
| §4.7, Algorithm 1 | Contiguous temporal mini-batches of 1,024 flows, Adam, early stopping on validation F1 | `train.train_graph_model`, `batching.training_ranges` |
| §4.7 | Exact chunked full-graph inference and input gradients (halo (L+1)K) | `batching.infer`, `batching.loss_gradient` |

## Experimental setup (Section 6)

| Manuscript | What | Code |
|---|---|---|
| §6.2–6.3 | Dataset files, cleaning, label mapping (benign → 0, attack → 1), metadata vs. features, feature roles | `datasets.py` (`load_unsw_nb15`, `load_cicids2017`, `load_cse_cicids2018`) |
| §6.5 | Chronological 70/10/20 split; graphs per partition; train-only encoding, scaling and valid ranges | `preprocess.py` (`split_indices`, `Encoder`, `prepare`), `graph_building.build_graph` |
| Table 2 | Files, row counts, duplicates, per-split class counts, feature counts, graph statistics | `preprocess.summarize`, `graph_building.graph_stats` → `data_summary.json` |
| §6.6 | Logistic Regression, Random Forest, XGBoost, MLP, windowed LSTM | `baselines.py` |
| §6.6 | GCN and GAT baselines (same graphs, batching and early stopping) | `model.GCNBaseline`, `model.GATBaseline` |
| §6.7 | Accuracy, precision, recall, F1 (malicious = positive class) | `metrics.classification_metrics` |
| §6.8 | TCS (E_t, unweighted), GSI (E, W0; learned-weight energy reported separately), SR, Δε (per seed) | `metrics.diagnostics`, `metrics.robustness_degradation` |
| §6.9 | Every hyperparameter value; software versions and hardware | `config.ExperimentConfig`, `configs/*.json`, `requirements.txt` |

## Results (Section 7)

| Manuscript | Code |
|---|---|
| Table 3, Figures 2–3 (main results) | `experiments.suite_main` → `aggregate.py` → `plots.py` |
| Table 13 (computational cost) | `experiments.suite_main` (timings, parameters) → `aggregate.py` |
| Tables 4–5, Figure 4 (adaptive edges, ablation) | `experiments.ABLATIONS`, `experiments.suite_ablation` → `aggregate.py` → `plots.py` |
| Table 6, Figures 5–6 (interpretability: t-SNE, separability, edge coefficients, φ weights) | `interpretability.py` |
| Table 7, Figure 7 (bounded random noise) | `experiments.suite_noise`, `robustness.evaluate_noise` → `aggregate.py` → `plots.py` |
| Table 8 (random noise at larger radii) | `interpretability.py` (`EXTENDED_EPS`) |
| Table 9 (FGSM, PGD, incl. w/o L_rob) | `experiments.suite_adversarial`, `robustness.fgsm`, `robustness.pgd` → `aggregate.py` |
| Table 10 (sensitivity of λ1–λ4) | `experiments.suite_sensitivity`, `LAMBDA_GRID` → `aggregate.py` |
| Tables 11–12 (cross-dataset, shared features) | `cross_dataset.py`, `datasets.SHARED_*`, `datasets._shared_view` |
| Figure 1 (framework overview) | `figures/framework_overview.tex` (LaTeX/TikZ source) |

The per-seed values behind every table are in `results_v2.0/`.
