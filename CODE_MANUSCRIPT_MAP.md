# Where each part of the manuscript is implemented (release v2.0)

Section numbers refer to the revised manuscript.

| Manuscript | What | Code |
|---|---|---|
| §4.2, §6.4 | Cleaning, label mapping, metadata vs. features, feature roles | `datasets.py` (`load_unsw_nb15`, `load_cicids2017`, `load_cse_cicids2018`) |
| §6.5 | Chronological 70/10/20 split; train-only encoding, scaling and valid ranges | `preprocess.py` (`split_indices`, `Encoder`, `prepare`) |
| Table 2 (data protocol) | Files, row counts, per-split class counts, feature counts, graph statistics | `preprocess.summarize`, `graph_building.graph_stats` → `data_summary.json` |
| §4.3.2, Eq. 16 | Relational edges E: next K flows within Δt, R(i,j) or cosine ≥ θ | `graph_building._relational_edges` |
| §3.2, Eq. 18 | Initial weights w0 = α·exp(−β\|Δt\|) + (1−α)·max(cos, 0) | `graph_building._relational_edges` |
| §4.3.3, Eq. 17 | Temporal edges E_t (same-session predecessor) and fixed weights ω | `graph_building._temporal_edges` |
| Eq. 19–20 | Adaptive coefficients a_ij = σ(φᵀ[x_i+x_j; \|x_i−x_j\|; x_i⊙x_j]), w~ = a·w0 | `model.ARTGL.edge_coefficients`, `ARTGL.forward` |
| §4.5 | Input projection h⁽⁰⁾, normalised messages, gate with h̄, gated update, classifier | `model.ARTGL.forward` |
| §4.6 | L_cls, L_temp (E_t), L_graph (E, adaptive), L_sparse, L_rob, all per-node normalised | `losses.composite_loss`, `losses.pairwise_energy` |
| Eq. 37 | Training perturbation: continuous features only, L2 radius ε, valid-range clipping | `perturbation.random_l2_perturbation`, `perturbation.clip_to_valid` |
| §4.7, Algorithm 1 | Contiguous temporal mini-batches, Adam, early stopping on validation F1 | `train.train_graph_model`, `batching.training_ranges` |
| §4.7 | Exact chunked full-graph inference and input gradients (halo (L+1)K) | `batching.infer`, `batching.loss_gradient` |
| §6.6 | Logistic Regression, Random Forest, XGBoost, MLP, windowed LSTM | `baselines.py` |
| §6.6 | GCN and GAT baselines | `model.GCNBaseline`, `model.GATBaseline` |
| §6.7 | Accuracy, precision, recall, F1 (malicious = positive class) | `metrics.classification_metrics` |
| §6.8 | TCS (E_t, unweighted), GSI (E, W0), SR (φ + classifier weights), Δε | `metrics.diagnostics`, `metrics.robustness_degradation` |
| §6.9 | Ablation variants (λ_k = 0; fixed edges a_ij = 1) | `experiments.ABLATIONS` |
| §6.10 | Sensitivity sweeps of λ1–λ4 | `experiments.suite_sensitivity`, `LAMBDA_GRID` |
| §6.11 | Bounded noise (5 draws), FGSM, PGD (masked, clipped, effective norms) | `robustness.py` |
| §6.12 | Every hyperparameter value | `config.ExperimentConfig`, `configs/*.json` |
| Table 3 (main), Table 11 (runtime) | | `experiments.suite_main` → `aggregate.py` |
| Tables 4–5 (adaptive edges, ablation) | | `experiments.suite_ablation` → `aggregate.py` |
| Table 6 (bounded noise) | | `experiments.suite_noise` → `aggregate.py` |
| Table 7 (adversarial, incl. w/o L_rob) | | `experiments.suite_adversarial` → `aggregate.py` |
| Table 8 (sensitivity) | | `experiments.suite_sensitivity` → `aggregate.py` |
| Tables 9–10 (cross-dataset, shared features) | | `cross_dataset.py`, `datasets.SHARED_*`, `datasets._shared_view` |
| Figures (classification, ablation, robustness) | | `plots.py` (reads only the aggregated tables) |

Table numbers follow the revised manuscript, where the new data-protocol table is Table 2;
check them once the journal class is applied.
