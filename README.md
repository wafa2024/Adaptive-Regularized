# Adaptive Regularized Temporal Graph Learning (ARTGL)

This folder contains a clean reference implementation for the paper:

**Adaptive Regularized Temporal Graph Learning for Robust Network Intrusion Detection**

The code is organized to match the manuscript pipeline:

1. preprocess flow records,
2. split data before graph construction,
3. construct temporal flow graphs,
4. train the adaptive temporal graph model,
5. compute classification metrics and mathematical diagnostics,
6. run ablation and robustness evaluations,
7. regenerate manuscript-style figures.

The code is intentionally written in a simple, readable style so it can be inspected and modified easily before public release.

## Folder contents

```text
config.py          Global experiment configuration
preprocess.py      CSV loading, cleaning, encoding, scaling, and splitting
graph_building.py  Temporal graph construction and PyG Data creation
model.py           Adaptive regularized temporal graph learning model
losses.py          Composite objective terms
metrics.py         Accuracy, precision, recall, F1, TCS, GSI, SR, Delta-epsilon
train.py           Training and evaluation loops
robustness.py      Bounded-noise, FGSM, and PGD robustness evaluation
baselines.py       Logistic Regression, Random Forest, XGBoost, MLP, and LSTM baselines
plots.py           Figures for classification, ablation, and robustness results
run_artgl.py       Example end-to-end run script
requirements.txt   Python dependencies
```

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run an experiment:

```bash
python run_artgl.py \
  --csv_path /path/to/dataset.csv \
  --label_col label \
  --time_col timestamp \
  --output_dir results_unsw
```

For datasets with source/destination/protocol fields, provide them so graph construction can use relational filtering:

```bash
python run_artgl.py \
  --csv_path /path/to/dataset.csv \
  --label_col label \
  --time_col timestamp \
  --src_col src_ip \
  --dst_col dst_ip \
  --protocol_col proto \
  --output_dir results_unsw
```

## Notes

- The split is performed before graph construction to avoid temporal or structural leakage.
- The graph is constructed independently for train, validation, and test partitions.
- The robustness term uses bounded random L2 perturbations during training.
- FGSM and PGD are used only for evaluation.
- Dataset column names differ across UNSW-NB15, CICIDS2017, and CSE-CIC-IDS2018, so the command-line arguments should be adjusted for each dataset.

