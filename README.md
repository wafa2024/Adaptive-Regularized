# ARTGL — Adaptive Regularized Temporal Graph Learning (release v2.0)

Reference implementation for *Adaptive Regularized Temporal Graph Learning for Robust
Network Intrusion Detection*. Release v2.0 is the version from which every number in the
revised manuscript is produced. `CODE_MANUSCRIPT_MAP.md` maps each equation and table to
the function that implements it, and `CHANGES.md` lists every difference from the v1
public code (commit `5aa768c`).

## 1. Install

```bash
pip install -r requirements.txt        # PyTorch >= 2.1, PyTorch Geometric >= 2.4
python tests/test_data_pipeline.py     # loaders, split, edge construction (no GPU needed)
python tests/test_model_torch.py       # model invariants + full pipeline on synthetic data (~minutes on CPU)
```

Run both tests before any real experiment.

## 2. Get the data (official files only)

If you downloaded the datasets as zip files, put them all in one folder and run
`python prepare_data.py --source <that folder>`. It finds the files listed below (also inside
nested zips), extracts only those into `data/`, and reports anything it skipped.

| Dataset | Files to download | Put them in |
|---|---|---|
| UNSW-NB15 | `UNSW-NB15_1.csv` … `UNSW-NB15_4.csv` (the full, headerless CSVs) | `data/UNSW-NB15/` |
| CICIDS2017 | `GeneratedLabelledFlows.zip` → `TrafficLabelling/*.pcap_ISCX.csv` (8 files) | `data/CICIDS2017/TrafficLabelling/` |
| CSE-CIC-IDS2018 | `Processed Traffic Data for ML Algorithms/*.csv` (10 files) | `data/CSE-CIC-IDS2018/Processed Traffic Data for ML Algorithms/` |

Two files that are commonly used **will not work**, deliberately: the UNSW-NB15
`training-set`/`testing-set` CSVs and the CICIDS2017 `MachineLearningCVE` CSVs. Neither
contains timestamps or IP addresses, so a temporal flow graph cannot be built from them.

## 3. Reproduce every table

In Jupyter or Google Colab, open `run_experiments.ipynb` and run its cells in order. From a terminal:

```bash
python experiments.py --config configs/unsw_nb15.json      # Tables 2-8 and 11
python experiments.py --config configs/cicids2017.json --suites main
python cross_dataset.py --train configs/unsw_nb15.json --test configs/cicids2017.json
python cross_dataset.py --train configs/cicids2017.json --test configs/unsw_nb15.json
python cross_dataset.py --train configs/unsw_nb15.json configs/cicids2017.json --test configs/cse_cicids2018.json
python interpretability.py --config configs/unsw_nb15.json      # t-SNE, edge coefficients, phi weights (uses the saved models)
python interpretability.py --config configs/cicids2017.json --skip_noise
python aggregate.py --results results/unsw_nb15 --cross results/cross_dataset
python aggregate.py --results results/cicids2017
python plots.py --tables results/unsw_nb15/tables --dataset UNSW-NB15
```

Outputs:

- `results/<dataset>/data_summary.json` — files read, rows before and after cleaning,
  per-split counts per class, label mapping, retained features with their roles, and graph
  statistics. This is the data-protocol table in the manuscript.
- `results/<dataset>/*_per_seed.csv` — every per-seed number (supplementary material).
- `results/<dataset>/tables/*.tex` — LaTeX rows to paste into the manuscript tables.
- `results/<dataset>/runs/` — one cached JSON and weights file per (model, configuration,
  seed). A model that appears in several tables is trained once, so its numbers are
  identical everywhere. Delete `runs/` to recompute from scratch.

Approximate cost on UNSW-NB15 with a single RTX 3060: the sensitivity suite is the most
expensive one (4 coefficients × 5 values × 5 seeds = 100 trainings). Use `--suites` to run
suites separately.

## 4. Files

```text
config.py           every hyperparameter (single source of truth)
configs/*.json      per-dataset paths and temporal windows
datasets.py         official-file loaders, feature roles, label mapping, cross-dataset features
preprocess.py       chronological split, train-only encoding/scaling, valid feature ranges
graph_building.py   relational edges E (message passing, L_graph) and temporal edges E_t (L_temp, TCS)
model.py            ARTGL; GCN and GAT baselines
losses.py           composite objective
perturbation.py     masked, clipped, budget-preserving perturbations
batching.py         temporal-subgraph mini-batches; exact chunked inference and gradients
train.py            optimisation and early stopping
metrics.py          accuracy/precision/recall/F1, TCS, GSI, SR, Delta_eps
robustness.py       bounded noise, FGSM, PGD
baselines.py        Logistic Regression, Random Forest, XGBoost, MLP, LSTM
experiments.py      all suites, run cache, per-seed CSVs
cross_dataset.py    cross-dataset protocol
interpretability.py interpretability analysis of the trained models (no retraining)
aggregate.py        per-seed -> mean ± std tables (CSV + LaTeX)
plots.py            figures from the aggregated tables
run_artgl.py        one quick single-seed run
prepare_data.py     pick the needed files out of downloaded zips into data/
tests/              synthetic data, pipeline tests, model tests
```
