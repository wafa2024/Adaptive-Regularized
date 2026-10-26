"""Splitting, encoding and scaling (manuscript Sections 6.3 and 6.5).

Order of operations, which the manuscript follows exactly:
  1. records are sorted by timestamp;
  2. the split is made (chronological by default: the first 70% of the time-ordered
     records train, the next 10% validate, the last 20% test);
  3. one-hot categories, the scaler, and the valid feature ranges are fitted on the
     training split only and applied unchanged to validation and test;
  4. graphs are then built separately inside each split (graph_building.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from datasets import LoadedData


@dataclass
class Split:
    x: np.ndarray            # (n, d) float32, standardised
    y: np.ndarray            # (n,) int64
    t: np.ndarray            # (n,) float64 seconds, non-decreasing
    src: np.ndarray          # (n,) object, may contain None
    dst: np.ndarray
    proto: np.ndarray
    raw_label: np.ndarray    # (n,) attack category strings


@dataclass
class Encoder:
    numeric_cols: list[str]
    discrete_cols: list[str]
    categorical_cols: list[str]
    categories: dict[str, list[str]] = field(default_factory=dict)
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    lower: np.ndarray | None = None     # valid range per feature (standardised units)
    upper: np.ndarray | None = None
    feature_names: list[str] = field(default_factory=list)
    continuous_mask: np.ndarray | None = None
    class_names: list[str] = field(default_factory=list)
    label_mode: str = "binary"

    # ------------------------------------------------------------------
    def fit(self, frame: pd.DataFrame) -> "Encoder":
        for c in self.categorical_cols:
            self.categories[c] = sorted(frame[c].astype(str).unique().tolist())
        raw = self._raw_matrix(frame)                      # float32, to limit memory
        self.mean = raw.mean(axis=0, dtype=np.float64)
        std = raw.std(axis=0, dtype=np.float64)
        self.std = np.where(std < 1e-12, 1.0, std)
        # valid range in standardised units (the transform is monotone per feature)
        self.lower = ((raw.min(axis=0) - self.mean) / self.std).astype(np.float32)
        self.upper = ((raw.max(axis=0) - self.mean) / self.std).astype(np.float32)
        return self

    def _raw_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        parts = []
        names: list[str] = []
        mask: list[bool] = []
        num = frame[self.numeric_cols].to_numpy(dtype=np.float32) if self.numeric_cols else np.zeros((len(frame), 0), np.float32)
        parts.append(np.nan_to_num(num, nan=0.0, posinf=0.0, neginf=0.0))
        names += self.numeric_cols
        mask += [True] * len(self.numeric_cols)
        dis = frame[self.discrete_cols].to_numpy(dtype=np.float32) if self.discrete_cols else np.zeros((len(frame), 0), np.float32)
        parts.append(np.nan_to_num(dis, nan=0.0, posinf=0.0, neginf=0.0))
        names += self.discrete_cols
        mask += [False] * len(self.discrete_cols)
        for c in self.categorical_cols:
            cats = self.categories[c]
            values = frame[c].astype(str).to_numpy()
            onehot = np.zeros((len(frame), len(cats)), dtype=np.float32)
            index = {v: i for i, v in enumerate(cats)}
            rows = np.array([index.get(v, -1) for v in values])
            ok = rows >= 0
            onehot[np.nonzero(ok)[0], rows[ok]] = 1.0          # unseen categories -> all zeros
            parts.append(onehot)
            names += [f"{c}={v}" for v in cats]
            mask += [False] * len(cats)
        self.feature_names = names
        self.continuous_mask = np.array(mask, dtype=bool)
        return np.concatenate(parts, axis=1)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self._raw_matrix(frame)
        raw -= self.mean.astype(np.float32)
        raw /= self.std.astype(np.float32)
        return raw

    def labels(self, frame: pd.DataFrame) -> np.ndarray:
        raw = frame["__label"].astype(str).to_numpy()
        if self.label_mode == "binary":
            return (raw != "benign").astype(np.int64)       # malicious = 1 = positive class
        index = {c: i for i, c in enumerate(self.class_names)}
        return np.array([index.get(v, -1) for v in raw], dtype=np.int64)


def _make_split(frame: pd.DataFrame, enc: Encoder) -> Split:
    def col(name):
        return frame[name].to_numpy(dtype=object) if name in frame.columns else np.full(len(frame), None, dtype=object)

    return Split(
        x=enc.transform(frame),
        y=enc.labels(frame),
        t=frame["__time"].to_numpy(dtype=np.float64),
        src=col("__src"),
        dst=col("__dst"),
        proto=col("__proto"),
        raw_label=frame["__label"].astype(str).to_numpy(),
    )


def split_indices(frame: pd.DataFrame, cfg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(frame)
    if cfg.split == "chronological":
        n_train = int(round(cfg.train_frac * n))
        n_val = int(round(cfg.val_frac * n))
        idx = np.arange(n)                                    # frame is already time-sorted
        return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]
    if cfg.split == "stratified":
        strat = frame["__label"].to_numpy()
        test_frac = 1.0 - cfg.train_frac - cfg.val_frac
        idx_tv, idx_te = train_test_split(np.arange(n), test_size=test_frac, random_state=cfg.split_seed, stratify=strat)
        rel = cfg.val_frac / (cfg.train_frac + cfg.val_frac)
        idx_tr, idx_va = train_test_split(idx_tv, test_size=rel, random_state=cfg.split_seed, stratify=strat[idx_tv])
        # keep each split in time order so graphs can be built within it
        return np.sort(idx_tr), np.sort(idx_va), np.sort(idx_te)
    raise ValueError(f"Unknown split '{cfg.split}'")


def prepare(data: LoadedData, cfg, encoder: Encoder | None = None) -> tuple[dict[str, Split], Encoder, dict]:
    """Split, fit the encoder on train (unless one is given), and transform all splits.

    Passing ``encoder`` is used by the cross-dataset protocol: the target dataset
    is transformed with the encoder fitted on the source training split.
    """

    frame = data.frame
    i_tr, i_va, i_te = split_indices(frame, cfg)
    parts = {"train": frame.iloc[i_tr], "val": frame.iloc[i_va], "test": frame.iloc[i_te]}

    if encoder is None:
        encoder = Encoder(data.numeric_cols, data.discrete_cols, data.categorical_cols, label_mode=cfg.label_mode)
        if cfg.label_mode == "multiclass":
            encoder.class_names = sorted(parts["train"]["__label"].unique().tolist())
        else:
            encoder.class_names = ["benign", "malicious"]
        encoder.fit(parts["train"])

    splits = {name: _make_split(part, encoder) for name, part in parts.items()}
    summary = summarize(data, parts, splits, encoder, cfg)
    return splits, encoder, summary


def summarize(data: LoadedData, parts: dict, splits: dict, enc: Encoder, cfg) -> dict:
    """Everything the reviewer asked to see: files, row counts, label mapping, features."""

    out = {
        "dataset": cfg.dataset,
        "feature_set": cfg.feature_set,
        "source_files": [Path(f).name for f in data.source_files],
        "rows_read": data.rows_read,
        "rows_after_cleaning": data.rows_after_cleaning,
        "cleaning_notes": data.notes,
        "split": cfg.split,
        "label_mode": cfg.label_mode,
        "label_mapping": "benign -> 0, every attack category -> 1" if cfg.label_mode == "binary" else enc.class_names,
        "features": {
            "continuous_perturbable": data.numeric_cols,
            "discrete_not_perturbed": data.discrete_cols,
            "categorical_one_hot_not_perturbed": data.categorical_cols,
            "encoded_dimension": int(len(enc.feature_names)),
        },
        "splits": {},
    }
    for name, part in parts.items():
        sp = splits[name]
        out["splits"][name] = {
            "rows": int(len(part)),
            "benign": int((sp.y == 0).sum()) if cfg.label_mode == "binary" else None,
            "malicious": int((sp.y == 1).sum()) if cfg.label_mode == "binary" else None,
            "per_category": part["__label"].value_counts().to_dict(),
            "time_range_s": [float(sp.t.min()) if len(sp.t) else None, float(sp.t.max()) if len(sp.t) else None],
        }
    return out


def write_summary(summary: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, default=str)
