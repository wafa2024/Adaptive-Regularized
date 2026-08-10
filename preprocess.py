from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class SplitData:
    """Preprocessed data splits used before graph construction."""

    x_train: np.ndarray
    y_train: np.ndarray
    t_train: np.ndarray
    meta_train: pd.DataFrame

    x_val: np.ndarray
    y_val: np.ndarray
    t_val: np.ndarray
    meta_val: pd.DataFrame

    x_test: np.ndarray
    y_test: np.ndarray
    t_test: np.ndarray
    meta_test: pd.DataFrame

    feature_names: list[str]
    class_names: list[str]


def _safe_datetime_to_seconds(series: pd.Series) -> np.ndarray:
    """Convert a timestamp column into numeric seconds.

    The datasets used in intrusion detection do not always store timestamps in the
    same format. This helper accepts either numeric timestamps or strings that can
    be parsed by pandas.
    """

    if pd.api.types.is_numeric_dtype(series):
        values = series.astype(float).to_numpy()
        return values - np.nanmin(values)

    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().all():
        raise ValueError("The timestamp column could not be parsed as numeric or datetime values.")

    seconds = parsed.astype("int64") / 1e9
    return seconds.to_numpy() - np.nanmin(seconds.to_numpy())


def load_and_prepare_splits(
    csv_path: str,
    label_col: str,
    time_col: str,
    meta_cols: Optional[Iterable[str]] = None,
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_seed: int = 42,
) -> SplitData:
    """Load a flow dataset, clean it, encode features, and split it safely.

    The split is done before graph construction. This is important because graph
    edges must not connect training nodes to validation or test nodes.
    """

    df = pd.read_csv(csv_path)
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan)

    required = {label_col, time_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    meta_cols = [c for c in (meta_cols or []) if c in df.columns]
    metadata = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)

    timestamps = _safe_datetime_to_seconds(df[time_col])

    labels = df[label_col].astype(str).fillna("unknown")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    # Remove target/time columns from the feature matrix. Metadata columns may
    # also appear as categorical features if they carry predictive information.
    feature_df = df.drop(columns=[label_col, time_col])
    feature_df = feature_df.fillna(0)

    categorical_cols = [c for c in feature_df.columns if feature_df[c].dtype == "object"]
    numeric_cols = [c for c in feature_df.columns if c not in categorical_cols]

    encoded_parts = []
    feature_names: list[str] = []

    if numeric_cols:
        numeric_values = feature_df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        encoded_parts.append(numeric_values.to_numpy(dtype=np.float32))
        feature_names.extend(numeric_cols)

    if categorical_cols:
        encoded_cat = pd.get_dummies(feature_df[categorical_cols].astype(str), dummy_na=True)
        encoded_parts.append(encoded_cat.to_numpy(dtype=np.float32))
        feature_names.extend(encoded_cat.columns.tolist())

    if not encoded_parts:
        raise ValueError("No usable feature columns were found after preprocessing.")

    x = np.concatenate(encoded_parts, axis=1).astype(np.float32)

    idx_all = np.arange(len(df))
    idx_train_val, idx_test = train_test_split(
        idx_all,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )

    relative_val = val_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(
        idx_train_val,
        test_size=relative_val,
        random_state=random_seed,
        stratify=y[idx_train_val],
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[idx_train]).astype(np.float32)
    x_val = scaler.transform(x[idx_val]).astype(np.float32)
    x_test = scaler.transform(x[idx_test]).astype(np.float32)

    return SplitData(
        x_train=x_train,
        y_train=y[idx_train],
        t_train=timestamps[idx_train],
        meta_train=metadata.iloc[idx_train].reset_index(drop=True),
        x_val=x_val,
        y_val=y[idx_val],
        t_val=timestamps[idx_val],
        meta_val=metadata.iloc[idx_val].reset_index(drop=True),
        x_test=x_test,
        y_test=y[idx_test],
        t_test=timestamps[idx_test],
        meta_test=metadata.iloc[idx_test].reset_index(drop=True),
        feature_names=feature_names,
        class_names=label_encoder.classes_.tolist(),
    )
