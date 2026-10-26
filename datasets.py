"""Dataset loaders for UNSW-NB15, CICIDS2017 and CSE-CIC-IDS2018.

Each loader documents exactly which official files it expects, which columns are
used as features, which are metadata only (used to build graph edges but never
fed to the classifier), and how labels are mapped. The manuscript's data
protocol section is written from this file.

All loaders return a ``LoadedData`` object whose ``frame`` contains:
    __time      float seconds since the earliest record in the dataset
    __label     raw label string (attack category or "benign")
    __src, __dst, __proto   metadata strings for edge construction (may be absent)
plus the feature columns listed in ``numeric_cols`` and ``categorical_cols``.
"""

from __future__ import annotations

import glob
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared (cross-dataset) feature set. Seven flow-level attributes that all three
# datasets record, mapped to common names and units.
# ---------------------------------------------------------------------------
SHARED_NUMERIC = ["duration_s", "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes"]
SHARED_DISCRETE = ["dst_port"]
SHARED_CATEGORICAL = ["protocol"]

_PROTO_NUM_TO_NAME = {"6": "tcp", "17": "udp", "1": "icmp"}


@dataclass
class LoadedData:
    frame: pd.DataFrame
    numeric_cols: list[str]                 # continuous: scaled and perturbable
    discrete_cols: list[str]                # numeric but never perturbed (ports, flags)
    categorical_cols: list[str]             # one-hot encoded, never perturbed
    has_src_dst: bool
    has_proto: bool
    source_files: list[str] = field(default_factory=list)
    rows_read: int = 0
    rows_after_cleaning: int = 0
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def expand_paths(paths: Iterable[str]) -> list[str]:
    out: list[str] = []
    for p in paths:
        matches = sorted(glob.glob(str(p)))
        if not matches and Path(p).exists():
            matches = [str(p)]
        if Path(p).is_dir():
            matches = sorted(glob.glob(str(Path(p) / "*.csv")))
        out.extend(matches)
    if not out:
        raise FileNotFoundError(f"No CSV files matched: {list(paths)}")
    return out


def _to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    s = series.astype(str).str.strip()
    # UNSW-NB15 stores some ports in hexadecimal (e.g. "0x000b").
    is_hex = s.str.match(r"^0x[0-9a-fA-F]+$", na=False)
    out = pd.to_numeric(s.where(~is_hex), errors="coerce")
    if is_hex.any():
        out[is_hex] = s[is_hex].apply(lambda v: int(v, 16)).astype(float)
    return out


_DAYFIRST_FORMATS = [
    "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y %I:%M:%S %p", "%d/%m/%Y %I:%M %p",
    "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
]


def _parse_datetime(series: pd.Series, what: str) -> pd.Series:
    """Parse day-first timestamps with explicit formats (works on pandas 1.x and 2.x/3.x).

    Raises an error, instead of silently dropping rows, if most timestamps cannot be parsed.
    """

    s = series.astype(str).str.strip()
    parsed = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in _DAYFIRST_FORMATS:
        todo = parsed.isna()
        if not todo.any():
            break
        parsed[todo] = pd.to_datetime(s[todo], format=fmt, errors="coerce")
    todo = parsed.isna()
    if todo.any():  # last resort for unusual formats
        parsed[todo] = pd.to_datetime(s[todo], dayfirst=True, errors="coerce")
    bad = float(parsed.isna().mean())
    if bad > 0.01:
        examples = s[parsed.isna()].head(5).tolist()
        raise ValueError(f"{what}: {bad:.1%} of timestamps could not be parsed, e.g. {examples}")
    return parsed


def _datetime_to_seconds(parsed: pd.Series) -> np.ndarray:
    """Seconds since the Unix epoch, independent of the datetime resolution (pandas 2 and 3)."""

    delta = (parsed - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
    return delta.to_numpy(dtype=float)


def _seconds_from_epoch(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values


def _finalize(frame: pd.DataFrame, numeric_cols, discrete_cols, categorical_cols, notes) -> pd.DataFrame:
    frame = frame.replace([np.inf, -np.inf], np.nan)
    for c in list(numeric_cols) + list(discrete_cols):
        frame[c] = _to_numeric(frame[c])
    for c in categorical_cols:
        frame[c] = frame[c].astype(str).str.strip().str.lower().replace({"nan": "missing", "": "missing", "-": "missing"})
    before = len(frame)
    frame = frame[np.isfinite(frame["__time"].to_numpy(dtype=float))]
    if len(frame) < before:
        notes.append(f"dropped {before - len(frame)} rows with unparseable timestamps")
    before = len(frame)
    frame = frame.drop_duplicates()
    notes.append(f"dropped {before - len(frame)} exact duplicate rows")
    frame["__time"] = frame["__time"] - frame["__time"].min()
    frame = frame.sort_values("__time", kind="mergesort").reset_index(drop=True)
    return frame


def _shared_view(frame: pd.DataFrame, dur, fp, bp, fb, bb, port, proto, duration_scale: float) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "__time": frame["__time"],
            "__label": frame["__label"],
            "duration_s": _to_numeric(frame[dur]) * duration_scale,
            "fwd_pkts": _to_numeric(frame[fp]),
            "bwd_pkts": _to_numeric(frame[bp]),
            "fwd_bytes": _to_numeric(frame[fb]),
            "bwd_bytes": _to_numeric(frame[bb]),
            "dst_port": _to_numeric(frame[port]),
            "protocol": frame[proto].astype(str).str.strip().str.lower().str.replace(r"\.0$", "", regex=True).map(lambda v: _PROTO_NUM_TO_NAME.get(v, v)),
        }
    )
    out["protocol"] = out["protocol"].where(out["protocol"].isin(["tcp", "udp", "icmp"]), "other")
    # log1p puts heavy-tailed counts/bytes/durations from different capture
    # environments on a comparable scale before standardisation.
    for c in SHARED_NUMERIC:
        out[c] = np.log1p(out[c].clip(lower=0))
    for c in ["__src", "__dst", "__proto"]:
        if c in frame.columns:
            out[c] = frame[c]
    return out


# ---------------------------------------------------------------------------
# UNSW-NB15
# ---------------------------------------------------------------------------
UNSW_COLUMNS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes", "sttl",
    "dttl", "sloss", "dloss", "service", "sload", "dload", "spkts", "dpkts", "swin", "dwin",
    "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "sjit", "djit",
    "stime", "ltime", "sintpkt", "dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports",
    "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst",
    "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "attack_cat", "label",
]
UNSW_CATEGORICAL = ["proto", "state", "service"]
UNSW_DISCRETE = ["dsport", "is_sm_ips_ports", "is_ftp_login"]
UNSW_METADATA_ONLY = ["srcip", "dstip", "sport", "stime", "ltime", "attack_cat", "label"]


def load_unsw_nb15(paths: Iterable[str], feature_set: str = "full") -> LoadedData:
    """Official full CSVs UNSW-NB15_1.csv ... UNSW-NB15_4.csv (headerless, 49 columns).

    The commonly used UNSW_NB15_training-set.csv / testing-set.csv files are NOT
    supported: they contain no timestamps or IP addresses, so a temporal flow
    graph cannot be built from them.
    """

    files = expand_paths(paths)
    notes: list[str] = []
    parts = []
    for f in files:
        part = pd.read_csv(f, header=None, names=UNSW_COLUMNS, low_memory=False, encoding="latin-1")
        if str(part.iloc[0, 0]).strip().lower() == "srcip":  # tolerate a header row
            part = part.iloc[1:]
        parts.append(part)
    raw = pd.concat(parts, ignore_index=True)
    rows_read = len(raw)

    frame = raw.copy()
    frame["__time"] = _seconds_from_epoch(frame["stime"])
    cat = frame["attack_cat"].astype(str).str.strip()
    cat = cat.replace({"nan": "", "Backdoors": "Backdoor"})
    lab = _to_numeric(frame["label"]).fillna(0).astype(int)
    frame["__label"] = np.where(lab == 0, "benign", np.where(cat == "", "unknown_attack", cat.str.lower()))
    frame["__src"] = frame["srcip"].astype(str).str.strip()
    frame["__dst"] = frame["dstip"].astype(str).str.strip()
    frame["__proto"] = frame["proto"].astype(str).str.strip().str.lower()

    if feature_set == "shared":
        frame = _shared_view(frame, "dur", "spkts", "dpkts", "sbytes", "dbytes", "dsport", "proto", 1.0)
        numeric, discrete, categorical = SHARED_NUMERIC, SHARED_DISCRETE, SHARED_CATEGORICAL
    else:
        feature_cols = [c for c in UNSW_COLUMNS if c not in UNSW_METADATA_ONLY]
        categorical = UNSW_CATEGORICAL
        discrete = UNSW_DISCRETE
        numeric = [c for c in feature_cols if c not in categorical and c not in discrete]
        frame = frame[["__time", "__label", "__src", "__dst", "__proto"] + feature_cols]

    frame = _finalize(frame, numeric, discrete, categorical, notes)
    return LoadedData(frame, list(numeric), list(discrete), list(categorical), True, True,
                      files, rows_read, len(frame), notes)


# ---------------------------------------------------------------------------
# CICIDS2017 (GeneratedLabelledFlows / TrafficLabelling CSVs)
# ---------------------------------------------------------------------------
CIC17_METADATA_ONLY = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Timestamp", "Label"]


def _parse_cicids2017_time(series: pd.Series, notes: list[str]) -> np.ndarray:
    """CICIDS2017 TrafficLabelling timestamps use a 12-hour clock without AM/PM.

    Captures ran from roughly 08:00 to 18:00, so hours 1-7 are afternoon hours and
    are shifted by 12. Without this correction, afternoon flows sort before
    morning flows and the temporal graph is built in the wrong order.
    """

    s = series.astype(str).str.strip()
    parsed = _parse_datetime(s, "CICIDS2017 Timestamp")
    fix = parsed.dt.hour.between(1, 7)
    parsed = parsed.where(~fix, parsed + pd.Timedelta(hours=12))
    notes.append(f"CICIDS2017: shifted {int(fix.sum())} timestamps with hour 1-7 by +12h (12-hour clock)")
    months = parsed.dropna().dt.to_period("M").astype(str).value_counts()
    notes.append(f"CICIDS2017: parsed months {months.to_dict()} (expected 2017-07 only; check the day/month order if not)")
    notes.append("CICIDS2017: several files record timestamps at one-minute resolution; flows within the same minute keep file order")
    return _datetime_to_seconds(parsed)


def load_cicids2017(paths: Iterable[str], feature_set: str = "full") -> LoadedData:
    """Official GeneratedLabelledFlows.zip -> TrafficLabelling/*.pcap_ISCX.csv files.

    The MachineLearningCVE CSVs are NOT supported: they omit Timestamp,
    Source/Destination IP and Protocol, which graph construction requires.
    """

    files = expand_paths(paths)
    notes: list[str] = []
    parts = []
    for f in files:
        part = pd.read_csv(f, low_memory=False, encoding="latin-1", skipinitialspace=True)
        part.columns = [c.strip() for c in part.columns]
        parts.append(part)
    raw = pd.concat(parts, ignore_index=True)
    rows_read = len(raw)
    raw = raw.dropna(how="all")
    raw = raw[raw["Label"].notna()]

    frame = raw.copy()
    frame["__time"] = _parse_cicids2017_time(frame["Timestamp"], notes)
    label = frame["Label"].astype(str).str.strip()
    label = label.str.replace(r"[^\x20-\x7E]", "-", regex=True).str.replace(r"\s+", " ", regex=True)
    frame["__label"] = np.where(label.str.upper() == "BENIGN", "benign", label.str.lower())
    frame["__src"] = frame["Source IP"].astype(str).str.strip()
    frame["__dst"] = frame["Destination IP"].astype(str).str.strip()
    frame["__proto"] = frame["Protocol"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    if feature_set == "shared":
        frame = _shared_view(frame, "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
                             "Total Length of Fwd Packets", "Total Length of Bwd Packets",
                             "Destination Port", "Protocol", 1e-6)
        numeric, discrete, categorical = SHARED_NUMERIC, SHARED_DISCRETE, SHARED_CATEGORICAL
    else:
        feature_cols = [c for c in raw.columns if c not in CIC17_METADATA_ONLY]
        categorical = ["Protocol"]
        discrete = ["Destination Port"] + [c for c in feature_cols if "Flag" in c]
        numeric = [c for c in feature_cols if c not in categorical and c not in discrete]
        frame = frame[["__time", "__label", "__src", "__dst", "__proto"] + feature_cols]

    frame = _finalize(frame, numeric, discrete, categorical, notes)
    return LoadedData(frame, list(numeric), list(discrete), list(categorical), True, True,
                      files, rows_read, len(frame), notes)


# ---------------------------------------------------------------------------
# CSE-CIC-IDS2018 ("Processed Traffic Data for ML Algorithms" CSVs)
# ---------------------------------------------------------------------------
CIC18_METADATA_ONLY = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp", "Label"]


def load_cse_cicids2018(paths: Iterable[str], feature_set: str = "full") -> LoadedData:
    """Official processed CSVs (e.g. Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv).

    Only the Tuesday-20-02-2018 file contains Src/Dst IP. For the other days the
    structural relation falls back to shared protocol and shared destination
    port, which is reported in the manuscript.
    """

    files = expand_paths(paths)
    notes: list[str] = []
    parts = []
    for f in files:
        part = pd.read_csv(f, low_memory=False, encoding="latin-1")
        part.columns = [c.strip() for c in part.columns]
        part = part[part["Label"].astype(str).str.strip() != "Label"]  # repeated header rows
        parts.append(part)
    raw = pd.concat(parts, ignore_index=True)
    rows_read = len(raw)

    frame = raw.copy()
    parsed = _parse_datetime(frame["Timestamp"], "CSE-CIC-IDS2018 Timestamp")
    frame["__time"] = _datetime_to_seconds(parsed)
    label = frame["Label"].astype(str).str.strip()
    frame["__label"] = np.where(label.str.lower() == "benign", "benign", label.str.lower())
    has_ip = "Src IP" in frame.columns and frame["Src IP"].notna().mean() > 0.99
    if "Src IP" in frame.columns:
        frame["__src"] = frame["Src IP"].astype(str).str.strip().replace({"nan": np.nan})
        frame["__dst"] = frame["Dst IP"].astype(str).str.strip().replace({"nan": np.nan})
    if not has_ip:
        notes.append("CSE-CIC-IDS2018: Src/Dst IP missing for most rows; relational edges use protocol and destination port")
        frame["__src"] = np.nan
        frame["__dst"] = "port:" + frame["Dst Port"].astype(str).str.strip()
    frame["__proto"] = frame["Protocol"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    if feature_set == "shared":
        frame = _shared_view(frame, "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
                             "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Dst Port", "Protocol", 1e-6)
        numeric, discrete, categorical = SHARED_NUMERIC, SHARED_DISCRETE, SHARED_CATEGORICAL
    else:
        feature_cols = [c for c in raw.columns if c not in CIC18_METADATA_ONLY]
        categorical = ["Protocol"]
        discrete = ["Dst Port"] + [c for c in feature_cols if "Flag" in c]
        numeric = [c for c in feature_cols if c not in categorical and c not in discrete]
        frame = frame[["__time", "__label", "__src", "__dst", "__proto"] + feature_cols]

    frame = _finalize(frame, numeric, discrete, categorical, notes)
    return LoadedData(frame, list(numeric), list(discrete), list(categorical), has_ip, True,
                      files, rows_read, len(frame), notes)


# ---------------------------------------------------------------------------
# Generic CSV (column names supplied in the config)
# ---------------------------------------------------------------------------
def load_generic(cfg) -> LoadedData:
    files = expand_paths(cfg.data_paths)
    notes: list[str] = []
    raw = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    raw.columns = [c.strip() for c in raw.columns]
    rows_read = len(raw)
    if cfg.label_col is None or cfg.time_col is None:
        raise ValueError("generic datasets need label_col and time_col in the config")

    frame = raw.copy()
    t = frame[cfg.time_col]
    if pd.api.types.is_numeric_dtype(t):
        frame["__time"] = t.astype(float)
    else:
        parsed = _parse_datetime(t, f"{cfg.time_col}")
        frame["__time"] = _datetime_to_seconds(parsed)
    label = frame[cfg.label_col].astype(str).str.strip()
    benign = {v.lower() for v in cfg.benign_values}
    frame["__label"] = np.where(label.str.lower().isin(benign), "benign", label.str.lower())
    meta = {"__src": cfg.src_col, "__dst": cfg.dst_col, "__proto": cfg.protocol_col}
    for new, old in meta.items():
        if old:
            frame[new] = frame[old].astype(str).str.strip()

    excluded = {cfg.label_col, cfg.time_col, cfg.src_col, cfg.dst_col, *cfg.drop_cols} - {None}
    feature_cols = [c for c in raw.columns if c not in excluded]
    categorical = [c for c in feature_cols if raw[c].dtype == "object" and c not in cfg.discrete_cols]
    discrete = [c for c in cfg.discrete_cols if c in feature_cols]
    numeric = [c for c in feature_cols if c not in categorical and c not in discrete]
    keep = ["__time", "__label"] + [c for c in meta if c in frame.columns] + feature_cols
    frame = _finalize(frame[keep], numeric, discrete, categorical, notes)
    return LoadedData(frame, numeric, discrete, categorical,
                      bool(cfg.src_col and cfg.dst_col), bool(cfg.protocol_col),
                      files, rows_read, len(frame), notes)


def load_dataset(cfg) -> LoadedData:
    name = cfg.dataset.lower()
    if name == "unsw_nb15":
        return load_unsw_nb15(cfg.data_paths, cfg.feature_set)
    if name == "cicids2017":
        return load_cicids2017(cfg.data_paths, cfg.feature_set)
    if name == "cse_cicids2018":
        return load_cse_cicids2018(cfg.data_paths, cfg.feature_set)
    if name == "generic":
        return load_generic(cfg)
    raise ValueError(f"Unknown dataset '{cfg.dataset}'")
