"""Small synthetic files in the raw layouts of the three datasets, for tests only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from datasets import UNSW_COLUMNS


def _flows(n: int, rng: np.random.Generator):
    """Benign background with bursts of attack flows from a few hosts."""

    t = np.sort(rng.uniform(0, 3600, n))
    src = rng.integers(0, 40, n)
    dst = rng.integers(0, 15, n)
    attack = np.zeros(n, dtype=bool)
    for start in rng.uniform(0, 3400, 6):
        burst = (t > start) & (t < start + 120) & (src % 7 == 0)
        attack |= burst
    dur = np.where(attack, rng.exponential(0.05, n), rng.exponential(2.0, n))
    fp = np.where(attack, rng.poisson(2, n), rng.poisson(12, n)) + 1
    bp = np.where(attack, rng.poisson(1, n), rng.poisson(10, n))
    fb = fp * np.where(attack, rng.normal(60, 5, n), rng.normal(500, 100, n)).clip(40)
    bb = bp * rng.normal(800, 200, n).clip(40)
    port = np.where(attack, rng.choice([22, 80, 445], n), rng.choice([80, 443, 53, 8080], n))
    proto = np.where(port == 53, 17, 6)
    return t, src, dst, attack, dur, fp, bp, fb, bb, port, proto


def write_unsw(folder: Path, n: int = 3000, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    t, src, dst, attack, dur, fp, bp, fb, bb, port, proto = _flows(n, rng)
    df = pd.DataFrame(0.0, index=range(n), columns=UNSW_COLUMNS)
    df["srcip"] = [f"10.0.0.{s}" for s in src]
    df["dstip"] = [f"149.171.126.{d}" for d in dst]
    df["sport"] = rng.integers(1024, 65535, n).astype(str)
    df.loc[:5, "sport"] = "0x000b"                     # hex ports appear in the real files
    df["dsport"] = port
    df["proto"] = np.where(proto == 17, "udp", "tcp")
    df["state"] = rng.choice(["FIN", "CON", "INT"], n)
    df["service"] = rng.choice(["-", "http", "dns"], n)
    df["dur"], df["spkts"], df["dpkts"], df["sbytes"], df["dbytes"] = dur, fp, bp, fb, bb
    df["sttl"] = np.where(attack, 254, 31)
    df["sload"] = fb / (dur + 1e-3)
    df["stime"] = (1421927414 + t).astype(np.int64)
    df["ltime"] = df["stime"] + dur.astype(np.int64)
    df["ct_ftp_cmd"] = " "                            # blank strings appear in the real files
    df["attack_cat"] = np.where(attack, rng.choice([" Fuzzers", "Exploits", "Backdoors"], n), "")
    df["label"] = attack.astype(int)
    paths = []
    for k, part in enumerate([df.iloc[: n // 2], df.iloc[n // 2:]], start=1):
        p = folder / f"UNSW-NB15_{k}.csv"
        part.to_csv(p, header=False, index=False)
        paths.append(str(p))
    pd.concat([df.iloc[:3]] * 1).to_csv(folder / "UNSW-NB15_1.csv", mode="a", header=False, index=False)  # duplicates
    return paths


def write_cicids2017(folder: Path, n: int = 3000, seed: int = 1) -> list[str]:
    rng = np.random.default_rng(seed)
    t, src, dst, attack, dur, fp, bp, fb, bb, port, proto = _flows(n, rng)
    base = pd.Timestamp("2017-07-05 11:00:00")
    times = base + pd.to_timedelta(t * 3.0, unit="s")       # 11:00-14:00, crosses noon
    hours12 = ((times.hour - 1) % 12) + 1                    # 12-hour clock, no AM/PM
    stamp = [f"{d:02d}/07/2017 {h}:{m:02d}" for d, h, m in zip(times.day, hours12, times.minute)]
    df = pd.DataFrame({
        "Flow ID": [f"f{i}" for i in range(n)],
        " Source IP": [f"192.168.10.{s}" for s in src],
        " Source Port": rng.integers(1024, 65535, n),
        " Destination IP": [f"205.174.165.{d}" for d in dst],
        " Destination Port": port,
        " Protocol": proto,
        " Timestamp": stamp,
        " Flow Duration": (dur * 1e6).astype(np.int64),
        " Total Fwd Packets": fp,
        " Total Backward Packets": bp,
        "Total Length of Fwd Packets": fb,
        " Total Length of Bwd Packets": bb,
        "Flow Bytes/s": np.where(rng.random(n) < 0.01, np.inf, fb / (dur + 1e-6)),
        " SYN Flag Count": rng.integers(0, 2, n),
        " Label": np.where(attack, rng.choice(["DoS Hulk", "Web Attack \x96 Brute Force"], n), "BENIGN"),
    })
    paths = []
    for k, part in enumerate([df.iloc[: n // 2], df.iloc[n // 2:]]):
        p = folder / f"Day{k}-WorkingHours.pcap_ISCX.csv"
        part.to_csv(p, index=False, encoding="latin-1")
        paths.append(str(p))
    return paths


def write_cse_cicids2018(folder: Path, n: int = 3000, seed: int = 2) -> list[str]:
    rng = np.random.default_rng(seed)
    t, src, dst, attack, dur, fp, bp, fb, bb, port, proto = _flows(n, rng)
    times = pd.Timestamp("2018-02-14 09:00:00") + pd.to_timedelta(t, unit="s")
    df = pd.DataFrame({
        "Dst Port": port, "Protocol": proto,
        "Timestamp": times.strftime("%d/%m/%Y %H:%M:%S"),
        "Flow Duration": (dur * 1e6).astype(np.int64),
        "Tot Fwd Pkts": fp, "Tot Bwd Pkts": bp, "TotLen Fwd Pkts": fb, "TotLen Bwd Pkts": bb,
        "FIN Flag Cnt": rng.integers(0, 2, n),
        "Label": np.where(attack, "FTP-BruteForce", "Benign"),
    })
    header_row = pd.DataFrame([df.columns.tolist()], columns=df.columns)       # repeated header inside file
    df = pd.concat([df.iloc[: n // 2], header_row, df.iloc[n // 2:]], ignore_index=True)
    p = folder / "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"
    df.to_csv(p, index=False)
    return [str(p)]
