"""Find the needed dataset files in a folder of downloads and extract them into data/.

    python prepare_data.py --source "D:/downloads/datasets"            # extract
    python prepare_data.py --source "D:/downloads/datasets" --list     # only show what would be used

The source folder may contain zip files (zips inside zips are fine), already-extracted
folders, or loose CSV files, in any arrangement. Files are recognised by their CONTENT
(the header line), not only by their names, and only the files the experiments need are
extracted:

    data/UNSW-NB15/UNSW-NB15_1.csv ... UNSW-NB15_4.csv
    data/CICIDS2017/TrafficLabelling/*.csv                               (8 files)
    data/CSE-CIC-IDS2018/Processed Traffic Data for ML Algorithms/*.csv   (10 files)

Files that look similar but cannot be used (UNSW-NB15 training/testing-set, CICIDS2017
MachineLearningCVE) are reported and skipped. Nothing in the source folder is modified.
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

TARGETS = {
    "unsw": (Path("UNSW-NB15"), 4, "UNSW-NB15"),
    "cic17": (Path("CICIDS2017") / "TrafficLabelling", 8, "CICIDS2017"),
    "cic18": (Path("CSE-CIC-IDS2018") / "Processed Traffic Data for ML Algorithms", 10, "CSE-CIC-IDS2018"),
}
NESTED_ZIP_IN_MEMORY = 200 * 1024 * 1024   # nested zips larger than this are unpacked to a temp file


@dataclass
class Found:
    kind: str            # unsw | cic17 | cic18 | skip
    name: str            # output file name
    where: str           # human-readable location
    opener: object       # callable returning a binary file object
    size: int
    reason: str = ""


def _header_fields(first_line: bytes) -> list[str]:
    text = first_line.decode("latin-1", errors="replace").strip().lstrip("\ufeff")
    return [f.strip() for f in text.split(",")]


def classify(name: str, first_line: bytes) -> tuple[str, str]:
    """Return (kind, reason) for a CSV, using its header line."""

    base = Path(name).name
    fields = _header_fields(first_line)
    lower = {f.lower() for f in fields}

    # UNSW-NB15 full records: no header, 49 fields, numeric start time in column 29
    if len(fields) == 49 and fields[0].count(".") == 3:
        try:
            float(fields[28])
            return "unsw", ""
        except ValueError:
            pass
    if "attack_cat" in lower and "stime" not in lower and ("sbytes" in lower or "dur" in lower):
        return "skip", "UNSW-NB15 training/testing-set file (no timestamps or IPs) - not usable"
    if "srcip" in lower and "stime" in lower and len(fields) == 49:
        return "unsw", ""   # a UNSW-NB15 full file someone saved with a header row

    if "label" in lower and ("flow duration" in lower):
        if "timestamp" in lower and ("source ip" in lower or "src ip" in lower) and "total fwd packets" in lower:
            return "cic17", ""
        if "timestamp" in lower and ("tot fwd pkts" in lower or "dst port" in lower):
            return "cic18", ""
        if "timestamp" not in lower:
            return "skip", "CIC file without Timestamp (e.g. MachineLearningCVE) - not usable"
    if base.lower().startswith("nusw-nb15_features") or "features" in base.lower():
        return "skip", "feature description file (not needed)"
    return "skip", "not one of the needed dataset files"


def _first_line(fp) -> bytes:
    return fp.readline(1_000_000)


def scan_zip(zf: zipfile.ZipFile, label: str, found: list[Found], tmpdir: Path) -> None:
    for info in zf.infolist():
        if info.is_dir():
            continue
        member = info.filename
        where = f"{label} -> {member}"
        low = member.lower()
        if low.endswith(".zip"):
            if info.file_size <= NESTED_ZIP_IN_MEMORY:
                inner = zipfile.ZipFile(io.BytesIO(zf.read(info)))
            else:
                path = tmpdir / f"nested_{len(list(tmpdir.iterdir()))}.zip"
                with zf.open(info) as src, open(path, "wb") as dst:
                    shutil.copyfileobj(src, dst, 16 * 1024 * 1024)
                inner = zipfile.ZipFile(path)
            scan_zip(inner, where, found, tmpdir)
        elif low.endswith(".csv"):
            with zf.open(info) as fp:
                kind, reason = classify(member, _first_line(fp))
            found.append(Found(kind, Path(member).name.strip(), where,
                               (lambda z=zf, i=info: z.open(i)), info.file_size, reason))
        elif low.endswith((".rar", ".7z", ".tar", ".gz", ".tgz")):
            found.append(Found("skip", Path(member).name, where, None, info.file_size,
                               "unsupported archive type - extract it manually, then run again"))


def scan(source: Path, tmpdir: Path, exclude: tuple[Path, ...] = ()) -> list[Found]:
    found: list[Found] = []
    exclude = tuple(e.resolve() for e in exclude)
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if any(e == path.resolve() or e in path.resolve().parents for e in exclude):
            continue
        low = path.name.lower()
        if low.endswith(".zip"):
            try:
                scan_zip(zipfile.ZipFile(path), str(path), found, tmpdir)
            except zipfile.BadZipFile:
                found.append(Found("skip", path.name, str(path), None, path.stat().st_size,
                                   "damaged or incomplete zip - download it again"))
        elif low.endswith(".csv"):
            with open(path, "rb") as fp:
                kind, reason = classify(path.name, _first_line(fp))
            found.append(Found(kind, path.name, str(path), (lambda p=path: open(p, "rb")), path.stat().st_size, reason))
        elif low.endswith((".rar", ".7z", ".tar", ".gz", ".tgz")):
            found.append(Found("skip", path.name, str(path), None, path.stat().st_size,
                               "unsupported archive type - extract it manually, then run again"))
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="folder containing the downloaded zips/CSVs")
    ap.add_argument("--dest", default="data", help="output folder (default: data)")
    ap.add_argument("--list", action="store_true", help="only report, do not extract")
    ap.add_argument("--exclude", nargs="*", default=[], help="folders inside --source to ignore")
    args = ap.parse_args()

    source, dest = Path(args.source), Path(args.dest)
    if not source.is_dir():
        print(f"Source folder not found: {source}")
        return 1
    if dest.resolve() == source.resolve():
        print("--dest must be a different folder from --source.")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        found = scan(source, Path(tmp), exclude=(dest, *[Path(e) for e in args.exclude]))

        chosen: dict[str, dict[str, Found]] = {k: {} for k in TARGETS}
        duplicates = []
        for f in found:
            if f.kind in chosen:
                if f.name in chosen[f.kind]:
                    duplicates.append(f)
                else:
                    chosen[f.kind][f.name] = f

        print("\n=== Files that will be used ===")
        for kind, (sub, expected, label) in TARGETS.items():
            files = chosen[kind]
            status = "OK" if len(files) == expected else f"expected {expected}"
            print(f"\n{label}: {len(files)} file(s) [{status}]  ->  {dest / sub}")
            for name, f in sorted(files.items()):
                print(f"   {name:60s} {f.size / 1e6:9.1f} MB   from {f.where}")

        skipped = [f for f in found if f.kind == "skip" and f.reason != "not one of the needed dataset files"]
        if skipped or duplicates:
            print("\n=== Skipped ===")
            for f in skipped:
                print(f"   {f.name}: {f.reason}   ({f.where})")
            for f in duplicates:
                print(f"   {f.name}: duplicate copy, first one kept   ({f.where})")

        if args.list:
            print("\n(--list given: nothing extracted)")
            return 0

        for kind, (sub, _, label) in TARGETS.items():
            if not chosen[kind]:
                continue
            out_dir = dest / sub
            out_dir.mkdir(parents=True, exist_ok=True)
            for name, f in sorted(chosen[kind].items()):
                target = out_dir / name
                if target.exists() and target.stat().st_size == f.size:
                    print(f"   exists, skipped: {target}")
                    continue
                with f.opener() as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst, 16 * 1024 * 1024)
                print(f"   extracted: {target}")

    missing = [TARGETS[k][2] for k in TARGETS if len(chosen[k]) != TARGETS[k][1]]
    print("\nDone." if not missing else f"\nDone, but check: {', '.join(missing)} "
          "(CSE-CIC-IDS2018 is only needed for the cross-dataset table).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
