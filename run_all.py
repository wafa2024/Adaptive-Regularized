"""ONE command for everything: data, tests, all experiments, tables, figures.

Put this file in the folder that holds your downloaded dataset zips, together with
Adaptive-Regularized-v2.0.zip (the code), then run ONE of:

    python run_all.py                  (command line, from inside that folder)
    %run run_all.py                    (Jupyter notebook opened in that folder)

Everything is written to the subfolder ARTGL_work/. The script can be stopped and
started again at any time: finished steps and finished training runs are skipped.

Optional:
    python run_all.py --only data        just prepare the data and stop
    python run_all.py --only tests       up to and including the tests
    python run_all.py --seeds 0 1 2      fewer seeds (state this in the paper)
    python run_all.py --skip-cse         skip CSE-CIC-IDS2018 (very large; cross-dataset only)
    python run_all.py --allow-partial-cse   use CSE-CIC-IDS2018 even with a file missing
    python run_all.py --torch-cuda cu128    install the CUDA 12.8 build of PyTorch instead of 12.6
    python run_all.py --only report      rebuild ALL_RESULTS.txt from whatever has finished so far
    python run_all.py --sensitivity-seeds 0 1 2    3 seeds for the sensitivity sweep only
    python run_all.py --work /content/ARTGL_work --results /content/drive/MyDrive/ARTGL_results   (Colab)

Steps:
  1. unzip the code into ARTGL_work/code
  2. install PyTorch (CUDA build if an NVIDIA GPU is found) and the other packages
  3. find the needed dataset files in your zips and extract them to ARTGL_work/data
  4. run the tests (the experiments are not started unless they pass)
  5. UNSW-NB15: main, ablation, noise, adversarial, sensitivity
  6. CICIDS2017: main
  7. cross-dataset (3 settings)
  7b. interpretability analysis (t-SNE, edge coefficients; evaluation only)
  8. tables (CSV + LaTeX) and figures
  9. write every table into one text file: <results>/ALL_RESULTS.txt
 10. pack the results to send back: <results>/send_to_claude.zip
A log of every step is written to <results>/run_log.txt (default ARTGL_work/results/).
"""

from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(os.getcwd()).resolve()
WORK = HERE / "ARTGL_work"
CODE = WORK / "code"
DATA = WORK / "data"
RESULTS = WORK / "results"
LOG = WORK / "run_log.txt"
HAS_CUDA = False

PATTERNS = {
    "unsw_nb15": ("UNSW-NB15/UNSW-NB15_[1-4].csv", 4),
    "cicids2017": ("CICIDS2017/TrafficLabelling/*.csv", 8),
    "cse_cicids2018": ("CSE-CIC-IDS2018/Processed Traffic Data for ML Algorithms/*.csv", 10),
}


# --------------------------------------------------------------------------- helpers
def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fp:
        fp.write(line + "\n")


def run(args: list[str], cwd: Path = None, allow_fail: bool = False) -> bool:
    """Run a command, streaming its output (works in a terminal and in Jupyter)."""

    log("RUN " + " ".join(f'"{a}"' if " " in a else a for a in args))
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", ARTGL_CACHE_DIR=str(WORK / "cache"))
    proc = subprocess.Popen(args, cwd=str(cwd or CODE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1, env=env)
    for line in proc.stdout:
        print(line, end="", flush=True)
        # open-append-close per line: on Google Drive this is what makes the line survive
        # if the Colab session is killed (a kept-open file loses its buffered contents)
        with open(LOG, "a", encoding="utf-8") as fp:
            fp.write(line)
    ok = proc.wait() == 0
    if not ok:
        log(f"FAILED (exit code {proc.returncode})")
        if not allow_fail:
            log("Stopped. Fix the error above (or send me ARTGL_work/run_log.txt), then run again; "
                "finished steps will be skipped.")
            raise SystemExit(1)
    return ok


def done_marker(name: str) -> Path:
    return RESULTS / f".done_{name}"   # kept with the results, so it survives a Colab disconnect


# --------------------------------------------------------------------------- steps
def _code_version(folder: Path) -> str:
    f = folder / "VERSION"
    return f.read_text().strip() if f.exists() else "unknown (older than 2.0.3)"


def step_code() -> None:
    """Unpack the NEWEST Adaptive-Regularized-v2*.zip (by modification time) into its own folder.

    Every distinct zip gets a fresh folder ARTGL_work/code_<id>, so an old, half-deleted or
    locked code folder can never be used by mistake. The folder in use is logged together
    with the code version.
    """

    global CODE
    import hashlib

    def zip_version(path: Path) -> tuple:
        try:
            with zipfile.ZipFile(path) as z:
                name = next((n for n in z.namelist() if n.endswith("/VERSION")), None)
                text = z.read(name).decode().strip() if name else "0"
            return tuple(int(x) for x in text.split()[0].split("."))
        except (ValueError, zipfile.BadZipFile):
            return (0,)

    # highest code version first; the newest file wins among equal versions
    zips = sorted(HERE.glob("Adaptive-Regularized-v2*.zip"),
                  key=lambda p: (zip_version(p), p.stat().st_mtime), reverse=True)
    if not zips:
        log("Could not find the code. Put Adaptive-Regularized-v2.0.zip in this folder and run again.")
        raise SystemExit(1)
    newest = zips[0]
    digest = hashlib.sha1(newest.read_bytes()).hexdigest()[:10]
    CODE = WORK / f"code_{digest}"
    if not (CODE / "experiments.py").exists():
        tmp = WORK / f"_unzip_{digest}"
        shutil.rmtree(tmp, ignore_errors=True)
        with zipfile.ZipFile(newest) as z:
            z.extractall(tmp)
        inner = next(p.parent for p in tmp.rglob("experiments.py"))
        inner.rename(CODE)
        shutil.rmtree(tmp, ignore_errors=True)
    if len(zips) > 1:
        log(f"Several code zips found; using the newest: {newest.name}")
    log(f"Code version {_code_version(CODE)} | zip {newest.name} | folder {CODE.name}")


def _torch_status() -> tuple[bool, bool, str]:
    """(installed, cuda_available, description) - checked in a fresh process."""

    code = ("import torch, sys; ok = torch.cuda.is_available(); "
            "print(torch.__version__, ok, torch.cuda.get_device_name(0) if ok else '-')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        return False, False, ""
    version, ok, name = r.stdout.strip().split(" ", 2)
    return True, ok == "True", f"PyTorch {version}, GPU: {name}"


def _nvidia_smi() -> str | None:
    candidates = ["nvidia-smi",
                  r"C:\Windows\System32\nvidia-smi.exe",
                  r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"]
    for exe in candidates:
        try:
            r = subprocess.run([exe, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                               capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        except (FileNotFoundError, OSError):
            continue
    return None


def _graphics_cards() -> str:
    """Names of all graphics adapters (Windows), to tell the user what the computer has."""

    if os.name != "nt":
        return "n/a"
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "(Get-CimInstance Win32_VideoController).Name -join '; '"],
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip() or "unknown"
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return "unknown"


def _has_nvidia_gpu() -> bool:
    info = _nvidia_smi()
    log(f"Graphics cards: {_graphics_cards()} | NVIDIA driver check: {info or 'no NVIDIA driver found'}")
    return info is not None


def step_packages(torch_cuda: str) -> None:
    log(f"Python {sys.version.split()[0]} at {sys.executable}")
    if sys.version_info < (3, 9):
        log("Python 3.9 or newer is needed. Install Python 3.11 from python.org and run this script with it.")
        raise SystemExit(1)
    # The newest PyTorch / PyTorch Geometric need Python 3.10+. On Python 3.9 use the last
    # releases that support it (PyTorch 2.6 has CUDA 12.6 builds).
    old_python = sys.version_info < (3, 10)
    torch_spec = "torch==2.6.0" if old_python else "torch"
    pyg_spec = "torch_geometric==2.6.1" if old_python else "torch_geometric"
    if old_python:
        log("Python 3.9 detected: using PyTorch 2.6.0 and PyTorch Geometric 2.6.1 (the last versions for 3.9).")
    installed, cuda, desc = _torch_status()
    gpu = _has_nvidia_gpu()
    if not installed or (gpu and not cuda):
        if gpu:
            index = f"https://download.pytorch.org/whl/{torch_cuda}"
            log(("PyTorch not found" if not installed else "PyTorch is installed without GPU support")
                + f" - installing the CUDA build ({torch_cuda}). This downloads about 2.5 GB.")
            extra = [] if not installed else ["--force-reinstall"]
        else:
            index = "https://download.pytorch.org/whl/cpu"
            log("No NVIDIA GPU detected - installing the CPU build of PyTorch (training will be slow).")
            extra = []
        run([sys.executable, "-m", "pip", "install", *extra, torch_spec, "--index-url", index], cwd=HERE,
            allow_fail=True)
        installed, cuda, desc = _torch_status()
        if not installed:
            log("PyTorch installation failed - see the messages above, or install it manually from "
                "https://pytorch.org/get-started/locally/ and run again.")
            raise SystemExit(1)
        if gpu and not cuda:
            log(f"PyTorch still cannot use the GPU. Try again with --torch-cuda cu128, or update the NVIDIA driver.")
            raise SystemExit(1)
    log(desc)
    global HAS_CUDA
    HAS_CUDA = cuda
    needed = {"torch_geometric": pyg_spec, "pandas": "pandas", "sklearn": "scikit-learn",
              "xgboost": "xgboost", "matplotlib": "matplotlib", "tqdm": "tqdm", "numpy": "numpy"}
    missing = []
    for module, package in needed.items():
        if subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True).returncode != 0:
            missing.append(package)
    if missing:
        log(f"Installing: {', '.join(missing)}")
        run([sys.executable, "-m", "pip", "install", *missing], cwd=HERE)
    code = ("import pandas, numpy, sklearn, torch, torch_geometric; "
            "print('versions: pandas', pandas.__version__, '| numpy', numpy.__version__, '| scikit-learn', "
            "sklearn.__version__, '| torch', torch.__version__, '| torch_geometric', torch_geometric.__version__)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    log(r.stdout.strip() or r.stderr.strip()[-500:])


def step_data(skip_cse: bool, partial_cse: bool) -> dict[str, list[str]]:
    run([sys.executable, str(CODE / "prepare_data.py"), "--source", str(HERE), "--dest", str(DATA),
         "--exclude", str(WORK), str(RESULTS)], cwd=HERE)
    available = {}
    for name, (pattern, expected) in PATTERNS.items():
        files = glob.glob(str(DATA / pattern))
        if len(files) == expected or (name == "cse_cicids2018" and partial_cse and files):
            available[name] = [str(DATA / pattern)]
            if len(files) != expected:
                log(f"NOTE: using {len(files)} of {expected} {name} files (--allow-partial-cse). "
                    "State this in the manuscript; data_summary.json lists the files used.")
        elif files:
            log(f"WARNING: {name}: found {len(files)} of {expected} files - it will be skipped. "
                f"Add the missing file(s) to this folder and run again"
                + (" (or use --allow-partial-cse)." if name == "cse_cicids2018" else "."))
        else:
            log(f"{name}: no files found - it will be skipped.")
    if skip_cse:
        available.pop("cse_cicids2018", None)
    if "unsw_nb15" not in available:
        log("UNSW-NB15 is required for the main experiments. Add its zip to this folder and run again.")
        raise SystemExit(1)
    return available


def write_configs(available: dict[str, list[str]]) -> Path:
    cfg_dir = WORK / "configs"
    cfg_dir.mkdir(exist_ok=True)
    for name, paths in available.items():
        cfg = json.loads((CODE / "configs" / f"{name}.json").read_text())
        cfg["data_paths"] = paths
        cfg["output_dir"] = str(RESULTS)
        (cfg_dir / f"{name}.json").write_text(json.dumps(cfg, indent=2))
    return cfg_dir


def step_tests() -> None:
    if done_marker("tests").exists():
        log("Tests already passed earlier - skipped.")
        return
    run([sys.executable, "tests/test_data_pipeline.py"])
    run([sys.executable, "tests/test_model_torch.py"])
    done_marker("tests").write_text(time.ctime())
    log("All tests passed.")


def step_experiments(cfg_dir: Path, available: dict, seeds: list[str], sens_seeds: list[str]) -> None:
    jobs = [("unsw_nb15", ["main"]), ("unsw_nb15", ["ablation", "noise"]), ("unsw_nb15", ["adversarial"]),
            ("cicids2017", ["main"]), ("unsw_nb15", ["sensitivity"])]
    for dataset, suites in jobs:
        if dataset not in available:
            continue
        key = f"{dataset}_{'_'.join(suites)}"
        if done_marker(key).exists():
            log(f"{dataset} {' + '.join(suites)}: already done - skipped.")
            continue
        log(f"=== {dataset}: {' + '.join(suites)} ===")
        run([sys.executable, "experiments.py", "--config", str(cfg_dir / f"{dataset}.json"),
             "--suites", *suites, "--seeds", *(sens_seeds if suites == ["sensitivity"] else seeds)])
        done_marker(key).write_text(time.ctime())


def step_cross(cfg_dir: Path, available: dict, seeds: list[str]) -> None:
    settings = [(["unsw_nb15"], "cicids2017"), (["cicids2017"], "unsw_nb15"),
                (["unsw_nb15", "cicids2017"], "cse_cicids2018")]
    for train, test in settings:
        if not all(d in available for d in train + [test]):
            log(f"cross-dataset {'+'.join(train)} -> {test}: data missing - skipped.")
            continue
        out = RESULTS / "cross_dataset" / f"{'+'.join(train)}__to__{test}" / "cross_dataset_per_seed.csv"
        if out.exists():
            log(f"cross-dataset {'+'.join(train)} -> {test}: already done - skipped.")
            continue
        log(f"=== cross-dataset {'+'.join(train)} -> {test} ===")
        ok = run([sys.executable, "cross_dataset.py", "--train", *[str(cfg_dir / f"{t}.json") for t in train],
                  "--test", str(cfg_dir / f"{test}.json"), "--seeds", *seeds,
                  "--output_dir", str(RESULTS / "cross_dataset")], allow_fail=True)
        if not ok:
            log("This cross-dataset setting failed (often: not enough memory for CSE-CIC-IDS2018). "
                "Continuing with the rest; send me the log.")


def step_interpret(cfg_dir: Path, available: dict) -> None:
    """Interpretability analysis of the trained models (t-SNE, edge coefficients, phi weights,
    extended noise radii). Evaluation only: uses the models saved by the experiments."""

    for dataset, extra in [("unsw_nb15", []), ("cicids2017", ["--skip_noise"])]:
        if dataset not in available:
            continue
        key = f"interpretability_{dataset}"
        if done_marker(key).exists():
            log(f"{dataset} interpretability: already done - skipped.")
            continue
        log(f"=== {dataset}: interpretability (evaluation only, no training) ===")
        ok = run([sys.executable, "interpretability.py", "--config", str(cfg_dir / f"{dataset}.json"), *extra],
                 allow_fail=True)
        if ok:
            done_marker(key).write_text(time.ctime())
        else:
            log("The interpretability step failed; the other results are not affected. Send me the log.")


def step_tables(available: dict) -> None:
    cross = RESULTS / "cross_dataset"
    for dataset, label in [("unsw_nb15", "UNSW-NB15"), ("cicids2017", "CICIDS2017")]:
        if not (RESULTS / dataset).exists():
            continue
        args = [sys.executable, "aggregate.py", "--results", str(RESULTS / dataset)]
        if dataset == "unsw_nb15" and cross.exists():
            args += ["--cross", str(cross)]
        run(args)
        run([sys.executable, "plots.py", "--tables", str(RESULTS / dataset / "tables"), "--dataset", label])


TABLE_TITLES = [
    ("table3_main", "Table 3. Classification results (binary; malicious = positive)"),
    ("table4_adaptive_edges", "Table 4. Effect of adaptive edge weighting"),
    ("table5_ablation", "Table 5. Ablation of the composite objective (Delta_eps at eps = 0.10)"),
    ("table6_noise", "Table 6. Bounded random noise (full model)"),
    ("table7_adversarial", "Table 7. FGSM / PGD at eps = 0.05"),
    ("table8_sensitivity", "Table 8. Sensitivity to lambda_1..lambda_4"),
    ("table9_cross_dataset", "Table 9. Cross-dataset F1"),
    ("table11_runtime", "Table 11. Computational cost"),
]


def _format_table(path: Path) -> str:
    import pandas as pd

    df = pd.read_csv(path)
    keys = [c for c in df.columns if not c.endswith(("_mean", "_std")) and c != "n_seeds"]
    metrics = [c[:-5] for c in df.columns if c.endswith("_mean")]
    rows = [keys + metrics + ["seeds"]]
    for _, r in df.iterrows():
        cells = [str(r[k]) for k in keys]
        for m in metrics:
            mean, std = r[f"{m}_mean"], r[f"{m}_std"]
            cells.append("n/a" if pd.isna(mean) else (f"{mean:.4f} +- {std:.4f}" if not pd.isna(std) else f"{mean:.4f}"))
        cells.append(str(r.get("n_seeds", "")))
        rows.append(cells)
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(row, widths)) for row in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def _summary_text(summary_path: Path) -> str:
    d = json.loads(summary_path.read_text())
    out = [f"files: {', '.join(d.get('source_files', []))}",
           f"records read: {d.get('rows_read')}, after cleaning: {d.get('rows_after_cleaning')}",
           f"features: {len(d['features']['continuous_perturbable'])} continuous, "
           f"{len(d['features']['discrete_not_perturbed'])} discrete, "
           f"{len(d['features']['categorical_one_hot_not_perturbed'])} categorical "
           f"(encoded dimension {d['features']['encoded_dimension']})"]
    for split, v in d.get("splits", {}).items():
        out.append(f"{split:5s}: {v['rows']} records (benign {v.get('benign')}, malicious {v.get('malicious')})")
    for split, g in d.get("graphs", {}).items():
        out.append(f"{split:5s} graph: {g['relational_edges']} relational edges (mean degree "
                   f"{g['mean_relational_degree']:.1f}), {g['temporal_edges']} temporal edges")
    out += [f"note: {n}" for n in d.get("cleaning_notes", [])]
    return "\n".join(out)


def step_report() -> Path:
    """All tables in one plain-text file: ARTGL_work/ALL_RESULTS.txt"""

    parts = ["ARTGL v2.0 - all results (mean +- sample std over seeds)",
             f"generated {time.strftime('%Y-%m-%d %H:%M')}", ""]
    for dataset, label in [("unsw_nb15", "UNSW-NB15"), ("cicids2017", "CICIDS2017")]:
        base = RESULTS / dataset
        if not base.exists():
            continue
        parts += ["=" * 100, label, "=" * 100]
        if (base / "data_summary.json").exists():
            parts += ["Data protocol (Table 2)", "-" * 23, _summary_text(base / "data_summary.json"), ""]
        for name, title in TABLE_TITLES:
            path = base / "tables" / f"{name}.csv"
            if path.exists():
                parts += [title, _format_table(path), ""]
    cross_dir = RESULTS / "cross_dataset"
    for summary in sorted(cross_dir.glob("*/data_summary.json")) if cross_dir.exists() else []:
        parts += [f"Cross-dataset data: {summary.parent.name}", summary.read_text()[:3000], ""]
    report = RESULTS / "ALL_RESULTS.txt"
    report.write_text("\n".join(parts), encoding="utf-8")
    log(f"All tables written to {report}")
    return report


def step_pack() -> None:
    target = RESULTS / "send_to_claude.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for path in RESULTS.rglob("*"):
            rel = path.relative_to(RESULTS)
            if path.is_dir() or "runs" in rel.parts or "cache" in rel.parts or path.suffix == ".zip":
                continue
            if path.suffix in (".csv", ".json", ".tex", ".pdf"):
                z.write(path, str(rel))
        if LOG.exists():
            z.write(LOG, "run_log.txt")
        if (RESULTS / "ALL_RESULTS.txt").exists():
            z.write(RESULTS / "ALL_RESULTS.txt", "ALL_RESULTS.txt")
    log(f"Results packed: {target}  <-- send this file to Claude")


# --------------------------------------------------------------------------- main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Run every ARTGL experiment end to end.")
    ap.add_argument("--only", choices=["data", "tests", "report"], default=None,
                    help="data: prepare data only; tests: stop after tests; report: rebuild tables + ALL_RESULTS.txt from finished runs")
    ap.add_argument("--seeds", nargs="+", default=["0", "1", "2", "3", "4"])
    ap.add_argument("--sensitivity-seeds", nargs="+", default=None,
                    help="seeds for the sensitivity sweep only (default: same as --seeds); e.g. 0 1 2 to save time")
    ap.add_argument("--work", default=None, help="folder for code, extracted data and caches (default: ./ARTGL_work)")
    ap.add_argument("--results", default=None,
                    help="folder for results, log and progress markers (default: <work>/results). "
                         "In Colab, put this on Google Drive so nothing is lost on disconnect.")
    ap.add_argument("--skip-cse", action="store_true")
    ap.add_argument("--allow-partial-cse", action="store_true",
                    help="use CSE-CIC-IDS2018 even if some of its 10 files are missing")
    ap.add_argument("--allow-cpu", action="store_true", help="run the experiments even without a GPU (very slow)")
    ap.add_argument("--torch-cuda", default="cu126", help="CUDA build of PyTorch to install (cu126 or cu128)")
    args = ap.parse_args(argv)

    global WORK, DATA, RESULTS, LOG
    if args.work:
        WORK = Path(args.work).resolve()
        DATA = WORK / "data"
        RESULTS = WORK / "results"
    if args.results:
        RESULTS = Path(args.results).resolve()
    LOG = RESULTS / "run_log.txt"
    WORK.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    log(f"Started in {HERE} | work: {WORK} | results: {RESULTS}")
    step_code()
    if args.only == "report":
        step_tables({})
        step_report()
        step_pack()
        return
    if args.only != "data":
        step_packages(args.torch_cuda)
    available = step_data(args.skip_cse, args.allow_partial_cse)
    cfg_dir = write_configs(available)
    log("Datasets ready: " + ", ".join(available))
    if args.only == "data":
        return
    step_tests()
    if args.only == "tests":
        return
    if not HAS_CUDA and not args.allow_cpu:
        log("STOPPED before the experiments: PyTorch cannot use an NVIDIA GPU on this computer. "
            "On a CPU the full experiments would take weeks. Run this on a computer with an NVIDIA GPU "
            "(the script installs the GPU version of PyTorch automatically there), or add --allow-cpu "
            "to run on the CPU anyway.")
        raise SystemExit(1)
    step_experiments(cfg_dir, available, args.seeds, args.sensitivity_seeds or args.seeds)
    step_cross(cfg_dir, available, args.seeds)
    step_interpret(cfg_dir, available)
    step_tables(available)
    step_report()
    step_pack()
    log("ALL DONE.")


if __name__ == "__main__":
    main(sys.argv[1:])
