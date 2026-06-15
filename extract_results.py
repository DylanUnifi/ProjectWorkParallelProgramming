"""
extract_results.py — Parse training logs and export a summary CSV.

Log discovery (in order of priority):
  1. logs/quantum/**/log_*.txt  (run_all_quantum.sh output)
  2. logs/classical/**/log_*.txt (run_all_classical.sh output)
  3. log_*.txt at project root   (legacy / manual runs)

Supported log filename format:
  log_<dataset>_<difficulty>_<backend>_<size>.txt
  where backend is: classical | torch | cuda_states

Usage:
  python3 extract_results.py                     # writes summary_results_v2.csv
  python3 extract_results.py --csv my_out.csv    # custom output path
  python3 extract_results.py --verbose           # print per-file details
"""

import os
import sys
import glob
import re
import csv
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Compiled regexes
# ---------------------------------------------------------------------------

# Matches the standard log filename produced by the batch scripts.
FILENAME_RE = re.compile(
    r"log_(?P<dataset>[^_]+)_(?P<diff>[^_]+)"
    r"_(?P<backend>classical|torch|cuda_states)"
    r"_(?P<size>[^\.]+)\.txt$"
)

# Quantum SVM final metrics (train_svm_qkernel.py)
Q_METRICS_RE = re.compile(r"test_F1=([0-9\.]+)\s+test_AUC=([0-9\.]+)")

# Classical SVM final metrics (train_svm_classical.py)
C_METRICS_RE = re.compile(
    r"AVERAGE RBF SVM \| F1:\s*([0-9\.]+)\s*\| AUC:\s*([0-9\.]+)"
)

# Wall-clock time – handles `real 4m33.210s` and `4:33.21elapsed`
TIME_RE_BASH = re.compile(r"real\s+([0-9]+m[0-9\.]+s)")
TIME_RE_GNU  = re.compile(r"([0-9]+:[0-9]+\.[0-9]+)elapsed")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_logs(root: Path) -> list[Path]:
    """Return all training log files, deduped and sorted."""
    patterns = [
        root / "logs" / "quantum"   / "**" / "log_*.txt",
        root / "logs" / "classical" / "**" / "log_*.txt",
        root / "log_*.txt",
    ]
    seen: set[Path] = set()
    files: list[Path] = []
    for pattern in patterns:
        for p in sorted(glob.glob(str(pattern), recursive=True)):
            resolved = Path(p).resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(resolved)
    return files


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_log(path: Path, verbose: bool = False) -> dict | None:
    """Extract metadata and metrics from a single log file.
    Returns None if the filename does not match the expected format.
    """
    m = FILENAME_RE.match(path.name)
    if not m:
        if verbose:
            print(f"  [skip] unrecognised filename: {path.name}", file=sys.stderr)
        return None

    dataset  = m.group("dataset")
    diff     = m.group("diff")
    backend  = m.group("backend")
    size     = m.group("size")

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"  [error] cannot read {path}: {exc}", file=sys.stderr)
        return None

    # ---- metrics -----------------------------------------------------------
    if backend == "classical":
        metric_matches = C_METRICS_RE.findall(content)
    else:
        metric_matches = Q_METRICS_RE.findall(content)

    if metric_matches:
        # Take the *last* occurrence: the final test evaluation, not CV folds
        f1, auc = metric_matches[-1]
    else:
        f1, auc = "FAILED", "FAILED"

    # ---- wall-clock time ---------------------------------------------------
    exec_time = "N/A"
    t1 = TIME_RE_BASH.findall(content)
    t2 = TIME_RE_GNU.findall(content)
    if t1:
        exec_time = t1[-1]
    elif t2:
        exec_time = t2[-1]

    # ---- run timestamp from parent directory name (logs/quantum/<ts>/) -----
    run_ts = ""
    parts = path.parts
    # The timestamp directory is 2 levels above the file: logs/quantum/<ts>/file.txt
    if len(parts) >= 3 and parts[-3] in ("quantum", "classical"):
        # path = .../logs/quantum/<ts>/log_*.txt → too many parts; just use parent name
        pass
    if path.parent.name not in (".", "logs", "quantum", "classical"):
        run_ts = path.parent.name  # e.g. 20260615_230502

    if verbose:
        status = "OK" if f1 not in ("FAILED", "N/A") else "FAILED"
        print(f"  [{status}] {path.relative_to(path.parent.parent.parent) if run_ts else path.name}"
              f"  F1={f1}  AUC={auc}  time={exec_time}")

    return {
        "Dataset":    dataset,
        "Difficulty": diff,
        "Backend":    backend,
        "Size":       size,
        "Run":        run_ts,
        "F1-Score":   f1,
        "AUC":        auc,
        "Time":       exec_time,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_logs(csv_out: str = "summary_results_v2.csv", verbose: bool = False) -> list[dict]:
    root = Path(__file__).resolve().parent
    log_files = discover_logs(root)

    if not log_files:
        print("Warning: no log_*.txt files found. Run the training scripts first.", file=sys.stderr)
        return []

    print(f"Found {len(log_files)} log file(s).")

    results = []
    for path in log_files:
        row = parse_log(path, verbose=verbose)
        if row is not None:
            results.append(row)

    # Sort: dataset → difficulty → backend → size → run timestamp
    results.sort(key=lambda r: (r["Dataset"], r["Difficulty"], r["Backend"], r["Size"], r["Run"]))

    # ---- CSV export --------------------------------------------------------
    fieldnames = ["Dataset", "Difficulty", "Backend", "Size", "Run", "F1-Score", "AUC", "Time"]
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{len(results)} results extracted → '{csv_out}'")

    # ---- inline summary ----------------------------------------------------
    failed = [r for r in results if r["F1-Score"] == "FAILED"]
    ok     = [r for r in results if r["F1-Score"] not in ("FAILED", "N/A")]
    print(f"  Successful runs : {len(ok)}")
    if failed:
        print(f"  Failed runs     : {len(failed)}")
        for r in failed:
            print(f"    - {r['Dataset']} {r['Difficulty']} {r['Backend']} size={r['Size']}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract training results from log files.")
    parser.add_argument("--csv", default="summary_results_v2.csv", help="Output CSV file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-file details")
    args = parser.parse_args()
    parse_logs(csv_out=args.csv, verbose=args.verbose)