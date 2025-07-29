import time
import csv
from typing import Optional
from datetime import datetime
from pathlib import Path

def log_benchmark(
    experiment: str,
    version: str,
    dataset: str,
    num_epochs: int,
    early_stop: bool,
    training_time: float,
    f1_score: float,
    accuracy: float,
    auc: float,
    balanced_accuracy: float,
    csv_path: str = "benchmark_results.csv",
    reference_time: Optional[float] = None
):
    speedup = round(reference_time / training_time, 2) if reference_time else None
    row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Experiment": experiment,
        "Version": version,
        "Dataset": dataset,
        "Num Epochs": num_epochs,
        "Early Stop": early_stop,
        "Training Time (s)": round(training_time, 2),
        "Speedup": speedup,
        "F1 Score": round(f1_score, 4),
        "Accuracy": round(accuracy, 4),
        "AUC": round(auc, 4),
        "Balanced Accuracy": round(balanced_accuracy, 4)
    }
    file_exists = Path(csv_path).is_file()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        results = func(*args, **kwargs)
        end = time.time()
        duration = end - start
        return results, duration
    return wrapper
