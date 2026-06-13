import os
import glob
import re
import csv

def parse_logs():
    # Find all files that start with log_ and end with .txt
    log_files = sorted(
        set(
            glob.glob("log_*.txt") +
            glob.glob("logs/**/log_*.txt", recursive=True)
        )
    )
    results = []

    # Regular expressions to capture metadata.
    # Keep the backend pattern aligned with the current log formats.
    filename_regex = re.compile(r"log_(?P<dataset>[^_]+)_(?P<diff>[^_]+)_(?P<backend>classical|torch|cuda_states)_(?P<size>[^\.]+)\.txt")
    
    # Quantum metrics
    q_metrics_regex = re.compile(r"test_F1=([0-9\.]+)\s+test_AUC=([0-9\.]+)")
    
    # Classical metrics
    c_metrics_regex = re.compile(r"AVERAGE RBF SVM \| F1:\s*([0-9\.]+)\s*\| AUC:\s*([0-9\.]+)")
    
    # Execution time (handles different 'time' formats)
    time_regex_1 = re.compile(r"real\s+([0-9]+m[0-9\.]+s)")
    time_regex_2 = re.compile(r"([0-9:]+\.[0-9]+)elapsed")

    for file in log_files:
        match = filename_regex.match(os.path.basename(file))
        if not match:
            continue
            
        dataset = match.group("dataset")
        diff = match.group("diff")
        backend = match.group("backend")
        size = match.group("size")
        
        f1, auc, exec_time = "N/A", "N/A", "N/A"
        
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            
            # 1. Search for F1 and AUC (use findall to list all occurrences)
            if backend == "classical":
                metrics_matches = c_metrics_regex.findall(content)
            else:
                metrics_matches = q_metrics_regex.findall(content)
                
            if metrics_matches:
                # Take the last index [-1] to read the LAST execution (the real run!)
                f1 = metrics_matches[-1][0]
                auc = metrics_matches[-1][1]
            else:
                f1, auc = "FAILED", "FAILED"  # If the script crashed before finishing
                
            # 2. Search for execution time (use findall and take the last value)
            time_matches_1 = time_regex_1.findall(content)
            time_matches_2 = time_regex_2.findall(content)
            
            if time_matches_1:
                exec_time = time_matches_1[-1]
            elif time_matches_2:
                exec_time = time_matches_2[-1]

        results.append({
            "Dataset": dataset,
            "Difficulty": diff,
            "Backend": backend,
            "Size": size,
            "F1-Score": f1,
            "AUC": auc,
            "Time": exec_time
        })

    # Sort results for cleaner output
    results = sorted(results, key=lambda x: (x["Dataset"], x["Difficulty"], x["Backend"], x["Size"]))

    # Export to CSV
    csv_file = "summary_results_v2.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Dataset", "Difficulty", "Backend", "Size", "F1-Score", "AUC", "Time"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Extraction v2 complete: {len(results)} logs analyzed.")
    print(f"Corrected results saved to '{csv_file}'.")

if __name__ == "__main__":
    parse_logs()