import os
import glob
import re
import csv

def parse_logs():
    # Chercher tous les fichiers qui commencent par log_ et finissent par .txt
    log_files = glob.glob("log_*.txt")
    results = []

    # Expressions régulières pour attraper les infos
    # Nom du fichier : log_dataset_diff_backend_size.txt
    filename_regex = re.compile(r"log_(?P<dataset>[^_]+)_(?P<diff>[^_]+)_(?P<backend>[^_]+)_(?P<size>[^\.]+)\.txt")
    
    # Métriques Quantiques
    q_metrics_regex = re.compile(r"test_F1=([0-9\.]+)\s+test_AUC=([0-9\.]+)")
    
    # Métriques Classiques
    c_metrics_regex = re.compile(r"AVERAGE RBF SVM \| F1:\s*([0-9\.]+)\s*\| AUC:\s*([0-9\.]+)")
    
    # Temps d'exécution (gère différents formats de 'time')
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
            
            # 1. Chercher F1 et AUC
            if backend in ["classique", "classical"]:
                metrics_match = c_metrics_regex.search(content)
            else:
                metrics_match = q_metrics_regex.search(content)
                
            if metrics_match:
                f1 = metrics_match.group(1)
                auc = metrics_match.group(2)
            else:
                f1, auc = "FAILED", "FAILED" # Si le script a planté avant la fin
                
            # 2. Chercher le temps
            time_match_1 = time_regex_1.search(content)
            time_match_2 = time_regex_2.search(content)
            
            if time_match_1:
                exec_time = time_match_1.group(1)
            elif time_match_2:
                exec_time = time_match_2.group(1)

        results.append({
            "Dataset": dataset,
            "Difficulty": diff,
            "Backend": backend,
            "Size": size,
            "F1-Score": f1,
            "AUC": auc,
            "Time": exec_time
        })

    # Trier les résultats pour que ce soit joli à lire
    results = sorted(results, key=lambda x: (x["Dataset"], x["Difficulty"], x["Backend"], x["Size"]))

    # Exporter en CSV
    csv_file = "summary_results.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Dataset", "Difficulty", "Backend", "Size", "F1-Score", "AUC", "Time"])
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ Extraction terminée ! {len(results)} logs analysés.")
    print(f"📄 Les résultats sont sauvegardés dans le fichier '{csv_file}'.")

if __name__ == "__main__":
    parse_logs()
