import os
import glob
import re
import csv

def parse_logs():
    # Chercher tous les fichiers qui commencent par log_ et finissent par .txt
    log_files = sorted(
        set(
            glob.glob("log_*.txt") +
            glob.glob("logs/**/log_*.txt", recursive=True)
        )
    )
    results = []

    # Expressions régulières pour attraper les infos
    # J'ai amélioré le regex pour qu'il reconnaisse spécifiquement "cuda_states" sans le couper !
    filename_regex = re.compile(r"log_(?P<dataset>[^_]+)_(?P<diff>[^_]+)_(?P<backend>classical|classique|torch|cuda_states)_(?P<size>[^\.]+)\.txt")
    
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
            
            # 1. Chercher F1 et AUC (On utilise findall pour tout lister)
            if backend in ["classique", "classical"]:
                metrics_matches = c_metrics_regex.findall(content)
            else:
                metrics_matches = q_metrics_regex.findall(content)
                
            if metrics_matches:
                # On prend l'index [-1] pour lire la DERNIÈRE exécution (le vrai run !)
                f1 = metrics_matches[-1][0]
                auc = metrics_matches[-1][1]
            else:
                f1, auc = "FAILED", "FAILED" # Si le script a planté avant la fin
                
            # 2. Chercher le temps (On utilise findall et on prend la dernière valeur)
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

    # Trier les résultats pour que ce soit joli à lire
    results = sorted(results, key=lambda x: (x["Dataset"], x["Difficulty"], x["Backend"], x["Size"]))

    # Exporter en CSV
    csv_file = "summary_results_v2.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Dataset", "Difficulty", "Backend", "Size", "F1-Score", "AUC", "Time"])
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ Extraction v2 terminée ! {len(results)} logs analysés.")
    print(f"📄 Les résultats corrigés sont sauvegardés dans '{csv_file}'.")

if __name__ == "__main__":
    parse_logs()