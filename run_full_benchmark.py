#!/usr/bin/env python3
"""
Benchmark Complet CPU vs GPU pour Quantum Kernels
==================================================

Ce script génère automatiquement:
1. Un CSV avec tous les résultats bruts
2. Un graphique dual (Scaling + Speedup)
3. Des statistiques détaillées pour le rapport

Auteur: [Dylan Fouepe]
Date: 2025-12-13
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
from pathlib import Path

# Ajout du dossier tools au path
sys.path.append(os.path.join(os.path.dirname(__file__), "tools"))

try:
    from benchmark_pl_kernel import run_once
except ImportError:
    print("❌ Erreur : Impossible d'importer 'run_once' depuis 'tools/benchmark_pl_kernel.py'.")
    print("Vérifie que le fichier existe et que tu lances ce script depuis la racine du projet.")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════
# CONFIGURATION DU TEST
# ═══════════════════════════════════════════════════════════

# Liste des qubits à tester (Scaling)
QUBITS_LIST = [4, 6, 8, 10, 12, 14, 16, 18, 20]

# Taille de la matrice (N x N)
N_SAMPLES = 2000 

# Limite CPU (Au-delà, le temps devient prohibitif)
CPU_QUBIT_LIMIT = 12 

# Configurations à comparer
CONFIGS = [
    {
        "name": "CPU (Lightning)",
        "device": "lightning.qubit",
        "gram_backend": "auto",
        "dtype": "float32",
        "color": "#ff7f0e",  # Orange
        "marker": "o"
    },
    {
        "name": "GPU (CUDA States)",
        "device": "default.qubit",
        "gram_backend": "cuda_states",
        "dtype": "float64",
        "color": "#2ca02c",  # Vert
        "marker": "s"
    }
]

# Outputs
OUTPUT_DIR = Path("benchmark_results")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / "final_benchmark_results.csv"
OUTPUT_PLOT = OUTPUT_DIR / "final_performance_benchmark.png"
OUTPUT_STATS = OUTPUT_DIR / "benchmark_summary.txt"

# ═══════════════════════════════════════════════════════════
# MOTEUR DE BENCHMARK
# ═══════════════════════════════════════════════════════════

def main():
    print(f"🚀 Lancement du Benchmark Comparatif Complet")
    print(f"📊 Configuration:")
    print(f"   - Samples: {N_SAMPLES}")
    print(f"   - Qubits: {QUBITS_LIST}")
    print(f"   - CPU limit: {CPU_QUBIT_LIMIT} qubits")
    print(f"   - Output: {OUTPUT_DIR}")
    print("="*70)

    results = []
    
    # Calcul du nombre total de tests
    total_tests = sum(
        1 for n_q in QUBITS_LIST 
        for cfg in CONFIGS 
        if not ("CPU" in cfg['name'] and n_q > CPU_QUBIT_LIMIT)
    )

    # Barre de progression globale
    with tqdm(total=total_tests, desc="Benchmark Progress", unit="test") as pbar:
        for n_qubits in QUBITS_LIST:
            for cfg in CONFIGS:
                # Skip CPU si trop de qubits
                if "CPU" in cfg['name'] and n_qubits > CPU_QUBIT_LIMIT:
                    continue
                
                pbar.set_description(f"Testing {cfg['name']:<20} {n_qubits:2d}Q")
                current_state_tile = 1024 if n_qubits >= 18 else 4096
                
                try:
                    # Appel de run_once avec tous les paramètres
                    res = run_once(
                        n_samples=N_SAMPLES,
                        n_qubits=n_qubits,
                        tile_size=10000,
                        state_tile=current_state_tile,
                        workers=1,
                        device_name=cfg['device'],
                        symmetric=True,
                        repeats=3,
                        seed=42,
                        layers=2,
                        dtype=cfg['dtype'],
                        return_dtype=cfg['dtype'],
                        gram_backend=cfg['gram_backend'],
                        angle_scale=1.0,
                        embed_mode="ryrz"
                    )
                    
                    # Enrichissement des résultats
                    res["config_name"] = cfg["name"]
                    res["color"] = cfg["color"]
                    results.append(res)
                    
                    # Log du résultat
                    time_str = f"{res['time_s']:.2f}s"
                    throughput_str = f"{res.get('throughput_mpairs_s', 0):.2f} Mpairs/s"
                    pbar.write(f"  ✓ {cfg['name']:<20} {n_qubits:2d}Q: {time_str:>10} | {throughput_str}")

                except Exception as e:
                    pbar.write(f"  ✗ {cfg['name']:<20} {n_qubits:2d}Q: FAILED ({str(e)[:50]})")
                    # On continue même en cas d'erreur
                
                pbar.update(1)

    # ═══════════════════════════════════════════════════════════
    # SAUVEGARDE & ANALYSE
    # ═══════════════════════════════════════════════════════════
    
    if not results:
        print("\n❌ Aucun résultat généré. Vérifiez la configuration.")
        return

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 Résultats bruts sauvegardés : {OUTPUT_CSV}")

    # Génération du rapport statistique
    generate_summary_report(df)

    # Génération des graphiques
    generate_plots(df)
    
    print(f"\n🎉 Benchmark terminé avec succès !")
    print(f"📁 Tous les fichiers sont dans : {OUTPUT_DIR}")

# ═══════════════════════════════════════════════════════════
# GÉNÉRATION DU RAPPORT STATISTIQUE
# ═══════════════════════════════════════════════════════════

def generate_summary_report(df):
    """Génère un rapport texte avec statistiques clés"""
    
    with open(OUTPUT_STATS, 'w') as f:
        f.write("="*70 + "\n")
        f.write("BENCHMARK SUMMARY - Quantum Kernel Performance\n")
        f.write("="*70 + "\n\n")
        
        # Statistiques par backend
        for config_name in df['config_name'].unique():
            subset = df[df['config_name'] == config_name]
            f.write(f"\n{config_name}:\n")
            f.write(f"  Qubits testés: {sorted(subset['n_qubits'].unique())}\n")
            f.write(f"  Temps moyen: {subset['time_s'].mean():.2f}s ± {subset['time_s'].std():.2f}s\n")
            f.write(f"  Temps min: {subset['time_s'].min():.2f}s ({subset.loc[subset['time_s'].idxmin(), 'n_qubits']:.0f} qubits)\n")
            f.write(f"  Temps max: {subset['time_s'].max():.2f}s ({subset.loc[subset['time_s'].idxmax(), 'n_qubits']:.0f} qubits)\n")
            
            if 'throughput_mpairs_s' in subset.columns:
                f.write(f"  Throughput moyen: {subset['throughput_mpairs_s'].mean():.2f} Mpairs/s\n")
        
        # Calcul du speedup
        f.write("\n" + "-"*70 + "\n")
        f.write("SPEEDUP ANALYSIS\n")
        f.write("-"*70 + "\n")
        
        try:
            df_pivot = df.pivot(index="n_qubits", columns="config_name", values="time_s")
            if "CPU (Lightning)" in df_pivot.columns and "GPU (CUDA States)" in df_pivot.columns:
                df_pivot["Speedup"] = df_pivot["CPU (Lightning)"] / df_pivot["GPU (CUDA States)"]
                
                f.write("\nSpeedup par nombre de qubits:\n")
                for qubits in sorted(df_pivot.index):
                    speedup = df_pivot.loc[qubits, "Speedup"]
                    if pd.notna(speedup):
                        f.write(f"  {qubits:2d} qubits: {speedup:6.2f}x\n")
                
                valid_speedups = df_pivot["Speedup"].dropna()
                if len(valid_speedups) > 0:
                    f.write(f"\nSpeedup moyen: {valid_speedups.mean():.2f}x\n")
                    f.write(f"Speedup max: {valid_speedups.max():.2f}x (à {valid_speedups.idxmax()} qubits)\n")
        except Exception as e:
            f.write(f"\nImpossible de calculer le speedup: {e}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"📄 Rapport statistique sauvegardé : {OUTPUT_STATS}")
    
    # Affichage console
    with open(OUTPUT_STATS, 'r') as f:
        print("\n" + f.read())

# ═══════════════════════════════════════════════════════════
# GÉNÉRATION DES GRAPHIQUES
# ═══════════════════════════════════════════════════════════

def generate_plots(df):
    """Génère les graphiques de performance"""
    
    # Configuration style publication
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- PLOT 1 : Scaling (Temps vs Qubits) ---
    for config_name in df['config_name'].unique():
        subset = df[df['config_name'] == config_name]
        color = subset['color'].iloc[0]
        
        ax1.plot(
            subset['n_qubits'], subset['time_s'],
            marker='o', linewidth=2.5, markersize=8,
            label=config_name, color=color
        )
    
    ax1.set_yscale("log")
    ax1.set_title("Weak Scaling : Temps de Calcul Kernel", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Temps (secondes) - Échelle Log", fontsize=12)
    ax1.set_xlabel("Nombre de Qubits", fontsize=12)
    ax1.legend(title="Backend", loc='upper left')
    ax1.grid(True, which="both", ls="--", alpha=0.3)

    # --- PLOT 2 : Speedup Factor ---
    try:
        df_pivot = df.pivot(index="n_qubits", columns="config_name", values="time_s")
        if "CPU (Lightning)" in df_pivot.columns and "GPU (CUDA States)" in df_pivot.columns:
            df_pivot["Speedup"] = df_pivot["CPU (Lightning)"] / df_pivot["GPU (CUDA States)"]
            speedup_data = df_pivot["Speedup"].dropna().reset_index()
            
            bars = ax2.bar(
                speedup_data['n_qubits'].astype(str), 
                speedup_data['Speedup'],
                color="#2ca02c", alpha=0.8, edgecolor='black'
            )
            
            ax2.set_title("Facteur d'Accélération GPU vs CPU", fontsize=14, fontweight='bold')
            ax2.set_ylabel("Speedup (× plus rapide)", fontsize=12)
            ax2.set_xlabel("Nombre de Qubits", fontsize=12)
            ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Baseline (1x)')
            ax2.legend()
            
            # Annotations sur les barres
            for bar, (_, row) in zip(bars, speedup_data.iterrows()):
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}×',
                    ha='center', va='bottom', fontweight='bold', fontsize=10
                )
        else:
            ax2.text(0.5, 0.5, "Données CPU insuffisantes\npour calculer le Speedup",
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
    except Exception as e:
        print(f"⚠️  Erreur lors du plot speedup: {e}")
        ax2.text(0.5, 0.5, f"Erreur: {str(e)[:50]}...",
                ha='center', va='center', transform=ax2.transAxes)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
    print(f"🖼️  Graphique généré : {OUTPUT_PLOT}")

# ═══════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrompu par l'utilisateur.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur fatale : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
