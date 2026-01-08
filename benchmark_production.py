#!/usr/bin/env python3
"""
Benchmark PRODUCTION - Pipeline Backends Réels
===============================================

Ce script teste directement votre implémentation cuda_states
au lieu de passer par l'API PennyLane générique.

Auteur: Fouepe Dylan
Date: 2025-12-13
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Ajout du chemin pour trouver vos scripts
sys.path.append(os.path.join(os.path.dirname(__file__)))

try:
    from scripts.pipeline_backends import compute_kernel_matrix
except ImportError:
    print("❌ Erreur: Impossible d'importer 'scripts.pipeline_backends'.")
    print("Lancez ce script depuis la racine du projet : python3 benchmark_production.py")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════
N_SAMPLES = 8000  # Assez grand pour amortir l'overhead GPU
QUBITS_GPU = [4, 6, 8, 10, 12, 14, 16, 18, 20]
QUBITS_CPU = [4, 6, 8, 10]  # S'arrête tôt (CPU trop lent)

# Outputs
OUTPUT_DIR = Path("benchmark_results")
OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_FILE = OUTPUT_DIR / "production_benchmark.csv"
PLOT_FILE = OUTPUT_DIR / "production_benchmark.png"

# ═══════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ═══════════════════════════════════════════════════════════

def estimate_cpu_time(time_small, n_small, n_large):
    """
    Extrapole le temps CPU basé sur complexité O(N^2 × 2^Q)
    
    Pour le kernel matrix:
    - Nombre de paires: N^2
    - Complexité par paire: O(2^Q) (simulation état quantique)
    
    Temps_large = Temps_small × (N_large / N_small)^2
    """
    ratio = (n_large / n_small) ** 2
    return time_small * ratio

def format_time(seconds):
    """Format temps en heures/minutes/secondes"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"

# ═══════════════════════════════════════════════════════════
# BENCHMARK PRINCIPAL
# ═══════════════════════════════════════════════════════════

def benchmark_run():
    print("="*70)
    print(f"🚀 Lancement du Benchmark PRODUCTION")
    print(f"📊 Configuration:")
    print(f"   - Samples (GPU): {N_SAMPLES}")
    print(f"   - Qubits (GPU): {QUBITS_GPU}")
    print(f"   - Qubits (CPU): {QUBITS_CPU}")
    print(f"   - Output: {OUTPUT_DIR}")
    print("="*70)
    
    results = []

    # ═══════════════════════════════════════════════════════════
    # 1. BENCHMARK GPU (Votre Code Optimisé)
    # ═══════════════════════════════════════════════════════════
    
    print("\n🟢 Test GPU (backend='cuda_states')...")
    print("-"*70)
    
    for n_q in QUBITS_GPU:
        print(f"   Testing {n_q:2d} qubits (N={N_SAMPLES})... ", end="", flush=True)
        
        # Génération données aléatoires
        np.random.seed(42)  # Reproductibilité
        X = np.random.uniform(0, np.pi, (N_SAMPLES, n_q)).astype(np.float64)
        weights = np.random.normal(0, 0.1, (2, n_q)).astype(np.float64)
        
        try:
            t0 = time.time()
            K = compute_kernel_matrix(
                X, X,  # Symmetric kernel
                weights=weights,
                device_name="cuda:0",
                gram_backend="cuda_states",
                tile_size=10000,
                dtype="float64",
                angle_scale=1.0,
                embed_mode="ryrz",
                progress=False
            )
            duration = time.time() - t0
            
            # Calcul throughput
            n_pairs = N_SAMPLES * N_SAMPLES
            throughput = n_pairs / duration / 1e6  # Mpairs/s
            
            print(f"✅ {format_time(duration):>12} | {throughput:.2f} Mpairs/s")
            
            results.append({
                "Backend": "GPU (CUDA States)",
                "Qubits": n_q,
                "N_Samples": N_SAMPLES,
                "Time_s": duration,
                "Throughput_Mpairs_s": throughput
            })
            
            # Cleanup mémoire
            del K, X, weights
            
        except Exception as e:
            print(f"❌ FAILED: {str(e)[:50]}")

    # ═══════════════════════════════════════════════════════════
    # 2. BENCHMARK CPU (Estimation par Extrapolation)
    # ═══════════════════════════════════════════════════════════
    
    print("\n🟠 Test CPU (backend='numpy' - Extrapolation)...")
    print("-"*70)
    print(f"   Note: Test sur N={500} puis extrapolation à N={N_SAMPLES}")
    print()
    
    N_SMALL = 500  # Taille gérable sur CPU
    
    for n_q in QUBITS_CPU:
        print(f"   Testing {n_q:2d} qubits (N={N_SMALL})... ", end="", flush=True)
        
        # Génération données réduites
        np.random.seed(42)
        X_small = np.random.uniform(0, np.pi, (N_SMALL, n_q)).astype(np.float64)
        weights = np.random.normal(0, 0.1, (2, n_q)).astype(np.float64)
        
        try:
            t0 = time.time()
            
            # Test sur petit dataset
            K_small = compute_kernel_matrix(
                X_small, X_small,
                weights=weights,
                device_name="default.qubit",
                gram_backend="numpy",  # Force CPU
                tile_size=500,
                dtype="float64",
                angle_scale=1.0,
                embed_mode="ryrz",
                progress=False
            )
            duration_small = time.time() - t0
            
            # Extrapolation vers N_SAMPLES
            estimated_duration = estimate_cpu_time(duration_small, N_SMALL, N_SAMPLES)
            
            print(f"⏱️  {format_time(duration_small):>12} → Est. {format_time(estimated_duration):>12} (N={N_SAMPLES})")
            
            results.append({
                "Backend": "CPU (Numpy - Estimated)",
                "Qubits": n_q,
                "N_Samples": N_SAMPLES,
                "Time_s": estimated_duration,
                "Throughput_Mpairs_s": (N_SAMPLES**2 / estimated_duration) / 1e6
            })
            
            # Cleanup
            del K_small, X_small, weights
            
        except Exception as e:
            print(f"❌ FAILED: {str(e)[:50]}")

    # ═══════════════════════════════════════════════════════════
    # 3. SAUVEGARDE & ANALYSE
    # ═══════════════════════════════════════════════════════════
    
    if not results:
        print("\n❌ Aucun résultat collecté.")
        return
    
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_FILE, index=False)
    print(f"\n💾 Résultats sauvegardés : {RESULTS_FILE}")
    
    # Calcul speedup
    print("\n" + "="*70)
    print("SPEEDUP ANALYSIS")
    print("="*70)
    
    df_pivot = df.pivot(index="Qubits", columns="Backend", values="Time_s")
    if "CPU (Numpy - Estimated)" in df_pivot.columns and "GPU (CUDA States)" in df_pivot.columns:
        df_pivot["Speedup"] = df_pivot["CPU (Numpy - Estimated)"] / df_pivot["GPU (CUDA States)"]
        
        print("\nSpeedup GPU vs CPU:")
        for qubits in sorted(df_pivot.index):
            if pd.notna(df_pivot.loc[qubits, "Speedup"]):
                speedup = df_pivot.loc[qubits, "Speedup"]
                print(f"  {qubits:2d} qubits: {speedup:7.1f}×")
        
        valid_speedups = df_pivot["Speedup"].dropna()
        if len(valid_speedups) > 0:
            print(f"\nSpeedup moyen: {valid_speedups.mean():.1f}×")
            print(f"Speedup max: {valid_speedups.max():.1f}× (à {valid_speedups.idxmax()} qubits)")
    
    # ═══════════════════════════════════════════════════════════
    # 4. GÉNÉRATION GRAPHIQUE
    # ═══════════════════════════════════════════════════════════
    
    generate_plots(df, df_pivot if "df_pivot" in locals() else None)
    
    print("\n" + "="*70)
    print("🎉 Benchmark terminé avec succès !")
    print(f"📁 Fichiers générés dans : {OUTPUT_DIR}")
    print("="*70)

# ═══════════════════════════════════════════════════════════
# GÉNÉRATION GRAPHIQUES
# ═══════════════════════════════════════════════════════════

def generate_plots(df, df_pivot):
    """Génère les graphiques de performance"""
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- PLOT 1: Temps vs Qubits ---
    for backend in df['Backend'].unique():
        subset = df[df['Backend'] == backend]
        color = "#2ca02c" if "GPU" in backend else "#ff7f0e"
        marker = "s" if "GPU" in backend else "o"
        linestyle = "-" if "GPU" in backend else "--"
        
        ax1.plot(
            subset['Qubits'], subset['Time_s'],
            marker=marker, linewidth=2.5, markersize=8,
            label=backend, color=color, linestyle=linestyle
        )
    
    ax1.set_yscale("log")
    ax1.set_title(f"Scaling Performance (N={N_SAMPLES})", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Temps (secondes) - Log Scale", fontsize=12)
    ax1.set_xlabel("Nombre de Qubits", fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, which="both", ls="--", alpha=0.3)
    
    # --- PLOT 2: Speedup ---
    if df_pivot is not None and "Speedup" in df_pivot.columns:
        speedup_data = df_pivot["Speedup"].dropna().reset_index()
        
        bars = ax2.bar(
            speedup_data['Qubits'].astype(str),
            speedup_data['Speedup'],
            color="#2ca02c", alpha=0.8, edgecolor='black'
        )
        
        ax2.set_title("Speedup GPU vs CPU (Estimated)", fontsize=14, fontweight='bold')
        ax2.set_ylabel("Speedup (× plus rapide)", fontsize=12)
        ax2.set_xlabel("Nombre de Qubits", fontsize=12)
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Baseline (1×)')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # Annotations
        for bar, (_, row) in zip(bars, speedup_data.iterrows()):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}×',
                ha='center', va='bottom', fontweight='bold', fontsize=10
            )
    else:
        ax2.text(0.5, 0.5, "Données insuffisantes\npour calculer le Speedup",
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300, bbox_inches='tight')
    print(f"🖼️  Graphique sauvegardé : {PLOT_FILE}")

# ═══════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        benchmark_run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrompu par l'utilisateur.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
