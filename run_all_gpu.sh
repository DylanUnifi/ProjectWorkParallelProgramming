#!/usr/bin/env bash

set -euo pipefail

datasets=("fashion" "cifar10" "svhn")
difficulties=("easy" "med" "hard")
sizes=("500" "1000" "2500" "5000" "all")
backends=("torch" "cuda_states") # 👈 La magie opère ici !

gpu_id=0
max_gpus=5
failures=0

batch_pids=()
batch_labels=()

wait_for_batch() {
  if [ ${#batch_pids[@]} -eq 0 ]; then
    return
  fi

  echo "⏳ Waiting for current GPU wave to complete..."
  for i in "${!batch_pids[@]}"; do
    pid="${batch_pids[$i]}"
    label="${batch_labels[$i]}"
    if wait "$pid"; then
      echo "✅ Completed: ${label}"
    else
      echo "❌ Failed: ${label}"
      failures=$((failures + 1))
    fi
  done

  batch_pids=()
  batch_labels=()
}

echo "🔥 Démarrage du cluster Quantique sur $max_gpus GPUs (Torch & CUDA States)..."

for ds in "${datasets[@]}"; do
  for diff in "${difficulties[@]}"; do
    for size in "${sizes[@]}"; do
      for backend in "${backends[@]}"; do
        
        echo "➔ Lancement $ds | $diff | $size | Backend: $backend sur le GPU n°$gpu_id"
        
        # Gestion du paramètre "all"
        cmd=(
          python3 train_svm_qkernel.py
          --config configs/${ds}_${diff}.yaml \
          --gram-backend $backend \
          --pca-components 16 \
          --embed-mode ryrz \
          --kernel-centering \
          --normalize-kernel
        )

        if [ "$size" != "all" ]; then
          cmd+=(--train-subset "$size")
        fi

        (time CUDA_VISIBLE_DEVICES=$gpu_id "${cmd[@]}") 2>&1 | tee -a "log_${ds}_${diff}_${backend}_${size}.txt" &
        batch_pids+=("$!")
        batch_labels+=("${ds}|${diff}|${size}|${backend}|gpu${gpu_id}")
        
        gpu_id=$(( (gpu_id + 1) % max_gpus ))
        
        # Pause tous les 5 lancements
        if [ $gpu_id -eq 0 ]; then
          wait_for_batch
        fi
        
      done
    done
  done
done

wait_for_batch

if [ "$failures" -gt 0 ]; then
  echo "❌ TÂCHES TERMINÉES AVEC ${failures} ÉCHEC(S)."
  exit 1
fi

echo "✅ TOUTES LES TÂCHES QUANTIQUES SONT TERMINÉES !"