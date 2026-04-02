#!/usr/bin/env bash

set -euo pipefail

datasets=("fashion" "cifar10" "svhn")
difficulties=("easy" "med" "hard")
sizes=("500" "1000" "2500" "5000" "all")

cpu_jobs=0
max_cpu_jobs=6
failures=0

batch_pids=()
batch_labels=()

wait_for_batch() {
  if [ ${#batch_pids[@]} -eq 0 ]; then
    return
  fi

  echo "CPU concurrency limit reached. Waiting..."
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

echo "Starting classical CPU benchmark..."

for ds in "${datasets[@]}"; do
  for diff in "${difficulties[@]}"; do
    for size in "${sizes[@]}"; do
      
      echo "Launching classical run: $ds | $diff | $size"
      
      cmd=(
        python3 train_svm_classical.py
        --config configs/${ds}_${diff}.yaml \
        --pca-components 16 \
        --kernel rbf
      )

      if [ "$size" != "all" ]; then
        cmd+=(--train-subset "$size")
      fi

      (time "${cmd[@]}") 2>&1 | tee -a "log_${ds}_${diff}_classical_${size}.txt" &
      batch_pids+=("$!")
      batch_labels+=("${ds}|${diff}|${size}|classical")
      
      cpu_jobs=$((cpu_jobs + 1))
      
      if [ $cpu_jobs -eq $max_cpu_jobs ]; then
        wait_for_batch
        cpu_jobs=0
      fi
      
    done
  done
done

wait_for_batch

if [ "$failures" -gt 0 ]; then
  echo "❌ Classical jobs completed with ${failures} failure(s)."
  exit 1
fi

echo "All classical jobs completed."
