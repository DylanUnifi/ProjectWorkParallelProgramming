#!/usr/bin/env bash
# Generic GPU launcher; hardware-specific selection stays in docker compose.

set -euo pipefail

datasets=("fashion" "cifar10" "svhn")
difficulties=("easy" "med" "hard")
sizes=("500" "1000" "2500" "5000" "all")
backends=("torch" "cuda_states") # backends to test

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="logs/quantum/${RUN_TS}"
mkdir -p "${LOG_ROOT}"

gpu_id=0
max_gpus=8
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
      echo "Success: Completed: ${label}"
    else
      echo "Failed: ${label}"
      failures=$((failures + 1))
    fi
  done

  batch_pids=()
  batch_labels=()
}

echo "Starting quantum cluster across $max_gpus GPUs (Torch and CUDA States)..."

for ds in "${datasets[@]}"; do
  for diff in "${difficulties[@]}"; do
    for size in "${sizes[@]}"; do
      for backend in "${backends[@]}"; do
        
        echo "Launching $ds | $diff | $size | Backend: $backend on GPU #$gpu_id"
        
        # Handle the "all" size parameter
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

        log_file="${LOG_ROOT}/log_${ds}_${diff}_${backend}_${size}.txt"
        (time CUDA_VISIBLE_DEVICES=$gpu_id "${cmd[@]}") 2>&1 | tee -a "${log_file}" &
        batch_pids+=("$!")
        batch_labels+=("${ds}|${diff}|${size}|${backend}|gpu${gpu_id}")
        
        gpu_id=$(( (gpu_id + 1) % max_gpus ))
        
        # Pause after each wave of max_gpus launches
        if [ $gpu_id -eq 0 ]; then
          wait_for_batch
        fi
        
      done
    done
  done
done

wait_for_batch

if [ "$failures" -gt 0 ]; then
  echo "Some tasks finished with ${failures} failure(s)."
  echo "Logs available in: ${LOG_ROOT}"
  exit 1
fi

echo "All quantum tasks completed successfully."
echo "Logs available in: ${LOG_ROOT}"