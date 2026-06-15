#!/usr/bin/env bash
# Generic GPU launcher; hardware-specific selection stays in docker compose.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${IN_DOCKER_CONTAINER:-}" ]]; then
  exec docker compose -f "$SCRIPT_DIR/docker-compose.yml" run --rm -T \
    -e IN_DOCKER_CONTAINER=1 trainer-quantum bash run_all_quantum.sh "$@"
fi

datasets=("fashion" "cifar10" "svhn")
difficulties=("easy" "med" "hard")
# Keep the default sweep short enough to produce actionable first results quickly.
sizes=("500" "1000")
backends=("torch" "cuda_states") # backends to test

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="logs/quantum/${RUN_TS}"
mkdir -p "${LOG_ROOT}"

# Detect available GPUs dynamically; fall back to 1 if nvidia-smi is unavailable.
max_gpus=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 1)
if [[ "$max_gpus" -eq 0 ]]; then max_gpus=1; fi
echo "Detected ${max_gpus} GPU(s)."
failures=0

batch_pids=()
batch_labels=()
batch_gpu_ids=()
cleanup_done=0

cleanup() {
  if [ "$cleanup_done" -eq 1 ]; then
    return 0
  fi

  cleanup_done=1
  local exit_code=$?

  if [ ${#batch_pids[@]} -gt 0 ]; then
    echo "Stopping running GPU jobs..."
    for pid in "${batch_pids[@]}"; do
      kill -- "-$pid" 2>/dev/null || true
    done

    for pid in "${batch_pids[@]}"; do
      wait "$pid" 2>/dev/null || true
    done
  fi

  batch_pids=()
  batch_labels=()
  batch_gpu_ids=()

  return "$exit_code"
}

trap 'cleanup; exit 130' INT TERM
trap cleanup EXIT

wait_for_all() {
  if [ ${#batch_pids[@]} -eq 0 ]; then
    return
  fi

  echo "Waiting for dataset workers to complete..."
  for i in "${!batch_pids[@]}"; do
    pid="${batch_pids[$i]}"
    label="${batch_labels[$i]}"
    if wait "$pid"; then
      echo "Success: Completed dataset worker: ${label}"
    else
      echo "Failed dataset worker: ${label}"
      failures=$((failures + 1))
    fi
  done
}

wait_for_dataset_slot() {
  if [ ${#batch_pids[@]} -lt "$max_gpus" ]; then
    return
  fi

  echo "All dataset slots busy. Waiting for the next dataset worker to finish..."

  local next_pids=()
  local next_labels=()
  local next_gpu_ids=()
  local freed_gpu_id=""

  while [ -z "$freed_gpu_id" ]; do
    wait -n || true

    for i in "${!batch_pids[@]}"; do
      pid="${batch_pids[$i]}"
      label="${batch_labels[$i]}"
      gpu_id="${batch_gpu_ids[$i]}"

      if kill -0 "$pid" 2>/dev/null; then
        next_pids+=("$pid")
        next_labels+=("$label")
        next_gpu_ids+=("$gpu_id")
        continue
      fi

      if wait "$pid"; then
        echo "Success: Completed dataset worker: ${label}"
      else
        echo "Failed dataset worker: ${label}"
        failures=$((failures + 1))
      fi

      if [ -z "$freed_gpu_id" ]; then
        freed_gpu_id="$gpu_id"
      fi
    done

    batch_pids=("${next_pids[@]}")
    batch_labels=("${next_labels[@]}")
    batch_gpu_ids=("${next_gpu_ids[@]}")
  done

  printf '%s\n' "$freed_gpu_id"
}

launch_dataset_worker() {
  local dataset_name="$1"
  local gpu_id="$2"
  local worker_label="$2"

  (
    set -euo pipefail
    worker_failed=0

    echo "Dataset ${dataset_name} pinned to GPU #${gpu_id}"

    for diff in "${difficulties[@]}"; do
      for size in "${sizes[@]}"; do
        for backend in "${backends[@]}"; do
          echo "Launching ${dataset_name} | ${diff} | ${size} | Backend: ${backend} on GPU #${gpu_id}"

          cmd=(
            python3 train_svm_qkernel.py
            --config configs/${dataset_name}_${diff}.yaml \
            --gram-backend "$backend" \
            --pca-components 16 \
            --embed-mode ryrz \
            --angle-scale 0.1 \
            --kernel-centering \
            --normalize-kernel
          )

          if [ "$backend" = "cuda_states" ]; then
            cmd+=(
              --dtype float64
              --state-tile -1
              --vram-fraction 0.95
              --num-streams 1
              --precompute-all-states
              --no-dynamic-batch
              --no-cuda-graphs
            )
          elif [ "$backend" = "torch" ]; then
            cmd+=(
              --dtype float64
              --torch-tile-size 512
              --torch-pinned-memory
              --torch-cuda-streams
            )
          fi

          if [ "$size" != "all" ]; then
            cmd+=(--train-subset "$size")
          fi

          log_file="${LOG_ROOT}/log_${dataset_name}_${diff}_${backend}_${size}.txt"
          echo "[$(date +%H:%M:%S)] START ${dataset_name}|${diff}|${size}|${backend}|gpu${gpu_id} -> ${log_file}" >> "$log_file"
          if ! { time CUDA_VISIBLE_DEVICES="$gpu_id" "${cmd[@]}"; } >> "$log_file" 2>&1; then
            echo "[$(date +%H:%M:%S)] FAIL  ${dataset_name}|${diff}|${size}|${backend}|gpu${gpu_id}" >> "$log_file"
            echo "Failed: ${dataset_name}|${diff}|${size}|${backend}|gpu${gpu_id}"
            worker_failed=1
          else
            echo "[$(date +%H:%M:%S)] OK    ${dataset_name}|${diff}|${size}|${backend}|gpu${gpu_id}" >> "$log_file"
            echo "Success: Completed: ${dataset_name}|${diff}|${size}|${backend}|gpu${gpu_id}"
          fi
        done
      done
    done

    exit "$worker_failed"
  ) &

  batch_pids+=("$!")
  batch_labels+=("$worker_label")
  batch_gpu_ids+=("$gpu_id")
}

echo "Starting quantum cluster across ${max_gpus} GPU(s) with one dataset queue per GPU..."

next_gpu_id=0

for ds in "${datasets[@]}"; do
  if [ ${#batch_pids[@]} -lt "$max_gpus" ]; then
    gpu_id="$next_gpu_id"
    next_gpu_id=$((next_gpu_id + 1))
  else
    gpu_id="$(wait_for_dataset_slot)"
  fi

  launch_dataset_worker "$ds" "$gpu_id" "${ds}|gpu${gpu_id}"
done

wait_for_all

if [ "$failures" -gt 0 ]; then
  echo "Some tasks finished with ${failures} failure(s)."
  echo "Logs available in: ${LOG_ROOT}"
  exit 1
fi

echo "All quantum tasks completed successfully."
echo "Logs available in: ${LOG_ROOT}"