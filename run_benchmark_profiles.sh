#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOG_DIR="benchmark_results/logs"
RUN_TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p "${LOG_DIR}"

run_profile() {
  local profile="$1"
  local log_file="${LOG_DIR}/benchmark_${profile}_${RUN_TS}.log"

  echo ""
  echo "============================================================"
  echo "Running benchmark profile: ${profile}"
  echo "Log file: ${log_file}"
  echo "============================================================"

  docker compose -f "$SCRIPT_DIR/docker-compose.yml" run --rm -T trainer-gpu130 python3 benchmark.py \
    --all \
    --parallel-gpus 5 \
    --dataset-profile "${profile}" \
    --output-dir "benchmark_results/${profile}" \
    --warmup-runs 2 \
    --benchmark-runs 2 2>&1 | tee "${log_file}"
}

run_profile fashion
run_profile cifar10
run_profile svhn

echo ""
echo "All benchmark profiles completed successfully."
echo "Logs written to: ${LOG_DIR}"
