#!/usr/bin/env bash
# Convenience wrapper to run the `trainer-quantum` service from any working directory.
#
# Usage:
#   ./run_trainer_quantum.sh                           # interactive bash shell
#   ./run_trainer_quantum.sh python3 train_svm_qkernel.py --config configs/fashion_easy.yaml
#   ./run_trainer_quantum.sh python3 train_svm_qkernel.py --config configs/fashion_easy.yaml --gram-backend cuda_states --pca-components 16 --embed-mode ryrz --angle-scale 0.1 --kernel-centering --normalize-kernel --dtype float64 --state-tile -1 --vram-fraction 0.95 --num-streams 1 --precompute-all-states --no-dynamic-batch --no-cuda-graphs
#   ./run_trainer_quantum.sh python3 train_svm_qkernel.py --config configs/fashion_easy.yaml --gram-backend torch --pca-components 16 --embed-mode ryrz --angle-scale 0.1 --kernel-centering --normalize-kernel --dtype float64 --torch-tile-size 512 --torch-pinned-memory --torch-cuda-streams
#   ./run_trainer_quantum.sh bash run_all_quantum.sh
#   ./run_trainer_quantum.sh python3 benchmark.py --all --parallel-gpus 3 --dataset-profile fashion
#   ./run_trainer_quantum.sh python3 benchmark.py --backend-comparison --warmup-runs 1 --benchmark-runs 1
#
# Environment variables:
#   GPU_SERVICE   Docker Compose service name (default: trainer-quantum)
#   CUDA_VISIBLE_DEVICES  GPUs to expose inside the container (default: all)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"

GPU_SERVICE="${GPU_SERVICE:-trainer-quantum}"

# Verify docker compose is available
if ! docker compose version &>/dev/null; then
  echo "Error: 'docker compose' not found. Install Docker Engine >= 23." >&2
  exit 1
fi

# Check nvidia-smi on the host (not inside a container)
if ! command -v nvidia-smi &>/dev/null || ! nvidia-smi &>/dev/null; then
  echo "Warning: nvidia-smi not found or no GPU detected on the host." >&2
  echo "         The container may crash if CUDA is required." >&2
fi

# Build image if not already present (fast no-op if up to date)
docker compose -f "$COMPOSE_FILE" build --quiet "$GPU_SERVICE"

exec docker compose -f "$COMPOSE_FILE" run --rm \
  ${CUDA_VISIBLE_DEVICES:+-e CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"} \
  "$GPU_SERVICE" "$@"
