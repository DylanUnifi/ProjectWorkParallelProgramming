#!/usr/bin/env bash
# Convenience wrapper to run the `trainer-gpu130` service from any working directory.
#
# Usage:
#   ./run_trainer_gpu.sh                           # interactive bash shell
#   ./run_trainer_gpu.sh python3 train_svm_qkernel.py --config configs/fashion_easy.yaml
#   ./run_trainer_gpu.sh bash run_all_gpu.sh
#   ./run_trainer_gpu.sh python3 benchmark.py --all --parallel-gpus 3 --dataset-profile fashion
#   ./run_trainer_gpu.sh python3 benchmark.py --backend-comparison --warmup-runs 1 --benchmark-runs 1
#
# Environment variables:
#   GPU_SERVICE   Docker Compose service name (default: trainer-gpu130)
#   CUDA_VISIBLE_DEVICES  GPUs to expose inside the container (default: all)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"

GPU_SERVICE="${GPU_SERVICE:-trainer-gpu130}"

# Verify docker compose is available
if ! docker compose version &>/dev/null; then
  echo "Error: 'docker compose' not found. Install Docker Engine >= 23." >&2
  exit 1
fi

# Verify at least one NVIDIA GPU is visible to docker
if ! docker run --rm --gpus all --entrypoint nvidia-smi "${GPU_SERVICE}" &>/dev/null 2>&1; then
  # Soft check: warn but continue (CI or CPU fallback)
  echo "Warning: NVIDIA runtime may not be available. The container may crash if CUDA is required." >&2
fi

# Build image if not already present (fast no-op if up to date)
docker compose -f "$COMPOSE_FILE" build --quiet "$GPU_SERVICE"

exec docker compose -f "$COMPOSE_FILE" run --rm \
  ${CUDA_VISIBLE_DEVICES:+-e CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"} \
  "$GPU_SERVICE" "$@"
