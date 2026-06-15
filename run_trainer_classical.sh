#!/usr/bin/env bash
# Convenience wrapper to run the `trainer-classical` service from any working directory.
#
# Usage:
#   ./run_trainer_classical.sh                          # interactive bash shell
#   ./run_trainer_classical.sh python3 train_svm_classical.py --config configs/fashion_easy.yaml
#   ./run_trainer_classical.sh bash run_all_classical.sh
#   ./run_trainer_classical.sh python3 extract_results.py
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"

# Verify docker compose is available
if ! docker compose version &>/dev/null; then
  echo "Error: 'docker compose' not found. Install Docker Engine >= 23." >&2
  exit 1
fi

# Build image if not already present (fast no-op if up to date)
docker compose -f "$COMPOSE_FILE" build --quiet trainer-classical

exec docker compose -f "$COMPOSE_FILE" run --rm trainer-classical "$@"
