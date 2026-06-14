#!/usr/bin/env bash
# Convenience wrapper to run the `trainer-cpu` service from any working directory.
set -euo pipefail

# Resolve this script directory (project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec docker compose -f "$SCRIPT_DIR/docker-compose.yml" run --rm trainer-cpu "$@"
