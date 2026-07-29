#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "${SCRIPT_DIRECTORY}/../scripts/create-env.sh" \
    --environment-name .venv-bench-cuda \
    --requirements envs/bench-cuda/requirements.txt \
    "$@"
