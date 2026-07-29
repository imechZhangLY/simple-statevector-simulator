#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Trailing "$@" lets a caller override the defaults; the last flag wins.
exec bash "${SCRIPT_DIRECTORY}/../scripts/create-env.sh" \
    --environment-name .venv-cuda \
    --requirements envs/cuda/requirements.txt \
    "$@"
