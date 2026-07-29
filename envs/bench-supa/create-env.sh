#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --system-site-packages is required: torch and torch_br come from the supa
# cloud image rather than from pip.
exec bash "${SCRIPT_DIRECTORY}/../scripts/create-env.sh" \
    --environment-name .venv-bench-supa \
    --requirements envs/bench-supa/requirements.txt \
    --system-site-packages \
    "$@"
