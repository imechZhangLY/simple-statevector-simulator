#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --system-site-packages is required: torch and torch_br are provided by the
# supa cloud image, not by pip, so an isolated venv could not import them.
exec bash "${SCRIPT_DIRECTORY}/../scripts/create-env.sh" \
    --environment-name .venv-supa \
    --requirements envs/supa/requirements.txt \
    --system-site-packages \
    "$@"
