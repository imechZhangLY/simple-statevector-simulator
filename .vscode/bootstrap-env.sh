#!/usr/bin/env bash
#
# Linux and macOS counterpart of bootstrap-env.ps1.
#
# Creates the virtual environment when it is missing and reinstalls only when
# the requirements file changed, so it is safe to run every time.
#
# bash is required rather than POSIX sh for `set -o pipefail` and `[[ ]]`.
# Only bash 3.2 features are used so the bash shipped with macOS works.

set -euo pipefail

ENVIRONMENT_NAME=".venv"
REQUIREMENTS=""

usage() {
    cat <<'EOF'
usage: bootstrap-env.sh [--environment-name NAME] [--requirements FILE]

  --environment-name  virtual environment directory (default: .venv)
  --requirements      requirements file; defaults to the CUDA build on Linux
                      and the PyPI build on macOS
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --environment-name) ENVIRONMENT_NAME="$2"; shift 2 ;;
        --requirements) REQUIREMENTS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIRECTORY}/.." && pwd)"

if [[ -z "${REQUIREMENTS}" ]]; then
    # The "+cu126" and "+cpu" wheels are published for Linux and Windows only.
    # macOS uses the plain PyPI build, which already covers CPU and MPS.
    if [[ "$(uname -s)" == "Darwin" ]]; then
        REQUIREMENTS="requirements-torch-macos.txt"
    else
        REQUIREMENTS="requirements-torch-cuda.txt"
    fi
fi

ENVIRONMENT_PATH="${REPOSITORY_ROOT}/${ENVIRONMENT_NAME}"
PYTHON="${ENVIRONMENT_PATH}/bin/python"
REQUIREMENTS_PATH="${REPOSITORY_ROOT}/${REQUIREMENTS}"

if [[ ! -f "${REQUIREMENTS_PATH}" ]]; then
    echo "Requirements file not found: ${REQUIREMENTS_PATH}" >&2
    exit 1
fi

if [[ ! -x "${PYTHON}" ]]; then
    echo "Creating virtual environment in ${ENVIRONMENT_NAME} ..."
    python3 -m venv "${ENVIRONMENT_PATH}"
fi

# Linux ships sha256sum, macOS ships shasum. The result is upper-cased so the
# marker matches the one written by Get-FileHash in bootstrap-env.ps1.
file_hash() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1 | tr '[:lower:]' '[:upper:]'
    else
        shasum -a 256 "$1" | cut -d' ' -f1 | tr '[:lower:]' '[:upper:]'
    fi
}

MARKER="${ENVIRONMENT_PATH}/.requirements-hash"
EXPECTED_HASH="$(file_hash "${REQUIREMENTS_PATH}")"
INSTALLED_HASH=""
if [[ -f "${MARKER}" ]]; then
    INSTALLED_HASH="$(cat "${MARKER}")"
fi

if [[ "${INSTALLED_HASH}" == "${EXPECTED_HASH}" ]]; then
    echo "${ENVIRONMENT_NAME} is up to date."
    exit 0
fi

echo "Installing dependencies from ${REQUIREMENTS} into ${ENVIRONMENT_NAME} ..."
"${PYTHON}" -m pip install --upgrade pip --quiet
"${PYTHON}" -m pip install -r "${REQUIREMENTS_PATH}"

printf '%s' "${EXPECTED_HASH}" > "${MARKER}"
echo "${ENVIRONMENT_NAME} ready."
