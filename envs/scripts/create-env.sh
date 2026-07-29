#!/usr/bin/env bash
#
# Shared virtual environment creator used by every envs/<name>/create-env.sh.
#
# Creates the environment when it is missing and reinstalls only when the
# requirements file changed, so it is safe to run every time.
#
# bash is required rather than POSIX sh for `set -o pipefail` and `[[ ]]`.

set -euo pipefail

ENVIRONMENT_NAME=""
REQUIREMENTS=""
SYSTEM_SITE_PACKAGES=0

usage() {
    cat <<'EOF'
usage: create-env.sh --environment-name NAME --requirements FILE [--system-site-packages]
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --environment-name) ENVIRONMENT_NAME="$2"; shift 2 ;;
        --requirements) REQUIREMENTS="$2"; shift 2 ;;
        --system-site-packages) SYSTEM_SITE_PACKAGES=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "${ENVIRONMENT_NAME}" || -z "${REQUIREMENTS}" ]]; then
    usage >&2
    exit 2
fi

SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIRECTORY}/../.." && pwd)"

ENVIRONMENT_PATH="${REPOSITORY_ROOT}/${ENVIRONMENT_NAME}"
PYTHON="${ENVIRONMENT_PATH}/bin/python"
REQUIREMENTS_PATH="${REPOSITORY_ROOT}/${REQUIREMENTS}"

if [[ ! -f "${REQUIREMENTS_PATH}" ]]; then
    echo "Requirements file not found: ${REQUIREMENTS_PATH}" >&2
    exit 1
fi

if [[ ! -x "${PYTHON}" ]]; then
    echo "Creating virtual environment in ${ENVIRONMENT_NAME} ..."
    if [[ "${SYSTEM_SITE_PACKAGES}" -eq 1 ]]; then
        python3 -m venv --system-site-packages "${ENVIRONMENT_PATH}"
    else
        python3 -m venv "${ENVIRONMENT_PATH}"
    fi
fi

# Linux ships sha256sum, macOS ships shasum. The result is upper-cased so the
# marker matches the one written by Get-FileHash in create-env.ps1.
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
else
    echo "Installing dependencies from ${REQUIREMENTS} into ${ENVIRONMENT_NAME} ..."
    "${PYTHON}" -m pip install --upgrade pip --quiet
    "${PYTHON}" -m pip install -r "${REQUIREMENTS_PATH}"

    printf '%s' "${EXPECTED_HASH}" > "${MARKER}"
    echo "${ENVIRONMENT_NAME} ready."
fi

echo "Interpreter: ${PYTHON}"
