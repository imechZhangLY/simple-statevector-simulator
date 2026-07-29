#!/usr/bin/env bash
#
# Provision the benchmark environment and run one cross-framework scenario.
#
#   cpu-single  every framework pinned to one thread
#   cpu-multi   every framework using all logical processors
#   gpu         Torch CUDA and qiskit-aer-gpu (Linux only)
#
# qiskit-aer is the fidelity reference in every scenario.
#
# bash is required rather than POSIX sh for `set -o pipefail`, arrays and
# `[[ ]]`. Only bash 3.2 features are used so macOS works without an upgrade.

set -euo pipefail

SCENARIO="cpu-single"
QUBITS="4,8,12,16,20"
REPEATS=5
SKIP_INSTALL=0

usage() {
    cat <<'EOF'
usage: run_benchmark.sh [options]

  --scenario <cpu-single|cpu-multi|gpu|supa>   default: cpu-single
  --qubits <list>                         default: 4,8,12,16,20
  --repeats <n>                           default: 5
  --skip-install                          reuse the environment as-is
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenario) SCENARIO="$2"; shift 2 ;;
        --qubits) QUBITS="$2"; shift 2 ;;
        --repeats) REPEATS="$2"; shift 2 ;;
        --skip-install) SKIP_INSTALL=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIRECTORY}/../.." && pwd)"
cd "${REPOSITORY_ROOT}"

logical_processors() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
    elif [[ "$(uname -s)" == "Darwin" ]]; then
        sysctl -n hw.logicalcpu
    else
        echo 1
    fi
}

# Linux ships sha256sum, macOS ships shasum. The result is upper-cased so the
# marker matches the one written by Get-FileHash in run_benchmark.ps1.
file_hash() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1 | tr '[:lower:]' '[:upper:]'
    else
        shasum -a 256 "$1" | cut -d' ' -f1 | tr '[:lower:]' '[:upper:]'
    fi
}

dependencies_ready() {
    local python="$1"
    local marker="$2"
    local expected="$3"
    shift 3

    [[ -f "${marker}" ]] || return 1
    [[ "$(cat "${marker}")" == "${expected}" ]] || return 1

    # The hash alone cannot tell that a package was removed by hand, so the
    # modules are located as well. find_spec avoids importing torch, which
    # would cost several seconds on its own.
    "${python}" "${SCRIPT_DIRECTORY}/check_dependencies.py" "$@" || return 1
    return 0
}

case "${SCENARIO}" in
    cpu-single)
        ENVIRONMENT_NAME=".venv-bench"
        REQUIREMENTS="requirements-bench.txt"
        THREADS=1
        IMPLEMENTATIONS="ours:numpy:complex128,ours:torch:cpu:complex128,qulacs:cpu:complex128,qiskit-aer:cpu:complex128"
        REFERENCE="qiskit-aer:cpu:complex128"
        TITLE="qulacs benchmark circuit, CPU single thread"
        MODULES="numpy torch qulacs qiskit_aer matplotlib"
        ;;
    cpu-multi)
        ENVIRONMENT_NAME=".venv-bench"
        REQUIREMENTS="requirements-bench.txt"
        THREADS="$(logical_processors)"
        IMPLEMENTATIONS="ours:numpy:complex128,ours:torch:cpu:complex128,qulacs:cpu:complex128,qiskit-aer:cpu:complex128"
        REFERENCE="qiskit-aer:cpu:complex128"
        TITLE="qulacs benchmark circuit, CPU ${THREADS} threads"
        MODULES="numpy torch qulacs qiskit_aer matplotlib"
        ;;
    gpu)
        if [[ "$(uname -s)" != "Linux" ]]; then
            echo "the gpu scenario requires Linux: qiskit-aer-gpu publishes" >&2
            echo "manylinux x86_64 wheels only." >&2
            exit 1
        fi
        ENVIRONMENT_NAME=".venv-bench-gpu"
        REQUIREMENTS="requirements-bench-gpu.txt"
        THREADS=0
        IMPLEMENTATIONS="ours:torch:cuda:complex64,ours:torch:cuda:complex128,qiskit-aer:gpu:complex128"
        REFERENCE="qiskit-aer:gpu:complex128"
        TITLE="qulacs benchmark circuit, GPU"
        MODULES="numpy torch qiskit_aer matplotlib"
        ;;
    supa)
        ENVIRONMENT_NAME=".venv-bench-supa"
        REQUIREMENTS="requirements-bench-supa.txt"
        THREADS="$(logical_processors)"
        IMPLEMENTATIONS="ours:numpy:complex64,ours:torch:cpu:complex64,ours:torch:supa:complex64,qulacs:cpu:complex128,qiskit-aer:cpu:complex128"
        REFERENCE="qiskit-aer:cpu:complex128"
        TITLE="supa benchmark circuit, CPU ${THREADS} threads"
        MODULES="numpy qulacs qiskit_aer matplotlib"
        ;;
    *)
        echo "unknown scenario: ${SCENARIO}" >&2
        usage
        exit 2
        ;;
esac

PYTHON="${REPOSITORY_ROOT}/${ENVIRONMENT_NAME}/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Creating virtual environment ${ENVIRONMENT_NAME} ..."
    # The --system-site-packages option is required for supa,
    # which is installed system-wide by the package manager.
    python3 -m venv "${REPOSITORY_ROOT}/${ENVIRONMENT_NAME}" --system-site-packages
fi

MARKER="${REPOSITORY_ROOT}/${ENVIRONMENT_NAME}/.requirements-hash"
EXPECTED_HASH="$(file_hash "${REQUIREMENTS}")"

if [[ "${SKIP_INSTALL}" -eq 1 ]]; then
    echo "Skipping the dependency check (--skip-install)."
elif dependencies_ready "${PYTHON}" "${MARKER}" "${EXPECTED_HASH}" ${MODULES}; then
    echo "Dependencies already satisfied for ${REQUIREMENTS}."
else
    echo "Installing ${REQUIREMENTS} ..."
    "${PYTHON}" -m pip install --upgrade pip --quiet
    "${PYTHON}" -m pip install -r "${REQUIREMENTS}" --quiet
    printf '%s' "${EXPECTED_HASH}" > "${MARKER}"
fi

# numpy, qulacs and the BLAS libraries read these at import time, so exporting
# them before Python starts is required. Passing --threads alone is not enough.
if [[ "${THREADS}" -gt 0 ]]; then
    export OMP_NUM_THREADS="${THREADS}"
    export MKL_NUM_THREADS="${THREADS}"
    export OPENBLAS_NUM_THREADS="${THREADS}"
fi

RESULTS_DIRECTORY="${REPOSITORY_ROOT}/benchmarks/results"
mkdir -p "${RESULTS_DIRECTORY}"
JSON_PATH="${RESULTS_DIRECTORY}/${SCENARIO}.json"
PLOT_PATH="${RESULTS_DIRECTORY}/${SCENARIO}.png"

echo
echo "scenario   : ${SCENARIO}"
echo "threads    : ${THREADS}"
echo "qubits     : ${QUBITS}"
echo "reference  : ${REFERENCE}"
echo

"${PYTHON}" "${REPOSITORY_ROOT}/benchmarks/framework_comparison.py" \
    --qubits "${QUBITS}" \
    --repeats "${REPEATS}" \
    --threads "${THREADS}" \
    --reference "${REFERENCE}" \
    --implementations "${IMPLEMENTATIONS}" \
    --output "${JSON_PATH}" \
    --plot "${PLOT_PATH}" \
    --title "${TITLE}"

echo
echo "results : ${JSON_PATH}"
echo "chart   : ${PLOT_PATH}"
