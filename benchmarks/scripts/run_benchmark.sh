#!/usr/bin/env bash
#
# Run one cross-framework benchmark scenario in the active environment.
#
#   cpu-single  every framework pinned to one thread
#   cpu-multi   every framework using all logical processors
#   gpu         Torch CUDA and qiskit-aer-gpu (Linux only)
#   supa        Torch supa (Linux only)
#
# Activate the matching environment from envs/ first; this script neither
# creates nor inspects it, and missing dependencies are reported by
# framework_comparison.py.
#
# qiskit-aer is the error reference in every scenario.
#
# bash is required rather than POSIX sh for `set -o pipefail`, arrays and
# `[[ ]]`. Only bash 3.2 features are used so macOS works without an upgrade.

set -euo pipefail

SCENARIO="cpu-single"
QUBITS="4,8,12,16,20"
REPEATS=5

usage() {
    cat <<'EOF'
usage: run_benchmark.sh [options]

  --scenario <cpu-single|cpu-multi|gpu|supa>   default: cpu-single
  --qubits <list>                         default: 4,8,12,16,20
  --repeats <n>                           default: 5
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenario) SCENARIO="$2"; shift 2 ;;
        --qubits) QUBITS="$2"; shift 2 ;;
        --repeats) REPEATS="$2"; shift 2 ;;
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

case "${SCENARIO}" in
    cpu-single)
        THREADS=1
        IMPLEMENTATIONS="ours:numpy:complex128,ours:numpy:complex128:fusion,ours:torch:cpu:complex128,ours:torch:cpu:complex128:fusion,qulacs:cpu:complex128,qiskit-aer:cpu:complex128"
        REFERENCE="qiskit-aer:cpu:complex128"
        TITLE="qulacs benchmark circuit, CPU single thread"
        ;;
    cpu-multi)
        THREADS="$(logical_processors)"
        IMPLEMENTATIONS="ours:numpy:complex128,ours:numpy:complex128:fusion,ours:torch:cpu:complex128,ours:torch:cpu:complex128:fusion,qulacs:cpu:complex128,qiskit-aer:cpu:complex128"
        REFERENCE="qiskit-aer:cpu:complex128"
        TITLE="qulacs benchmark circuit, CPU ${THREADS} threads"
        ;;
    gpu)
        if [[ "$(uname -s)" != "Linux" ]]; then
            echo "the gpu scenario requires Linux: qiskit-aer-gpu publishes" >&2
            echo "manylinux x86_64 wheels only." >&2
            exit 1
        fi
        THREADS=0
        IMPLEMENTATIONS="ours:torch:cuda:complex64,ours:torch:cuda:complex64:fusion,ours:torch:cuda:complex128,ours:torch:cuda:complex128:fusion,qiskit-aer:gpu:complex128"
        REFERENCE="qiskit-aer:gpu:complex128"
        TITLE="qulacs benchmark circuit, GPU"
        ;;
    supa)
        THREADS="$(logical_processors)"
        IMPLEMENTATIONS="ours:torch:cpu:complex64,ours:torch:cpu:complex64:fusion,ours:torch:supa:complex64,ours:torch:supa:complex64:fusion,qiskit-aer:cpu:complex128"
        REFERENCE="qiskit-aer:cpu:complex128"
        TITLE="supa benchmark circuit, CPU ${THREADS} threads"
        ;;
    *)
        echo "unknown scenario: ${SCENARIO}" >&2
        usage
        exit 2
        ;;
esac

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
echo "interpreter: $(command -v python)"
echo

python "${REPOSITORY_ROOT}/benchmarks/framework_comparison.py" \
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
