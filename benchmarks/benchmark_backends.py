import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from numpy_backend import NumpyBackend  # noqa: E402
from single_qubit_gates import H, RX  # noqa: E402
from statevector import StateVector  # noqa: E402
from three_qubit_gates import CCX  # noqa: E402
from two_qubit_gates import CX  # noqa: E402

PEAK_MEMORY_FACTOR = 3
BYTES_PER_AMPLITUDE = {"complex64": 8, "complex128": 16}


def build_backends(selected: list[str] | None) -> list:
    backends = [
        ("numpy:complex128", lambda: NumpyBackend()),
        ("numpy:complex64", lambda: NumpyBackend(dtype=np.complex64)),
    ]

    try:
        import torch
    except ImportError:
        torch = None

    try:
        import torch_br  # noqa: F401  registers torch.supa
    except ImportError:
        torch_br = None

    if torch is not None:
        from torch_backend import TorchBackend

        backends.extend(
            [
                ("torch:cpu:complex128", lambda: TorchBackend(dtype="complex128")),
                ("torch:cpu:complex64", lambda: TorchBackend(dtype="complex64")),
            ]
        )
        if torch.cuda.is_available():
            backends.extend(
                [
                    (
                        "torch:cuda:complex128",
                        lambda: TorchBackend(device="cuda", dtype="complex128"),
                    ),
                    (
                        "torch:cuda:complex64",
                        lambda: TorchBackend(device="cuda", dtype="complex64"),
                    ),
                ]
            )
        if torch_br is not None and torch.supa.is_available():
            backends.append(
                (
                    "torch:supa:complex64",
                    lambda: TorchBackend(device="supa", dtype="complex64"),
                )
            )

    if selected is None:
        return [factory() for _, factory in backends]

    known = dict(backends)
    unknown = [name for name in selected if name not in known]
    if unknown:
        raise SystemExit(
            f"unknown backend(s): {', '.join(unknown)}\n"
            f"available: {', '.join(known)}"
        )
    return [known[name]() for name in selected]


def amplitude_bytes(backend) -> int:
    dtype_name = backend.name.rsplit(":", 1)[-1]
    return BYTES_PER_AMPLITUDE[dtype_name]


def device_type(backend) -> str | None:
    device = getattr(backend, "device", None)
    return getattr(device, "type", None)


def make_synchronize(backend):
    backend_device = device_type(backend)
    if backend_device == "cuda":
        import torch

        return torch.cuda.synchronize
    if backend_device == "supa":
        import torch
        import torch_br  # noqa: F401  registers torch.supa

        return torch.supa.synchronize
    return lambda: None


def available_memory(backend) -> int:
    backend_device = device_type(backend)
    if backend_device == "cuda":
        import torch

        return torch.cuda.mem_get_info()[0]
    if backend_device == "supa":
        import torch
        import torch_br  # noqa: F401  registers torch.supa

        return torch.supa.mem_get_info()[0]
    return 8 * 1024**3


def fits_in_memory(backend, num_qubits: int) -> bool:
    required = (1 << num_qubits) * amplitude_bytes(backend) * PEAK_MEMORY_FACTOR
    return required < available_memory(backend)


def time_operations(state, operations, repeats: int, synchronize) -> tuple[float, float]:
    for operation in operations:
        state.apply(operation)
    synchronize()

    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        for operation in operations:
            state.apply(operation)
        synchronize()
        durations.append((time.perf_counter() - start) / len(operations))

    return min(durations), statistics.median(durations)


def time_callable(function, repeats: int, inner: int) -> float:
    for _ in range(inner):
        function()

    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(inner):
            function()
        durations.append((time.perf_counter() - start) / inner)

    return min(durations)


def effective_inner(inner: int, num_qubits: int, reference: int = 16) -> int:
    if num_qubits <= reference:
        return inner
    return max(inner >> (num_qubits - reference), 2)


def gate_workloads(num_qubits: int, inner: int) -> dict[str, list]:
    last = num_qubits - 1
    return {
        "1q H(0)": [H(0)] * inner,
        f"1q H({last})": [H(last)] * inner,
        "2q CX(0,1)": [CX(0, 1)] * inner,
        f"2q CX({last},0)": [CX(last, 0)] * inner,
        "3q CCX(0,1,2)": [CCX(0, 1, 2)] * inner,
    }


def run_gate_benchmark(backends, qubit_counts, repeats, inner) -> None:
    print("\n=== apply() cost per gate (microseconds, min of repeats) ===\n")

    for num_qubits in qubit_counts:
        scaled_inner = effective_inner(inner, num_qubits)
        workloads = gate_workloads(num_qubits, scaled_inner)
        headers = list(workloads)
        print(f"n = {num_qubits} qubits (inner={scaled_inner})")
        print(f"{'backend':<24}" + "".join(f"{header:>16}" for header in headers))

        for backend in backends:
            if not fits_in_memory(backend, num_qubits):
                print(f"{backend.name:<24}{'skipped (memory)':>16}")
                continue

            synchronize = make_synchronize(backend)
            state = StateVector(num_qubits, backend=backend)
            cells = []
            for header in headers:
                best, _ = time_operations(
                    state, workloads[header], repeats, synchronize
                )
                cells.append(f"{best * 1e6:>16.1f}")
            print(f"{backend.name:<24}" + "".join(cells))
        print()


def run_cache_benchmark(num_qubits, repeats, inner) -> None:
    try:
        import torch
    except ImportError:
        return

    try:
        import torch_br  # noqa: F401  registers torch.supa
    except ImportError:
        torch_br = None

    from torch_backend import TorchBackend

    configurations = [
        ("torch:cpu:complex128", {"device": "cpu", "dtype": "complex128"}),
        ("torch:cpu:complex64", {"device": "cpu", "dtype": "complex64"}),
    ]
    if torch.cuda.is_available():
        configurations.extend(
            [
                ("torch:cuda:complex128", {"device": "cuda", "dtype": "complex128"}),
                ("torch:cuda:complex64", {"device": "cuda", "dtype": "complex64"}),
            ]
        )
    if torch_br is not None and torch.supa.is_available():
        configurations.append(
            ("torch:supa:complex64", {"device": "supa", "dtype": "complex64"})
        )

    print("\n=== matrix cache: guaranteed hit vs guaranteed miss ===")
    print(
        f"(n = {num_qubits} qubits; misses are forced with matrix_cache_size=1 "
        "and two alternating angles)\n"
    )

    cache_inner = max(inner * 10, 2)
    cache_repeats = repeats * 3
    operations = [RX(0.3, 0), RX(0.7, 0)] * (cache_inner // 2)

    print(f"{'backend':<24}{'hit':>14}{'miss':>14}{'overhead':>14}")
    for label, keywords in configurations:
        hit_backend = TorchBackend(matrix_cache_size=8, **keywords)
        miss_backend = TorchBackend(matrix_cache_size=1, **keywords)
        synchronize = make_synchronize(hit_backend)

        hit_state = StateVector(num_qubits, backend=hit_backend)
        miss_state = StateVector(num_qubits, backend=miss_backend)
        hit_best, _ = time_operations(
            hit_state, operations, cache_repeats, synchronize
        )
        miss_best, _ = time_operations(
            miss_state, operations, cache_repeats, synchronize
        )

        if hit_backend.cached_matrix_count != 2:
            raise RuntimeError("hit configuration did not cache both matrices")
        if miss_backend.cached_matrix_count != 1:
            raise RuntimeError("miss configuration retained more than one matrix")

        print(
            f"{label:<24}"
            f"{hit_best * 1e6:>14.2f}"
            f"{miss_best * 1e6:>14.2f}"
            f"{(miss_best - hit_best) * 1e6:>+14.2f}"
        )
    print("\nmicroseconds per apply, min of repeats")


def run_construction_benchmark(repeats, inner) -> None:
    print("\n=== gate function cost (microseconds per call) ===\n")

    measurements = {
        "H(0) shared gate": lambda: H(0),
        "CX(0,1) shared gate": lambda: CX(0, 1),
        "RX(0.3,0) bound gate": lambda: RX(0.3, 0),
        "CCX(0,1,2) shared gate": lambda: CCX(0, 1, 2),
    }

    for label, function in measurements.items():
        best = time_callable(function, repeats * 3, inner * 10)
        print(f"{label:<26}{best * 1e6:>10.2f}")
    print()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Statevector backend benchmarks")
    parser.add_argument(
        "--qubits",
        default="10,16,20,22",
        help="comma-separated qubit counts",
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--inner", type=int, default=20)
    parser.add_argument(
        "--backends",
        default=None,
        help="comma-separated backend names; defaults to all available",
    )
    parser.add_argument(
        "--cache-qubits",
        type=int,
        default=4,
        help="qubit count used for the matrix cache benchmark",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    qubit_counts = [int(value) for value in arguments.qubits.split(",")]
    selected = arguments.backends.split(",") if arguments.backends else None
    backends = build_backends(selected)

    print("backends:", ", ".join(backend.name for backend in backends))
    print(f"repeats={arguments.repeats} inner={arguments.inner}")

    run_gate_benchmark(backends, qubit_counts, arguments.repeats, arguments.inner)
    run_cache_benchmark(
        arguments.cache_qubits, arguments.repeats, arguments.inner
    )
    run_construction_benchmark(arguments.repeats, arguments.inner)


if __name__ == "__main__":
    main()
