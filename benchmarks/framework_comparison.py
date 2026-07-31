import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

DEFAULT_DEPTH = 9
MINIMUM_QUBITS = 2
IMPLEMENTATION_WIDTH = 40


def build_instructions(num_qubits: int, depth: int, seed: int) -> list[tuple]:
    """Reproduce the qulacs benchmark circuit as backend-neutral instructions.

    first_rotation -> entangler -> depth x (mid_rotation -> entangler)
    -> last_rotation, which yields 41 * num_qubits gates at depth 9.
    """
    generator = np.random.default_rng(seed)
    pairs = [(index, (index + 1) % num_qubits) for index in range(num_qubits)]
    instructions: list[tuple] = []

    def rotation(name: str, qubit: int) -> None:
        instructions.append((name, (qubit,), float(generator.random())))

    def entangler() -> None:
        instructions.extend(("cnot", pair, None) for pair in pairs)

    for qubit in range(num_qubits):
        rotation("rx", qubit)
        rotation("rz", qubit)
    entangler()

    for _ in range(depth):
        for qubit in range(num_qubits):
            rotation("rz", qubit)
            rotation("rx", qubit)
            rotation("rz", qubit)
        entangler()

    for qubit in range(num_qubits):
        rotation("rz", qubit)
        rotation("rx", qubit)

    return instructions


class OurImplementation:
    def __init__(self, label: str, backend_factory, *, fusion: bool = False) -> None:
        self.label = label
        self._backend_factory = backend_factory
        self._fusion = fusion

    def build(self, num_qubits: int, instructions: list[tuple], threads: int):
        from circuit import Circuit
        from simulator import StateVectorSimulator
        from single_qubit_gates import RX, RZ
        from two_qubit_gates import CX

        if threads > 0 and self.label.startswith("ours:torch"):
            import torch

            torch.set_num_threads(threads)

        circuit = Circuit(num_qubits)
        for name, qubits, parameter in instructions:
            if name == "rx":
                circuit.append(RX(parameter, qubits[0]))
            elif name == "rz":
                circuit.append(RZ(parameter, qubits[0]))
            else:
                circuit.append(CX(qubits[0], qubits[1]))

        simulator = StateVectorSimulator(
            self._backend_factory(len(instructions)),
            fusion=self._fusion,
        )
        state = {}

        def run() -> None:
            state["result"] = simulator.run(circuit)

        def final_state() -> np.ndarray:
            return state["result"].amplitudes

        return run, _make_synchronize(simulator.backend), final_state


class QulacsImplementation:
    def __init__(self, label: str = "qulacs:cpu:complex128", use_gpu: bool = False) -> None:
        self.label = label
        self._use_gpu = use_gpu

    def build(self, num_qubits: int, instructions: list[tuple], threads: int):
        from qulacs import QuantumCircuit as QulacsCircuit

        if self._use_gpu:
            from qulacs import QuantumStateGpu as StateType
        else:
            from qulacs import QuantumState as StateType

        circuit = QulacsCircuit(num_qubits)
        for name, qubits, parameter in instructions:
            # qulacs rotations use exp(+i theta P / 2), the opposite sign of
            # this project and Qiskit, so the angle is negated to make the
            # frameworks compute the same state.
            if name == "rx":
                circuit.add_RX_gate(qubits[0], -parameter)
            elif name == "rz":
                circuit.add_RZ_gate(qubits[0], -parameter)
            else:
                circuit.add_CNOT_gate(qubits[0], qubits[1])

        quantum_state = StateType(num_qubits)

        def run() -> None:
            quantum_state.set_zero_state()
            circuit.update_quantum_state(quantum_state)

        def final_state() -> np.ndarray:
            return quantum_state.get_vector()

        return run, _no_synchronize, final_state


class AerImplementation:
    def __init__(
        self,
        label: str = "qiskit-aer:cpu:complex128",
        device: str = "CPU",
        precision: str = "double",
    ) -> None:
        self.label = label
        self._device = device
        self._precision = precision

    def build(self, num_qubits: int, instructions: list[tuple], threads: int):
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator

        circuit = QuantumCircuit(num_qubits)
        for name, qubits, parameter in instructions:
            if name == "rx":
                circuit.rx(parameter, qubits[0])
            elif name == "rz":
                circuit.rz(parameter, qubits[0])
            else:
                circuit.cx(qubits[0], qubits[1])
        circuit.save_statevector()

        simulator = AerSimulator(
            method="statevector",
            device=self._device,
            precision=self._precision,
            fusion_enable=False,
            max_parallel_threads=threads,
        )
        compiled = transpile(circuit, simulator, optimization_level=0)

        expected = _instruction_counts(instructions)
        applied = {
            name: compiled.count_ops().get(name, 0) for name in ("rx", "rz", "cx")
        }
        if applied != expected:
            raise RuntimeError(
                f"transpilation changed the gate counts from {expected} "
                f"to {applied}; the measurement would be meaningless"
            )

        result = {}

        def run() -> None:
            result["value"] = simulator.run(compiled, shots=1).result()

        def final_state() -> np.ndarray:
            return np.asarray(result["value"].get_statevector())

        return run, _no_synchronize, final_state


def _instruction_counts(instructions: list[tuple]) -> dict:
    counts = {"rx": 0, "rz": 0, "cx": 0}
    for name, _, _ in instructions:
        counts["cx" if name == "cnot" else name] += 1
    return counts


def _no_synchronize() -> None:
    return None


def _make_synchronize(backend):
    device = getattr(backend, "device", None)
    # backend.device is a torch.device, which never compares equal to a string.
    device_type = getattr(device, "type", None)
    if device_type == "cuda":
        import torch
        return torch.cuda.synchronize

    if device_type == "supa":
        import torch
        import torch_br  # noqa: F401  registers torch.supa
        return torch.supa.synchronize

    return _no_synchronize


def _unavailable_reason(label: str) -> str:
    """Say why a requested implementation could not be built."""
    from importlib.util import find_spec

    if label.startswith("qulacs"):
        if find_spec("qulacs") is None:
            return "qulacs is not installed"
        if label.startswith("qulacs:gpu"):
            return "the installed qulacs has no QuantumStateGpu"
    elif label.startswith("qiskit-aer"):
        if find_spec("qiskit_aer") is None:
            return "qiskit-aer is not installed"
        if label.startswith("qiskit-aer:gpu"):
            return "the installed qiskit-aer reports no GPU device"
    elif label.startswith("ours:torch"):
        if find_spec("torch") is None:
            return "torch is not installed"
        if ":supa:" in label:
            if find_spec("torch_br") is None:
                return "torch_br is not installed"
            return "torch_br is installed but no supa device is available"
        if ":cuda:" in label:
            return "torch is installed but no CUDA device is available"
    return "unknown label"


def available_implementations(selected: list[str] | None) -> list:
    from numpy_backend import NumpyBackend

    implementations = [
        OurImplementation("ours:numpy:complex128", lambda _: NumpyBackend()),
        OurImplementation(
            "ours:numpy:complex128:fusion",
            lambda _: NumpyBackend(),
            fusion=True,
        ),
        OurImplementation(
            "ours:numpy:complex64", lambda _: NumpyBackend(dtype=np.complex64)
        ),
        OurImplementation(
            "ours:numpy:complex64:fusion",
            lambda _: NumpyBackend(dtype=np.complex64),
            fusion=True,
        ),
    ]

    try:
        import torch
    except ImportError:
        torch = None

    try:
        import torch_br
    except ImportError:
        torch_br = None

    if torch is not None:
        from torch_backend import TorchBackend

        implementations.extend(
            [
                OurImplementation(
                    "ours:torch:cpu:complex128",
                    lambda gates: TorchBackend(
                        dtype="complex128", matrix_cache_size=max(gates, 256)
                    ),
                ),
                OurImplementation(
                    "ours:torch:cpu:complex128:fusion",
                    lambda gates: TorchBackend(
                        dtype="complex128", matrix_cache_size=max(gates, 256)
                    ),
                    fusion=True,
                ),
                OurImplementation(
                    "ours:torch:cpu:complex64",
                    lambda gates: TorchBackend(
                        dtype="complex64", matrix_cache_size=max(gates, 256)
                    ),
                ),
                OurImplementation(
                    "ours:torch:cpu:complex64:fusion",
                    lambda gates: TorchBackend(
                        dtype="complex64", matrix_cache_size=max(gates, 256)
                    ),
                    fusion=True,
                ),
            ]
        )
        if torch.cuda.is_available():
            implementations.extend(
                [
                    OurImplementation(
                        "ours:torch:cuda:complex128",
                        lambda gates: TorchBackend(
                            device="cuda",
                            dtype="complex128",
                            matrix_cache_size=max(gates, 256),
                        ),
                    ),
                    OurImplementation(
                        "ours:torch:cuda:complex128:fusion",
                        lambda gates: TorchBackend(
                            device="cuda",
                            dtype="complex128",
                            matrix_cache_size=max(gates, 256),
                        ),
                        fusion=True,
                    ),
                    OurImplementation(
                        "ours:torch:cuda:complex64",
                        lambda gates: TorchBackend(
                            device="cuda",
                            dtype="complex64",
                            matrix_cache_size=max(gates, 256),
                        ),
                    ),
                    OurImplementation(
                        "ours:torch:cuda:complex64:fusion",
                        lambda gates: TorchBackend(
                            device="cuda",
                            dtype="complex64",
                            matrix_cache_size=max(gates, 256),
                        ),
                        fusion=True,
                    ),
                ]
            )
        if torch_br is not None and torch.supa.is_available():
            implementations.extend(
                [
                    OurImplementation(
                        "ours:torch:supa:complex128",
                        lambda gates: TorchBackend(
                            device="supa",
                            dtype="complex128",
                            matrix_cache_size=max(gates, 256),
                        ),
                    ),
                    OurImplementation(
                        "ours:torch:supa:complex128:fusion",
                        lambda gates: TorchBackend(
                            device="supa",
                            dtype="complex128",
                            matrix_cache_size=max(gates, 256),
                        ),
                        fusion=True,
                    ),
                    OurImplementation(
                        "ours:torch:supa:complex64",
                        lambda gates: TorchBackend(
                            device="supa",
                            dtype="complex64",
                            matrix_cache_size=max(gates, 256),
                        ),
                    ),
                    OurImplementation(
                        "ours:torch:supa:complex64:fusion",
                        lambda gates: TorchBackend(
                            device="supa",
                            dtype="complex64",
                            matrix_cache_size=max(gates, 256),
                        ),
                        fusion=True,
                    ),
                ]
            )

    try:
        import qulacs

        implementations.append(QulacsImplementation("qulacs:cpu:complex128"))
        if hasattr(qulacs, "QuantumStateGpu"):
            implementations.append(
                QulacsImplementation("qulacs:gpu:complex128", use_gpu=True)
            )
    except ImportError:
        pass

    try:
        from qiskit_aer import AerSimulator

        implementations.append(
            AerImplementation("qiskit-aer:cpu:complex128", "CPU", "double")
        )
        if "GPU" in AerSimulator().available_devices():
            implementations.append(
                AerImplementation("qiskit-aer:gpu:complex128", "GPU", "double")
            )
    except ImportError:
        pass

    if selected is None:
        return implementations

    known = {implementation.label: implementation for implementation in implementations}
    missing = [label for label in selected if label not in known]
    if missing:
        details = "\n".join(
            f"  {label}: {_unavailable_reason(label)}" for label in missing
        )
        raise SystemExit(
            "unavailable implementation(s):\n"
            f"{details}\n\n"
            "activate the environment that matches the scenario, for example\n"
            "  envs/bench-cpu -> .venv-bench-cpu\n"
            "see envs/README.md for how to create and activate it.\n\n"
            f"available in this interpreter: {', '.join(sorted(known))}"
        )
    return [known[label] for label in selected]


def measure(run, synchronize, repeats: int) -> float:
    run()
    synchronize()

    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        synchronize()
        durations.append(time.perf_counter() - start)

    return min(durations)


def environment_metadata() -> dict:
    import os

    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
        "numpy": np.__version__,
    }

    try:
        import torch

        metadata["torch"] = torch.__version__
        if torch.cuda.is_available():
            device_index = torch.cuda.current_device()
            device_properties = torch.cuda.get_device_properties(device_index)
            metadata["cuda_device"] = device_properties.name
            metadata["cuda_memory_bytes"] = device_properties.total_memory
            metadata["cuda_memory_gib"] = round(
                device_properties.total_memory / (1024**3), 2
            )
    except ImportError:
        pass
    
    try:
        import torch
        import torch_br

        metadata["torch_br"] = torch_br.__version__
        if torch.supa.is_available():
            device_index = torch.supa.current_device()
            device_properties = torch.supa.get_device_properties(device_index)
            metadata["supa_device"] = device_properties.name
            metadata["supa_memory_bytes"] = device_properties.total_memory
            metadata["supa_memory_gib"] = round(
                device_properties.total_memory / (1024**3), 2
            )
    except ImportError:
        pass

    try:
        import qulacs

        metadata["qulacs"] = getattr(qulacs, "__version__", "unknown")
    except ImportError:
        pass

    try:
        import qiskit
        import qiskit_aer

        metadata["qiskit"] = qiskit.__version__
        metadata["qiskit_aer"] = qiskit_aer.__version__
    except ImportError:
        pass

    return metadata


def max_amplitude_deviation(reference: np.ndarray, amplitudes: np.ndarray) -> float:
    """Largest absolute amplitude difference, with the global phase removed.

    Preferred over fidelity because fidelity sits so close to 1 that double
    precision cannot represent the difference: at a 1e-12 amplitude error it
    already returns exactly 1.0, and rounding can even push it above 1. This
    deviation is linear in the error, so it stays readable in scientific
    notation all the way down to 1e-16.

    The phase alignment keeps a framework that differs only by a global phase,
    which is physically the same state, from reporting a large error.
    """
    overlap = complex(np.vdot(reference, amplitudes))
    if overlap != 0:
        amplitudes = amplitudes * (abs(overlap) / overlap)
    return float(np.max(np.abs(reference - amplitudes)))


def run_benchmark(arguments) -> dict:
    implementations = available_implementations(arguments.implementations)
    qubit_counts = [int(value) for value in arguments.qubits.split(",")]
    threads = arguments.threads or 0

    for num_qubits in qubit_counts:
        if num_qubits < MINIMUM_QUBITS:
            raise SystemExit(f"--qubits values must be at least {MINIMUM_QUBITS}")

    reference_label = arguments.reference
    if reference_label is not None:
        known = {implementation.label for implementation in implementations}
        if reference_label not in known:
            raise SystemExit(
                f"--reference {reference_label!r} is not among the selected "
                f"implementations: {', '.join(sorted(known))}"
            )
        # Stable sort keeps the original order while moving the reference first.
        implementations = sorted(
            implementations, key=lambda item: item.label != reference_label
        )

    records = []
    print(
        f"{'implementation':<{IMPLEMENTATION_WIDTH}}{'qubits':>7}{'gates':>8}"
        f"{'circuit ms':>13}{'per gate us':>13}{'max error':>14}",
        flush=True,
    )

    for num_qubits in qubit_counts:
        instructions = build_instructions(num_qubits, arguments.depth, arguments.seed)
        reference_amplitudes = None

        for implementation in implementations:
            try:
                run, synchronize, final_state = implementation.build(
                    num_qubits, instructions, threads
                )
                seconds = measure(run, synchronize, arguments.repeats)
                amplitudes = np.asarray(final_state(), dtype=np.complex128)
            except Exception as error:  # noqa: BLE001
                print(
                    f"skipped {implementation.label} n={num_qubits}: "
                    f"{type(error).__name__}: {error}",
                    file=sys.stderr,
                )
                continue

            if implementation.label == reference_label:
                reference_amplitudes = amplitudes

            deviation = None
            if reference_amplitudes is not None:
                deviation = max_amplitude_deviation(reference_amplitudes, amplitudes)

            records.append(
                {
                    "implementation": implementation.label,
                    "qubits": num_qubits,
                    "gates": len(instructions),
                    "seconds": seconds,
                    "deviation": deviation,
                }
            )
            print(
                f"{implementation.label:<{IMPLEMENTATION_WIDTH}}"
                f"{num_qubits:>7}{len(instructions):>8}"
                f"{seconds * 1e3:>13.3f}"
                f"{seconds / len(instructions) * 1e6:>13.3f}"
                f"{'-' if deviation is None else format(deviation, '.3e'):>14}",
                flush=True,
            )

    return {
        "environment": environment_metadata(),
        "circuit": {
            "depth": arguments.depth,
            "seed": arguments.seed,
            "threads": threads,
            "reference": reference_label,
        },
        "records": records,
    }


def write_plot(records: list[dict], destination: Path, title: str) -> None:
    try:
        import matplotlib
    except ImportError as error:
        raise SystemExit(
            "--plot requires matplotlib: "
            "pip install -r requirements-bench.txt"
        ) from error

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    implementations = sorted({record["implementation"] for record in records})
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    figure, axes = plt.subplots(figsize=(9, 6))
    for index, implementation in enumerate(implementations):
        points = sorted(
            (record["qubits"], record["seconds"] * 1e3)
            for record in records
            if record["implementation"] == implementation
        )
        if not points:
            continue

        axes.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker=markers[index % len(markers)],
            markersize=5,
            linewidth=1.6,
            linestyle="--" if implementation.startswith("ours:") else "-",
            label=implementation,
        )

    qubit_counts = sorted({record["qubits"] for record in records})
    axes.set_xticks(qubit_counts)
    axes.set_yscale("log")
    axes.set_xlabel("qubits")
    axes.set_ylabel("milliseconds per circuit execution")
    axes.set_title(title)
    axes.grid(True, which="both", linewidth=0.3, alpha=0.6)
    axes.legend(fontsize=8, loc="upper left")
    figure.tight_layout()

    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    print(f"wrote {destination}")


def print_report(paths: list[str]) -> list[dict]:
    records = []
    environments = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        records.extend(payload["records"])
        environments.append((path, payload["environment"]))

    for path, environment in environments:
        print(f"# {path}")
        for key, value in environment.items():
            print(f"    {key}: {value}")
    print()

    implementations = sorted({record["implementation"] for record in records})
    qubit_counts = sorted({record["qubits"] for record in records})
    lookup = {
        (record["implementation"], record["qubits"]): record["seconds"]
        for record in records
    }

    print("=== qulacs benchmark circuit, milliseconds per execution ===\n")
    header = "".join(f"{count:>11}" for count in qubit_counts)
    print(f"{'implementation':<{IMPLEMENTATION_WIDTH}}{header}")
    for implementation in implementations:
        cells = []
        for count in qubit_counts:
            seconds = lookup.get((implementation, count))
            cells.append("-" if seconds is None else f"{seconds * 1e3:.3f}")
        print(
            f"{implementation:<{IMPLEMENTATION_WIDTH}}"
            + "".join(f"{cell:>11}" for cell in cells)
        )
    print()

    return records


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-framework statevector benchmark using the qulacs benchmark "
            "circuit: random RX/RZ rotation layers separated by a CNOT ring"
        )
    )
    parser.add_argument("--qubits", default="4,8,12,16,20,22")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--reference",
        default=None,
        help=(
            "implementation label whose final state is the error reference; "
            "it is executed first for every qubit count"
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="thread count passed to frameworks that accept one (0 = automatic)",
    )
    parser.add_argument(
        "--implementations",
        default=None,
        type=lambda value: value.split(","),
        help="comma-separated labels; defaults to everything importable",
    )
    parser.add_argument("--output", default=None, help="write results as JSON")
    parser.add_argument(
        "--plot",
        default=None,
        help="write a comparison line chart to this path",
    )
    parser.add_argument(
        "--title",
        default="qulacs benchmark circuit (depth 9)",
        help="chart title",
    )
    parser.add_argument(
        "--report",
        nargs="+",
        default=None,
        help="merge JSON result files into a comparison table",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if arguments.report:
        records = print_report(arguments.report)
        if arguments.plot:
            write_plot(records, Path(arguments.plot), arguments.title)
        return

    payload = run_benchmark(arguments)

    if arguments.output:
        destination = Path(arguments.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {destination}")

    if arguments.plot:
        write_plot(payload["records"], Path(arguments.plot), arguments.title)


if __name__ == "__main__":
    main()
