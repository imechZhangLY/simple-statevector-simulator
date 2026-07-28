import argparse

import numpy as np

from backend_option import add_backend_argument, create_backend

from circuit import Circuit
from simulator import StateVectorSimulator
from single_qubit_gates import H, X
from two_qubit_gates import CP, SWAP


def qft_circuit(num_qubits: int, prepared_value: int) -> Circuit:
    circuit = Circuit(num_qubits)

    for qubit in range(num_qubits):
        if (prepared_value >> qubit) & 1:
            circuit.append(X(qubit))

    for target in reversed(range(num_qubits)):
        circuit.append(H(target))
        for control in reversed(range(target)):
            circuit.append(
                CP(np.pi / (2 ** (target - control)), control, target)
            )

    for qubit in range(num_qubits // 2):
        circuit.append(SWAP(qubit, num_qubits - 1 - qubit))

    return circuit


def exact_transform(num_qubits: int, prepared_value: int) -> np.ndarray:
    dimension = 1 << num_qubits
    indices = np.arange(dimension)
    phases = np.exp(2j * np.pi * prepared_value * indices / dimension)
    return phases / np.sqrt(dimension)


def fidelity(simulated: np.ndarray, expected: np.ndarray) -> float:
    return float(abs(np.vdot(expected, simulated)) ** 2)


def print_amplitudes(simulated: np.ndarray, expected: np.ndarray) -> None:
    print(f"\n{'index':>7}{'simulated':>34}{'expected':>34}")
    for index, (value, reference) in enumerate(zip(simulated, expected)):
        print(f"{index:>7}{value:>34.6f}{reference:>34.6f}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantum Fourier transform demo with a fidelity check"
    )
    parser.add_argument("--qubits", type=int, default=4)
    parser.add_argument(
        "--value",
        type=int,
        default=3,
        help="computational basis state |value> used as input",
    )
    parser.add_argument(
        "--show-amplitudes",
        action="store_true",
        help="print every amplitude next to the analytic value",
    )
    add_backend_argument(parser)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    dimension = 1 << arguments.qubits
    if not 0 <= arguments.value < dimension:
        raise SystemExit(f"--value must be in [0, {dimension})")

    backend = create_backend(arguments.backend)
    circuit = qft_circuit(arguments.qubits, arguments.value)
    state = StateVectorSimulator(backend).run(circuit)

    simulated = state.amplitudes
    expected = exact_transform(arguments.qubits, arguments.value)

    print(f"backend        : {backend.name}")
    print(f"qubits         : {arguments.qubits}")
    print(f"input state    : |{arguments.value}> = |{arguments.value:0{arguments.qubits}b}>")
    print(f"gate count     : {len(circuit)}")
    print(f"norm           : {np.linalg.norm(simulated):.12f}")
    print(f"fidelity       : {fidelity(simulated, expected):.12f}")
    print(f"max abs error  : {np.max(np.abs(simulated - expected)):.3e}")

    if arguments.show_amplitudes:
        print_amplitudes(simulated, expected)


if __name__ == "__main__":
    main()
