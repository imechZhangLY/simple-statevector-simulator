import argparse

import numpy as np

from backend_option import add_backend_argument, create_backend

from circuit import Circuit
from observable import Observable
from simulator import StateVectorSimulator
from single_qubit_gates import H, RX, RZ
from two_qubit_gates import CX

PAULI_MATRICES = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def dense_term(factors: dict, num_qubits: int) -> np.ndarray:
    operator = np.array([[1]], dtype=np.complex128)
    for qubit in reversed(range(num_qubits)):
        operator = np.kron(operator, PAULI_MATRICES[factors.get(qubit, "I")])
    return operator


def ising_hamiltonian(num_qubits: int, coupling: float, field: float) -> np.ndarray:
    dimension = 1 << num_qubits
    hamiltonian = np.zeros((dimension, dimension), dtype=np.complex128)

    for qubit in range(num_qubits - 1):
        hamiltonian -= coupling * dense_term({qubit: "Z", qubit + 1: "Z"}, num_qubits)
    for qubit in range(num_qubits):
        hamiltonian -= field * dense_term({qubit: "X"}, num_qubits)

    return hamiltonian


def ising_observable(num_qubits: int, coupling: float, field: float) -> Observable:
    terms = [
        (-coupling, {qubit: "Z", qubit + 1: "Z"})
        for qubit in range(num_qubits - 1)
    ]
    terms.extend((-field, {qubit: "X"}) for qubit in range(num_qubits))
    return Observable(terms)


def dense_energy(amplitudes: np.ndarray, hamiltonian: np.ndarray) -> float:
    return float(np.vdot(amplitudes, hamiltonian @ amplitudes).real)


def preparation_circuit(num_qubits: int) -> Circuit:
    circuit = Circuit(num_qubits)
    circuit.append(H(0))
    return circuit


def trotter_circuit(
    num_qubits: int,
    coupling: float,
    field: float,
    total_time: float,
    steps: int,
) -> Circuit:
    circuit = preparation_circuit(num_qubits)
    interval = total_time / steps

    for _ in range(steps):
        for qubit in range(num_qubits):
            circuit.append(RX(-2 * field * interval, qubit))
        for qubit in range(num_qubits - 1):
            circuit.append(CX(qubit, qubit + 1))
            circuit.append(RZ(-2 * coupling * interval, qubit + 1))
            circuit.append(CX(qubit, qubit + 1))

    return circuit


def exact_evolution(
    hamiltonian: np.ndarray,
    initial: np.ndarray,
    total_time: float,
) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    phases = np.exp(-1j * eigenvalues * total_time)
    return eigenvectors @ (phases * (eigenvectors.conj().T @ initial))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transverse-field Ising Trotter simulation compared with the exact "
            "matrix exponential"
        )
    )
    parser.add_argument("--qubits", type=int, default=4)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--field", type=float, default=0.7)
    parser.add_argument(
        "--steps",
        default="1,2,4,8,16,32,64",
        help="comma-separated Trotter step counts",
    )
    add_backend_argument(parser)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    backend = create_backend(arguments.backend)
    simulator = StateVectorSimulator(backend)

    hamiltonian = ising_hamiltonian(
        arguments.qubits, arguments.coupling, arguments.field
    )
    observable = ising_observable(
        arguments.qubits, arguments.coupling, arguments.field
    )

    initial_state = simulator.run(preparation_circuit(arguments.qubits))
    initial = initial_state.amplitudes
    reference = exact_evolution(hamiltonian, initial, arguments.time)

    observable_energy = initial_state.expectation(observable)
    matrix_energy = dense_energy(initial, hamiltonian)
    evolved_energy = dense_energy(reference, hamiltonian)

    print(f"backend    : {backend.name}")
    print(f"qubits     : {arguments.qubits}")
    print(f"H          : -{arguments.coupling} * sum ZZ - {arguments.field} * sum X")
    print(f"time       : {arguments.time}")

    print("\nenergy reference")
    print(f"  <H> from Observable       : {observable_energy:+.12f}")
    print(f"  <H> from dense matrix     : {matrix_energy:+.12f}")
    print(f"  <H> after exact evolution : {evolved_energy:+.12f}  (conserved)")

    print(
        f"\n{'steps':>7}{'gates':>9}{'infidelity':>16}"
        f"{'2-norm error':>16}{'error ratio':>13}{'step ratio':>12}"
        f"{'<H> error':>14}"
    )

    previous_error = None
    previous_steps = None
    for steps in [int(value) for value in arguments.steps.split(",")]:
        circuit = trotter_circuit(
            arguments.qubits,
            arguments.coupling,
            arguments.field,
            arguments.time,
            steps,
        )
        state = simulator.run(circuit)
        evolved = state.amplitudes

        infidelity = 1.0 - float(abs(np.vdot(reference, evolved)) ** 2)
        error = float(np.linalg.norm(evolved - reference))
        energy_error = abs(state.expectation(observable) - evolved_energy)
        if previous_error is None:
            error_ratio = ""
            step_ratio = ""
        else:
            error_ratio = f"{previous_error / error:>13.2f}"
            step_ratio = f"{steps / previous_steps:>12.2f}"
        previous_error = error
        previous_steps = steps

        print(
            f"{steps:>7}{len(circuit):>9}"
            f"{infidelity:>16.3e}{error:>16.3e}{error_ratio}{step_ratio}"
            f"{energy_error:>14.3e}"
        )

    print(
        "\nfirst-order Trotter error scales as O(1/steps), "
        "so the error ratio should approach the step ratio"
    )
    print(
        "<H> is conserved by the exact evolution, so the energy error "
        "measures the Trotter deviation alone"
    )


if __name__ == "__main__":
    main()
