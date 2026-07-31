import unittest

import numpy as np

from circuit import Circuit
from gate_fusion import fuse_circuit, fuse_operations
from single_qubit_gates import H, RX, T, X
from statevector import StateVector
from three_qubit_gates import CCX
from two_qubit_gates import CX


def entangling_circuit() -> Circuit:
    return (
        Circuit(3)
        .append(H(0))
        .append(CX(0, 1))
        .append(T(2))
        .append(RX(0.4, 1))
        .append(CCX(0, 1, 2))
    )


def run(circuit: Circuit, amplitudes: np.ndarray | None = None) -> StateVector:
    state = StateVector(circuit.num_qubits, amplitudes)
    for operation in circuit:
        state.apply(operation)
    return state


def nontrivial_state(num_qubits: int) -> np.ndarray:
    dimension = 1 << num_qubits
    amplitudes = np.arange(1, dimension + 1) + 1j * np.arange(dimension, 0, -1)
    return amplitudes / np.linalg.norm(amplitudes)


class CircuitTests(unittest.TestCase):
    def test_appends_operations_fluently(self) -> None:
        circuit = Circuit(2)

        returned = circuit.append(H(0)).append(CX(0, 1))

        self.assertIs(returned, circuit)
        self.assertEqual(len(circuit), 2)
        self.assertEqual(circuit.operations[0].name, "H")
        self.assertEqual(circuit.operations[1].name, "CX")

    def test_operations_are_exposed_as_an_immutable_snapshot(self) -> None:
        circuit = Circuit(2).append(H(0))

        snapshot = circuit.operations
        circuit.append(X(1))

        self.assertIsInstance(snapshot, tuple)
        self.assertEqual(len(snapshot), 1)
        self.assertEqual(len(circuit.operations), 2)

    def test_rejects_qubit_outside_circuit(self) -> None:
        with self.assertRaisesRegex(IndexError, "outside the circuit"):
            Circuit(2).append(X(2))
        with self.assertRaisesRegex(IndexError, "outside the circuit"):
            Circuit(2).append(CX(0, 5))

    def test_rejects_non_operation(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be an Operation"):
            Circuit(2).append(H)

    def test_rejects_invalid_num_qubits(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be an integer"):
            Circuit(1.5)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            Circuit(0)

    def test_dagger_reverses_order_and_directions(self) -> None:
        circuit = Circuit(2).append(H(0)).append(T(1)).append(CX(0, 1))

        inverted = circuit.dagger()

        self.assertEqual(inverted.num_qubits, 2)
        self.assertEqual(
            [operation.name for operation in inverted],
            ["CX†", "T†", "H†"],
        )
        self.assertTrue(all(operation.is_dagger for operation in inverted))
        self.assertEqual([operation.name for operation in circuit], ["H", "T", "CX"])

    def test_dagger_preserves_qubits(self) -> None:
        circuit = Circuit(3).append(CX(2, 0))

        self.assertEqual(circuit.dagger().operations[0].qubits, (2, 0))

    def test_double_dagger_restores_circuit(self) -> None:
        circuit = entangling_circuit()

        restored = circuit.dagger().dagger()

        self.assertEqual(restored.operations, circuit.operations)

    def test_dagger_restores_initial_state(self) -> None:
        circuit = entangling_circuit()

        state = run(circuit)
        for operation in circuit.dagger():
            state.apply(operation)

        expected = np.zeros(8)
        expected[0] = 1
        np.testing.assert_allclose(state.amplitudes, expected, atol=1e-12)

    def test_dagger_without_reversal_would_not_restore_state(self) -> None:
        circuit = entangling_circuit()

        state = run(circuit)
        for operation in circuit:
            state.apply(operation.dagger())

        self.assertLess(abs(state.amplitudes[0]), 0.99)

    def test_copy_is_independent(self) -> None:
        circuit = Circuit(2).append(H(0))

        copied = circuit.copy().append(X(1))

        self.assertEqual(len(circuit), 1)
        self.assertEqual(len(copied), 2)
        self.assertEqual(copied.num_qubits, circuit.num_qubits)

    def test_fuses_two_qubit_gate_followed_by_gate_on_its_first_qubit(self) -> None:
        circuit = Circuit(4).append(CX(2, 3)).append(X(2))

        fused = fuse_circuit(circuit)

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused.operations[0].qubits, (2, 3))
        amplitudes = nontrivial_state(4)
        np.testing.assert_allclose(
            run(fused, amplitudes).amplitudes,
            run(circuit, amplitudes).amplitudes,
        )

    def test_fuses_non_adjacent_pair_into_three_qubit_gate(self) -> None:
        circuit = Circuit(5).append(CX(2, 4)).append(CCX(2, 3, 4))

        fused = fuse_circuit(circuit)

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused.operations[0].qubits, (2, 3, 4))
        amplitudes = nontrivial_state(5)
        np.testing.assert_allclose(
            run(fused, amplitudes).amplitudes,
            run(circuit, amplitudes).amplitudes,
        )

    def test_does_not_fuse_partially_overlapping_operations(self) -> None:
        circuit = Circuit(3).append(CX(0, 1)).append(CX(1, 2))

        fused = fuse_circuit(circuit)

        self.assertEqual(fused.operations, circuit.operations)

    def test_fused_gate_names_use_unique_incrementing_ids(self) -> None:
        first = fuse_operations(H(0), X(0))
        second = fuse_operations(H(0), X(0))

        first_id = int(first.name.removeprefix("FUSED_"))
        second_id = int(second.name.removeprefix("FUSED_"))
        self.assertEqual(second_id, first_id + 1)

    def test_fused_circuit_names_are_stable_across_repeated_fusion(self) -> None:
        circuit = Circuit(1).append(H(0)).append(X(0))

        first = fuse_circuit(circuit)
        second = fuse_circuit(circuit)

        self.assertEqual(first.operations[0].name, second.operations[0].name)

    def test_different_circuits_use_different_fusion_namespaces(self) -> None:
        first = fuse_circuit(Circuit(1).append(H(0)).append(X(0)))
        second = fuse_circuit(Circuit(1).append(H(0)).append(X(0)))

        self.assertNotEqual(first.operations[0].name, second.operations[0].name)


if __name__ == "__main__":
    unittest.main()
