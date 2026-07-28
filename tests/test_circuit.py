import unittest

import numpy as np

from circuit import Circuit
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


def run(circuit: Circuit) -> StateVector:
    state = StateVector(circuit.num_qubits)
    for operation in circuit:
        state.apply(operation)
    return state


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


if __name__ == "__main__":
    unittest.main()
