import unittest

import numpy as np

from single_qubit_gates import H, RX, X
from statevector import StateVector
from three_qubit_gates import CCX
from two_qubit_gates import CX


class StateVectorTests(unittest.TestCase):
    def test_initializes_to_zero_state(self) -> None:
        state = StateVector(3)

        np.testing.assert_array_equal(state.amplitudes, [1, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(state.probabilities, [1, 0, 0, 0, 0, 0, 0, 0])

    def test_copies_and_validates_initial_amplitudes(self) -> None:
        amplitudes = np.array([1 / np.sqrt(2), 1j / np.sqrt(2)])
        state = StateVector(1, amplitudes)

        amplitudes[0] = 0

        np.testing.assert_allclose(
            state.amplitudes,
            [1 / np.sqrt(2), 1j / np.sqrt(2)],
        )
        self.assertFalse(state.amplitudes.flags.writeable)

    def test_qubit_zero_is_least_significant_bit(self) -> None:
        state = StateVector(2).apply(X(0))

        np.testing.assert_array_equal(state.amplitudes, [0, 1, 0, 0])

    def test_creates_bell_state(self) -> None:
        state = StateVector(2).apply(H(0)).apply(CX(0, 1))

        np.testing.assert_allclose(
            state.amplitudes,
            [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)],
            atol=1e-15,
        )

    def test_applies_local_gate_to_entangled_state(self) -> None:
        state = StateVector(2).apply(H(0)).apply(CX(0, 1)).apply(X(0))

        np.testing.assert_allclose(
            state.amplitudes,
            [0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0],
            atol=1e-15,
        )

    def test_applies_gate_to_non_adjacent_qubits(self) -> None:
        state = StateVector(3).apply(X(2)).apply(CX(2, 0))

        expected = np.zeros(8)
        expected[5] = 1
        np.testing.assert_array_equal(state.amplitudes, expected)

    def test_applies_three_qubit_gate(self) -> None:
        state = StateVector(3).apply(X(0)).apply(X(1)).apply(CCX(0, 1, 2))

        expected = np.zeros(8)
        expected[7] = 1
        np.testing.assert_array_equal(state.amplitudes, expected)

    def test_dagger_restores_state(self) -> None:
        operation = RX(0.3, 1)
        state = StateVector(2).apply(operation).apply(operation.dagger())

        np.testing.assert_allclose(state.amplitudes, [1, 0, 0, 0], atol=1e-15)

    def test_copy_is_independent(self) -> None:
        original = StateVector(1)
        copied = original.copy().apply(X(0))

        np.testing.assert_array_equal(original.amplitudes, [1, 0])
        np.testing.assert_array_equal(copied.amplitudes, [0, 1])

    def test_rejects_invalid_amplitudes(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            StateVector(2, [1, 0])
        with self.assertRaisesRegex(ValueError, "normalized"):
            StateVector(1, [1, 1])
        with self.assertRaisesRegex(ValueError, "finite"):
            StateVector(1, [np.nan, 0])

    def test_rejects_operation_outside_statevector(self) -> None:
        with self.assertRaisesRegex(IndexError, "outside"):
            StateVector(2).apply(X(2))


if __name__ == "__main__":
    unittest.main()