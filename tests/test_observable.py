import unittest

import numpy as np

from circuit import Circuit
from observable import Observable, PauliTerm
from simulator import StateVectorSimulator
from single_qubit_gates import H, RX, RY
from statevector import StateVector
from two_qubit_gates import CX

PAULI_MATRICES = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def dense_operator(paulis: dict, num_qubits: int) -> np.ndarray:
    operator = np.array([[1]], dtype=np.complex128)
    for qubit in reversed(range(num_qubits)):
        operator = np.kron(operator, PAULI_MATRICES[paulis.get(qubit, "I")])
    return operator


def dense_expectation(state: StateVector, terms: list, num_qubits: int) -> float:
    amplitudes = state.amplitudes
    total = 0.0
    for coefficient, paulis in terms:
        operator = dense_operator(paulis, num_qubits)
        total += coefficient * np.vdot(amplitudes, operator @ amplitudes).real
    return total


def entangled_state() -> StateVector:
    circuit = (
        Circuit(3)
        .append(H(0))
        .append(RY(0.9, 1))
        .append(CX(0, 2))
        .append(RX(0.4, 2))
        .append(CX(1, 0))
    )
    return StateVectorSimulator().run(circuit)


class PauliTermTests(unittest.TestCase):
    def test_normalizes_and_sorts_paulis(self) -> None:
        term = PauliTerm(1.0, {1: "z", 0: "X"})

        self.assertEqual(term.paulis, ((0, "X"), (1, "Z")))
        self.assertEqual(term.coefficient, 1.0)

    def test_drops_identity_entries(self) -> None:
        term = PauliTerm(1.0, {0: "I", 1: "Z"})

        self.assertEqual(term.paulis, ((1, "Z"),))

    def test_builds_operations_for_each_pauli(self) -> None:
        term = PauliTerm(1.0, {0: "X", 2: "Y"})

        operations = term.operations

        self.assertEqual([operation.name for operation in operations], ["X", "Y"])
        self.assertEqual([operation.qubits for operation in operations], [(0,), (2,)])

    def test_rejects_invalid_pauli_letter(self) -> None:
        with self.assertRaisesRegex(ValueError, "one of I, X, Y, Z"):
            PauliTerm(1.0, {0: "H"})

    def test_rejects_duplicate_qubits(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            PauliTerm(1.0, [(0, "X"), (0, "Z")])

    def test_rejects_negative_qubit(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            PauliTerm(1.0, {-1: "Z"})

    def test_rejects_invalid_coefficient(self) -> None:
        with self.assertRaisesRegex(TypeError, "real number"):
            PauliTerm("1.0", {0: "Z"})
        with self.assertRaisesRegex(ValueError, "finite"):
            PauliTerm(np.inf, {0: "Z"})


class ObservableTests(unittest.TestCase):
    def test_accepts_pairs_and_pauli_terms(self) -> None:
        observable = Observable(
            [(1.0, {0: "Z"}), PauliTerm(0.5, {1: "X"})]
        )

        self.assertEqual(len(observable), 2)
        self.assertEqual(observable.terms[0].paulis, ((0, "Z"),))
        self.assertEqual(observable.terms[1].coefficient, 0.5)

    def test_identity_term_contributes_its_coefficient(self) -> None:
        observable = Observable([(2.5, {})])

        self.assertAlmostEqual(StateVector(2).expectation(observable), 2.5)

    def test_single_pauli_on_basis_state(self) -> None:
        observable = Observable([(1.0, {0: "Z"})])

        self.assertAlmostEqual(StateVector(1).expectation(observable), 1.0)

    def test_weighted_sum_on_bell_state(self) -> None:
        state = StateVectorSimulator().run(Circuit(2).append(H(0)).append(CX(0, 1)))
        observable = Observable([(0.5, {0: "Z", 1: "Z"}), (0.5, {0: "X", 1: "X"})])

        self.assertAlmostEqual(state.expectation(observable), 1.0)

    def test_weights_scale_the_result(self) -> None:
        state = StateVector(1)

        self.assertAlmostEqual(
            state.expectation(Observable([(3.0, {0: "Z"})])), 3.0
        )
        self.assertAlmostEqual(
            state.expectation(Observable([(-2.0, {0: "Z"})])), -2.0
        )

    def test_matches_dense_matrix_expectation(self) -> None:
        state = entangled_state()
        terms = [
            (0.7, {0: "Z", 2: "Z"}),
            (-1.3, {1: "X"}),
            (0.25, {0: "Y", 1: "Y", 2: "Y"}),
            (2.0, {}),
        ]

        value = state.expectation(Observable(terms))

        self.assertAlmostEqual(value, dense_expectation(state, terms, 3), places=10)

    def test_empty_observable_is_zero(self) -> None:
        self.assertAlmostEqual(StateVector(1).expectation(Observable([])), 0.0)

    def test_expectation_does_not_mutate_the_state(self) -> None:
        state = entangled_state()
        before = state.amplitudes.copy()

        state.expectation(Observable([(1.0, {0: "X", 1: "Z"})]))

        np.testing.assert_allclose(state.amplitudes, before, atol=1e-12)

    def test_rejects_qubit_outside_statevector(self) -> None:
        with self.assertRaisesRegex(IndexError, "outside the statevector"):
            StateVector(2).expectation(Observable([(1.0, {5: "Z"})]))


if __name__ == "__main__":
    unittest.main()
