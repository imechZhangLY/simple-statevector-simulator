import unittest

import numpy as np

from gate import Gate
from operation import Operation


class OperationTests(unittest.TestCase):
    def setUp(self) -> None:
        matrix = np.diag([1, 1j])
        dagger_matrix = np.diag([1, -1j])
        self.gate = Gate(
            "S",
            1,
            matrix,
            dagger_matrix,
            qasm_name="s",
            dagger_qasm_name="sdg",
        )

    def test_constructs_operation(self) -> None:
        operation = Operation(self.gate, (2,))

        self.assertIs(operation.gate, self.gate)
        self.assertEqual(operation.qubits, (2,))
        self.assertFalse(operation.is_dagger)
        self.assertEqual(operation.name, "S")
        self.assertEqual(operation.qasm_name, "s")
        self.assertEqual(operation.parameters, ())
        self.assertIs(operation.matrix, self.gate.matrix)

    def test_dagger_selects_precomputed_matrix(self) -> None:
        operation = Operation(self.gate, (2,)).dagger()

        self.assertIs(operation.gate, self.gate)
        self.assertEqual(operation.qubits, (2,))
        self.assertTrue(operation.is_dagger)
        self.assertEqual(operation.name, "S†")
        self.assertEqual(operation.qasm_name, "sdg")
        self.assertEqual(operation.parameters, ())
        self.assertIs(operation.matrix, self.gate.dagger_matrix)

    def test_double_dagger_restores_operation(self) -> None:
        operation = Operation(self.gate, (2,))

        self.assertEqual(operation.dagger().dagger(), operation)

    def test_normalizes_qubits_to_tuple(self) -> None:
        operation = Operation(self.gate, [2])

        self.assertEqual(operation.qubits, (2,))

    def test_rejects_wrong_number_of_qubits(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires 1 qubits"):
            Operation(self.gate, (0, 1))

    def test_rejects_negative_qubits(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            Operation(self.gate, (-1,))

    def test_rejects_duplicate_qubits(self) -> None:
        two_qubit_gate = Gate("CX", 2, np.eye(4), np.eye(4), qasm_name="cx")

        with self.assertRaisesRegex(ValueError, "unique"):
            Operation(two_qubit_gate, (1, 1))

    def test_rejects_non_integer_qubits(self) -> None:
        with self.assertRaisesRegex(TypeError, "only integers"):
            Operation(self.gate, (1.5,))


if __name__ == "__main__":
    unittest.main()