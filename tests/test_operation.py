import unittest

import numpy as np

from gate import Gate
from operation import Operation
from single_qubit_gates import H, RX, RY, S
from two_qubit_gates import CX


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


class OperationMatrixKeyTests(unittest.TestCase):
    def test_key_is_hashable(self) -> None:
        cache = {H(0).matrix_key: "h"}

        self.assertEqual(cache[H(1).matrix_key], "h")

    def test_key_ignores_qubit_placement(self) -> None:
        self.assertEqual(RX(0.3, 0).matrix_key, RX(0.3, 5).matrix_key)

    def test_key_distinguishes_parameters(self) -> None:
        self.assertNotEqual(RX(0.3, 0).matrix_key, RX(0.7, 0).matrix_key)

    def test_key_distinguishes_dagger_of_constant_gate(self) -> None:
        operation = S(0)

        self.assertNotEqual(operation.matrix_key, operation.dagger().matrix_key)

    def test_key_distinguishes_gates_sharing_parameters(self) -> None:
        self.assertNotEqual(RX(0.3, 0).matrix_key, RY(0.3, 0).matrix_key)

    def test_equal_keys_imply_equal_matrices(self) -> None:
        operations = [
            H(0),
            H(1),
            S(0),
            S(0).dagger(),
            RX(0.3, 0),
            RX(0.3, 2),
            RX(0.7, 0),
            RY(0.3, 0),
            CX(0, 1),
            CX(2, 0),
        ]

        matrices: dict[tuple, np.ndarray] = {}
        for operation in operations:
            with self.subTest(operation=operation.name):
                key = operation.matrix_key
                if key in matrices:
                    np.testing.assert_array_equal(matrices[key], operation.matrix)
                else:
                    matrices[key] = operation.matrix

        self.assertEqual(len(matrices), 7)


if __name__ == "__main__":
    unittest.main()