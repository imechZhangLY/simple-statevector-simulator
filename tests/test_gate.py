import unittest

import numpy as np

from gate import Gate


class GateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.identity = np.eye(2, dtype=np.complex128)

    def test_constructs_gate_with_normalized_matrices(self) -> None:
        gate = Gate("I", 1, self.identity, self.identity, qasm_name="id")

        self.assertEqual(gate.name, "I")
        self.assertEqual(gate.qasm_name, "id")
        self.assertEqual(gate.num_qubits, 1)
        self.assertEqual(gate.parameters, ())
        self.assertEqual(gate.dagger_qasm_name, "id")
        self.assertEqual(gate.dagger_parameters, ())
        self.assertEqual(gate.matrix.dtype, np.complex128)
        self.assertFalse(gate.matrix.flags.writeable)
        self.assertFalse(gate.dagger_matrix.flags.writeable)

    def test_copies_input_matrices(self) -> None:
        gate = Gate("I", 1, self.identity, self.identity, qasm_name="id")

        self.identity[0, 0] = 0

        self.assertEqual(gate.matrix[0, 0], 1)
        self.assertEqual(gate.dagger_matrix[0, 0], 1)

    def test_rejects_empty_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "name must not be empty"):
            Gate("", 1, self.identity, self.identity, qasm_name="id")

    def test_rejects_empty_qasm_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "qasm_name must not be empty"):
            Gate("I", 1, self.identity, self.identity, qasm_name="")

    def test_rejects_non_positive_num_qubits(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_qubits must be positive"):
            Gate("I", 0, self.identity, self.identity, qasm_name="id")

    def test_rejects_matrix_with_wrong_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, r"matrix must have shape \(4, 4\)"):
            Gate("I", 2, self.identity, np.eye(4), qasm_name="id")

    def test_rejects_non_finite_matrix_values(self) -> None:
        matrix = self.identity.copy()
        matrix[0, 0] = np.nan

        with self.assertRaisesRegex(ValueError, "only finite values"):
            Gate("I", 1, matrix, self.identity, qasm_name="id")

    def test_stores_direction_specific_serialization_metadata(self) -> None:
        gate = Gate(
            "RX",
            1,
            self.identity,
            self.identity,
            qasm_name="rx",
            parameters=(np.pi / 2,),
            dagger_parameters=(-np.pi / 2,),
        )

        self.assertEqual(gate.parameters, (float(np.pi / 2),))
        self.assertEqual(gate.dagger_parameters, (-float(np.pi / 2),))

    def test_rejects_non_finite_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "only finite values"):
            Gate(
                "RX",
                1,
                self.identity,
                self.identity,
                qasm_name="rx",
                parameters=(np.inf,),
            )


if __name__ == "__main__":
    unittest.main()