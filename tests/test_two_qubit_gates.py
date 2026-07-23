import unittest

import numpy as np

from two_qubit_gates import CH, CNOT, CP, CRX, CRY, CRZ, CX, CY, CZ, SWAP


class TwoQubitGateTests(unittest.TestCase):
    def test_cx_matrix_and_alias(self) -> None:
        expected = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
        )

        np.testing.assert_array_equal(CX(0, 1).matrix, expected)
        self.assertIs(CNOT(0, 1).gate, CX(0, 1).gate)
        self.assertEqual(CNOT(0, 1).qasm_name, "cx")

    def test_swap_matrix(self) -> None:
        expected = np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
        )

        np.testing.assert_array_equal(SWAP(0, 1).matrix, expected)

    def test_controlled_phase_matrix(self) -> None:
        np.testing.assert_allclose(
            CP(np.pi, 0, 1).matrix,
            np.diag([1, 1, 1, -1]),
            atol=1e-15,
        )

    def test_all_matrices_are_unitary(self) -> None:
        operations = [
            CX(0, 1),
            CY(0, 1),
            CZ(0, 1),
            CH(0, 1),
            SWAP(0, 1),
            CP(0.3, 0, 1),
            CRX(0.3, 0, 1),
            CRY(0.3, 0, 1),
            CRZ(0.3, 0, 1),
        ]

        for operation in operations:
            with self.subTest(gate=operation.name):
                np.testing.assert_allclose(
                    operation.gate.dagger_matrix @ operation.gate.matrix,
                    np.eye(4),
                    atol=1e-15,
                )

    def test_dagger_serialization_metadata(self) -> None:
        for operation in (
            CP(0.3, 0, 1),
            CRX(0.3, 0, 1),
            CRY(0.3, 0, 1),
            CRZ(0.3, 0, 1),
        ):
            with self.subTest(gate=operation.name):
                self.assertEqual(operation.dagger().parameters, (-0.3,))
                np.testing.assert_allclose(
                    operation.dagger().matrix,
                    operation.matrix.conj().T,
                )

    def test_qubit_order_is_preserved(self) -> None:
        self.assertEqual(CX(3, 1).qubits, (3, 1))

    def test_rejects_same_control_and_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            CX(1, 1)

    def test_rejects_invalid_angle(self) -> None:
        with self.assertRaisesRegex(TypeError, "real number"):
            CRX("pi", 0, 1)


if __name__ == "__main__":
    unittest.main()