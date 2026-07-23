import unittest

import numpy as np

from three_qubit_gates import CCX, CSWAP, FREDKIN, TOFFOLI


class ThreeQubitGateTests(unittest.TestCase):
    def test_ccx_swaps_110_and_111(self) -> None:
        expected = np.eye(8)
        expected[[6, 7]] = expected[[7, 6]]

        np.testing.assert_array_equal(CCX(0, 1, 2).matrix, expected)

    def test_cswap_swaps_101_and_110(self) -> None:
        expected = np.eye(8)
        expected[[5, 6]] = expected[[6, 5]]

        np.testing.assert_array_equal(CSWAP(0, 1, 2).matrix, expected)

    def test_aliases_share_gate_instances(self) -> None:
        self.assertIs(TOFFOLI(0, 1, 2).gate, CCX(0, 1, 2).gate)
        self.assertIs(FREDKIN(0, 1, 2).gate, CSWAP(0, 1, 2).gate)
        self.assertEqual(TOFFOLI(0, 1, 2).qasm_name, "ccx")
        self.assertEqual(FREDKIN(0, 1, 2).qasm_name, "cswap")

    def test_gates_are_unitary_and_self_adjoint(self) -> None:
        for operation in (CCX(0, 1, 2), CSWAP(0, 1, 2)):
            with self.subTest(gate=operation.name):
                np.testing.assert_array_equal(operation.dagger().matrix, operation.matrix)
                np.testing.assert_array_equal(
                    operation.matrix @ operation.matrix,
                    np.eye(8),
                )

    def test_qubit_order_is_preserved(self) -> None:
        self.assertEqual(CCX(4, 2, 0).qubits, (4, 2, 0))
        self.assertEqual(CSWAP(3, 1, 2).qubits, (3, 1, 2))

    def test_rejects_duplicate_qubits(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            CCX(0, 0, 1)
        with self.assertRaisesRegex(ValueError, "unique"):
            CSWAP(0, 1, 1)


if __name__ == "__main__":
    unittest.main()