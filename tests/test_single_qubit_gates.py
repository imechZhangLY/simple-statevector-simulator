import unittest

import numpy as np

from single_qubit_gates import H, I, P, RX, RY, RZ, S, T, U1, U2, U3, X, Y, Z


class SingleQubitGateTests(unittest.TestCase):
    def test_constant_gate_matrices(self) -> None:
        expected_matrices = {
            "I": np.eye(2),
            "X": np.array([[0, 1], [1, 0]]),
            "Y": np.array([[0, -1j], [1j, 0]]),
            "Z": np.array([[1, 0], [0, -1]]),
            "H": np.array([[1, 1], [1, -1]]) / np.sqrt(2),
            "S": np.diag([1, 1j]),
            "T": np.diag([1, np.exp(1j * np.pi / 4)]),
        }

        for operation in (I(0), X(0), Y(0), Z(0), H(0), S(0), T(0)):
            with self.subTest(gate=operation.name):
                np.testing.assert_allclose(
                    operation.matrix, expected_matrices[operation.name]
                )

    def test_constant_gate_instances_are_shared(self) -> None:
        self.assertIs(H(0).gate, H(1).gate)

    def test_all_gate_matrices_are_unitary(self) -> None:
        operations = [
            I(0),
            X(0),
            Y(0),
            Z(0),
            H(0),
            S(0),
            T(0),
            RX(0.3, 0),
            RY(0.3, 0),
            RZ(0.3, 0),
            P(0.3, 0),
            U1(0.3, 0),
            U2(0.2, 0.3, 0),
            U3(0.1, 0.2, 0.3, 0),
        ]

        for operation in operations:
            with self.subTest(gate=operation.name):
                np.testing.assert_allclose(
                    operation.gate.dagger_matrix @ operation.gate.matrix,
                    np.eye(2),
                    atol=1e-15,
                )

    def test_rotation_gate_matrices(self) -> None:
        np.testing.assert_allclose(
            RX(np.pi, 0).matrix, [[0, -1j], [-1j, 0]], atol=1e-15
        )
        np.testing.assert_allclose(
            RY(np.pi, 0).matrix, [[0, -1], [1, 0]], atol=1e-15
        )
        np.testing.assert_allclose(
            RZ(np.pi, 0).matrix, [[-1j, 0], [0, 1j]], atol=1e-15
        )
        np.testing.assert_allclose(
            P(np.pi, 0).matrix, [[1, 0], [0, -1]], atol=1e-15
        )

    def test_u1_is_equivalent_to_phase_gate(self) -> None:
        operation = U1(0.3, 0)

        np.testing.assert_allclose(operation.matrix, P(0.3, 0).matrix)
        self.assertEqual(operation.qasm_name, "u1")
        self.assertEqual(operation.parameters, (0.3,))

    def test_u2_is_u3_with_fixed_theta(self) -> None:
        np.testing.assert_allclose(
            U2(0.2, 0.3, 0).matrix,
            U3(np.pi / 2, 0.2, 0.3, 0).matrix,
        )

    def test_u3_matrix(self) -> None:
        operation = U3(np.pi, 0, np.pi, 0)

        np.testing.assert_allclose(operation.matrix, [[0, 1], [1, 0]], atol=1e-15)

    def test_dagger_serialization_metadata(self) -> None:
        self.assertEqual(S(0).dagger().qasm_name, "sdg")
        self.assertEqual(T(0).dagger().qasm_name, "tdg")

        operation = RX(0.3, 0).dagger()
        self.assertEqual(operation.qasm_name, "rx")
        self.assertEqual(operation.parameters, (-0.3,))

        u2_dagger = U2(0.2, 0.3, 0).dagger()
        self.assertEqual(u2_dagger.qasm_name, "u2")
        self.assertEqual(u2_dagger.parameters, (np.pi - 0.3, -np.pi - 0.2))
        np.testing.assert_allclose(
            U2(*u2_dagger.parameters, 0).matrix,
            U2(0.2, 0.3, 0).dagger().matrix,
        )

        u3_dagger = U3(0.1, 0.2, 0.3, 0).dagger()
        self.assertEqual(u3_dagger.qasm_name, "u3")
        self.assertEqual(u3_dagger.parameters, (-0.1, -0.3, -0.2))
        np.testing.assert_allclose(
            U3(*u3_dagger.parameters, 0).matrix,
            U3(0.1, 0.2, 0.3, 0).dagger().matrix,
        )

    def test_rejects_invalid_angles(self) -> None:
        with self.assertRaisesRegex(TypeError, "real number"):
            RX("pi", 0)
        with self.assertRaisesRegex(ValueError, "finite"):
            RY(np.inf, 0)

    def test_rejects_invalid_qubits(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            H(-1)


if __name__ == "__main__":
    unittest.main()