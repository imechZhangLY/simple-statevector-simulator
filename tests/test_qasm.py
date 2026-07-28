import unittest

import numpy as np

from circuit import Circuit
from qasm_exporter import export_qasm
from qasm_parser import QasmError, evaluate_expression, parse_qasm
from simulator import StateVectorSimulator
from single_qubit_gates import H, RX, S, T, U2, U3, X
from three_qubit_gates import CCX
from two_qubit_gates import CNOT, CP, CX, SWAP

BELL_PROGRAM = """
OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg c[2];

h q[0];
cx q[0], q[1];

measure q -> c;
"""


def final_state(circuit: Circuit) -> np.ndarray:
    return StateVectorSimulator().run(circuit).amplitudes


class ExporterTests(unittest.TestCase):
    def test_exports_header_register_and_gates(self) -> None:
        circuit = Circuit(2).append(H(0)).append(CX(0, 1))

        qasm = export_qasm(circuit)

        self.assertEqual(
            qasm,
            'OPENQASM 2.0;\ninclude "qelib1.inc";\n\nqreg q[2];\n\n'
            "h q[0];\ncx q[0], q[1];\n",
        )

    def test_exports_parameters(self) -> None:
        circuit = Circuit(2).append(RX(0.3, 0)).append(CP(np.pi, 0, 1))

        qasm = export_qasm(circuit)

        self.assertIn("rx(0.3) q[0];", qasm)
        self.assertIn(f"cp({np.pi!r}) q[0], q[1];", qasm)

    def test_exports_dagger_metadata(self) -> None:
        circuit = Circuit(1).append(S(0).dagger()).append(T(0).dagger())
        circuit.append(RX(0.3, 0).dagger())

        qasm = export_qasm(circuit)

        self.assertIn("sdg q[0];", qasm)
        self.assertIn("tdg q[0];", qasm)
        self.assertIn("rx(-0.3) q[0];", qasm)

    def test_aliases_use_the_canonical_name(self) -> None:
        qasm = export_qasm(Circuit(2).append(CNOT(0, 1)))

        self.assertIn("cx q[0], q[1];", qasm)

    def test_optional_measurement_block(self) -> None:
        qasm = export_qasm(Circuit(2).append(H(0)), measure_all=True)

        self.assertIn("creg c[2];", qasm)
        self.assertIn("measure q -> c;", qasm)

    def test_rejects_non_circuit(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be a Circuit"):
            export_qasm([H(0)])


class ExpressionTests(unittest.TestCase):
    def test_evaluates_arithmetic_and_pi(self) -> None:
        self.assertAlmostEqual(evaluate_expression("pi/2"), np.pi / 2)
        self.assertAlmostEqual(evaluate_expression("-2*pi"), -2 * np.pi)
        self.assertAlmostEqual(evaluate_expression("1 + 2*3"), 7.0)
        self.assertAlmostEqual(evaluate_expression("sqrt(4)"), 2.0)

    def test_rejects_unknown_identifier(self) -> None:
        with self.assertRaisesRegex(QasmError, "unknown identifier"):
            evaluate_expression("theta")

    def test_does_not_execute_arbitrary_code(self) -> None:
        with self.assertRaises(QasmError):
            evaluate_expression("__import__('os').getcwd()")
        with self.assertRaises(QasmError):
            evaluate_expression("open('secret.txt').read()")


class ParserTests(unittest.TestCase):
    def test_parses_registers_gates_and_measurements(self) -> None:
        program = parse_qasm(BELL_PROGRAM)

        self.assertEqual(program.circuit.num_qubits, 2)
        self.assertEqual(
            [operation.name for operation in program.circuit], ["H", "CX"]
        )
        self.assertEqual(program.measurements, ((0, 0), (1, 1)))
        self.assertEqual(program.num_clbits, 2)

    def test_measurements_stay_outside_the_circuit(self) -> None:
        program = parse_qasm(BELL_PROGRAM)

        self.assertTrue(
            all(not operation.is_dagger for operation in program.circuit)
        )
        self.assertEqual(len(program.circuit), 2)

    def test_parses_parameter_expressions(self) -> None:
        program = parse_qasm(
            "OPENQASM 2.0; qreg q[1]; rx(pi/2) q[0]; u2(0, pi) q[0];"
        )

        operations = program.circuit.operations
        self.assertAlmostEqual(operations[0].parameters[0], np.pi / 2)
        self.assertAlmostEqual(operations[1].parameters[1], np.pi)

    def test_parses_dagger_gates(self) -> None:
        program = parse_qasm("OPENQASM 2.0; qreg q[1]; sdg q[0]; tdg q[0];")

        self.assertEqual(
            [operation.name for operation in program.circuit], ["S†", "T†"]
        )

    def test_broadcasts_single_qubit_gates_over_a_register(self) -> None:
        program = parse_qasm("OPENQASM 2.0; qreg q[3]; h q;")

        self.assertEqual(
            [operation.qubits for operation in program.circuit],
            [(0,), (1,), (2,)],
        )

    def test_ignores_comments_and_barriers(self) -> None:
        program = parse_qasm(
            "OPENQASM 2.0;\nqreg q[1];\n// a comment\nbarrier q;\nx q[0];"
        )

        self.assertEqual(len(program.circuit), 1)

    def test_rejects_mid_circuit_measurement(self) -> None:
        program = (
            "OPENQASM 2.0; qreg q[2]; creg c[2];"
            " h q[0]; measure q[0] -> c[0]; x q[1];"
        )

        with self.assertRaisesRegex(QasmError, "mid-circuit measurement"):
            parse_qasm(program)

    def test_rejects_unsupported_gate_and_names_it(self) -> None:
        with self.assertRaisesRegex(QasmError, "unsupported gate 'rzz'"):
            parse_qasm("OPENQASM 2.0; qreg q[2]; rzz(0.3) q[0], q[1];")

    def test_rejects_unsupported_statements(self) -> None:
        for program, message in [
            ("OPENQASM 2.0; qreg q[1]; reset q[0];", "reset"),
            ("OPENQASM 2.0; qreg q[1]; opaque foo a;", "opaque"),
            ("OPENQASM 2.0; qreg q[1]; gate my a { x a; }", "user defined"),
        ]:
            with self.subTest(program=program):
                with self.assertRaisesRegex(QasmError, message):
                    parse_qasm(program)

    def test_rejects_wrong_operand_and_parameter_counts(self) -> None:
        with self.assertRaisesRegex(QasmError, "expects 2 qubits"):
            parse_qasm("OPENQASM 2.0; qreg q[2]; cx q[0];")
        with self.assertRaisesRegex(QasmError, "expects 1 parameters"):
            parse_qasm("OPENQASM 2.0; qreg q[1]; rx q[0];")

    def test_rejects_out_of_range_index(self) -> None:
        with self.assertRaisesRegex(QasmError, "outside register"):
            parse_qasm("OPENQASM 2.0; qreg q[2]; x q[5];")

    def test_rejects_unknown_register(self) -> None:
        with self.assertRaisesRegex(QasmError, "unknown quantum register"):
            parse_qasm("OPENQASM 2.0; qreg q[2]; x r[0];")

    def test_rejects_multiple_registers(self) -> None:
        with self.assertRaisesRegex(QasmError, "multiple quantum registers"):
            parse_qasm("OPENQASM 2.0; qreg q[1]; qreg r[1];")

    def test_rejects_other_versions(self) -> None:
        with self.assertRaisesRegex(QasmError, "only OpenQASM 2.x"):
            parse_qasm("OPENQASM 3.0; qreg q[1];")

    def test_requires_a_quantum_register(self) -> None:
        with self.assertRaisesRegex(QasmError, "does not declare a quantum register"):
            parse_qasm("OPENQASM 2.0;")


class RoundTripTests(unittest.TestCase):
    def test_round_trip_preserves_the_final_state(self) -> None:
        circuit = (
            Circuit(3)
            .append(H(0))
            .append(X(1))
            .append(S(2).dagger())
            .append(T(0).dagger())
            .append(RX(0.37, 1))
            .append(U3(0.2, 0.4, 0.6, 2))
            .append(U2(0.15, 0.25, 0).dagger())
            .append(CX(0, 1))
            .append(CP(0.55, 2, 0))
            .append(SWAP(1, 2))
            .append(CCX(0, 1, 2))
        )

        restored = parse_qasm(export_qasm(circuit)).circuit

        self.assertEqual(restored.num_qubits, circuit.num_qubits)
        self.assertEqual(len(restored), len(circuit))
        np.testing.assert_allclose(
            final_state(restored), final_state(circuit), atol=1e-12
        )

    def test_round_trip_preserves_qubit_order(self) -> None:
        circuit = Circuit(3).append(CX(2, 0)).append(CCX(2, 0, 1))

        restored = parse_qasm(export_qasm(circuit)).circuit

        self.assertEqual(
            [operation.qubits for operation in restored], [(2, 0), (2, 0, 1)]
        )

    def test_round_trip_preserves_parameters_exactly(self) -> None:
        circuit = Circuit(1).append(RX(0.1 + 0.2, 0))

        restored = parse_qasm(export_qasm(circuit)).circuit

        self.assertEqual(
            restored.operations[0].parameters, circuit.operations[0].parameters
        )

    def test_exported_measurements_are_parsed_back(self) -> None:
        circuit = Circuit(2).append(H(0)).append(CX(0, 1))

        program = parse_qasm(export_qasm(circuit, measure_all=True))

        self.assertEqual(program.measurements, ((0, 0), (1, 1)))


if __name__ == "__main__":
    unittest.main()
