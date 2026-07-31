import unittest
from unittest.mock import patch

import numpy as np

from circuit import Circuit
from gate_fusion import fuse_circuit as actual_fuse_circuit
from observable import Observable
from simulator import StateVectorSimulator
from single_qubit_gates import H, RY, X
from statevector import StateVector
from two_qubit_gates import CX

TOLERANCE = 1e-12


def bell_circuit() -> Circuit:
    return Circuit(2).append(H(0)).append(CX(0, 1))


class SimulatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.simulator = StateVectorSimulator()

    def test_run_produces_bell_state(self) -> None:
        state = self.simulator.run(bell_circuit())

        np.testing.assert_allclose(
            state.amplitudes,
            [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)],
            atol=TOLERANCE,
        )

    def test_run_starts_from_zero_state(self) -> None:
        state = self.simulator.run(Circuit(2))

        np.testing.assert_allclose(state.amplitudes, [1, 0, 0, 0], atol=TOLERANCE)

    def test_fusion_is_disabled_by_default(self) -> None:
        with patch("simulator.fuse_circuit") as fuse:
            self.simulator.run(bell_circuit())

        self.assertFalse(self.simulator.fusion)
        fuse.assert_not_called()

    def test_run_uses_gate_fusion_when_enabled(self) -> None:
        circuit = Circuit(2).append(H(0)).append(X(0)).append(CX(0, 1))
        simulator = StateVectorSimulator(fusion=True)

        with patch(
            "simulator.fuse_circuit",
            wraps=actual_fuse_circuit,
        ) as fuse:
            state = simulator.run(circuit)

        self.assertTrue(simulator.fusion)
        fuse.assert_called_once_with(circuit)
        np.testing.assert_allclose(
            state.amplitudes,
            self.simulator.run(circuit).amplitudes,
            atol=TOLERANCE,
        )

    def test_rejects_non_boolean_fusion_option(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be a boolean"):
            StateVectorSimulator(fusion=1)

    def test_run_accepts_initial_state_without_mutating_it(self) -> None:
        initial_state = StateVector(2).apply(X(0))

        result = self.simulator.run(Circuit(2).append(X(0)), initial_state)

        np.testing.assert_allclose(result.amplitudes, [1, 0, 0, 0], atol=TOLERANCE)
        np.testing.assert_allclose(
            initial_state.amplitudes, [0, 1, 0, 0], atol=TOLERANCE
        )

    def test_run_rejects_register_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "3 qubits"):
            self.simulator.run(Circuit(3), StateVector(2))

    def test_run_rejects_non_circuit(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be a Circuit"):
            self.simulator.run([H(0)])

    def test_circuit_dagger_round_trip(self) -> None:
        circuit = bell_circuit()

        state = self.simulator.run(circuit)
        for operation in circuit.dagger():
            state.apply(operation)

        np.testing.assert_allclose(state.amplitudes, [1, 0, 0, 0], atol=TOLERANCE)


class SamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.simulator = StateVectorSimulator()

    def test_sampling_is_reproducible_with_a_seeded_generator(self) -> None:
        first = self.simulator.run(bell_circuit()).sample(
            500, np.random.default_rng(7)
        )
        second = self.simulator.run(bell_circuit()).sample(
            500, np.random.default_rng(7)
        )

        self.assertEqual(first, second)

    def test_bell_state_only_produces_correlated_outcomes(self) -> None:
        counts = self.simulator.run(bell_circuit()).sample(
            2000, np.random.default_rng(1)
        )

        self.assertEqual(set(counts), {0, 3})
        self.assertEqual(sum(counts.values()), 2000)

    def test_sampling_follows_the_probability_distribution(self) -> None:
        counts = self.simulator.run(bell_circuit()).sample(
            20000, np.random.default_rng(2)
        )

        self.assertAlmostEqual(counts[0] / 20000, 0.5, places=1)

    def test_sampling_respects_little_endian_indexing(self) -> None:
        counts = self.simulator.run(Circuit(2).append(X(0))).sample(
            10, np.random.default_rng(3)
        )

        self.assertEqual(counts, {1: 10})

    def test_sampling_reproduces_a_non_uniform_entangled_distribution(self) -> None:
        circuit = Circuit(2).append(RY(np.pi / 3, 0)).append(CX(0, 1))

        counts = self.simulator.run(circuit).sample(4000, np.random.default_rng(17))

        self.assertEqual(set(counts), {0, 3})
        self.assertAlmostEqual(counts[0] / 4000, 0.75, places=1)

    def test_sampling_does_not_modify_the_state(self) -> None:
        state = self.simulator.run(bell_circuit())
        before = state.amplitudes.copy()

        state.sample(500, np.random.default_rng(4))

        np.testing.assert_allclose(state.amplitudes, before, atol=TOLERANCE)

    def test_repeated_sampling_of_one_state_is_allowed(self) -> None:
        state = self.simulator.run(bell_circuit())

        first = state.sample(200, np.random.default_rng(9))
        second = state.sample(200, np.random.default_rng(9))

        self.assertEqual(first, second)

    def test_destructive_measurement_is_not_exposed(self) -> None:
        state = self.simulator.run(bell_circuit())

        self.assertFalse(hasattr(state, "measure"))
        self.assertFalse(hasattr(state, "measure_qubits"))
        self.assertFalse(hasattr(state.backend, "collapse"))

    def test_rejects_invalid_shots(self) -> None:
        state = StateVector(1)

        with self.assertRaisesRegex(ValueError, "must be positive"):
            state.sample(0)
        with self.assertRaisesRegex(TypeError, "must be an integer"):
            state.sample(1.5)


class ExpectationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.simulator = StateVectorSimulator()

    def test_pauli_z_on_basis_states(self) -> None:
        observable = Observable([(1.0, {0: "Z"})])

        self.assertAlmostEqual(
            self.simulator.run(Circuit(1)).expectation(observable), 1.0
        )
        self.assertAlmostEqual(
            self.simulator.run(Circuit(1).append(X(0))).expectation(observable),
            -1.0,
        )

    def test_pauli_z_on_superposition(self) -> None:
        state = self.simulator.run(Circuit(1).append(H(0)))

        value = state.expectation(Observable([(1.0, {0: "Z"})]))

        self.assertAlmostEqual(value, 0.0)

    def test_pauli_x_and_y_on_superposition(self) -> None:
        state = self.simulator.run(Circuit(1).append(H(0)))

        self.assertAlmostEqual(
            state.expectation(Observable([(1.0, {0: "X"})])), 1.0
        )
        self.assertAlmostEqual(
            state.expectation(Observable([(1.0, {0: "Y"})])), 0.0
        )

    def test_correlated_observables_on_bell_state(self) -> None:
        state = self.simulator.run(bell_circuit())

        self.assertAlmostEqual(
            state.expectation(Observable([(1.0, {0: "Z", 1: "Z"})])), 1.0
        )
        self.assertAlmostEqual(
            state.expectation(Observable([(1.0, {0: "X", 1: "X"})])), 1.0
        )
        self.assertAlmostEqual(
            state.expectation(Observable([(1.0, {0: "Y", 1: "Y"})])), -1.0
        )

    def test_expectation_does_not_mutate_the_state(self) -> None:
        state = StateVectorSimulator().run(bell_circuit())
        before = state.amplitudes.copy()

        state.expectation(Observable([(1.0, {0: "Z", 1: "Z"})]))

        np.testing.assert_allclose(state.amplitudes, before, atol=TOLERANCE)

    def test_inner_product_of_normalized_state_is_one(self) -> None:
        state = self.simulator.run(bell_circuit())

        self.assertAlmostEqual(abs(state.inner_product(state)), 1.0)

    def test_inner_product_rejects_size_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "same number of qubits"):
            StateVector(2).inner_product(StateVector(3))


if __name__ == "__main__":
    unittest.main()
