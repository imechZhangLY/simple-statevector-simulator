import unittest

import numpy as np

from numpy_backend import NumpyBackend
from single_qubit_gates import H, RX, X
from statevector import StateVector
from three_qubit_gates import CCX
from two_qubit_gates import CX

TOLERANCE = 1e-6


def available_backends() -> list:
    return [NumpyBackend(), NumpyBackend(dtype=np.complex64)] + torch_backends()


def torch_backends() -> list:
    try:
        import torch
    except ImportError:
        return []

    from torch_backend import TorchBackend

    backends = [
        TorchBackend(dtype="complex128"),
        TorchBackend(dtype="complex64"),
    ]
    if torch.cuda.is_available():
        backends.append(TorchBackend(device="cuda", dtype="complex64"))
    return backends


class BackendConformanceTests(unittest.TestCase):
    def test_initializes_to_zero_state(self) -> None:
        for backend in available_backends():
            with self.subTest(backend=backend.name):
                state = StateVector(3, backend=backend)

                np.testing.assert_allclose(
                    state.amplitudes, [1, 0, 0, 0, 0, 0, 0, 0], atol=TOLERANCE
                )

    def test_qubit_zero_is_least_significant_bit(self) -> None:
        for backend in available_backends():
            with self.subTest(backend=backend.name):
                state = StateVector(2, backend=backend).apply(X(0))

                np.testing.assert_allclose(
                    state.amplitudes, [0, 1, 0, 0], atol=TOLERANCE
                )

    def test_creates_bell_state(self) -> None:
        for backend in available_backends():
            with self.subTest(backend=backend.name):
                state = StateVector(2, backend=backend).apply(H(0)).apply(CX(0, 1))

                np.testing.assert_allclose(
                    state.amplitudes,
                    [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)],
                    atol=TOLERANCE,
                )

    def test_applies_local_gate_to_entangled_state(self) -> None:
        for backend in available_backends():
            with self.subTest(backend=backend.name):
                state = (
                    StateVector(2, backend=backend)
                    .apply(H(0))
                    .apply(CX(0, 1))
                    .apply(X(0))
                )

                np.testing.assert_allclose(
                    state.amplitudes,
                    [0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0],
                    atol=TOLERANCE,
                )

    def test_applies_gate_to_non_adjacent_qubits(self) -> None:
        for backend in available_backends():
            with self.subTest(backend=backend.name):
                state = StateVector(3, backend=backend).apply(X(2)).apply(CX(2, 0))

                expected = np.zeros(8)
                expected[5] = 1
                np.testing.assert_allclose(
                    state.amplitudes, expected, atol=TOLERANCE
                )

    def test_applies_three_qubit_gate(self) -> None:
        for backend in available_backends():
            with self.subTest(backend=backend.name):
                state = (
                    StateVector(3, backend=backend)
                    .apply(X(0))
                    .apply(X(1))
                    .apply(CCX(0, 1, 2))
                )

                expected = np.zeros(8)
                expected[7] = 1
                np.testing.assert_allclose(
                    state.amplitudes, expected, atol=TOLERANCE
                )

    def test_dagger_restores_state(self) -> None:
        operation = RX(0.3, 1)

        for backend in available_backends():
            with self.subTest(backend=backend.name):
                state = (
                    StateVector(2, backend=backend)
                    .apply(operation)
                    .apply(operation.dagger())
                )

                np.testing.assert_allclose(
                    state.amplitudes, [1, 0, 0, 0], atol=TOLERANCE
                )

    def test_matches_reference_backend_on_mixed_circuit(self) -> None:
        operations = [
            H(0),
            CX(0, 1),
            RX(0.7, 2),
            CCX(0, 1, 2),
            X(2),
            CX(2, 0),
            RX(0.4, 1).dagger(),
        ]

        reference = StateVector(3, backend=NumpyBackend())
        for operation in operations:
            reference.apply(operation)

        for backend in available_backends():
            with self.subTest(backend=backend.name):
                state = StateVector(3, backend=backend)
                for operation in operations:
                    state.apply(operation)

                np.testing.assert_allclose(
                    state.amplitudes, reference.amplitudes, atol=TOLERANCE
                )

    def test_probabilities_are_normalized(self) -> None:
        for backend in available_backends():
            with self.subTest(backend=backend.name):
                state = StateVector(3, backend=backend).apply(H(0)).apply(CX(0, 2))

                probabilities = state.probabilities

                self.assertEqual(probabilities.dtype, np.float64)
                self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=5)

    def test_copy_is_independent_and_preserves_backend(self) -> None:
        for backend in available_backends():
            with self.subTest(backend=backend.name):
                original = StateVector(1, backend=backend)
                copied = original.copy().apply(X(0))

                self.assertIs(copied.backend, backend)
                np.testing.assert_allclose(
                    original.amplitudes, [1, 0], atol=TOLERANCE
                )
                np.testing.assert_allclose(copied.amplitudes, [0, 1], atol=TOLERANCE)

    def test_amplitudes_are_read_only_numpy_arrays(self) -> None:
        for backend in available_backends():
            with self.subTest(backend=backend.name):
                state = StateVector(1, backend=backend)

                amplitudes = state.amplitudes

                self.assertEqual(amplitudes.dtype, np.complex128)
                self.assertFalse(amplitudes.flags.writeable)

    def test_rejects_invalid_amplitudes(self) -> None:
        for backend in available_backends():
            with self.subTest(backend=backend.name):
                with self.assertRaisesRegex(ValueError, "shape"):
                    StateVector(2, [1, 0], backend=backend)
                with self.assertRaisesRegex(ValueError, "normalized"):
                    StateVector(1, [1, 1], backend=backend)
                with self.assertRaisesRegex(ValueError, "finite"):
                    StateVector(1, [np.nan, 0], backend=backend)

    def test_rejects_operation_outside_statevector(self) -> None:
        for backend in available_backends():
            with self.subTest(backend=backend.name):
                with self.assertRaisesRegex(IndexError, "outside"):
                    StateVector(2, backend=backend).apply(X(2))


class NumpyBackendTests(unittest.TestCase):
    def test_rejects_non_complex_dtype(self) -> None:
        with self.assertRaisesRegex(ValueError, "complex"):
            NumpyBackend(dtype=np.float64)

    def test_name_reports_dtype(self) -> None:
        self.assertEqual(NumpyBackend().name, "numpy:complex128")
        self.assertEqual(NumpyBackend(dtype=np.complex64).name, "numpy:complex64")

    def test_raw_amplitudes_use_backend_dtype(self) -> None:
        state = StateVector(2, backend=NumpyBackend(dtype=np.complex64))

        self.assertEqual(state.raw_amplitudes.dtype, np.complex64)


@unittest.skipUnless(torch_backends(), "torch is not installed")
class TorchBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch

        from torch_backend import TorchBackend

        self.torch = torch
        self.TorchBackend = TorchBackend

    def test_rejects_non_complex_dtype(self) -> None:
        with self.assertRaisesRegex(ValueError, "complex"):
            self.TorchBackend(dtype="float64")

    def test_name_reports_device_and_dtype(self) -> None:
        backend = self.TorchBackend(dtype="complex64")

        self.assertEqual(backend.name, "torch:cpu:complex64")

    def test_raw_amplitudes_are_tensors_on_backend_device(self) -> None:
        backend = self.TorchBackend(dtype="complex64")
        state = StateVector(2, backend=backend)

        self.assertIsInstance(state.raw_amplitudes, self.torch.Tensor)
        self.assertEqual(state.raw_amplitudes.dtype, self.torch.complex64)
        self.assertEqual(state.raw_amplitudes.device.type, backend.device.type)

    def test_matrix_cache_reuses_entries_for_equal_keys(self) -> None:
        backend = self.TorchBackend()
        state = StateVector(2, backend=backend)

        state.apply(RX(0.3, 0)).apply(RX(0.3, 1)).apply(RX(0.3, 0))

        self.assertEqual(backend.cached_matrix_count, 1)

    def test_matrix_cache_is_bounded(self) -> None:
        backend = self.TorchBackend(matrix_cache_size=4)
        state = StateVector(1, backend=backend)

        for angle in np.linspace(0.1, 1.0, 20):
            state.apply(RX(float(angle), 0))

        self.assertLessEqual(backend.cached_matrix_count, 4)

    def test_eviction_does_not_corrupt_results(self) -> None:
        angles = [float(angle) for angle in np.linspace(0.1, 1.0, 12)]
        reference = StateVector(1, backend=NumpyBackend())
        state = StateVector(1, backend=self.TorchBackend(matrix_cache_size=2))

        for angle in angles + angles:
            reference.apply(RX(angle, 0))
            state.apply(RX(angle, 0))

        np.testing.assert_allclose(
            state.amplitudes, reference.amplitudes, atol=TOLERANCE
        )

    def test_rejects_cuda_when_unavailable(self) -> None:
        if self.torch.cuda.is_available():
            self.skipTest("CUDA is available on this machine")

        with self.assertRaisesRegex(RuntimeError, "CUDA is not available"):
            self.TorchBackend(device="cuda")

    def test_matches_numpy_on_cuda(self) -> None:
        if not self.torch.cuda.is_available():
            self.skipTest("CUDA is not available on this machine")

        operations = [H(0), CX(0, 1), RX(0.7, 2), CCX(0, 1, 2), CX(2, 0)]
        reference = StateVector(3, backend=NumpyBackend())
        state = StateVector(
            3, backend=self.TorchBackend(device="cuda", dtype="complex64")
        )

        for operation in operations:
            reference.apply(operation)
            state.apply(operation)

        self.assertEqual(state.raw_amplitudes.device.type, "cuda")
        np.testing.assert_allclose(
            state.amplitudes, reference.amplitudes, atol=TOLERANCE
        )


if __name__ == "__main__":
    unittest.main()
