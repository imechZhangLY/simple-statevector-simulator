from numbers import Integral
from typing import Any

import numpy as np

from backend import Amplitudes, Backend, ComplexVector, ProbabilityVector
from numpy_backend import NumpyBackend
from observable import Observable
from operation import Operation

DEFAULT_BACKEND: Backend = NumpyBackend()


class StateVector:
    def __init__(
        self,
        num_qubits: int,
        amplitudes: Any | None = None,
        *,
        backend: Backend | None = None,
    ) -> None:
        if not isinstance(num_qubits, Integral) or isinstance(num_qubits, bool):
            raise TypeError("num_qubits must be an integer")
        if num_qubits <= 0:
            raise ValueError("num_qubits must be positive")

        self._num_qubits = int(num_qubits)
        self._backend = DEFAULT_BACKEND if backend is None else backend

        if amplitudes is None:
            self._amplitudes = self._backend.zero_state(self._num_qubits)
            return

        dimension = 1 << self._num_qubits
        normalized_amplitudes = self._backend.as_amplitudes(amplitudes)
        actual_shape = self._backend.shape(normalized_amplitudes)
        if actual_shape != (dimension,):
            raise ValueError(
                f"amplitudes must have shape ({dimension},), "
                f"but received {actual_shape}"
            )
        if not self._backend.is_finite(normalized_amplitudes):
            raise ValueError("amplitudes must contain only finite values")
        if not np.isclose(self._backend.squared_norm(normalized_amplitudes), 1):
            raise ValueError("amplitudes must be normalized")

        self._amplitudes = normalized_amplitudes

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def raw_amplitudes(self) -> Amplitudes:
        return self._amplitudes

    @property
    def amplitudes(self) -> ComplexVector:
        amplitudes = self._backend.to_numpy(self._amplitudes).view()
        amplitudes.flags.writeable = False
        return amplitudes

    @property
    def probabilities(self) -> ProbabilityVector:
        return self._backend.probabilities(self._amplitudes)

    def apply(self, operation: Operation) -> "StateVector":
        if any(qubit >= self._num_qubits for qubit in operation.qubits):
            raise IndexError("operation qubit is outside the statevector")

        self._amplitudes = self._backend.apply(
            self._amplitudes,
            operation,
            self._num_qubits,
        )
        return self

    def inner_product(self, other: "StateVector") -> complex:
        if other.num_qubits != self._num_qubits:
            raise ValueError(
                f"statevectors must have the same number of qubits, "
                f"but received {self._num_qubits} and {other.num_qubits}"
            )

        return self._backend.inner_product(
            self._amplitudes,
            self._backend.as_amplitudes(other.raw_amplitudes),
        )

    def expectation(self, observable: "Observable") -> float:
        total = 0.0
        for term in observable:
            transformed = self.copy()
            for operation in term.operations:
                transformed.apply(operation)
            total += term.coefficient * self.inner_product(transformed).real
        return total

    def sample(
        self,
        shots: int,
        generator: "np.random.Generator | None" = None,
    ) -> dict[int, int]:
        if not isinstance(shots, Integral) or isinstance(shots, bool):
            raise TypeError("shots must be an integer")
        if shots <= 0:
            raise ValueError("shots must be positive")

        probabilities = self.probabilities
        probabilities = probabilities / probabilities.sum()
        generator = np.random.default_rng() if generator is None else generator

        outcomes = generator.choice(
            probabilities.size, size=int(shots), p=probabilities
        )
        values, counts = np.unique(outcomes, return_counts=True)
        return {int(value): int(count) for value, count in zip(values, counts)}

    def copy(self) -> "StateVector":
        return StateVector(
            self._num_qubits,
            self._amplitudes,
            backend=self._backend,
        )


__all__ = ["DEFAULT_BACKEND", "StateVector"]