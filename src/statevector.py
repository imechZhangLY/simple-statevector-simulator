from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from operation import Operation


ComplexVector = NDArray[np.complex128]
ProbabilityVector = NDArray[np.float64]


class StateVector:
    def __init__(self, num_qubits: int, amplitudes: ArrayLike | None = None) -> None:
        if not isinstance(num_qubits, Integral) or isinstance(num_qubits, bool):
            raise TypeError("num_qubits must be an integer")
        if num_qubits <= 0:
            raise ValueError("num_qubits must be positive")

        self._num_qubits = int(num_qubits)
        dimension = 1 << self._num_qubits
        if amplitudes is None:
            normalized_amplitudes = np.zeros(dimension, dtype=np.complex128)
            normalized_amplitudes[0] = 1
        else:
            normalized_amplitudes = np.array(
                amplitudes, dtype=np.complex128, copy=True
            )
            if normalized_amplitudes.shape != (dimension,):
                raise ValueError(
                    f"amplitudes must have shape ({dimension},), "
                    f"but received {normalized_amplitudes.shape}"
                )
            if not np.isfinite(normalized_amplitudes).all():
                raise ValueError("amplitudes must contain only finite values")
            if not np.isclose(np.vdot(normalized_amplitudes, normalized_amplitudes), 1):
                raise ValueError("amplitudes must be normalized")

        self._amplitudes = normalized_amplitudes

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def amplitudes(self) -> ComplexVector:
        amplitudes = self._amplitudes.view()
        amplitudes.flags.writeable = False
        return amplitudes

    @property
    def probabilities(self) -> ProbabilityVector:
        return np.abs(self._amplitudes) ** 2

    def apply(self, operation: Operation) -> "StateVector":
        if any(qubit >= self._num_qubits for qubit in operation.qubits):
            raise IndexError("operation qubit is outside the statevector")

        target_axes = [self._num_qubits - 1 - qubit for qubit in operation.qubits]
        remaining_axes = [
            axis for axis in range(self._num_qubits) if axis not in target_axes
        ]
        axes = target_axes + remaining_axes
        inverse_axes = np.argsort(axes)

        amplitude_tensor = self._amplitudes.reshape((2,) * self._num_qubits)
        target_dimension = 1 << len(operation.qubits)
        batched_amplitudes = amplitude_tensor.transpose(axes).reshape(
            target_dimension, -1
        )
        updated = operation.matrix @ batched_amplitudes
        self._amplitudes = updated.reshape((2,) * self._num_qubits).transpose(
            inverse_axes
        ).reshape(-1)
        return self

    def copy(self) -> "StateVector":
        return StateVector(self._num_qubits, self._amplitudes)