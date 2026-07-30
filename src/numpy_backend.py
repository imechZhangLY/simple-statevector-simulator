from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import DTypeLike

from backend import Amplitudes, ComplexVector, ProbabilityVector

if TYPE_CHECKING:
    from operation import Operation


class NumpyBackend:
    def __init__(self, dtype: DTypeLike = np.complex128) -> None:
        resolved_dtype = np.dtype(dtype)
        if resolved_dtype.kind != "c":
            raise ValueError("dtype must be a complex type")

        self._dtype = resolved_dtype

    @property
    def name(self) -> str:
        return f"numpy:{self._dtype.name}"

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def zero_state(self, num_qubits: int) -> Amplitudes:
        amplitudes = np.zeros((2,) * num_qubits, dtype=self._dtype)
        amplitudes[(0,) * num_qubits] = 1
        return amplitudes

    def as_amplitudes(self, amplitudes: Any) -> Amplitudes:
        array = np.array(amplitudes, dtype=self._dtype, copy=True)
        return _as_qubit_axes(array)

    def shape(self, amplitudes: Amplitudes) -> tuple[int, ...]:
        return tuple(amplitudes.shape)

    def is_finite(self, amplitudes: Amplitudes) -> bool:
        return bool(np.isfinite(amplitudes).all())

    def squared_norm(self, amplitudes: Amplitudes) -> float:
        # np.vdot flattens its operands, so the axis layout does not matter.
        return float(np.vdot(amplitudes, amplitudes).real)

    def apply(
        self,
        amplitudes: Amplitudes,
        operation: "Operation",
        num_qubits: int,
    ) -> Amplitudes:
        qubits = operation.qubits
        target_axes = [num_qubits - 1 - qubit for qubit in qubits]
        remaining_axes = [
            axis for axis in range(num_qubits) if axis not in target_axes
        ]
        axes = target_axes + remaining_axes
        inverse_axes = np.argsort(axes)

        amplitude_tensor = amplitudes.reshape((2,) * num_qubits)
        batched_amplitudes = amplitude_tensor.transpose(axes).reshape(
            1 << len(qubits), -1
        )
        matrix = np.asarray(operation.matrix, dtype=self._dtype)
        updated = matrix @ batched_amplitudes
        return updated.reshape((2,) * num_qubits).transpose(inverse_axes)

    def probabilities(self, amplitudes: Amplitudes) -> ProbabilityVector:
        probabilities = np.abs(amplitudes) ** 2
        return probabilities.astype(np.float64, copy=False).reshape(-1)

    def inner_product(self, left: Amplitudes, right: Amplitudes) -> complex:
        return complex(np.vdot(left, right))

    def copy(self, amplitudes: Amplitudes) -> Amplitudes:
        return amplitudes.copy()

    def to_numpy(self, amplitudes: Amplitudes) -> ComplexVector:
        return np.asarray(amplitudes, dtype=np.complex128).reshape(-1)


def _as_qubit_axes(array: Amplitudes) -> Amplitudes:
    """Reshape to one axis per qubit, leaving anything else for StateVector to reject."""
    size = array.size
    num_qubits = int(size).bit_length() - 1
    if size and (1 << num_qubits) == size:
        return array.reshape((2,) * num_qubits)
    return array


__all__ = ["NumpyBackend"]
