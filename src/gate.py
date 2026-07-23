from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray


ComplexMatrix = NDArray[np.complex128]


@dataclass(frozen=True)
class Gate:
    name: str
    qasm_name: str
    num_qubits: int
    parameters: tuple[float, ...]
    dagger_qasm_name: str
    dagger_parameters: tuple[float, ...]
    matrix: ComplexMatrix
    dagger_matrix: ComplexMatrix

    def __init__(
        self,
        name: str,
        num_qubits: int,
        matrix: ArrayLike,
        dagger_matrix: ArrayLike,
        *,
        qasm_name: str,
        parameters: tuple[float, ...] = (),
        dagger_qasm_name: str | None = None,
        dagger_parameters: tuple[float, ...] | None = None,
    ) -> None:
        if not name:
            raise ValueError("name must not be empty")
        if not qasm_name:
            raise ValueError("qasm_name must not be empty")
        if num_qubits <= 0:
            raise ValueError("num_qubits must be positive")

        normalized_parameters = self._normalize_parameters("parameters", parameters)
        normalized_dagger_parameters = self._normalize_parameters(
            "dagger_parameters",
            parameters if dagger_parameters is None else dagger_parameters,
        )
        dimension = 1 << num_qubits
        expected_shape = (dimension, dimension)
        normalized_matrix = self._normalize_matrix("matrix", matrix, expected_shape)
        normalized_dagger_matrix = self._normalize_matrix(
            "dagger_matrix", dagger_matrix, expected_shape
        )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "qasm_name", qasm_name)
        object.__setattr__(self, "num_qubits", num_qubits)
        object.__setattr__(self, "parameters", normalized_parameters)
        object.__setattr__(
            self, "dagger_qasm_name", dagger_qasm_name or qasm_name
        )
        object.__setattr__(self, "dagger_parameters", normalized_dagger_parameters)
        object.__setattr__(self, "matrix", normalized_matrix)
        object.__setattr__(self, "dagger_matrix", normalized_dagger_matrix)

    @staticmethod
    def _normalize_parameters(
        field_name: str,
        parameters: tuple[float, ...],
    ) -> tuple[float, ...]:
        normalized: list[float] = []
        for parameter in parameters:
            if not isinstance(parameter, Real) or isinstance(parameter, bool):
                raise TypeError(f"{field_name} must contain only real numbers")
            value = float(parameter)
            if not np.isfinite(value):
                raise ValueError(f"{field_name} must contain only finite values")
            normalized.append(value)
        return tuple(normalized)

    @staticmethod
    def _normalize_matrix(
        field_name: str,
        matrix: ArrayLike,
        expected_shape: tuple[int, int],
    ) -> ComplexMatrix:
        normalized = np.array(matrix, dtype=np.complex128, copy=True)
        if normalized.shape != expected_shape:
            raise ValueError(
                f"{field_name} must have shape {expected_shape}, "
                f"but received {normalized.shape}"
            )
        if not np.isfinite(normalized).all():
            raise ValueError(f"{field_name} must contain only finite values")

        normalized.flags.writeable = False
        return normalized