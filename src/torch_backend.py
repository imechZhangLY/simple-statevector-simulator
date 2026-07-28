from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import numpy as np

from backend import Amplitudes, ComplexVector, ProbabilityVector

if TYPE_CHECKING:
    from operation import Operation

DEFAULT_MATRIX_CACHE_SIZE = 256


class TorchBackend:
    def __init__(
        self,
        device: Any = "cpu",
        dtype: Any = None,
        matrix_cache_size: int = DEFAULT_MATRIX_CACHE_SIZE,
    ) -> None:
        import torch

        self._torch = torch
        self._device = torch.device(device)
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this machine")

        if dtype is None:
            dtype = torch.complex128
        elif isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if not isinstance(dtype, torch.dtype) or not dtype.is_complex:
            raise ValueError("dtype must be a complex type")
        self._dtype = dtype

        if matrix_cache_size <= 0:
            raise ValueError("matrix_cache_size must be positive")
        self._matrix_cache_size = int(matrix_cache_size)
        self._matrix_cache: "OrderedDict[tuple, Any]" = OrderedDict()

    @property
    def name(self) -> str:
        dtype_name = str(self._dtype).removeprefix("torch.")
        return f"torch:{self._device}:{dtype_name}"

    @property
    def device(self) -> Any:
        return self._device

    @property
    def dtype(self) -> Any:
        return self._dtype

    @property
    def matrix_cache_size(self) -> int:
        return self._matrix_cache_size

    @property
    def cached_matrix_count(self) -> int:
        return len(self._matrix_cache)

    def zero_state(self, num_qubits: int) -> Amplitudes:
        amplitudes = self._torch.zeros(
            1 << num_qubits, dtype=self._dtype, device=self._device
        )
        amplitudes[0] = 1
        return amplitudes

    def as_amplitudes(self, amplitudes: Any) -> Amplitudes:
        if isinstance(amplitudes, self._torch.Tensor):
            return amplitudes.detach().clone().to(
                device=self._device, dtype=self._dtype
            )

        array = np.array(amplitudes, dtype=np.complex128, copy=True)
        return self._torch.as_tensor(
            array, dtype=self._dtype, device=self._device
        )

    def shape(self, amplitudes: Amplitudes) -> tuple[int, ...]:
        return tuple(amplitudes.shape)

    def is_finite(self, amplitudes: Amplitudes) -> bool:
        return bool(self._torch.isfinite(amplitudes).all())

    def squared_norm(self, amplitudes: Amplitudes) -> float:
        return float(self._torch.vdot(amplitudes, amplitudes).real)

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
        inverse_axes = [0] * num_qubits
        for position, axis in enumerate(axes):
            inverse_axes[axis] = position

        matrix = self._device_matrix(operation)
        amplitude_tensor = amplitudes.reshape((2,) * num_qubits)
        batched_amplitudes = amplitude_tensor.permute(axes).reshape(
            1 << len(qubits), -1
        )
        updated = matrix @ batched_amplitudes
        return (
            updated.reshape((2,) * num_qubits)
            .permute(inverse_axes)
            .reshape(-1)
        )

    def probabilities(self, amplitudes: Amplitudes) -> ProbabilityVector:
        return (
            amplitudes.detach()
            .abs()
            .pow(2)
            .to(device="cpu", dtype=self._torch.float64)
            .numpy()
        )

    def copy(self, amplitudes: Amplitudes) -> Amplitudes:
        return amplitudes.clone()

    def inner_product(self, left: Amplitudes, right: Amplitudes) -> complex:
        return complex(self._torch.vdot(left, right))

    def to_numpy(self, amplitudes: Amplitudes) -> ComplexVector:
        return (
            amplitudes.detach()
            .to(device="cpu", dtype=self._torch.complex128)
            .numpy()
        )

    def _device_matrix(self, operation: "Operation") -> Any:
        key = operation.matrix_key
        cached = self._matrix_cache.get(key)
        if cached is not None:
            self._matrix_cache.move_to_end(key)
            return cached

        matrix = np.array(operation.matrix, dtype=np.complex128, copy=True)
        tensor = self._torch.as_tensor(
            matrix, dtype=self._dtype, device=self._device
        )

        self._matrix_cache[key] = tensor
        if len(self._matrix_cache) > self._matrix_cache_size:
            self._matrix_cache.popitem(last=False)
        return tensor


__all__ = ["DEFAULT_MATRIX_CACHE_SIZE", "TorchBackend"]
