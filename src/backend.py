from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from operation import Operation

Amplitudes = Any
ComplexVector = NDArray[np.complex128]
ProbabilityVector = NDArray[np.float64]


class Backend(Protocol):
    @property
    def name(self) -> str: ...

    def zero_state(self, num_qubits: int) -> Amplitudes: ...

    def as_amplitudes(self, amplitudes: Any) -> Amplitudes: ...

    def shape(self, amplitudes: Amplitudes) -> tuple[int, ...]: ...

    def is_finite(self, amplitudes: Amplitudes) -> bool: ...

    def squared_norm(self, amplitudes: Amplitudes) -> float: ...

    def apply(
        self,
        amplitudes: Amplitudes,
        operation: "Operation",
        num_qubits: int,
    ) -> Amplitudes: ...

    def probabilities(self, amplitudes: Amplitudes) -> ProbabilityVector: ...

    def inner_product(self, left: Amplitudes, right: Amplitudes) -> complex: ...

    def copy(self, amplitudes: Amplitudes) -> Amplitudes: ...

    def to_numpy(self, amplitudes: Amplitudes) -> ComplexVector: ...


__all__ = ["Amplitudes", "Backend", "ComplexVector", "ProbabilityVector"]
