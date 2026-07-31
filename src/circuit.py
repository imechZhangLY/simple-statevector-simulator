from numbers import Integral
from typing import Iterator

from operation import Operation


class Circuit:
    def __init__(self, num_qubits: int) -> None:
        if not isinstance(num_qubits, Integral) or isinstance(num_qubits, bool):
            raise TypeError("num_qubits must be an integer")
        if num_qubits <= 0:
            raise ValueError("num_qubits must be positive")

        self._num_qubits = int(num_qubits)
        self._operations: list[Operation] = []
        self._operations_mat: list[list[Operation | None]] = [
            [] for _ in range(num_qubits)
        ]

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def operations(self) -> tuple[Operation, ...]:
        return tuple(self._operations)

    # Each row corresponds to a qubit, and each column corresponds to an operation in the circuit.
    # If a qubit is not involved in an operation, the corresponding entry is None.
    # The operations in the same column are in the same layer, and can be applied in parallel.
    @property
    def operations_mat(self) -> list[list[Operation | None]]:
        return self._operations_mat

    def append(self, operation: Operation) -> "Circuit":
        if not isinstance(operation, Operation):
            raise TypeError("operation must be an Operation")
        if any(qubit >= self._num_qubits for qubit in operation.qubits):
            raise IndexError("operation qubit is outside the circuit")

        self._operations.append(operation)
        target = max(
            operation.qubits,
            key=lambda qubit: len(self._operations_mat[qubit]),
        )
        target_length = len(self._operations_mat[target])
        for qubit in operation.qubits:
            if qubit != target:
                self._operations_mat[qubit] += [None] * (
                    target_length - len(self._operations_mat[qubit])
                )
            self._operations_mat[qubit].append(operation)

        return self

    def dagger(self) -> "Circuit":
        inverted = Circuit(self._num_qubits)
        for operation in reversed(self._operations):
            inverted.append(operation.dagger())
        return inverted

    def copy(self) -> "Circuit":
        copied = Circuit(self._num_qubits)
        copied._operations = list(self._operations)
        copied._operations_mat = [list(row) for row in self._operations_mat]
        return copied

    def __len__(self) -> int:
        return len(self._operations)

    def __iter__(self) -> Iterator[Operation]:
        return iter(self._operations)


__all__ = ["Circuit"]
