from dataclasses import dataclass

from gate import ComplexMatrix, Gate


@dataclass(frozen=True)
class Operation:
    gate: Gate
    qubits: tuple[int, ...]
    is_dagger: bool = False

    def __post_init__(self) -> None:
        qubits = tuple(self.qubits)
        if len(qubits) != self.gate.num_qubits:
            raise ValueError(
                f"{self.gate.name} requires {self.gate.num_qubits} qubits, "
                f"but received {len(qubits)}"
            )
        if any(not isinstance(qubit, int) or isinstance(qubit, bool) for qubit in qubits):
            raise TypeError("qubits must contain only integers")
        if any(qubit < 0 for qubit in qubits):
            raise ValueError("qubits must be non-negative")
        if len(set(qubits)) != len(qubits):
            raise ValueError("qubits must be unique")

        object.__setattr__(self, "qubits", qubits)

    @property
    def name(self) -> str:
        return f"{self.gate.name}†" if self.is_dagger else self.gate.name

    @property
    def qasm_name(self) -> str:
        return self.gate.dagger_qasm_name if self.is_dagger else self.gate.qasm_name

    @property
    def parameters(self) -> tuple[float, ...]:
        if self.is_dagger:
            return self.gate.dagger_parameters
        return self.gate.parameters

    @property
    def matrix(self) -> ComplexMatrix:
        return self.gate.dagger_matrix if self.is_dagger else self.gate.matrix

    @property
    def matrix_key(self) -> tuple[str, int, tuple[float, ...], bool]:
        return (
            self.gate.name,
            self.gate.num_qubits,
            self.parameters,
            self.is_dagger,
        )

    def dagger(self) -> "Operation":
        return Operation(self.gate, self.qubits, not self.is_dagger)