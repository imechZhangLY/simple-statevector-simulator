from dataclasses import dataclass
from numbers import Integral, Real
from typing import Iterable, Mapping

import numpy as np

from operation import Operation
from single_qubit_gates import X, Y, Z

PAULI_GATES = {"X": X, "Y": Y, "Z": Z}
PAULI_LETTERS = frozenset({"I", "X", "Y", "Z"})


@dataclass(frozen=True)
class PauliTerm:
    coefficient: float
    paulis: tuple[tuple[int, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.coefficient, Real) or isinstance(
            self.coefficient, bool
        ):
            raise TypeError("coefficient must be a real number")
        coefficient = float(self.coefficient)
        if not np.isfinite(coefficient):
            raise ValueError("coefficient must be finite")

        entries = (
            self.paulis.items()
            if isinstance(self.paulis, Mapping)
            else tuple(self.paulis)
        )

        normalized: dict[int, str] = {}
        for qubit, letter in entries:
            if not isinstance(qubit, Integral) or isinstance(qubit, bool):
                raise TypeError("qubit must be an integer")
            if qubit < 0:
                raise ValueError("qubit must be non-negative")
            if not isinstance(letter, str):
                raise TypeError("Pauli must be a string")

            pauli = letter.upper()
            if pauli not in PAULI_LETTERS:
                raise ValueError(
                    f"Pauli must be one of I, X, Y, Z, but received {letter!r}"
                )
            if int(qubit) in normalized:
                raise ValueError("qubits must be unique within a term")

            normalized[int(qubit)] = pauli

        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(
            self,
            "paulis",
            tuple(
                (qubit, normalized[qubit])
                for qubit in sorted(normalized)
                if normalized[qubit] != "I"
            ),
        )

    @property
    def operations(self) -> tuple[Operation, ...]:
        return tuple(PAULI_GATES[letter](qubit) for qubit, letter in self.paulis)


class Observable:
    def __init__(self, terms: Iterable) -> None:
        normalized: list[PauliTerm] = []
        for term in terms:
            if isinstance(term, PauliTerm):
                normalized.append(term)
            else:
                coefficient, paulis = term
                normalized.append(PauliTerm(coefficient, paulis))

        self._terms = tuple(normalized)

    @property
    def terms(self) -> tuple[PauliTerm, ...]:
        return self._terms

    def __len__(self) -> int:
        return len(self._terms)

    def __iter__(self):
        return iter(self._terms)


__all__ = ["Observable", "PauliTerm"]
