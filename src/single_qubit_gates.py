from numbers import Real

import numpy as np

from gate import Gate
from operation import Operation


_I_MATRIX = np.eye(2, dtype=np.complex128)
_X_MATRIX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y_MATRIX = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z_MATRIX = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_H_MATRIX = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
_S_MATRIX = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
_S_DAGGER_MATRIX = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
_T_MATRIX = np.array(
    [[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128
)
_T_DAGGER_MATRIX = np.array(
    [[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=np.complex128
)

_I_GATE = Gate("I", 1, _I_MATRIX, _I_MATRIX, qasm_name="id")
_X_GATE = Gate("X", 1, _X_MATRIX, _X_MATRIX, qasm_name="x")
_Y_GATE = Gate("Y", 1, _Y_MATRIX, _Y_MATRIX, qasm_name="y")
_Z_GATE = Gate("Z", 1, _Z_MATRIX, _Z_MATRIX, qasm_name="z")
_H_GATE = Gate("H", 1, _H_MATRIX, _H_MATRIX, qasm_name="h")
_S_GATE = Gate(
    "S", 1, _S_MATRIX, _S_DAGGER_MATRIX, qasm_name="s", dagger_qasm_name="sdg"
)
_T_GATE = Gate(
    "T", 1, _T_MATRIX, _T_DAGGER_MATRIX, qasm_name="t", dagger_qasm_name="tdg"
)


def I(qubit: int) -> Operation:
    return Operation(_I_GATE, (qubit,))


def X(qubit: int) -> Operation:
    return Operation(_X_GATE, (qubit,))


def Y(qubit: int) -> Operation:
    return Operation(_Y_GATE, (qubit,))


def Z(qubit: int) -> Operation:
    return Operation(_Z_GATE, (qubit,))


def H(qubit: int) -> Operation:
    return Operation(_H_GATE, (qubit,))


def S(qubit: int) -> Operation:
    return Operation(_S_GATE, (qubit,))


def T(qubit: int) -> Operation:
    return Operation(_T_GATE, (qubit,))


def RX(theta: float, qubit: int) -> Operation:
    theta = _normalize_angle(theta)
    cosine = np.cos(theta / 2)
    sine = np.sin(theta / 2)
    matrix = np.array(
        [[cosine, -1j * sine], [-1j * sine, cosine]], dtype=np.complex128
    )
    dagger_matrix = np.array(
        [[cosine, 1j * sine], [1j * sine, cosine]], dtype=np.complex128
    )
    gate = Gate(
        "RX",
        1,
        matrix,
        dagger_matrix,
        qasm_name="rx",
        parameters=(theta,),
        dagger_parameters=(-theta,),
    )
    return Operation(gate, (qubit,))


def RY(theta: float, qubit: int) -> Operation:
    theta = _normalize_angle(theta)
    cosine = np.cos(theta / 2)
    sine = np.sin(theta / 2)
    matrix = np.array([[cosine, -sine], [sine, cosine]], dtype=np.complex128)
    dagger_matrix = np.array(
        [[cosine, sine], [-sine, cosine]], dtype=np.complex128
    )
    gate = Gate(
        "RY",
        1,
        matrix,
        dagger_matrix,
        qasm_name="ry",
        parameters=(theta,),
        dagger_parameters=(-theta,),
    )
    return Operation(gate, (qubit,))


def RZ(theta: float, qubit: int) -> Operation:
    theta = _normalize_angle(theta)
    negative_phase = np.exp(-1j * theta / 2)
    positive_phase = np.exp(1j * theta / 2)
    matrix = np.diag([negative_phase, positive_phase]).astype(np.complex128)
    dagger_matrix = np.diag([positive_phase, negative_phase]).astype(np.complex128)
    gate = Gate(
        "RZ",
        1,
        matrix,
        dagger_matrix,
        qasm_name="rz",
        parameters=(theta,),
        dagger_parameters=(-theta,),
    )
    return Operation(gate, (qubit,))


def P(theta: float, qubit: int) -> Operation:
    theta = _normalize_angle(theta)
    matrix = np.diag([1, np.exp(1j * theta)]).astype(np.complex128)
    dagger_matrix = np.diag([1, np.exp(-1j * theta)]).astype(np.complex128)
    gate = Gate(
        "P",
        1,
        matrix,
        dagger_matrix,
        qasm_name="p",
        parameters=(theta,),
        dagger_parameters=(-theta,),
    )
    return Operation(gate, (qubit,))


def U1(lam: float, qubit: int) -> Operation:
    lam = _normalize_angle(lam)
    matrix = np.diag([1, np.exp(1j * lam)]).astype(np.complex128)
    dagger_matrix = np.diag([1, np.exp(-1j * lam)]).astype(np.complex128)
    gate = Gate(
        "U1",
        1,
        matrix,
        dagger_matrix,
        qasm_name="u1",
        parameters=(lam,),
        dagger_parameters=(-lam,),
    )
    return Operation(gate, (qubit,))


def U2(phi: float, lam: float, qubit: int) -> Operation:
    phi = _normalize_angle(phi)
    lam = _normalize_angle(lam)
    matrix = _u_matrix(np.pi / 2, phi, lam)
    dagger_matrix = matrix.conj().T
    gate = Gate(
        "U2",
        1,
        matrix,
        dagger_matrix,
        qasm_name="u2",
        parameters=(phi, lam),
        dagger_parameters=(np.pi - lam, -np.pi - phi),
    )
    return Operation(gate, (qubit,))


def U3(theta: float, phi: float, lam: float, qubit: int) -> Operation:
    theta = _normalize_angle(theta)
    phi = _normalize_angle(phi)
    lam = _normalize_angle(lam)
    matrix = _u_matrix(theta, phi, lam)
    dagger_matrix = matrix.conj().T
    gate = Gate(
        "U3",
        1,
        matrix,
        dagger_matrix,
        qasm_name="u3",
        parameters=(theta, phi, lam),
        dagger_parameters=(-theta, -lam, -phi),
    )
    return Operation(gate, (qubit,))


def _u_matrix(theta: float, phi: float, lam: float) -> np.ndarray:
    cosine = np.cos(theta / 2)
    sine = np.sin(theta / 2)
    return np.array(
        [
            [cosine, -np.exp(1j * lam) * sine],
            [np.exp(1j * phi) * sine, np.exp(1j * (phi + lam)) * cosine],
        ],
        dtype=np.complex128,
    )


def _normalize_angle(theta: float) -> float:
    if not isinstance(theta, Real) or isinstance(theta, bool):
        raise TypeError("theta must be a real number")
    normalized = float(theta)
    if not np.isfinite(normalized):
        raise ValueError("theta must be finite")
    return normalized


__all__ = [
    "H",
    "I",
    "P",
    "RX",
    "RY",
    "RZ",
    "S",
    "T",
    "U1",
    "U2",
    "U3",
    "X",
    "Y",
    "Z",
]