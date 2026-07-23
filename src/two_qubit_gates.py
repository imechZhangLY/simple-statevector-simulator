from numbers import Real

import numpy as np

from gate import Gate
from operation import Operation


_I_MATRIX = np.eye(2, dtype=np.complex128)
_X_MATRIX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y_MATRIX = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z_MATRIX = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_H_MATRIX = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)

_CX_MATRIX = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    dtype=np.complex128,
)
_CY_MATRIX = np.block(
    [[_I_MATRIX, np.zeros((2, 2))], [np.zeros((2, 2)), _Y_MATRIX]]
).astype(np.complex128)
_CZ_MATRIX = np.diag([1, 1, 1, -1]).astype(np.complex128)
_CH_MATRIX = np.block(
    [[_I_MATRIX, np.zeros((2, 2))], [np.zeros((2, 2)), _H_MATRIX]]
).astype(np.complex128)
_SWAP_MATRIX = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    dtype=np.complex128,
)

_CX_GATE = Gate("CX", 2, _CX_MATRIX, _CX_MATRIX, qasm_name="cx")
_CY_GATE = Gate("CY", 2, _CY_MATRIX, _CY_MATRIX, qasm_name="cy")
_CZ_GATE = Gate("CZ", 2, _CZ_MATRIX, _CZ_MATRIX, qasm_name="cz")
_CH_GATE = Gate("CH", 2, _CH_MATRIX, _CH_MATRIX, qasm_name="ch")
_SWAP_GATE = Gate("SWAP", 2, _SWAP_MATRIX, _SWAP_MATRIX, qasm_name="swap")


def CX(control: int, target: int) -> Operation:
    return Operation(_CX_GATE, (control, target))


def CNOT(control: int, target: int) -> Operation:
    return CX(control, target)


def CY(control: int, target: int) -> Operation:
    return Operation(_CY_GATE, (control, target))


def CZ(control: int, target: int) -> Operation:
    return Operation(_CZ_GATE, (control, target))


def CH(control: int, target: int) -> Operation:
    return Operation(_CH_GATE, (control, target))


def SWAP(first: int, second: int) -> Operation:
    return Operation(_SWAP_GATE, (first, second))


def CP(theta: float, control: int, target: int) -> Operation:
    theta = _normalize_angle(theta)
    matrix = np.diag([1, 1, 1, np.exp(1j * theta)]).astype(np.complex128)
    dagger_matrix = np.diag([1, 1, 1, np.exp(-1j * theta)]).astype(
        np.complex128
    )
    gate = Gate(
        "CP",
        2,
        matrix,
        dagger_matrix,
        qasm_name="cp",
        parameters=(theta,),
        dagger_parameters=(-theta,),
    )
    return Operation(gate, (control, target))


def CRX(theta: float, control: int, target: int) -> Operation:
    theta = _normalize_angle(theta)
    cosine = np.cos(theta / 2)
    sine = np.sin(theta / 2)
    target_matrix = np.array(
        [[cosine, -1j * sine], [-1j * sine, cosine]], dtype=np.complex128
    )
    return _controlled_rotation("CRX", "crx", theta, target_matrix, control, target)


def CRY(theta: float, control: int, target: int) -> Operation:
    theta = _normalize_angle(theta)
    cosine = np.cos(theta / 2)
    sine = np.sin(theta / 2)
    target_matrix = np.array([[cosine, -sine], [sine, cosine]])
    return _controlled_rotation("CRY", "cry", theta, target_matrix, control, target)


def CRZ(theta: float, control: int, target: int) -> Operation:
    theta = _normalize_angle(theta)
    target_matrix = np.diag(
        [np.exp(-1j * theta / 2), np.exp(1j * theta / 2)]
    )
    return _controlled_rotation("CRZ", "crz", theta, target_matrix, control, target)


def _controlled_rotation(
    name: str,
    qasm_name: str,
    theta: float,
    target_matrix: np.ndarray,
    control: int,
    target: int,
) -> Operation:
    matrix = _controlled_matrix(target_matrix)
    gate = Gate(
        name,
        2,
        matrix,
        matrix.conj().T,
        qasm_name=qasm_name,
        parameters=(theta,),
        dagger_parameters=(-theta,),
    )
    return Operation(gate, (control, target))


def _controlled_matrix(target_matrix: np.ndarray) -> np.ndarray:
    zero_matrix = np.zeros((2, 2), dtype=np.complex128)
    return np.block([[_I_MATRIX, zero_matrix], [zero_matrix, target_matrix]]).astype(
        np.complex128
    )


def _normalize_angle(theta: float) -> float:
    if not isinstance(theta, Real) or isinstance(theta, bool):
        raise TypeError("theta must be a real number")
    normalized = float(theta)
    if not np.isfinite(normalized):
        raise ValueError("theta must be finite")
    return normalized


__all__ = ["CH", "CNOT", "CP", "CRX", "CRY", "CRZ", "CX", "CY", "CZ", "SWAP"]