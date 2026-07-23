import numpy as np

from gate import Gate
from operation import Operation


_CCX_MATRIX = np.eye(8, dtype=np.complex128)
_CCX_MATRIX[[6, 7]] = _CCX_MATRIX[[7, 6]]

_CSWAP_MATRIX = np.eye(8, dtype=np.complex128)
_CSWAP_MATRIX[[5, 6]] = _CSWAP_MATRIX[[6, 5]]

_CCX_GATE = Gate("CCX", 3, _CCX_MATRIX, _CCX_MATRIX, qasm_name="ccx")
_CSWAP_GATE = Gate(
    "CSWAP", 3, _CSWAP_MATRIX, _CSWAP_MATRIX, qasm_name="cswap"
)


def CCX(control1: int, control2: int, target: int) -> Operation:
    return Operation(_CCX_GATE, (control1, control2, target))


def TOFFOLI(control1: int, control2: int, target: int) -> Operation:
    return CCX(control1, control2, target)


def CSWAP(control: int, target1: int, target2: int) -> Operation:
    return Operation(_CSWAP_GATE, (control, target1, target2))


def FREDKIN(control: int, target1: int, target2: int) -> Operation:
    return CSWAP(control, target1, target2)


__all__ = ["CCX", "CSWAP", "FREDKIN", "TOFFOLI"]