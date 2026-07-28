import ast
import math
import re
from dataclasses import dataclass

from circuit import Circuit
from single_qubit_gates import H, I, P, RX, RY, RZ, S, T, U1, U2, U3, X, Y, Z
from three_qubit_gates import CCX, CSWAP
from two_qubit_gates import CH, CP, CRX, CRY, CRZ, CX, CY, CZ, SWAP

GATE_TABLE = {
    "id": (I, 0, 1, False),
    "x": (X, 0, 1, False),
    "y": (Y, 0, 1, False),
    "z": (Z, 0, 1, False),
    "h": (H, 0, 1, False),
    "s": (S, 0, 1, False),
    "sdg": (S, 0, 1, True),
    "t": (T, 0, 1, False),
    "tdg": (T, 0, 1, True),
    "p": (P, 1, 1, False),
    "rx": (RX, 1, 1, False),
    "ry": (RY, 1, 1, False),
    "rz": (RZ, 1, 1, False),
    "u1": (U1, 1, 1, False),
    "u2": (U2, 2, 1, False),
    "u3": (U3, 3, 1, False),
    "cx": (CX, 0, 2, False),
    "cy": (CY, 0, 2, False),
    "cz": (CZ, 0, 2, False),
    "ch": (CH, 0, 2, False),
    "swap": (SWAP, 0, 2, False),
    "cp": (CP, 1, 2, False),
    "crx": (CRX, 1, 2, False),
    "cry": (CRY, 1, 2, False),
    "crz": (CRZ, 1, 2, False),
    "ccx": (CCX, 0, 3, False),
    "cswap": (CSWAP, 0, 3, False),
}

UNSUPPORTED_STATEMENTS = {
    "reset": "reset is not supported",
    "if": "classically controlled statements are not supported",
    "gate": "user defined gate declarations are not supported",
    "opaque": "opaque gate declarations are not supported",
}

ALLOWED_CONSTANTS = {"pi": math.pi}
ALLOWED_FUNCTIONS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "ln": math.log,
    "sqrt": math.sqrt,
}

IDENTIFIER = r"[A-Za-z][A-Za-z0-9_]*"


class QasmError(ValueError):
    pass


@dataclass(frozen=True)
class QasmProgram:
    circuit: Circuit
    measurements: tuple[tuple[int, int], ...]
    num_clbits: int


def evaluate_expression(text: str) -> float:
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as error:
        raise QasmError(f"invalid parameter expression: {text.strip()!r}") from error

    return _evaluate_node(tree.body, text)


def _evaluate_node(node: ast.AST, text: str) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise QasmError(f"invalid parameter expression: {text.strip()!r}")
        return float(node.value)

    if isinstance(node, ast.Name):
        if node.id in ALLOWED_CONSTANTS:
            return ALLOWED_CONSTANTS[node.id]
        raise QasmError(f"unknown identifier in expression: {node.id!r}")

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_node(node.operand, text)
        return value if isinstance(node.op, ast.UAdd) else -value

    if isinstance(node, ast.BinOp):
        left = _evaluate_node(node.left, text)
        right = _evaluate_node(node.right, text)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function = ALLOWED_FUNCTIONS.get(node.func.id)
        if function is None:
            raise QasmError(f"unknown function in expression: {node.func.id!r}")
        if len(node.args) != 1 or node.keywords:
            raise QasmError(f"{node.func.id} expects exactly one argument")
        return function(_evaluate_node(node.args[0], text))

    raise QasmError(f"unsupported parameter expression: {text.strip()!r}")


def _statements(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        comment = line.find("//")
        lines.append(line if comment < 0 else line[:comment])

    joined = " ".join(lines)
    return [statement.strip() for statement in joined.split(";") if statement.strip()]


def _split_arguments(text: str) -> list[str]:
    depth = 0
    current = ""
    arguments = []
    for character in text:
        if character == "," and depth == 0:
            arguments.append(current)
            current = ""
            continue
        if character in "([":
            depth += 1
        elif character in ")]":
            depth -= 1
        current += character
    arguments.append(current)
    return [argument.strip() for argument in arguments if argument.strip()]


class _Parser:
    def __init__(self) -> None:
        self._qreg_name: str | None = None
        self._qreg_size = 0
        self._creg_name: str | None = None
        self._creg_size = 0
        self._circuit: Circuit | None = None
        self._measurements: list[tuple[int, int]] = []

    def parse(self, text: str) -> QasmProgram:
        for statement in _statements(text):
            self._parse_statement(statement)

        if self._circuit is None:
            raise QasmError("program does not declare a quantum register")

        return QasmProgram(
            circuit=self._circuit,
            measurements=tuple(self._measurements),
            num_clbits=self._creg_size,
        )

    def _parse_statement(self, statement: str) -> None:
        keyword = statement.split("(")[0].split()[0].split("[")[0]

        if keyword == "OPENQASM":
            self._parse_version(statement)
            return
        if keyword == "include":
            return
        if keyword == "barrier":
            return
        if keyword in UNSUPPORTED_STATEMENTS:
            raise QasmError(UNSUPPORTED_STATEMENTS[keyword])
        if keyword == "qreg":
            self._parse_qreg(statement)
            return
        if keyword == "creg":
            self._parse_creg(statement)
            return
        if keyword == "measure":
            self._parse_measure(statement)
            return

        self._parse_gate(statement)

    def _parse_version(self, statement: str) -> None:
        version = statement.split(maxsplit=1)[1].strip()
        if not version.startswith("2"):
            raise QasmError(f"only OpenQASM 2.x is supported, but found {version!r}")

    def _parse_qreg(self, statement: str) -> None:
        if self._qreg_name is not None:
            raise QasmError("multiple quantum registers are not supported")

        match = re.fullmatch(rf"qreg\s+({IDENTIFIER})\s*\[\s*(\d+)\s*\]", statement)
        if match is None:
            raise QasmError(f"invalid qreg declaration: {statement!r}")

        self._qreg_name = match.group(1)
        self._qreg_size = int(match.group(2))
        self._circuit = Circuit(self._qreg_size)

    def _parse_creg(self, statement: str) -> None:
        if self._creg_name is not None:
            raise QasmError("multiple classical registers are not supported")

        match = re.fullmatch(rf"creg\s+({IDENTIFIER})\s*\[\s*(\d+)\s*\]", statement)
        if match is None:
            raise QasmError(f"invalid creg declaration: {statement!r}")

        self._creg_name = match.group(1)
        self._creg_size = int(match.group(2))

    def _parse_measure(self, statement: str) -> None:
        match = re.fullmatch(r"measure\s+(.+?)\s*->\s*(.+)", statement)
        if match is None:
            raise QasmError(f"invalid measure statement: {statement!r}")

        qubits = self._resolve_qubits(match.group(1))
        clbits = self._resolve_clbits(match.group(2))
        if len(qubits) != len(clbits):
            raise QasmError(
                "measure requires the same number of qubits and classical bits"
            )

        self._measurements.extend(zip(qubits, clbits))

    def _parse_gate(self, statement: str) -> None:
        match = re.fullmatch(
            rf"({IDENTIFIER})\s*(?:\(([^)]*)\))?\s+(.+)", statement, re.S
        )
        if match is None:
            raise QasmError(f"unsupported statement: {statement!r}")

        name = match.group(1)
        entry = GATE_TABLE.get(name)
        if entry is None:
            raise QasmError(
                f"unsupported gate {name!r}; "
                f"supported gates are {', '.join(sorted(GATE_TABLE))}"
            )
        if self._circuit is None:
            raise QasmError("a gate is applied before any qreg declaration")
        if self._measurements:
            raise QasmError(
                "mid-circuit measurement is not supported: "
                f"gate {name!r} appears after a measure statement"
            )

        factory, parameter_count, qubit_count, is_dagger = entry

        expressions = _split_arguments(match.group(2) or "")
        if len(expressions) != parameter_count:
            raise QasmError(
                f"gate {name!r} expects {parameter_count} parameters, "
                f"but received {len(expressions)}"
            )
        parameters = [evaluate_expression(expression) for expression in expressions]

        operands = [
            self._resolve_qubits(operand)
            for operand in _split_arguments(match.group(3))
        ]
        if len(operands) != qubit_count:
            raise QasmError(
                f"gate {name!r} expects {qubit_count} qubits, "
                f"but received {len(operands)}"
            )

        for qubits in self._broadcast(name, operands):
            operation = factory(*parameters, *qubits)
            self._circuit.append(operation.dagger() if is_dagger else operation)

    def _broadcast(self, name: str, operands: list[list[int]]) -> list[tuple[int, ...]]:
        if all(len(qubits) == 1 for qubits in operands):
            return [tuple(qubits[0] for qubits in operands)]

        if len(operands) != 1:
            raise QasmError(
                f"register broadcasting is only supported for single-qubit gates, "
                f"but {name!r} received a whole register"
            )
        return [(qubit,) for qubit in operands[0]]

    def _resolve_qubits(self, text: str) -> list[int]:
        return self._resolve(text, self._qreg_name, self._qreg_size, "quantum")

    def _resolve_clbits(self, text: str) -> list[int]:
        return self._resolve(text, self._creg_name, self._creg_size, "classical")

    @staticmethod
    def _resolve(text: str, name: str | None, size: int, kind: str) -> list[int]:
        text = text.strip()
        if name is None:
            raise QasmError(f"no {kind} register has been declared")

        indexed = re.fullmatch(rf"({IDENTIFIER})\s*\[\s*(\d+)\s*\]", text)
        if indexed is not None:
            if indexed.group(1) != name:
                raise QasmError(f"unknown {kind} register: {indexed.group(1)!r}")
            index = int(indexed.group(2))
            if index >= size:
                raise QasmError(
                    f"{kind} index {index} is outside register {name!r} of size {size}"
                )
            return [index]

        if re.fullmatch(IDENTIFIER, text):
            if text != name:
                raise QasmError(f"unknown {kind} register: {text!r}")
            return list(range(size))

        raise QasmError(f"invalid {kind} operand: {text!r}")


def parse_qasm(text: str) -> QasmProgram:
    return _Parser().parse(text)


__all__ = [
    "GATE_TABLE",
    "QasmError",
    "QasmProgram",
    "evaluate_expression",
    "parse_qasm",
]
