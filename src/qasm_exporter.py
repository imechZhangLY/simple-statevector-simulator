from circuit import Circuit
from operation import Operation

QASM_VERSION = "2.0"


def format_parameter(value: float) -> str:
    return repr(float(value))


def format_operation(operation: Operation, register: str) -> str:
    operands = ", ".join(f"{register}[{qubit}]" for qubit in operation.qubits)
    if not operation.parameters:
        return f"{operation.qasm_name} {operands};"

    arguments = ", ".join(
        format_parameter(parameter) for parameter in operation.parameters
    )
    return f"{operation.qasm_name}({arguments}) {operands};"


def export_qasm(
    circuit: Circuit,
    *,
    register: str = "q",
    measure_all: bool = False,
) -> str:
    if not isinstance(circuit, Circuit):
        raise TypeError("circuit must be a Circuit")

    lines = [
        f"OPENQASM {QASM_VERSION};",
        'include "qelib1.inc";',
        "",
        f"qreg {register}[{circuit.num_qubits}];",
    ]
    if measure_all:
        lines.append(f"creg c[{circuit.num_qubits}];")
    lines.append("")

    lines.extend(format_operation(operation, register) for operation in circuit)

    if measure_all:
        lines.append("")
        lines.append(f"measure {register} -> c;")

    return "\n".join(lines) + "\n"


__all__ = ["QASM_VERSION", "export_qasm", "format_operation", "format_parameter"]
