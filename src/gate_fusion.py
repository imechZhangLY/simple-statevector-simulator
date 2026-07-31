from itertools import count
from weakref import WeakKeyDictionary

import numpy as np

from circuit import Circuit
from gate import ComplexMatrix, Gate
from operation import Operation

_standalone_fused_gate_ids = count(1)
_circuit_ids = count(1)
_circuit_namespaces: WeakKeyDictionary[Circuit, int] = WeakKeyDictionary()


def _expand_matrix(
    operation: Operation,
    target_qubits: tuple[int, ...],
) -> ComplexMatrix:
    if not set(target_qubits).issuperset(operation.qubits):
        raise ValueError("target_qubits must contain all operation qubits")

    num_target_qubits = len(target_qubits)
    dimension = 1 << num_target_qubits
    expanded = np.zeros((dimension, dimension), dtype=np.complex128)
    positions = tuple(target_qubits.index(qubit) for qubit in operation.qubits)

    for input_basis in range(dimension):
        local_input = 0
        for position in positions:
            local_input = (local_input << 1) | (
                (input_basis >> (num_target_qubits - 1 - position)) & 1
            )

        for local_output in range(1 << len(operation.qubits)):
            output_basis = input_basis
            for local_position, target_position in enumerate(positions):
                bit_position = num_target_qubits - 1 - target_position
                output_bit = (
                    local_output >> (len(operation.qubits) - 1 - local_position)
                ) & 1
                output_basis = (output_basis & ~(1 << bit_position)) | (
                    output_bit << bit_position
                )
            expanded[output_basis, input_basis] = operation.matrix[
                local_output, local_input
            ]

    return expanded


def fuse_operations(
    first: Operation,
    second: Operation,
    *,
    name: str | None = None,
) -> Operation:
    first_qubits = set(first.qubits)
    second_qubits = set(second.qubits)
    if not (
        first_qubits.issuperset(second_qubits)
        or second_qubits.issuperset(first_qubits)
    ):
        raise ValueError("fused operation must not use more qubits than its inputs")

    target_qubits = (
        first.qubits if len(first.qubits) >= len(second.qubits) else second.qubits
    )
    matrix = (
        _expand_matrix(second, target_qubits)
        @ _expand_matrix(first, target_qubits)
    )
    if name is None:
        name = f"FUSED_{next(_standalone_fused_gate_ids)}"
    gate = Gate(
        name,
        len(target_qubits),
        matrix,
        matrix.conj().T,
        qasm_name=name,
    )
    return Operation(gate, target_qubits)


def fuse_circuit(circuit: Circuit) -> Circuit:
    operations_mat = [list(row) for row in circuit.operations_mat]
    namespace = _circuit_namespaces.get(circuit)
    if namespace is None:
        namespace = next(_circuit_ids)
        _circuit_namespaces[circuit] = namespace
    fused_gate_ids = count(1)

    for operations in operations_mat:
        previous: tuple[int, Operation] | None = None
        for column, operation in enumerate(operations):
            if operation is None:
                continue

            if previous is None:
                previous = (column, operation)
                continue

            previous_column, previous_operation = previous
            try:
                fused_operation = fuse_operations(
                    previous_operation,
                    operation,
                    name=f"FUSED_{namespace}_{next(fused_gate_ids)}",
                )
            except ValueError:
                previous = (column, operation)
                continue

            previous_qubits = set(previous_operation.qubits)
            operation_qubits = set(operation.qubits)
            anchor_column = (
                previous_column
                if previous_qubits.issuperset(operation_qubits)
                else column
            )

            for target_qubit in previous_operation.qubits:
                operations_mat[target_qubit][previous_column] = None
            for target_qubit in operation.qubits:
                operations_mat[target_qubit][column] = None
            for target_qubit in fused_operation.qubits:
                operations_mat[target_qubit][anchor_column] = fused_operation

            previous = (anchor_column, fused_operation)

    fused = Circuit(circuit.num_qubits)
    depth = max((len(row) for row in operations_mat), default=0)
    for column in range(depth):
        appended: set[int] = set()
        for row in operations_mat:
            operation = row[column] if column < len(row) else None
            if operation is None or id(operation) in appended:
                continue
            fused.append(operation)
            appended.add(id(operation))
    return fused


__all__ = ["fuse_circuit", "fuse_operations"]