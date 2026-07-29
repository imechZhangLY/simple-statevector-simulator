# test supda reshape performance
import torch
import torch_br
from time import perf_counter

REPEATS = 10

for num_qubits in [16, 20, 24, 28]:
    statevector = torch.zeros(
        1 << num_qubits, dtype=torch.complex64, device=torch.device("supa")
    )
    statevector[0] = 1.0
    qubits = [num_qubits - 1]
    target_axes = [num_qubits - 1 - qubit for qubit in qubits]
    remaining_axes = [
        axis for axis in range(num_qubits) if axis not in target_axes
    ]
    axes = target_axes + remaining_axes
    inverse_axes = [0] * num_qubits
    for position, axis in enumerate(axes):
        inverse_axes[axis] = position
    start_time = perf_counter()
    for i in range(REPEATS):
        updated = statevector.reshape((2, ) * num_qubits)
        updated = updated.permute(axes).reshape(2, -1)
        updated = updated.reshape((2,) * num_qubits).permute(inverse_axes).reshape(-1)
    
    cost_before_sync = (perf_counter() - start_time) / REPEATS * 1000
    torch.supa.synchronize()
    cost = (perf_counter() - start_time) / REPEATS * 1000

    print(f"{num_qubits} reshape cost {cost} ms, before sync {cost_before_sync} ms")


    