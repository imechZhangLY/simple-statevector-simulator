# test supda reshape performance
import torch
import torch_br
from time import perf_counter

REPEATS = 10

for num_qubits in [16, 20, 24, 28]:
    statevector = torch.zeros(1 << num_qubits, dtype="complex64", device="supa")
    statevector[0] = 1.0
    start_time = perf_counter()
    for i in range(REPEATS):
        updated = statevector.reshape((2, ) * num_qubits)
    
    cost_before_sync = (perf_counter() - start_time) / REPEATS * 1000
    torch.supa.synchronize()
    cost = (perf_counter() - start_time) / REPEATS * 1000

    print(f"{num_qubits} reshape cost {cost} ms, before sync {cost_before_sync} ms")


    