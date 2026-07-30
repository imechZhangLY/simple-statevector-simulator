import argparse
from time import perf_counter

import numpy
import torch


parser = argparse.ArgumentParser(description="Test tensor reshape performance.")
parser.add_argument(
    "--device",
    choices=("cpu", "cuda", "supa"),
    required=True,
)
args = parser.parse_args()

if args.device == "supa":
    import torch_br

REPEATS = 10

for num_qubits in [16, 20, 24]:
    arr = numpy.random.rand(1 << num_qubits).astype(numpy.complex64)
    matrix = numpy.random.rand(2, 2).astype(numpy.complex64)
    statevector = torch.from_numpy(arr)
    matrix_tensor = torch.from_numpy(matrix)
    if args.device == "cuda":
        statevector = statevector.cuda()
        matrix_tensor = matrix_tensor.cuda()
    elif args.device == "supa":
        statevector = statevector.supa()
        matrix_tensor = matrix_tensor.supa()

    qubits = [0]
    target_axes = [num_qubits - 1 - qubit for qubit in qubits]
    remaining_axes = [
        axis for axis in range(num_qubits) if axis not in target_axes
    ]
    axes = target_axes + remaining_axes
    inverse_axes = [0] * num_qubits
    for position, axis in enumerate(axes):
        inverse_axes[axis] = position

    # 冷启动一次
    updated = statevector.reshape((2, ) * num_qubits)
    updated = updated.permute(axes).reshape(2, -1)
    updated = matrix_tensor @ updated
    updated = updated.reshape((2,) * num_qubits).permute(inverse_axes).reshape(-1)

    start_time = perf_counter()
    for i in range(REPEATS):
        updated = statevector.reshape((2, ) * num_qubits)
        updated = updated.permute(axes).reshape(2, -1)
        updated = matrix_tensor @ updated
        updated = updated.reshape((2,) * num_qubits)
    cost_before_sync = (perf_counter() - start_time) / REPEATS * 1000
    if args.device != "cpu":
        getattr(torch, args.device).synchronize()
    cost = (perf_counter() - start_time) / REPEATS * 1000

    print(f"{num_qubits} reshape cost {cost} ms, before sync {cost_before_sync} ms")
