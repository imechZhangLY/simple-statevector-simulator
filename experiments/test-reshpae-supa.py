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
parser.add_argument("--log-copy", action="store_true")
args = parser.parse_args()

if args.device == "supa":
    import torch_br

REPEATS = 10


def log_tensor(label, tensor, previous):
    storage_ptr = tensor.untyped_storage().data_ptr()
    previous_storage_ptr = previous.untyped_storage().data_ptr()
    print(
        f"  {label}: copied={storage_ptr != previous_storage_ptr}, "
        f"shape={tuple(tensor.shape)}, stride={tensor.stride()}, "
        f"contiguous={tensor.is_contiguous()}, storage_ptr={storage_ptr}"
    )


for num_qubits in [16, 20, 24, 28]:
    arr = numpy.random.rand(1 << num_qubits).astype(numpy.complex64)
    statevector = torch.from_numpy(arr)
    if args.device == "cuda":
        statevector = statevector.cuda()
    elif args.device == "supa":
        statevector = statevector.supa()

    qubits = [0]
    target_axes = [num_qubits - 1 - qubit for qubit in qubits]
    remaining_axes = [
        axis for axis in range(num_qubits) if axis not in target_axes
    ]
    axes = target_axes + remaining_axes
    inverse_axes = [0] * num_qubits
    for position, axis in enumerate(axes):
        inverse_axes[axis] = position

    if args.log_copy:
        print(f"{num_qubits} qubits:")
        reshaped = statevector.reshape((2,) * num_qubits)
        log_tensor("reshape to tensor", reshaped, statevector)
        permuted = reshaped.permute(axes)
        log_tensor("permute target axes", permuted, reshaped)
        flattened = permuted.reshape(2, -1)
        log_tensor("reshape to matrix", flattened, permuted)
        restored = flattened.reshape((2,) * num_qubits)
        log_tensor("reshape from matrix", restored, flattened)
        inverse_permuted = restored.permute(inverse_axes)
        log_tensor("permute inverse axes", inverse_permuted, restored)
        final = inverse_permuted.reshape(-1)
        log_tensor("reshape to vector", final, inverse_permuted)

    start_time = perf_counter()
    for i in range(REPEATS):
        updated = statevector.reshape((2, ) * num_qubits)
        updated = updated.permute(axes).reshape(2, -1)
        updated = updated.reshape((2,) * num_qubits).permute(inverse_axes).reshape(-1)
    
    cost_before_sync = (perf_counter() - start_time) / REPEATS * 1000
    if args.device != "cpu":
        getattr(torch, args.device).synchronize()
    cost = (perf_counter() - start_time) / REPEATS * 1000

    print(f"{num_qubits} reshape cost {cost} ms, before sync {cost_before_sync} ms")


    