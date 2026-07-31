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


def synchronize() -> None:
    if args.device != "cpu":
        getattr(torch, args.device).synchronize()


def benchmark(function) -> tuple[float, torch.Tensor]:
    result = function()
    synchronize()

    start_time = perf_counter()
    for _ in range(REPEATS):
        result = function()
    synchronize()
    elapsed_ms = (perf_counter() - start_time) / REPEATS * 1000
    return elapsed_ms, result


for num_qubits in [16, 20, 24]:
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
    tensor_shape = (2,) * num_qubits

    measurements = {
        "reshape((2,) * num_qubits)": lambda: statevector.reshape(tensor_shape),
        "reshape(2, -1)": lambda: statevector.reshape(2, -1),
        "reshape(tensor).permute": lambda: statevector.reshape(
            tensor_shape
        ).permute(axes),
        "reshape(tensor).permute.reshape(2, -1)": lambda: statevector.reshape(
            tensor_shape
        ).permute(axes).reshape(2, -1),
    }

    print(f"\n{num_qubits} qubits")
    print(f"{'operation':<44}{'time (ms)':>12}{'shares storage':>17}")
    source_storage = statevector.untyped_storage().data_ptr()
    for label, function in measurements.items():
        elapsed_ms, result = benchmark(function)
        shares_storage = result.untyped_storage().data_ptr() == source_storage
        print(f"{label:<44}{elapsed_ms:>12.6f}{str(shares_storage):>17}")
