import argparse
from time import perf_counter

import numpy as np
import torch


parser = argparse.ArgumentParser(
    description="Compare reshape and as_strided for a one-qubit gate."
)
parser.add_argument(
    "--device",
    choices=("cpu", "cuda", "supa"),
    required=True,
)
parser.add_argument("--repeats", type=int, default=10)
args = parser.parse_args()

if args.device == "supa":
    import torch_br  # noqa: F401  registers torch.supa


def synchronize() -> None:
    if args.device != "cpu":
        getattr(torch, args.device).synchronize()


def to_device(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(args.device)


def benchmark(function) -> tuple[float, torch.Tensor]:
    result = function()
    synchronize()

    durations = []
    for _ in range(args.repeats):
        start = perf_counter()
        result = function()
        synchronize()
        durations.append((perf_counter() - start) * 1000)
    return min(durations), result


def evaluate(function) -> torch.Tensor:
    result = function()
    synchronize()
    host_result = result.to("cpu")
    synchronize()
    return host_result


generator = np.random.default_rng(7)

for num_qubits in (16, 20, 24):
    amplitudes = (
        generator.standard_normal(1 << num_qubits)
        + 1j * generator.standard_normal(1 << num_qubits)
    ).astype(np.complex64)
    matrix = (
        generator.standard_normal((2, 2))
        + 1j * generator.standard_normal((2, 2))
    ).astype(np.complex64)

    statevector = to_device(torch.from_numpy(amplitudes)).reshape(
        (2,) * num_qubits
    )
    matrix_tensor = to_device(torch.from_numpy(matrix))

    target_axis = num_qubits - 1
    axes = [target_axis] + [
        axis for axis in range(num_qubits) if axis != target_axis
    ]
    inverse_axes = [0] * num_qubits
    for position, axis in enumerate(axes):
        inverse_axes[axis] = position

    permuted = statevector.permute(axes)
    columns = 1 << (num_qubits - 1)

    def reshape_view() -> torch.Tensor:
        return permuted.reshape(2, -1)

    def strided_view() -> torch.Tensor:
        return torch.as_strided(
            permuted,
            size=(2, columns),
            stride=(permuted.stride(0), permuted.stride(-1)),
            storage_offset=permuted.storage_offset(),
        )

    def reshape_matmul() -> torch.Tensor:
        return matrix_tensor @ reshape_view()

    def strided_matmul() -> torch.Tensor:
        return matrix_tensor @ strided_view()

    def reshape_apply() -> torch.Tensor:
        updated = reshape_matmul()
        return updated.reshape((2,) * num_qubits).permute(inverse_axes)

    def strided_apply() -> torch.Tensor:
        updated = strided_matmul()
        return updated.reshape((2,) * num_qubits).permute(inverse_axes)

    reshape_view_ms, reshaped = benchmark(reshape_view)
    strided_view_ms, strided = benchmark(strided_view)
    reshape_apply_ms, _ = benchmark(reshape_apply)
    expected_host = evaluate(reshape_matmul)

    strided_apply_ms: float | None = None
    unsupported_reason: str | None = None
    try:
        actual_host = evaluate(strided_matmul)
        torch.testing.assert_close(
            actual_host,
            expected_host,
            rtol=1e-5,
            atol=1e-5,
        )
        strided_apply_ms, _ = benchmark(strided_apply)
    except RuntimeError as error:
        unsupported_reason = str(error).splitlines()[0]
    source_storage = statevector.untyped_storage().data_ptr()

    print(f"\n{num_qubits} qubits, target qubit 0")
    print(f"{'implementation':<24}{'time (ms)':>12}{'shares storage':>17}")
    print(
        f"{'reshape view':<24}{reshape_view_ms:>12.6f}"
        f"{str(reshaped.untyped_storage().data_ptr() == source_storage):>17}"
    )
    print(
        f"{'as_strided view':<24}{strided_view_ms:>12.6f}"
        f"{str(strided.untyped_storage().data_ptr() == source_storage):>17}"
    )
    print(f"{'reshape + matmul':<24}{reshape_apply_ms:>12.6f}{'-':>17}")
    if strided_apply_ms is None:
        print(f"{'as_strided + matmul':<24}{'unsupported':>12}{'-':>17}")
        print(f"reason: {unsupported_reason}")
    else:
        print(f"{'as_strided + matmul':<24}{strided_apply_ms:>12.6f}{'-':>17}")

    if strided_apply_ms is None and args.device == "supa":
        print("stopping: the SUPA matmul kernel does not support this strided view")
        break
