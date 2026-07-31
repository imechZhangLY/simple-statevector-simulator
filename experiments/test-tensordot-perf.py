import argparse
from time import perf_counter

import numpy as np
import torch


parser = argparse.ArgumentParser(
    description="Compare reshape plus matmul with a multi-axis tensordot."
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


def make_axes(
    num_qubits: int,
    qubits: tuple[int, ...],
) -> tuple[list[int], list[int], list[int]]:
    target_axes = [num_qubits - 1 - qubit for qubit in qubits]
    remaining_axes = [
        axis for axis in range(num_qubits) if axis not in target_axes
    ]
    axes = target_axes + remaining_axes
    inverse_axes = [0] * num_qubits
    for position, axis in enumerate(axes):
        inverse_axes[axis] = position
    return target_axes, remaining_axes, inverse_axes


generator = np.random.default_rng(7)

for num_qubits in (16, 20, 24):
    amplitudes = (
        generator.standard_normal(1 << num_qubits)
        + 1j * generator.standard_normal(1 << num_qubits)
    ).astype(np.complex64)
    statevector = to_device(torch.from_numpy(amplitudes)).reshape(
        (2,) * num_qubits
    )

    workloads = (
        ("1q", (0,)),
        ("2q non-adjacent", (num_qubits - 1, 0)),
        ("3q non-adjacent", (num_qubits - 1, num_qubits // 2, 0)),
    )

    print(f"\n{num_qubits} qubits")
    print(f"{'workload':<20}{'reshape + matmul':>20}{'tensordot':>15}")

    for label, qubits in workloads:
        target_axes, remaining_axes, inverse_axes = make_axes(num_qubits, qubits)
        width = len(target_axes)
        dimension = 1 << width
        matrix = (
            generator.standard_normal((dimension, dimension))
            + 1j * generator.standard_normal((dimension, dimension))
        ).astype(np.complex64)
        matrix_tensor = to_device(torch.from_numpy(matrix))
        matrix_axes = matrix_tensor.reshape((2,) * (2 * width))

        axes = target_axes + remaining_axes

        def reshape_matmul_output() -> torch.Tensor:
            batched = statevector.permute(axes).reshape(dimension, -1)
            return matrix_tensor @ batched

        def reshape_matmul() -> torch.Tensor:
            updated = reshape_matmul_output()
            return updated.reshape((2,) * num_qubits).permute(inverse_axes)

        output_axes = []
        for axis in range(num_qubits):
            if axis in target_axes:
                output_axes.append(target_axes.index(axis))
            else:
                output_axes.append(width + remaining_axes.index(axis))

        def contract_output() -> torch.Tensor:
            return torch.tensordot(
                matrix_axes,
                statevector,
                dims=(list(range(width, 2 * width)), target_axes),
            )

        def contract() -> torch.Tensor:
            updated = contract_output()
            return updated.permute(output_axes)

        reshape_ms, _ = benchmark(reshape_matmul)
        expected_host = evaluate(reshape_matmul_output)
        try:
            actual_host = evaluate(contract_output).reshape(dimension, -1)
            torch.testing.assert_close(
                actual_host,
                expected_host,
                rtol=1e-5,
                atol=1e-5,
            )
            contract_ms, _ = benchmark(contract)
            contract_result = f"{contract_ms:.6f}"
        except RuntimeError as error:
            contract_result = "unsupported"
            print(f"  {label} reason: {str(error).splitlines()[0]}")

        print(f"{label:<20}{reshape_ms:>20.6f}{contract_result:>15}")