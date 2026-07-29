# Test torch performance on converting numpy arrays to cuda tensors
from time import perf_counter

import torch
import torch_br
import numpy as np

REPEAT = 1000

for size in [2, 4, 8]:
    # Create a numpy array of the given size
    np_array = np.random.rand(size, size).astype(np.complex128)

    # Warm up SUPA initialization and the allocator outside the timed region.
    cuda_tensor = torch.from_numpy(np_array).supa()
    torch.supa.synchronize()

    # Measure host wall time and CUDA stream time over the same operations.
    start_time = torch.supa.Event(enable_timing=True)
    end_time = torch.supa.Event(enable_timing=True)

    cpu_start_time = perf_counter()
    start_time.record()
    for _ in range(REPEAT):
        tensor = torch.from_numpy(np_array).supa(non_blocking=False)
    end_time.record()
    cpu_elapsed_time_ms = (perf_counter() - cpu_start_time) * 1000

    # Synchronization is required before reading either completed duration.
    torch.supa.synchronize()

    gpu_elapsed_time_ms = start_time.elapsed_time(end_time)
    cpu_avg_time_ms = cpu_elapsed_time_ms / REPEAT
    gpu_avg_time_ms = gpu_elapsed_time_ms / REPEAT
    host_overhead_ms = cpu_avg_time_ms - gpu_avg_time_ms

    print(
        f"Size: {size:>7}, CPU wall: {cpu_avg_time_ms:.6f} ms, "
        f"GPU: {gpu_avg_time_ms:.6f} ms, difference: {host_overhead_ms:.6f} ms"
    )