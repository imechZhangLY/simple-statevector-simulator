# Test torch performance on converting numpy arrays to cuda tensors
from time import perf_counter

import torch
import numpy as np

REPEAT = 1000

for size in [100, 1000, 10000, 100000, 1000000]:
    # Create a numpy array of the given size
    np_array = np.random.rand(size).astype(np.float32)

    # Warm up CUDA initialization and the allocator outside the timed region.
    cuda_tensor = torch.from_numpy(np_array).cuda()
    torch.cuda.synchronize()

    # Measure host wall time and CUDA stream time over the same operations.
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    cpu_start_time = perf_counter()
    start_time.record()
    for _ in range(REPEAT):
        cpu_tensor = torch.from_numpy(np_array).pin_memory()  # pin memory for faster transfer
        cuda_tensor = cpu_tensor.cuda(non_blocking=True)
    end_time.record()
    cpu_elapsed_time_ms = (perf_counter() - cpu_start_time) * 1000

    # Synchronization is required before reading either completed duration.
    torch.cuda.synchronize()

    gpu_elapsed_time_ms = start_time.elapsed_time(end_time)
    cpu_avg_time_ms = cpu_elapsed_time_ms / REPEAT
    gpu_avg_time_ms = gpu_elapsed_time_ms / REPEAT
    host_overhead_ms = cpu_avg_time_ms - gpu_avg_time_ms

    print(
        f"Size: {size:>7}, CPU wall: {cpu_avg_time_ms:.6f} ms, "
        f"GPU: {gpu_avg_time_ms:.6f} ms, difference: {host_overhead_ms:.6f} ms"
    )