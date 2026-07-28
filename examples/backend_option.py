import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from numpy_backend import NumpyBackend  # noqa: E402

BACKEND_CHOICES = (
    "numpy:complex128",
    "numpy:complex64",
    "torch:cpu:complex128",
    "torch:cpu:complex64",
    "torch:cuda:complex128",
    "torch:cuda:complex64",
)


def create_backend(name: str):
    library, _, remainder = name.partition(":")

    if library == "numpy":
        dtype = np.complex64 if remainder == "complex64" else np.complex128
        return NumpyBackend(dtype=dtype)

    if library == "torch":
        device, _, dtype_name = remainder.partition(":")
        from torch_backend import TorchBackend

        return TorchBackend(device=device, dtype=dtype_name)

    raise ValueError(f"unknown backend: {name}")


def add_backend_argument(parser) -> None:
    parser.add_argument(
        "--backend",
        default="numpy:complex128",
        choices=BACKEND_CHOICES,
        help="computation backend (default: numpy:complex128)",
    )


__all__ = ["BACKEND_CHOICES", "add_backend_argument", "create_backend"]
