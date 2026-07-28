"""Post-processing helpers for simple-statevector-simulator runs.

Import these from a simulation program instead of hand-rolling CSV or JSON
formatting, so every run reports results the same way.

Add this directory to PYTHONPATH alongside ``src``:

    PowerShell: $env:PYTHONPATH = "$PWD\\src;$PWD\\.agents\\skills\\quantum-simulator\\scripts"
    bash:       export PYTHONPATH="$PWD/src:$PWD/.agents/skills/quantum-simulator/scripts"

Bit order follows the project convention: the statevector index is read as
|q[n-1]...q[1]q[0]>, so in the emitted bitstring the rightmost character is
qubit 0.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_RESULTS_DIRECTORY = Path("results")

__all__ = [
    "DEFAULT_RESULTS_DIRECTORY",
    "format_rows",
    "print_expectation",
    "to_bitstring",
    "write_amplitudes_json",
    "write_sampling_csv",
]


def to_bitstring(index: int, num_qubits: int) -> str:
    """Render a basis-state index as |q[n-1]...q[0]>, qubit 0 rightmost."""
    if index < 0 or index >= (1 << num_qubits):
        raise ValueError(
            f"index {index} is outside a {num_qubits}-qubit register"
        )
    return format(index, f"0{num_qubits}b")


def _prepare_path(path: Any, default_name: str) -> Path:
    resolved = DEFAULT_RESULTS_DIRECTORY / default_name if path is None else Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def write_sampling_csv(
    counts: Mapping[int, int],
    num_qubits: int,
    path: Any = None,
    top: int = 10,
) -> tuple[Path, list[tuple[str, int, int, float]]]:
    """Write sampling counts to CSV, ordered by count descending.

    Outcomes with a zero count are omitted. ``StateVector.sample()`` already
    returns only observed outcomes, but a merged or hand-built dictionary can
    still carry zeros.

    Ties are broken by ascending basis index so the file is deterministic.

    Returns the output path and the first ``top`` rows as
    ``(bitstring, index, count, probability)``.
    """
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")

    rows: list[tuple[str, int, int, float]] = []
    total = 0
    for index, count in counts.items():
        index = int(index)
        count = int(count)
        if count < 0:
            raise ValueError(f"count for index {index} is negative")
        if count == 0:
            continue
        total += count
        rows.append((to_bitstring(index, num_qubits), index, count, 0.0))

    if total == 0:
        raise ValueError("counts contain no non-zero outcome")

    rows = [
        (bitstring, index, count, count / total)
        for bitstring, index, count, _ in rows
    ]
    rows.sort(key=lambda row: (-row[2], row[1]))

    output = _prepare_path(path, "sampling.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bitstring", "index", "count", "probability"])
        for bitstring, index, count, probability in rows:
            writer.writerow([bitstring, index, count, repr(probability)])

    return output, rows[:top]


def format_rows(rows: Sequence[tuple[str, int, int, float]]) -> str:
    """Render sampling rows as a fixed-width table for the chat transcript."""
    if not rows:
        return "(no outcomes)"

    width = max(len(row[0]) for row in rows)
    width = max(width, len("bitstring"))
    lines = [f"{'bitstring':<{width}}  {'count':>8}  {'probability':>12}"]
    for bitstring, _, count, probability in rows:
        lines.append(f"{bitstring:<{width}}  {count:>8}  {probability:>12.6f}")
    return "\n".join(lines)


def print_expectation(value: float, label: str = "expectation") -> float:
    """Print an expectation value to stdout so the agent can relay it."""
    print(f"{label} = {float(value):.12f}")
    return float(value)


def write_amplitudes_json(
    amplitudes: Iterable[complex],
    path: Any = None,
    num_qubits: int | None = None,
    backend: str | None = None,
) -> Path:
    """Write the full amplitude vector to JSON.

    The file holds every one of the 2**n amplitudes, so it grows exponentially:
    roughly 100 bytes per entry means about 100 MB at 20 qubits. Prefer
    sampling or expectation values for large registers.
    """
    values = [complex(amplitude) for amplitude in amplitudes]
    dimension = len(values)
    inferred = dimension.bit_length() - 1
    if dimension == 0 or (1 << inferred) != dimension:
        raise ValueError("amplitude count must be a power of two")
    if num_qubits is None:
        num_qubits = inferred
    elif num_qubits != inferred:
        raise ValueError(
            f"{dimension} amplitudes describe {inferred} qubits, "
            f"but num_qubits={num_qubits} was given"
        )

    payload: dict[str, Any] = {
        "num_qubits": num_qubits,
        "dimension": dimension,
        "amplitudes": [
            {
                "index": index,
                "bitstring": to_bitstring(index, num_qubits),
                "real": value.real,
                "imag": value.imag,
                "probability": abs(value) ** 2,
            }
            for index, value in enumerate(values)
        ],
    }
    if backend is not None:
        payload["backend"] = backend

    output = _prepare_path(path, "amplitudes.json")
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return output
