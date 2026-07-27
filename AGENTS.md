# AGENTS.md

## Project Overview

This repository implements a small pure-state quantum simulator in Python and NumPy. The simulation backend is a full statevector, and gates are applied through local unitary matrices.

Read [docs/architecture.md](docs/architecture.md) before changing mathematical conventions or module boundaries. That document is the source of truth for the architecture and planned layers.

## Current Status

Implemented:

- immutable `Gate` definitions with precomputed forward/dagger matrices;
- immutable `Operation` values that bind gates to ordered qubits;
- common one-, two-, and three-qubit gate functions;
- direction-aware metadata intended for future OpenQASM serialization;
- mutable `StateVector` with generic application of arbitrary local gates;
- a pluggable `Backend` protocol with `NumpyBackend` and `TorchBackend` implementations, covering CPU and CUDA;
- a bounded LRU cache of device matrices keyed by `Operation.matrix_key`;
- backend benchmarks under `benchmarks/`;
- `unittest` coverage for matrices, unitarity, dagger behavior, indexing, entanglement, backend conformance, and input validation.

Not implemented yet:

- `Circuit`;
- `StateVectorSimulator` execution layer;
- measurement, collapse, and sampling;
- OpenQASM parser/exporter;
- packaging metadata and an installable Python package;
- density matrices, noise, or alternate simulation backends.

Do not describe planned features as implemented.

## Repository Layout

```text
benchmarks/
  benchmark_backends.py
src/
  backend.py
  numpy_backend.py
  torch_backend.py
  gate.py
  operation.py
  single_qubit_gates.py
  two_qubit_gates.py
  three_qubit_gates.py
  statevector.py
tests/
  test_*.py
docs/
  architecture.md
```

The current source layout is intentionally flat. Imports look like:

```python
from gate import Gate
from operation import Operation
from statevector import StateVector
from numpy_backend import NumpyBackend
```

Do not migrate to a nested package or add packaging infrastructure unless the task explicitly calls for it.

## Environment and Validation

The project uses two Git-ignored virtual environments whose definitions are committed. Never run project code with the system, conda, or any other interpreter.

| Environment | Purpose | Requirements file |
|---|---|---|
| `.venv` | primary environment, CUDA PyTorch | [requirements-torch-cuda.txt](requirements-torch-cuda.txt) |
| `.venv-cpu` | CPU-only PyTorch comparison | [requirements-torch-cpu.txt](requirements-torch-cpu.txt) |

Both use Python 3.10 and install NumPy from [requirements.txt](requirements.txt).

### Check the environments before debugging

Before running, debugging, or profiling anything, verify both interpreters exist:

```powershell
Test-Path .\.venv\Scripts\python.exe
Test-Path .\.venv-cpu\Scripts\python.exe
```

If either returns `False`, provision it. The bootstrap script creates the environment when it is missing and reinstalls only when the requirements file changed, so it is safe to run every time:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\.vscode\bootstrap-env.ps1 -EnvironmentName .venv -Requirements requirements-torch-cuda.txt
powershell -NoProfile -ExecutionPolicy Bypass -File .\.vscode\bootstrap-env.ps1 -EnvironmentName .venv-cpu -Requirements requirements-torch-cpu.txt
```

Never install packages into an environment ad hoc. Add the dependency to the correct requirements file and re-run the bootstrap script, so the environment stays reproducible from the repository.

### Run tests

Tests use the standard library `unittest`; pytest is not required. Always call the environment interpreter by path instead of a bare `python`:

```powershell
$env:PYTHONPATH = Join-Path $PWD 'src'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Run a focused test module with:

```powershell
$env:PYTHONPATH = Join-Path $PWD 'src'
.\.venv\Scripts\python.exe -m unittest tests.test_statevector -v
```

After every behavioral change, run the narrowest relevant tests first and then the full suite in `.venv`. Any change touching a backend, `StateVector`, or gate matrices must also pass the full suite in `.venv-cpu`:

```powershell
$env:PYTHONPATH = Join-Path $PWD 'src'
.\.venv-cpu\Scripts\python.exe -m unittest discover -s tests -v
```

Use `numpy.testing.assert_allclose` with an explicit absolute tolerance when expected values include mathematical zeros produced by trigonometric functions.

### Benchmarks

Performance claims must be measured, not assumed. [benchmarks/benchmark_backends.py](benchmarks/benchmark_backends.py) adds `src` to `sys.path` itself, so it runs without `PYTHONPATH`:

```powershell
.\.venv\Scripts\python.exe benchmarks\benchmark_backends.py --qubits 16,20,22
```

Benchmarks are not part of the test suite and must stay out of `tests/`. When adding one, keep the existing methodology:

- call `torch.cuda.synchronize()` before reading the clock, or GPU timings measure only kernel launches;
- run an untimed warm-up pass so CUDA context creation and cache population are excluded;
- report the minimum across repeats, not the mean;
- construct gate objects outside the timed region unless construction is what you are measuring;
- make cache hits and misses deterministic by controlling cache capacity. Reusing one operation list across repeats silently turns every workload into a cache hit.

### GPU notes

The reference machine has a Quadro P620: Pascal, compute capability 6.1, 2 GiB VRAM. The installed CUDA build ships `sm_61` kernels and executes both `complex64` and `complex128`, but Pascal runs FP64 at 1/32 rate, so GPU work should default to `complex64`. Two GiB of VRAM limits GPU statevectors to roughly 24 qubits once temporaries are counted.

Never import torch at module import time in `src/`, and never assume CUDA is present. Torch-dependent tests must skip cleanly when torch or a CUDA device is unavailable, so the suite still passes in a NumPy-only environment.

## Non-Negotiable Mathematical Conventions

### Global Indexing

The global statevector uses little-endian qubit indexing:

- qubit 0 is the least significant bit of a statevector index;
- index `i` represents the basis state `|q[n-1]...q[1]q[0]>`.

For two qubits:

```text
index 0 = |00>
index 1 = |01>
index 2 = |10>
index 3 = |11>
```

Do not change this convention without updating all gate-application tests and the architecture document.

### Local Gate Basis

A local gate matrix follows the order of `operation.qubits`, with the first listed qubit acting as the most significant local bit.

Examples:

- `CX(control, target)` uses local basis `|control,target>`;
- `CCX(control1, control2, target)` uses local basis `|control1,control2,target>`;
- `CSWAP(control, target1, target2)` uses local basis `|control,target1,target2>`.

The `StateVector.apply()` axis permutation reconciles this local convention with global little-endian indexing.

### Entanglement

Gate application must always operate on the complete amplitude tensor. The backend's temporary matrix has shape `2**k x 2**(n-k)` and still contains all `2**n` amplitudes. It does not extract an independent pure state for the target qubits.

Any rewrite of the application algorithm, in any backend, must retain tests for:

- applying a local gate to a Bell state;
- non-adjacent qubits such as `CX(2, 0)`;
- a three-qubit operation;
- dagger followed by the original direction restoring the state.

## Core Ownership Boundaries

### Gate

`Gate` owns the mathematical and serialization definition of a bound gate:

- internal/display name;
- OpenQASM name;
- qubit arity;
- forward and dagger parameters;
- forward and dagger matrices.

Parameters belong to `Gate`, not `Operation`, because they determine the matrix. `Gate` copies matrices to `numpy.complex128`, validates shape/finite values, and makes them read-only.

Constant gates should use one module-level shared `Gate` object. Parameterized gate functions may construct a bound `Gate` per call.

Do not make `Gate` import `Operation`. User-facing gate functions construct operations.

### Operation

`Operation` owns execution placement and direction:

- a `Gate` reference;
- an ordered tuple of unique, non-negative qubits;
- `is_dagger`.

Direction-aware `name`, `qasm_name`, `parameters`, and `matrix` are delegated to the gate metadata. `Operation.dagger()` only toggles direction and must not calculate matrices or infer serialization rules.

### StateVector

`StateVector` owns quantum semantics and validation, not numerics. It currently:

- initializes to `|0...0>`;
- validates optional normalized amplitudes;
- exposes a read-only `numpy.complex128` amplitude view and a probability vector;
- checks that operation qubits are inside the register;
- delegates gate application to its backend and returns `self`;
- supports independent copying that preserves the backend.

`amplitudes` must always return read-only `numpy.complex128`, regardless of backend. Backend-native arrays are exposed separately through `raw_amplitudes`.

### Backend

`Backend` is a `Protocol` in [src/backend.py](src/backend.py) that owns all numerical work: state creation, conversion, finiteness, squared norm, gate application, probabilities, copying, and NumPy conversion.

Rules:

- backends must contain no quantum semantics; endianness, qubit bounds, and normalization checks stay in `StateVector`;
- device and dtype are backend constructor parameters, never new `StateVector` subclasses;
- backends implement the full `apply` algorithm, not thin array-namespace forwarding, so each library can use its own optimal execution path;
- every backend must produce results identical to `NumpyBackend` within tolerance;
- new backends must be added to `available_backends()` in [tests/test_backend.py](tests/test_backend.py);
- optional dependencies such as torch must be imported lazily and must not be added to [requirements.txt](requirements.txt).

Do not add gate-specific execution branches to any backend's `apply()`. The generic tensor-axis algorithm must handle every gate arity unless profiling demonstrates a justified optimization.

## Gate Function Conventions

Gate functions return `Operation` directly:

```python
H(qubit)
RX(theta, qubit)
CX(control, target)
CRX(theta, control, target)
CCX(control1, control2, target)
```

Keep continuous parameters before qubit arguments. Preserve control/target order in `Operation.qubits`.

Aliases must share the same `Gate` object and serialize to the canonical OpenQASM name:

- `CNOT` aliases `CX` and serializes as `cx`;
- `TOFFOLI` aliases `CCX` and serializes as `ccx`;
- `FREDKIN` aliases `CSWAP` and serializes as `cswap`.

When adding a gate:

1. confirm its exact matrix under the repository's local basis order;
2. define both forward and dagger matrices;
3. define forward and dagger serialization names/parameters;
4. validate all angles as finite real numbers;
5. export the function through the module's `__all__`;
6. test the matrix, unitarity, dagger metadata, qubit order, and invalid inputs.

Do not reconstruct serialization parameters from floating-point matrices. Preserve them structurally in `Gate`.

## Dagger Rules

Dagger data is precomputed and stored. Self-adjoint constant gates may reuse the same matrix object for both directions.

Do not assume dagger always means negating every parameter or adding a name suffix. Examples already encoded by the project include:

- `S` -> `sdg`;
- `T` -> `tdg`;
- `RX(theta)` -> `RX(-theta)`;
- `U3(theta, phi, lambda)` -> `U3(-theta, -lambda, -phi)`.

For a future `Circuit.dagger()`, reverse operation order and dagger every operation.

## Serialization Boundary

Core objects expose structured metadata but do not emit OpenQASM strings. A future exporter should consume:

```python
operation.qasm_name
operation.parameters
operation.qubits
```

Keep OpenQASM version syntax, register declarations, parameter formatting, and complete circuit output in a separate exporter module. Measurement should not be modeled as a unitary `Gate`.

## Coding Style

- Follow the existing concise Python style and type annotations.
- Use ASCII in source code unless mathematical documentation clearly benefits from Unicode.
- Keep public APIs explicit with `__all__` in gate modules.
- Prefer immutable dataclasses for value objects and defensive copies for input arrays.
- Use `numpy.complex128` consistently until configurable simulator precision is deliberately introduced.
- Avoid unrelated refactors when adding behavior.
- Add no gate-specific comments unless the matrix ordering would otherwise be ambiguous.

## Test Expectations

Every new mathematical behavior needs a focused regression test. Favor semantic checks over implementation details:

- expected basis-state mappings;
- exact matrices for permutation gates;
- approximate matrices for trigonometric/phase gates;
- `U.conj().T @ U == I`;
- correct OpenQASM metadata in both directions;
- correct behavior on entangled states and non-adjacent qubits;
- identical results across every registered backend;
- rejection of malformed dimensions, duplicate qubits, invalid indices, and non-finite parameters.

Do not weaken numeric tolerances broadly to hide a formula or ordering error. Reduced-precision backends may use a looser tolerance, but only in the shared backend conformance tests.

## Recommended Next Work

Unless the user directs otherwise, the next architectural step is `Circuit`:

1. immutable or carefully encapsulated operation storage;
2. circuit-level qubit-bound validation;
3. fluent `append()`;
4. correct circuit dagger behavior;
5. tests before adding a separate `StateVectorSimulator` runner.

After Circuit, proceed to the execution layer, measurement/sampling, and then OpenQASM export. Keep these concerns in separate modules.