---
name: quantum-simulator
description: 'Run quantum circuits on this repository''s statevector simulator. Use when asked to build, simulate, or execute a quantum circuit, apply gates, sample measurement outcomes or shots, compute an expectation value of an observable, inspect the amplitude vector or state vector, set up the .venv/.venv-cpu environment, or export and parse OpenQASM. Covers environment bootstrap (CUDA/CPU/macOS aware), writing programs against the flat src/ API, and reporting results as CSV, JSON, or chat output.'
argument-hint: 'describe the circuit and whether you want sampling, an expectation value, or amplitudes'
---

# Quantum Simulator Workflow

Three stages: provision the environment, write the program, run it and report the results.

Never run project code with the system, conda, or any other interpreter. Only the
repository virtual environments are valid.

## Stage 1 — Provision the environment

Run the dispatcher with any interpreter. It detects the platform and whether an
NVIDIA driver is present, then calls the matching bootstrap script with the
matching requirements file:

```powershell
python .agents/skills/quantum-simulator/scripts/setup_environment.py
```

| Platform | CUDA | Script | Requirements |
|---|---|---|---|
| Windows | yes | `.vscode/bootstrap-env.ps1` | `requirements-torch-cuda.txt` |
| Windows | no | `.vscode/bootstrap-env.ps1` | `requirements-torch-cpu.txt` |
| Linux | yes | `.vscode/bootstrap-env.sh` | `requirements-torch-cuda.txt` |
| Linux | no | `.vscode/bootstrap-env.sh` | `requirements-torch-cpu.txt` |
| macOS | n/a | `.vscode/bootstrap-env.sh` | `requirements-torch-macos.txt` |

Useful flags:

- `--print-only` reports the detected plan without changing anything;
- `--force-cpu` skips CUDA detection;
- `--environment-name .venv-cpu` provisions the CPU comparison environment.

The script is idempotent. It creates the environment only when missing and
reinstalls only when the requirements file changed, so it is safe to run every
time before working.

CUDA is detected with `nvidia-smi`, because torch is not yet installed at that
point. On Windows the dispatcher prefers `pwsh`: launching Windows PowerShell 5.1
from a non-PowerShell parent leaks PowerShell 7 module paths, which makes
`Get-FileHash` disappear and the bootstrap fail.

Never install packages ad hoc. Add the dependency to the correct requirements
file and re-run the dispatcher.

## Stage 2 — Write the program

Read [docs/api.md](../../../docs/api.md) for types and signatures and
[docs/gates.md](../../../docs/gates.md) for the gate list, matrices and dagger
metadata. Do not guess a gate name or argument order; both files are generated
against the code.

`src/` is a flat layout, so import by module name:

```python
from circuit import Circuit
from observable import Observable, PauliTerm
from simulator import StateVectorSimulator
from single_qubit_gates import H, RX
from two_qubit_gates import CX
```

Conventions that change results if ignored:

- **Little-endian global index.** Qubit 0 is the least significant bit, so index
  `i` is the basis state `|q[n-1]...q[1]q[0]>`.
- **Continuous parameters come before qubits**: `RX(theta, qubit)`,
  `CRX(theta, control, target)`.
- **Local gate basis follows `operation.qubits`**, first qubit most significant:
  `CX(control, target)` uses basis `|control,target>`.
- **A circuit is unitary evolution only.** There is no measurement operation and
  no `measure()` or `collapse()` on the state. Readout never mutates the state.
- `Circuit.append()` is fluent; `Circuit.dagger()` reverses order and daggers
  each operation.

Select a backend only when asked for GPU or reduced precision:

```python
from torch_backend import TorchBackend
simulator = StateVectorSimulator(TorchBackend("cuda", "complex64"))
```

Default to `complex64` on GPU. The reference Quadro P620 runs FP64 at 1/32 rate,
and 2 GiB VRAM caps GPU statevectors near 24 qubits.

## Stage 3 — Run and report

Put both `src` and this skill's `scripts` directory on `PYTHONPATH`, then run
with the environment interpreter:

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD\.agents\skills\quantum-simulator\scripts"
.\.venv\Scripts\python.exe your_program.py
```

```bash
export PYTHONPATH="$PWD/src:$PWD/.agents/skills/quantum-simulator/scripts"
./.venv/bin/python your_program.py
```

Use [qsim_report.py](./scripts/qsim_report.py) instead of hand-writing output
formatting. Pick the branch that matches what the user asked for.

### Sampling

Write every outcome to CSV and show only the top 10 in chat.

```python
from qsim_report import write_sampling_csv, format_rows

counts = state.sample(2000, np.random.default_rng(7))
path, top = write_sampling_csv(counts, state.num_qubits)
print(f"csv: {path}")
print(format_rows(top))
```

`write_sampling_csv` sorts by count descending, breaks ties by ascending index,
drops zero-count outcomes, and converts each index to a bitstring. Columns are
`bitstring,index,count,probability`; the default path is `results/sampling.csv`.

Report the printed table in the reply and state where the full file went.

Two things to keep in mind: `StateVector.sample()` already returns only observed
outcomes, so the zero filter matters mainly for merged or hand-built
dictionaries; and a spreadsheet may read the `bitstring` column as a number and
strip leading zeros, so use the `index` column when reloading programmatically.

### Expectation value

Print it; do not write a file.

```python
from qsim_report import print_expectation

observable = Observable([PauliTerm(1.0, [(0, "Z"), (1, "Z")])])
print_expectation(state.expectation(observable), "<Z0 Z1>")
```

`Observable` guarantees Hermiticity structurally. `expectation()` rejects raw
operation sequences on purpose, because a non-Hermitian gate such as `RX(0.3)`
would otherwise produce a meaningless number without any error. Use
`state.inner_product(other)` for arbitrary operators.

Relay the value in the reply.

### Amplitude vector

Write JSON; do not dump the vector into chat.

```python
from qsim_report import write_amplitudes_json

path = write_amplitudes_json(
    state.amplitudes,
    num_qubits=state.num_qubits,
    backend=state.backend.name,
)
print(f"json: {path}")
```

Each entry carries `index`, `bitstring`, `real`, `imag` and `probability`; the
default path is `results/amplitudes.json`. The file holds all `2**n` amplitudes
and therefore grows exponentially — roughly 100 MB at 20 qubits. Warn the user
and suggest sampling or an expectation value instead for large registers.

Summarise in the reply: qubit count, backend, output path, and at most a couple
of notable amplitudes.

## Validation

After changing simulator code rather than just driving it, run the suite:

```powershell
$env:PYTHONPATH = Join-Path $PWD 'src'
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Anything touching a backend, `StateVector` or gate matrices must also pass in
`.venv-cpu`.

Sanity checks worth running when a result looks surprising:

- a Bell circuit must sample only `00` and `11`, never `01` or `10`;
- applying `Circuit.dagger()` to the produced state must restore `|0...0>`;
- probabilities must sum to 1.

## Reference

- [docs/api.md](../../../docs/api.md) — types, signatures, exceptions
- [docs/gates.md](../../../docs/gates.md) — gates, matrices, dagger metadata
- [docs/architecture.md](../../../docs/architecture.md) — design decisions
- [AGENTS.md](../../../AGENTS.md) — repository-wide constraints
