---
name: quantum-simulator
description: 'Run quantum circuits on this repository''s statevector simulator. Use when asked to build, simulate, or execute a quantum circuit, apply gates, sample measurement outcomes or shots, compute an expectation value of an observable, inspect the amplitude vector or state vector, create or activate a project environment under envs/, or export and parse OpenQASM. Covers environment setup (SUPA/CUDA/CPU aware), writing programs against the flat src/ API, and reporting results as CSV, JSON, or chat output.'
argument-hint: 'describe the circuit and whether you want sampling, an expectation value, or amplitudes'
---

# Quantum Simulator Workflow

Three stages: provision the environment, write the program, run it and report the results.

## Stage 1 — Provision the environment

Every environment is defined under [envs/](../../../envs/), one directory each,
and `envs/<name>` always creates `.venv-<name>`. Full list and platform limits:
[envs/README.md](../../../envs/README.md).

Pick the environment for the task, then run its creator directly.

| Task | Environment | Interpreter |
|---|---|---|
| day-to-day work, tests | `envs/cpu` | `.venv-cpu` |
| GPU simulation | `envs/cuda` | `.venv-cuda` |
| SUPA tests | `envs/supa` | `.venv-supa` |
| cross-framework benchmarks | `envs/bench-cpu` | `.venv-bench-cpu` |

```powershell
.\envs\cpu\create-env.ps1
```

```bash
bash envs/cpu/create-env.sh
```

Choose `envs/cuda` only when an NVIDIA GPU is present; run `nvidia-smi` to check,
because torch is not installed yet at that point. There is no macOS environment,
as the `+cpu` and `+cu126` wheels are published for Linux and Windows only.

Choose `envs/supa` only on the SUPA cloud image, because it depends on SUPA
hardware. `brsmi` is the SUPA counterpart of `nvidia-smi`, and the probe confirms
the image's Python stack can actually reach the device. Run both with the
**system** interpreter: this decides whether to create `envs/supa`, so no venv
exists yet. The image ships torch and torch_br as system packages, which is why
`envs/supa` is created with `--system-site-packages`. Exit code 0 means SUPA is
usable, 1 means it is not:

```bash
brsmi
python .agents/skills/quantum-simulator/scripts/check_supa.py
```

The probe prints the interpreter it used. Bare `python` resolves to whatever
venv is active, so deactivate first if that line is not the image's Python.

The creators are idempotent: they build the environment only when it is missing
and reinstall only when `requirements.txt` changed, so running before each task
is safe and usually instant. They also work from any working directory — the
venv always lands in the repository root.

**Never run project code with the system, conda, or any other interpreter**, and
never `pip install` into an environment by hand. Add the dependency to
`envs/<name>/requirements.txt` and re-run the creator, so the environment stays
reproducible from the repository.

## Stage 2 — Write the program

Read [docs/api.md](../../../docs/api.md) for types and signatures and
[docs/gates.md](../../../docs/gates.md) for the gate list, matrices and dagger
metadata. Do not guess a gate name or argument order; both files are generated
against the code.

Write the program under `workspace/` in the repository root, creating the folder
if needed. It is Git-ignored, so generated programs never end up in a commit.
Keep `src/` for simulator code only. Store every generated result under
`workspace/results/`; do not write simulation outputs to the repository root,
`results/`, or source directories.

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
.\.venv-cpu\Scripts\python.exe workspace\your_program.py
```

```bash
export PYTHONPATH="$PWD/src:$PWD/.agents/skills/quantum-simulator/scripts"
./.venv-cpu/bin/python workspace/your_program.py
```

Use [qsim_report.py](./scripts/qsim_report.py) instead of hand-writing output
formatting. Pick the branch that matches what the user asked for.

### Sampling

Write every outcome to CSV, draw the top 10 as a bar chart, and show the same
top 10 in chat.

```python
from qsim_report import format_rows, write_sampling_csv, write_sampling_plot

counts = state.sample(2000, np.random.default_rng(7))
csv_path, top = write_sampling_csv(counts, state.num_qubits)
plot_path = write_sampling_plot(counts, state.num_qubits)
print(f"csv: {csv_path}")
print(f"plot: {plot_path}")
print(format_rows(top))
```

`write_sampling_csv` sorts by count descending, breaks ties by ascending index,
drops zero-count outcomes, and converts each index to a bitstring. Columns are
`bitstring,index,count,probability`; the default path is
`workspace/results/sampling.csv`.

`write_sampling_plot` uses the same ordering and writes at most 10 observed
outcomes to `workspace/results/sampling.png` by default. The horizontal axis
contains the zero-padded binary basis states, the vertical axis contains sample
counts, and the title includes the total number of shots. If fewer than 10
states were observed, plot only those states.

Report the printed table in the reply and state where both output files went.

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
default path is `workspace/results/amplitudes.json`. The file holds all `2**n`
amplitudes and therefore grows exponentially — roughly 100 MB at 20 qubits.
Warn the user and suggest sampling or an expectation value instead for large
registers.

Summarise in the reply: qubit count, backend, output path, and at most a couple
of notable amplitudes.

## Validation

After changing simulator code rather than just driving it, run the suite:

```powershell
$env:PYTHONPATH = Join-Path $PWD 'src'
.\.venv-cpu\Scripts\python.exe -m unittest discover -s tests
```

`PYTHONPATH` is needed here because Python puts the *script's* directory on
`sys.path`, not the working directory, so `unittest discover` cannot see `src/`
without it.

Anything touching a backend, `StateVector` or gate matrices must pass in both
`.venv-cpu` and `.venv-cuda`, since the torch backend behaves differently on GPU.

Sanity checks worth running when a result looks surprising:

- a Bell circuit must sample only `00` and `11`, never `01` or `10`;
- applying `Circuit.dagger()` to the produced state must restore `|0...0>`;
- probabilities must sum to 1.

## Reference

- [docs/api.md](../../../docs/api.md) — types, signatures, exceptions
- [docs/gates.md](../../../docs/gates.md) — gates, matrices, dagger metadata
- [docs/architecture.md](../../../docs/architecture.md) — design decisions
- [AGENTS.md](../../../AGENTS.md) — repository-wide constraints
