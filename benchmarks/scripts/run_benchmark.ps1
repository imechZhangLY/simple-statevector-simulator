#Requires -Version 5.1
<#
.SYNOPSIS
    Run one cross-framework benchmark scenario in the active environment.

.DESCRIPTION
    Activate the matching environment from envs/ first; this script neither
    creates nor inspects it, and missing dependencies are reported by
    framework_comparison.py.

    Scenario cpu-single pins every framework to a single thread, cpu-multi uses
    all logical processors. Both compare NumPy, Torch CPU, qulacs and
    qiskit-aer, using qiskit-aer as the error reference.

    The GPU scenario is Linux only because qiskit-aer-gpu ships manylinux
    wheels exclusively; use run_benchmark.sh there.
#>
[CmdletBinding()]
param(
    [ValidateSet('cpu-single', 'cpu-multi')]
    [string]$Scenario = 'cpu-single',
    [string]$Qubits = '4,8,12,16,20',
    [int]$Repeats = 5
)

$ErrorActionPreference = 'Stop'

$repositoryRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Push-Location $repositoryRoot
try {
    $implementations = @(
        'ours:numpy:complex128',
        'ours:numpy:complex128:fusion',
        'ours:torch:cpu:complex128',
        'ours:torch:cpu:complex128:fusion',
        'qulacs:cpu:complex128',
        'qiskit-aer:cpu:complex128'
    ) -join ','
    $reference = 'qiskit-aer:cpu:complex128'

    if ($Scenario -eq 'cpu-single') {
        $threads = 1
        $title = 'qulacs benchmark circuit, CPU single thread'
    }
    else {
        $threads = [Environment]::ProcessorCount
        $title = "qulacs benchmark circuit, CPU $threads threads"
    }

    # numpy, qulacs and the BLAS libraries read these at import time, so they
    # must be set before Python starts. Passing --threads alone is not enough.
    $env:OMP_NUM_THREADS = "$threads"
    $env:MKL_NUM_THREADS = "$threads"
    $env:OPENBLAS_NUM_THREADS = "$threads"

    $resultsDirectory = Join-Path $repositoryRoot 'benchmarks\results'
    New-Item -ItemType Directory -Path $resultsDirectory -Force | Out-Null
    $jsonPath = Join-Path $resultsDirectory "$Scenario.json"
    $plotPath = Join-Path $resultsDirectory "$Scenario.png"

    Write-Host ""
    Write-Host "scenario   : $Scenario"
    Write-Host "threads    : $threads"
    Write-Host "qubits     : $Qubits"
    Write-Host "reference  : $reference"
    Write-Host "interpreter: $((Get-Command python).Source)"
    Write-Host ""

    & python (Join-Path $repositoryRoot 'benchmarks\framework_comparison.py') `
        --qubits $Qubits `
        --repeats $Repeats `
        --threads $threads `
        --reference $reference `
        --implementations $implementations `
        --output $jsonPath `
        --plot $plotPath `
        --title $title

    if ($LASTEXITCODE -ne 0) {
        throw "benchmark failed with exit code $LASTEXITCODE"
    }

    Write-Host ""
    Write-Host "results : $jsonPath"
    Write-Host "chart   : $plotPath"
}
finally {
    Pop-Location
}
