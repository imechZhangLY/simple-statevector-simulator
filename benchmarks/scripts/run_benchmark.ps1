#Requires -Version 5.1
<#
.SYNOPSIS
    Provision the benchmark environment and run one cross-framework scenario.

.DESCRIPTION
    Scenario cpu-single pins every framework to a single thread, cpu-multi uses
    all logical processors. Both compare NumPy, Torch CPU, qulacs and
    qiskit-aer, using qiskit-aer as the fidelity reference.

    The GPU scenario is Linux only because qiskit-aer-gpu ships manylinux
    wheels exclusively; use run_benchmark.sh there.
#>
[CmdletBinding()]
param(
    [ValidateSet('cpu-single', 'cpu-multi')]
    [string]$Scenario = 'cpu-single',
    [string]$Qubits = '4,8,12,16,20',
    [int]$Repeats = 5,
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'

function Test-DependenciesReady {
    param(
        [string]$PythonPath,
        [string]$MarkerPath,
        [string]$ExpectedHash,
        [string]$ProbePath,
        [string[]]$Modules
    )

    if (-not (Test-Path $MarkerPath)) { return $false }
    if ((Get-Content -Path $MarkerPath -Raw).Trim() -ne $ExpectedHash) { return $false }

    # The hash alone cannot tell that a package was removed by hand, so the
    # modules are located as well. find_spec avoids importing torch, which
    # would cost several seconds on its own.
    & $PythonPath $ProbePath @Modules
    return $LASTEXITCODE -eq 0
}

$repositoryRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Push-Location $repositoryRoot
try {
    $environmentName = '.venv-bench'
    $requirements = 'requirements-bench.txt'
    $implementations = @(
        'ours:numpy:complex128',
        'ours:torch:cpu:complex128',
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

    $pythonPath = Join-Path $repositoryRoot "$environmentName\Scripts\python.exe"
    $modules = @('numpy', 'torch', 'qulacs', 'qiskit_aer', 'matplotlib')

    if (-not (Test-Path $pythonPath)) {
        Write-Host "Creating virtual environment $environmentName ..."
        python -m venv (Join-Path $repositoryRoot $environmentName)
    }

    $markerPath = Join-Path $repositoryRoot "$environmentName\.requirements-hash"
    $expectedHash = (Get-FileHash -Path $requirements -Algorithm SHA256).Hash
    $probePath = Join-Path $PSScriptRoot 'check_dependencies.py'

    if ($SkipInstall) {
        Write-Host 'Skipping the dependency check (-SkipInstall).'
    }
    elseif (Test-DependenciesReady -PythonPath $pythonPath -MarkerPath $markerPath `
            -ExpectedHash $expectedHash -ProbePath $probePath -Modules $modules) {
        Write-Host "Dependencies already satisfied for $requirements."
    }
    else {
        Write-Host "Installing $requirements ..."
        & $pythonPath -m pip install --upgrade pip --quiet
        & $pythonPath -m pip install -r $requirements --quiet
        Set-Content -Path $markerPath -Value $expectedHash
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
    Write-Host ""

    & $pythonPath (Join-Path $repositoryRoot 'benchmarks\framework_comparison.py') `
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
