#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$EnvironmentName = '.venv',
    [string]$Requirements = 'requirements-torch-cuda.txt'
)

$ErrorActionPreference = 'Stop'

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$environmentPath = Join-Path $repositoryRoot $EnvironmentName
$pythonPath = Join-Path $environmentPath 'Scripts\python.exe'
$requirementsPath = Join-Path $repositoryRoot $Requirements

if (-not (Test-Path $requirementsPath)) {
    throw "Requirements file not found: $requirementsPath"
}

if (-not (Test-Path $pythonPath)) {
    Write-Host "Creating virtual environment in $EnvironmentName ..."
    python -m venv $environmentPath
}

$markerPath = Join-Path $environmentPath '.requirements-hash'
$expectedHash = (Get-FileHash -Path $requirementsPath -Algorithm SHA256).Hash
$installedHash = if (Test-Path $markerPath) { (Get-Content -Path $markerPath -Raw).Trim() } else { '' }

if ($installedHash -eq $expectedHash) {
    Write-Host "$EnvironmentName is up to date."
    exit 0
}

Write-Host "Installing dependencies from $Requirements into $EnvironmentName ..."
& $pythonPath -m pip install --upgrade pip --quiet
& $pythonPath -m pip install -r $requirementsPath

Set-Content -Path $markerPath -Value $expectedHash
Write-Host "$EnvironmentName ready."
