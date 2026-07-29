#Requires -Version 5.1
<#
.SYNOPSIS
    Shared virtual environment creator used by every envs/<name>/create-env.ps1.

.DESCRIPTION
    Creates the environment when it is missing and reinstalls only when the
    requirements file changed, so it is safe to run on every workspace open.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$EnvironmentName,
    [Parameter(Mandatory = $true)][string]$Requirements,
    [switch]$SystemSitePackages
)

$ErrorActionPreference = 'Stop'

$repositoryRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$environmentPath = Join-Path $repositoryRoot $EnvironmentName
$pythonPath = Join-Path $environmentPath 'Scripts\python.exe'
$requirementsPath = Join-Path $repositoryRoot $Requirements

if (-not (Test-Path $requirementsPath)) {
    throw "Requirements file not found: $requirementsPath"
}

if (-not (Test-Path $pythonPath)) {
    Write-Host "Creating virtual environment in $EnvironmentName ..."
    if ($SystemSitePackages) {
        python -m venv --system-site-packages $environmentPath
    }
    else {
        python -m venv $environmentPath
    }
}

$markerPath = Join-Path $environmentPath '.requirements-hash'
$expectedHash = (Get-FileHash -Path $requirementsPath -Algorithm SHA256).Hash
$installedHash = if (Test-Path $markerPath) { (Get-Content -Path $markerPath -Raw).Trim() } else { '' }

if ($installedHash -eq $expectedHash) {
    Write-Host "$EnvironmentName is up to date."
}
else {
    Write-Host "Installing dependencies from $Requirements into $EnvironmentName ..."
    & $pythonPath -m pip install --upgrade pip --quiet
    & $pythonPath -m pip install -r $requirementsPath

    Set-Content -Path $markerPath -Value $expectedHash
    Write-Host "$EnvironmentName ready."
}

Write-Host "Interpreter: $pythonPath"
