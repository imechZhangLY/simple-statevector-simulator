#Requires -Version 5.1
[CmdletBinding()]
param([string]$EnvironmentName = '.venv-cpu')

& (Join-Path $PSScriptRoot '..\scripts\create-env.ps1') `
    -EnvironmentName $EnvironmentName `
    -Requirements 'envs/cpu/requirements.txt'
