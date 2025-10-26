#!/usr/bin/env pwsh

$ErrorActionPreference = "Stop"

if ($?)
{
    Write-Host "Found nvidia-smi in PATH, preparing for CUDA..."
    uv sync --extra cuda
}
else
{
    Write-Host "Didn't find nvidia-smi in PATH, preparing for CPU..."
    uv sync --extra cpu
}

Write-Host "Success!"
