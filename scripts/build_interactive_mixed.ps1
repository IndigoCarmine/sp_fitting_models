$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

Write-Host "[1/2] Sync project environment with uv..."
uv sync

Write-Host "[2/2] Build Windows app (no console) with PyInstaller..."
uv run --with pyinstaller pyinstaller `
  --noconfirm `
  --clean `
  --windowed `
  --onefile `
  --name interactive_mixed `
  --hidden-import sp_fitting_models._core `
  examples/interactive_mixed.py

Write-Host "Build complete: dist/interactive_mixed/interactive_mixed.exe"
