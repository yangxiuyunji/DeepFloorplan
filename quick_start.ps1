# Quick Start Script - Ensure Proper Environment Setup
# PowerShell script for DeepFloorplan environment

Write-Host "=== DeepFloorplan Quick Start ===" -ForegroundColor Green

# 1. Set project directory
$ProjectDir = "D:\ws\DeepFloorplan"
if (Get-Location | Select-String $ProjectDir) {
    Write-Host "OK: Already in project directory" -ForegroundColor Green
} else {
    Write-Host "Switching to project directory..." -ForegroundColor Yellow
    Set-Location $ProjectDir
}

# 2. Activate virtual environment
if ($env:VIRTUAL_ENV) {
    Write-Host "OK: Virtual environment active: $env:VIRTUAL_ENV" -ForegroundColor Green
} else {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    if (Test-Path ".\dfp\Scripts\Activate.ps1") {
        & ".\dfp\Scripts\Activate.ps1"
        if ($env:VIRTUAL_ENV) {
            Write-Host "OK: Virtual environment activated" -ForegroundColor Green
        } else {
            Write-Error "ERROR: Failed to activate virtual environment"
            exit 1
        }
    } else {
        Write-Error "ERROR: Virtual environment script not found: .\dfp\Scripts\Activate.ps1"
        exit 1
    }
}

# 3. Verify Python environment
Write-Host "Verifying Python environment..." -ForegroundColor Yellow
python -c "
import sys
import os
print('Python version:', sys.version.split()[0])
print('Virtual env:', os.environ.get('VIRTUAL_ENV', 'Not activated'))

try:
    import cv2, PIL, numpy
    print('OK: Key modules installed (opencv, PIL, numpy)')
except ImportError as e:
    print('ERROR: Missing module:', e)
    sys.exit(1)
"

if ($LASTEXITCODE -eq 0) {
    Write-Host "OK: Environment verification passed" -ForegroundColor Green
    Write-Host "`nEnvironment ready! You can start working." -ForegroundColor Cyan
    Write-Host "`nMain Scripts:" -ForegroundColor Yellow
    Write-Host "  Room detection (refactored): python demo_refactored_clean.py demo\demo1.jpg"
    Write-Host "  Room detection (original):   python demo.py demo\demo1.jpg"
    Write-Host "  Generate fengshui analysis:  python luoshu_visualizer.py .\output\demo1_result_edited.json"
    Write-Host "  Open room editor:            python -m editor.main --json .\output\demo1_result_edited.json"
    Write-Host "  Batch process all demos:     python batch_run_demos.py"
    Write-Host "`nUtility Scripts:" -ForegroundColor Cyan
    Write-Host "  Environment check:           python environment_checker.py"
    Write-Host "  Quick environment setup:     .\quick_start.ps1"
    Write-Host "  Auto-run with env check:     .\activate_and_run.ps1"
    Write-Host "`nDebug/Test files are now in: .\debug\" -ForegroundColor Magenta
    Write-Host "`nNote: Room editor requires PySide6 (pip install PySide6)" -ForegroundColor Yellow
} else {
    Write-Error "ERROR: Environment verification failed"
    exit 1
}
