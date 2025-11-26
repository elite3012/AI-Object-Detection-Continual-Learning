# PowerShell runner for continual learning demo
param(
    [string]$Method = "all"
)

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "CONTINUAL LEARNING DEMO" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

if ($Method -eq "all") {
    Write-Host "[1/3] Training with Finetune (baseline)..." -ForegroundColor Yellow
    python main.py --mode continual --method finetune --epochs 2 --batch 128
    
    Write-Host ""
    Write-Host "[2/3] Training with Experience Replay..." -ForegroundColor Green
    python main.py --mode continual --method ER --epochs 2 --batch 128
    
    Write-Host ""
    Write-Host "[3/3] Training with EWC..." -ForegroundColor Blue
    python main.py --mode continual --method EWC --epochs 2 --batch 128
    
    Write-Host ""
    Write-Host "==================================" -ForegroundColor Cyan
    Write-Host "ALL EXPERIMENTS COMPLETED!" -ForegroundColor Cyan
    Write-Host "==================================" -ForegroundColor Cyan
} else {
    Write-Host "Training with $Method..." -ForegroundColor Green
    python main.py --mode continual --method $Method --epochs 2 --batch 128
}
