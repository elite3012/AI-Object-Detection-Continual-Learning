# PowerShell runner for Windows
param(
    [string[]]$Args
)

# Default args if none provided
if (-not $Args) {
    $Args = @('--task', '0', '--epochs', '2', '--batch', '128', '--method', 'ER')
}

# Use `python` on Windows (user's python installation)
$python = 'python'

Write-Host "Running: $python main.py $($Args -join ' ')"
& $python main.py @Args
