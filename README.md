# AI-Object-Dection-Continual-Learning
AI project Object Dection Continual Learning using Detectron2 (Meta)

## Downloading CIFAR-100 (Windows)

If the default `./data` directory in the repository is not writable, torchvision will fail to download the dataset with a FileNotFoundError. The project includes a helper script to download CIFAR-100 to a location you choose.

Run from PowerShell:

```powershell
# Download into the repo data folder (may fail if ./data is not writable)
python .\scripts\download_cifar100.py --root .\data

# Force download to a custom folder (example)
python .\scripts\download_cifar100.py --root C:\\Users\\Admin\\datasets\\cifar100
```

If the script detects `./data` is not writable it will automatically fall back to `C:\\Users\\<you>\\.cache\\cifar100` and print a warning message with the fallback path.

You can also call the project's loader directly and pass a custom root in Python:

```python
from data.cifar100_split import get_task_loaders
train_loader, test_loader, classes = get_task_loaders(0, batch_size=128, root='C:\\path\\to\\writable\\dir')
```

## Running the project on Windows

The repository includes a Unix shell runner `scripts/run.sh` which uses `python3`. On Windows you have a few options:

- Use PowerShell (recommended): the repository contains `scripts/run.ps1` which forwards arguments to your `python` executable. Example:

```powershell
# Use defaults from the script
powershell -File .\scripts\run.ps1

# Override arguments
powershell -File .\scripts\run.ps1 -- --task 0 --epochs 1 --batch 8
```

- Use Git Bash or WSL if you prefer to run the original `scripts/run.sh`:

```powershell
# Git Bash
bash scripts/run.sh

# WSL
wsl bash scripts/run.sh
```

If `bash` is not available you will get an error like `bash : The term 'bash' is not recognized...` — use the PowerShell runner or install Git Bash/WSL.

