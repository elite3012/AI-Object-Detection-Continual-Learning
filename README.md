# Continual Learning System for Clothing Classification

A recruiter-friendly showcase of a multi-phase continual learning project built with PyTorch and Streamlit, covering replay-based learning, PEFT/LoRA, multimodal fusion, and deployment-aware compression.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.5.1-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.31.0-green.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/status-smoke_checked-1f9d55.svg)](#validation-on-the-current-codebase)

> Built as a 3-member AI course project at HCMIU. This README focuses on the implemented system, measured outcomes, and current codebase status.

> Scope note: despite the repository name, the current implementation is a continual clothing classification benchmark on Fashion-MNIST rather than a full object detection pipeline.

![Pipeline overview](assets/pipeline-overview.svg)
![Results snapshot](assets/results-snapshot.svg)

## If you have 30 seconds

- This project addresses catastrophic forgetting when a model must learn new classes sequentially.
- It combines Experience Replay, PEFT/LoRA, vision-text multimodal fusion, and hardware-aware compression in one end-to-end system.
- It includes a usable Streamlit demo with training controls, Plotly dashboards, and image-based inference.
- The current repository has been smoke-checked locally for imports, dataloading, model forward pass, and key module initialization.

## Why this project matters

Most ML demos stop at a single training run on a fixed dataset. This project is more interesting because it explores a real engineering problem:

- new classes arrive over time
- retraining from scratch is expensive
- old knowledge should not disappear
- deployment targets may require smaller, faster models

In short, the project is not just about getting accuracy on Fashion-MNIST. It is about managing the trade-offs between retention, efficiency, multimodal context, and deployability.

## What is implemented

### 1. Experience Replay

- Fixed-capacity ring buffer for replayed samples
- Balanced sampling across learned classes
- FIFO replacement to keep memory bounded
- Sequential task setup over Fashion-MNIST class groups

### 2. PEFT / LoRA

- LoRA adapters injected into convolutional and linear layers
- Rank and alpha configurable from the UI
- Parameter-efficient updates instead of full retraining

### 3. Multimodal Learning

- Vision encoder + text encoder pipeline
- Fusion strategies available in the app: `concat`, `gated`, `cross_attention`
- Text descriptions used to complement visual features

### 4. Hardware Optimization

- Structured pruning and quantization utilities
- Presets for `mobile`, `gpu`, `edge`, and `cloud`
- Benchmark-oriented workflow for size and latency trade-offs

### 5. Demo Application

- Streamlit interface for training and testing
- Plotly charts for accuracy, forgetting, timing, and buffer distribution
- Image upload with preprocessing-based inference flow

## Reported results

The figures below come from the two project reports dated 19/12/2025 and 26/12/2025. They should be treated as reported experiment results, not as claims re-verified by a full benchmark rerun in this review session.

| Module | Reported outcome | Why it matters |
|---|---|---|
| Experience Replay | ~86% knowledge retention with a fixed 5,000-sample replay buffer | Shows the system can preserve prior knowledge while learning sequential tasks |
| PEFT / LoRA | 88.2% final accuracy at rank 24 / alpha 48, with 85% optimizer-state memory reduction vs full fine-tuning | Demonstrates a practical efficiency trade-off |
| Multimodal Fusion | 99.49% accuracy, +10.25 points over the vision-only baseline | Shows text descriptions can materially improve classification |
| Hardware Optimization | Mobile preset reported at 5.6 MB, 45 ms CPU, 4 ms GPU | Highlights deployment-minded optimization rather than benchmark-only thinking |

The Experience Replay report also documents:

- a fixed total buffer size of 5,000 samples
- class-balanced replay allocation as tasks grow
- a bounded-memory design that avoids unscaled data retention

## Validation on the current codebase

I reviewed the repository and ran smoke checks on 2026-05-13. Current status:

- `python -m compileall .` passes across `app.py`, `data`, `eval`, `models`, `optimizers`, `replay`, and `trainers`
- core dependencies import successfully in the local environment: `torch`, `streamlit`, `plotly`, `cv2`, `Pillow`
- `app.py` imports successfully
- task-0 dataloader returns the expected tensor shapes from `get_task_loaders_true_continual(...)`
- model forward pass works for `SimpleCNNMulticlass`
- LoRA insertion runs successfully
- hardware optimization preset initialization runs successfully

Important honesty note:

- this repository does not currently include an automated test suite
- the checks above are smoke checks, not a full experiment reproduction
- some report numbers reflect experiment configurations captured in the reports and may differ from what a fresh rerun on the current code revision produces

## Recruiter takeaways

This project is valuable because it demonstrates more than a standard model-training notebook:

- end-to-end ML thinking: data pipeline, model design, evaluation, UI, and deployment trade-offs
- practical continual learning: not just classification accuracy, but retention under sequential tasks
- efficiency mindset: LoRA, quantization, pruning, and target-hardware presets
- product sense: a working Streamlit layer makes the project easier to demo and explain
- documentation quality: the project is backed by detailed technical reports rather than a code dump

## Quick start

### Requirements

- Python 3.10+
- CPU is enough for smoke checks; GPU is recommended for training

### Install

```bash
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

On Windows, you can also use:

```bash
run_app.bat
```

The application opens at `http://localhost:8501`.

## Demo features to try

### Training side

- toggle Experience Replay on/off
- change LoRA rank and alpha
- compare multimodal fusion strategies
- switch hardware presets for compression experiments

### Testing side

- upload a clothing image
- inspect the preprocessing flow
- view top-k predictions and confidence scores

## Repository layout

```text
AI-Object-Detection-Continual-Learning-main/
|-- app.py
|-- data/
|   |-- fashion_mnist_true_continual.py
|   `-- fashion_text.py
|-- models/
|   |-- simple_cnn_multiclass.py
|   |-- peft_lora.py
|   |-- text_encoder.py
|   `-- multimodal_fusion.py
|-- trainers/
|   |-- continual_trainer.py
|   |-- peft_trainer.py
|   |-- multimodal_trainer.py
|   `-- hardware_trainer.py
|-- replay/
|   `-- buffer.py
|-- optimizers/
|   |-- pruning.py
|   |-- quantization.py
|   |-- hardware_optimizer.py
|   `-- benchmark.py
`-- eval/
    |-- metrics.py
    `-- logger.py
```

## Known limitations

- current benchmark scope is Fashion-MNIST, which is much simpler than real-world production data
- the repository does not yet ship with automated tests
- full result reproduction still requires rerunning the training phases end to end
- the project would be stronger with a larger continual-learning benchmark and cleaner experiment tracking

## Suggested next upgrades

- add automated tests for dataloaders, trainers, and metric utilities
- log experiment runs to a reproducible tracker
- add a CLI benchmark script for one-command result reproduction
- extend beyond Fashion-MNIST to a harder continual-learning dataset
- add model cards or exported checkpoints for demo-ready sharing

## License

Add a license file before publishing the repository publicly so usage terms are explicit for recruiters, collaborators, and future employers.
