# Continual Learning System for Fashion-MNIST

A PyTorch and Streamlit project about one question I kept running into while studying applied AI:

> What happens when a model is not allowed to learn everything at once?

This repository turns Fashion-MNIST into a small continual-learning lab. The model sees classes in sequential tasks, keeps a bounded replay memory, experiments with LoRA-style parameter-efficient updates, combines image features with class text descriptions, and includes pruning/quantization utilities for deployment-aware thinking.

The repository name still says "Object Dection" from the original course repo. The implemented benchmark here is clothing classification, not full object detection. I keep that scope explicit because a clean project should not overclaim.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/ui-streamlit-green.svg)](https://streamlit.io/)

![Pipeline overview](assets/pipeline-overview.svg)
![Results snapshot](assets/results-snapshot.svg)

## Why I Built This

Most beginner ML projects train once, evaluate once, and stop. That is useful, but it hides a harder problem: real systems change. New labels arrive, data distributions move, and retraining from scratch is expensive.

I used this project to study the engineering around catastrophic forgetting:

- how to split one dataset into sequential tasks
- how much old data a fixed replay buffer should keep
- how LoRA-style adapters behave when only a small part of the model is trainable
- whether text descriptions can help a vision model reason about class meaning
- what compression does to a model that may need to run outside a notebook

The most important part of the project for me was not one final accuracy number. It was learning how to build an experiment loop that makes forgetting visible.

## What Is Implemented

| Area | Implementation |
|---|---|
| True continual split | 5 tasks, 2 Fashion-MNIST classes per task |
| Experience replay | Fixed-budget replay buffer with class-balanced sampling |
| PEFT / LoRA | Low-rank adapters for convolutional and linear layers |
| Multi-modal fusion | Vision encoder, lightweight character text encoder, concat/gated/cross-attention fusion |
| Evaluation | Per-task accuracy, per-class accuracy, forgetting estimate, replay-buffer statistics |
| Deployment thinking | Pruning, FP16 conversion, INT8/QAT utility path, hardware presets |
| Demo layer | Streamlit UI for training, charts, checkpoints, upload inference, and batch evaluation |

## Project Results

These are the outcomes reported in my project report/demo runs. They are useful as project context, not as fresh benchmark claims from this cleanup commit.

| Experiment | Reported observation | Why I cared |
|---|---|---|
| Experience Replay | Knowledge retention stayed much stronger with a fixed replay buffer than with plain finetuning | Shows catastrophic forgetting directly instead of only reporting final accuracy |
| PEFT / LoRA | Rank/alpha tuning reduced trainable state while keeping the experiment usable | Connects continual learning with practical fine-tuning constraints |
| Multi-modal fusion | Text descriptions improved the model in the reported setup | Tests whether semantic class hints can help a visual classifier |
| Hardware optimization | Pruning/quantization workflow produced smaller checkpoints for demo deployment | Forces the project to think beyond notebook accuracy |

## What Was Cleaned Up

The current codebase has been tightened for public review:

- removed tracked `__pycache__` bytecode
- removed tracked Fashion-MNIST raw files; the dataset now downloads through `torchvision`
- removed local checkpoints from the repository surface
- reduced `requirements.txt` to dependencies actually used by the code
- fixed Fashion-MNIST test loading to use the real test split
- made validation use deterministic, non-augmented transforms
- fixed PEFT trainer so the LoRA-wrapped model is not accidentally discarded
- fixed the Streamlit app so UI learning rate is passed into training
- made Testing use the phase that produced the trained model, not the current sidebar selection

## Quick Start

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Windows users can also run:

```bat
run_app.bat
```

The app opens at `http://localhost:8501`.

Fashion-MNIST is downloaded automatically into `data/FashionMNIST/` on first run. Checkpoints are written to `checkpoints/`. Both are ignored by Git because they are generated artifacts.

## Repository Map

```text
app.py                         Streamlit demo and experiment dashboard
data/
  fashion_mnist_true_continual.py
  fashion_text.py              Task splits and class text descriptions
models/
  simple_cnn_multiclass.py     CNN image classifier
  peft_lora.py                 LoRA adapter implementation
  text_encoder.py              Lightweight character-level text encoder
  multimodal_fusion.py         Fusion classifiers
trainers/
  trainer.py                   Single-task training loop
  continual_trainer.py         Replay-based continual trainer
  peft_trainer.py              LoRA continual trainer
  multimodal_trainer.py        Vision-text continual trainer
  hardware_trainer.py          Compression wrapper
replay/
  buffer.py                    Fixed-budget replay buffer
optimizers/
  pruning.py
  quantization.py
  hardware_optimizer.py
  benchmark.py                 Compression and profiling utilities
eval/
  metrics.py
  logger.py
assets/
  pipeline-overview.svg
  results-snapshot.svg
```

## Validation

Useful smoke checks:

```bash
python -B -c "import pathlib; [compile(p.read_text(encoding='utf-8'), str(p), 'exec') for p in pathlib.Path('.').rglob('*.py') if '.git' not in p.parts]"
python -B -c "import torch; from models.simple_cnn_multiclass import SimpleCNNMulticlass; from models.peft_lora import apply_lora_to_model; from models.text_encoder import SimpleTextEncoder, get_tokenizer, encode_texts; from models.multimodal_fusion import MultiModalClassifier; model=SimpleCNNMulticlass(10); assert model(torch.randn(2,1,28,28)).shape==(2,10); _, trainable, total=apply_lora_to_model(SimpleCNNMulticlass(10), rank=4, alpha=8); assert trainable < total; text_encoder=SimpleTextEncoder(vocab_size=get_tokenizer().vocab_size); input_ids, mask=encode_texts(['a casual shirt','ankle boot']); multi=MultiModalClassifier(SimpleCNNMulticlass(10), text_encoder); assert multi(torch.randn(2,1,28,28), input_ids, mask).shape==(2,10)"
```

## Current Limitations

- Fashion-MNIST is intentionally small; it is good for studying mechanics, not for claiming production robustness.
- There is no full automated test suite yet.
- Reported experiment numbers should be reproduced with a fresh run before being used in a formal benchmark.
- The multi-modal setup uses class descriptions as controlled semantic hints; a harder version should use noisier real-world text.
- Compression utilities are research/demo utilities, not a polished deployment pipeline.

## Next Things I Would Improve

- add a one-command benchmark script that records config, seed, metrics, and checkpoint paths
- add pytest coverage for data splits, replay sampling, LoRA wrapping, and multimodal forward passes
- export experiment logs to CSV/JSON so reports can be regenerated without manual screenshots
- try a harder continual-learning benchmark beyond Fashion-MNIST
- add a small model card for the best checkpoint generated by a reproducible run

## License

No license is included yet. Add one before reusing or publishing this as a shared package.
