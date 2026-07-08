# BindCOREpLM

LoRA fine-tuning of **ESM‑C** (6B‑parameter) for residue‑level binding‑site prediction on LIP / MoRF datasets.

## Installation

```bash
# Create a fresh environment (conda or venv)
conda create -n bindcoreplm python=3.11 -y
conda activate bindcoreplm

# Install the package in editable mode (recommended for development)
pip install -e .

# Or install from the repository directly
pip install git+https://github.com/your-org/BindCOREpLM.git
```

## Quick start

### Training

Prepare a config YAML (see `example_config.yaml`) and run:

```bash
bindcoreplm-train --config configs/my_experiment.yaml
bindcoreplm-train --config configs/my_experiment.yaml --final   # combine train+valid
```

### Prediction

```bash
bindcoreplm-predict --checkpoint outputs/run1/best.pt               \\
                    --config outputs/run1/best.config.yaml          \\
                    --input data/LIP_dataset/test.txt               \\
                    --output predictions.csv
```

### Evaluation

```bash
bindcoreplm-evaluate --predictions predictions.csv                   \\
                     --true-labels data/LIP_dataset/test.txt
```

### Merge LoRA weights

```bash
bindcoreplm-merge-lora --checkpoint outputs/run1/best.pt             \\
                       --config outputs/run1/best.config.yaml        \\
                       --output merged_model
```

## Programmatic usage

```python
from BindCOREpLM import (
    ExperimentConfig,
    ESMCResidueBindingModel,
    LIPDataset,
    LIPCollator,
)

cfg = ExperimentConfig.from_yaml("config.yaml")
model = ESMCResidueBindingModel(cfg.model)
# ...
```

## Project structure

```
BindCOREpLM/
├── config.py       – Dataclass configs (round‑trip YAML)
├── dataset.py      – LIP‑format dataset & collator
├── lora.py         – LoRA adapter injection (no ``peft`` dependency)
├── model.py        – ESMCResidueBindingModel (backbone + LoRA + CNN/MLP)
├── train.py        – Training loop with MLflow
├── predict.py      – Inference on LIP‑format files
├── evaluate.py     – Metric evaluation (MCC, F1, AUROC, …)
└── merge_lora.py   – Fuse LoRA → full‑rank weights
pyproject.toml      – Package metadata, dependencies, CLI entry points
```

## License

MIT