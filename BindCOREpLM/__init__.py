"""
BindCOREpLM - LoRA fine-tuning of ESM-C for residue-level binding-site prediction.

Usage
-----
    # Training
    python -m BindCOREpLM.training.train --config config.yaml
    
    # Inference
    python -m BindCOREpLM.inference.predict --checkpoint best.pt --config config.yaml \
                            --input test.txt --output predictions.csv
    
    # Evaluation
    python -m BindCOREpLM.inference.evaluate --predictions predictions.csv \
                            --true_labels test.txt
    
    # Merge LoRA weights
    python -m BindCOREpLM.models.merge_lora --checkpoint best.pt --config config.yaml
"""

from __future__ import annotations

__version__ = "0.1.0"

# Core config
from BindCOREpLM.config import (
    ExperimentConfig,
    ESMCFineTuneConfig,
    LoRAConfig,
    CNNHeadConfig,
    MLPHeadConfig,
    DataConfig,
    SchedulerConfig,
    TrainingConfig,
)

# Core model
from BindCOREpLM.models.model import ESMCResidueBindingModel

# Subpackage-level imports (for convenience)
from BindCOREpLM.data.dataset import LIPDataset, LIPCollator, build_datasets, parse_lip_file
from BindCOREpLM.models.lora import (
    LoRALinear, LoRAFusedQKV, inject_lora_adapters, count_trainable_parameters,
    list_linear_module_names, list_fused_qkv_module_names,
)
from BindCOREpLM.models.merge_lora import merge_lora_into_base, save_merged_model

__all__ = [
    "__version__",
    "ExperimentConfig",
    "ESMCFineTuneConfig",
    "LoRAConfig",
    "CNNHeadConfig",
    "MLPHeadConfig",
    "DataConfig",
    "SchedulerConfig",
    "TrainingConfig",
    "ESMCResidueBindingModel",
    "LIPDataset",
    "LIPCollator",
    "build_datasets",
    "parse_lip_file",
    "LoRALinear",
    "LoRAFusedQKV",
    "inject_lora_adapters",
    "count_trainable_parameters",
    "list_linear_module_names",
    "list_fused_qkv_module_names",
    "merge_lora_into_base",
    "save_merged_model",
]
