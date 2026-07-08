"""
Configuration objects for LoRA fine-tuning of ESM-C on residue-level
binding-site prediction (LIP/MoRF-style tasks).

Everything a run needs lives in `ESMCFineTuneConfig`, which nests the
smaller sub-configs below. The whole thing round-trips to/from YAML so
you can keep one config file per experiment (e.g. per Ray Tune trial
or per ablation).

Example
-------
>>> cfg = ESMCFineTuneConfig.from_yaml("example_config.yaml")
>>> cfg.lora.rank
8
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional

import yaml

# --------------------------------------------------------------------------- #
# Sub-configs
# --------------------------------------------------------------------------- #


@dataclass
class LoRAConfig:
    """LoRA adapters on the attention K/V projections."""

    rank: int = 8
    alpha: int = 16
    # LoRA dropout applied to the *input* of the low-rank path only
    # (the frozen base path is untouched). Independent from the head
    # dropout below.
    dropout: float = 0.05
    # Which linear submodules to wrap. Names are matched by suffix, so
    # this works whether the underlying model calls them "k_proj"/"v_proj",
    # "key"/"value", etc. Check `list_linear_module_names(model)` in
    # lora.py if your checkpoint uses different names.
    target_modules: List[str] = field(default_factory=lambda: ["layernorm_qkv"])
    # Restrict LoRA to a subset of transformer layers (0-indexed).
    # None = apply to every layer. With only 800 proteins, restricting
    # to the last N layers (e.g. list(range(30, 36)) for a 36-layer model)
    # is a reasonable way to cut trainable params further and reduce
    # overfitting risk.
    layer_indices: Optional[List[int]] = None


@dataclass
class CNNHeadConfig:
    """Parallel 1D-CNN branches over the per-residue embedding sequence.

    Each branch is a single Conv1d with 'same' padding (so the sequence
    length is preserved -- this is a residue-level task, not a pooled
    sequence-level one). Branch outputs are concatenated channel-wise.
    """

    kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    out_channels_per_branch: int = 128
    # Dropout applied to the ESM-C hidden states before they enter the
    # CNN branches (separate from the head MLP dropout below).
    input_dropout: float = 0.1


@dataclass
class MLPHeadConfig:
    """Small per-residue MLP applied after the CNN branches are concatenated."""

    # Each entry is a hidden layer width; should shrink quickly given
    # the small dataset, e.g. [256, 64] or just [128].
    hidden_dims: List[int] = field(default_factory=lambda: [256, 64])
    activation: str = "gelu"  # gelu | relu | silu
    # High dropout in the head, as requested -- this is where most of
    # the model's trainable capacity for a small dataset lives, so it's
    # the most important place to regularize.
    dropout: float = 0.4
    num_labels: int = 1  # 1 -> binary binding/non-binding logit per residue


@dataclass
class ESMCFineTuneConfig:
    model_name_or_path: str = "biohub/ESMC-6B"

    # Freeze all base-model weights except the injected LoRA params.
    # Strongly recommended with 800 proteins and a 6B backbone.
    freeze_base_model: bool = True
    gradient_checkpointing: bool = True

    lora: LoRAConfig = field(default_factory=LoRAConfig)
    cnn: CNNHeadConfig = field(default_factory=CNNHeadConfig)
    mlp: MLPHeadConfig = field(default_factory=MLPHeadConfig)

    # fp32 | fp16 | bf16. bf16 is the usual choice for fine-tuning a
    # 6B model on modern GPUs (A100/H100/Grid5000 nodes with Ampere+),
    # since it avoids the loss-scaling headaches of fp16 while halving
    # memory vs fp32. fp16 is offered too in case your nodes are older
    # (V100) and lack native bf16 support.
    precision: str = "bf16"

    max_seq_length: int = 1024

    def __post_init__(self):
        # Allow plain dicts to be passed in (e.g. straight from yaml.safe_load)
        if isinstance(self.lora, dict):
            self.lora = LoRAConfig(**self.lora)
        if isinstance(self.cnn, dict):
            self.cnn = CNNHeadConfig(**self.cnn)
        if isinstance(self.mlp, dict):
            self.mlp = MLPHeadConfig(**self.mlp)
        if self.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError(
                f"precision must be one of fp32/fp16/bf16, got {self.precision!r}"
            )

    # ------------------------------------------------------------------ #
    # (De)serialization
    # ------------------------------------------------------------------ #

    @classmethod
    def from_yaml(cls, path: str) -> "ESMCFineTuneConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)

    def to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False, default_flow_style=False)

    def to_dict(self) -> dict:
        return asdict(self)


PRECISION_TORCH_DTYPE_NAMES = {
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
}


# --------------------------------------------------------------------------- #
# Training-side configs
# --------------------------------------------------------------------------- #


@dataclass
class DataConfig:
    """Points at the LIP_dataset folder (3 lines per protein: >id, sequence, labels)."""

    data_dir: str = "data/LIP_dataset"
    train_file: str = "train.txt"
    valid_file: str = "valid.txt"
    test_file: str = "test.txt"
    # If True, valid.txt is merged into the training set (e.g. for a final
    # run after hyperparameters have already been chosen using a real
    # train/valid split). Validation-based logging/early-stopping/best-
    # checkpoint-selection are disabled in that case since there is no
    # held-out split left; training just runs for num_epochs.
    combine_train_valid_for_final: bool = False


@dataclass
class SchedulerConfig:
    """LR scheduler applied per optimizer step (except reduce_on_plateau,
    which steps per epoch on validation loss)."""

    name: str = (
        "linear_warmup"  # linear_warmup | cosine_warmup | constant | reduce_on_plateau | none
    )
    # Warmup as a fraction of total optimizer steps; ignored if warmup_steps is set.
    warmup_ratio: float = 0.06
    warmup_steps: Optional[int] = None
    # reduce_on_plateau-only params:
    factor: float = 0.5
    patience: int = 3
    min_lr: float = 1e-7

    def __post_init__(self):
        valid = {
            "linear_warmup",
            "cosine_warmup",
            "constant",
            "reduce_on_plateau",
            "none",
        }
        if self.name not in valid:
            raise ValueError(
                f"scheduler.name must be one of {valid}, got {self.name!r}"
            )


@dataclass
class TrainingConfig:
    data: DataConfig = field(default_factory=DataConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    batch_size: int = 4
    # Effective batch size = batch_size * grad_accumulation_steps. Useful to
    # simulate a larger batch than fits in memory with a 6B backbone.
    grad_accumulation_steps: int = 8

    lr: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 20
    max_grad_norm: float = 1.0

    # BCEWithLogitsLoss pos_weight for the positive (binding) class. If
    # None, it's computed automatically from the training file(s) actually
    # used (train only, or train+valid if combine_train_valid_for_final),
    # as num_negative / num_positive over non-ignored residues.
    pos_weight: Optional[float] = None

    seed: int = 42
    output_dir: str = "outputs/run1"
    num_workers: int = 2

    log_every_n_steps: int = 10
    eval_every_n_epochs: int = 1
    save_best_only: bool = True
    # Metric used to pick the best checkpoint: "loss" (lower better) or "f1" (higher better).
    best_metric: str = "f1"
    # Epochs without validation improvement before stopping early. None disables.
    early_stopping_patience: Optional[int] = 5

    def __post_init__(self):
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.scheduler, dict):
            self.scheduler = SchedulerConfig(**self.scheduler)
        if self.best_metric not in ("loss", "f1"):
            raise ValueError(
                f"best_metric must be 'loss' or 'f1', got {self.best_metric!r}"
            )


@dataclass
class ExperimentConfig:
    """Top-level config: one YAML file with a `model:` section (as before)
    and a `training:` section, so a single file fully specifies a run.
    """

    model: ESMCFineTuneConfig = field(default_factory=ESMCFineTuneConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        if isinstance(self.model, dict):
            self.model = ESMCFineTuneConfig(**self.model)
        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return cls(
            model=ESMCFineTuneConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
        )

    def to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(
                {"model": asdict(self.model), "training": asdict(self.training)},
                f,
                sort_keys=False,
                default_flow_style=False,
            )
