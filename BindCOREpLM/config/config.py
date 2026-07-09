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
from typing import List, Optional, Union

import yaml

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _parse_layer_indices(
    value: Optional[Union[List[int], str]],
) -> Optional[List[int]]:
    """Accept ``layer_indices`` as either a list of ints or a range string.

    Range string formats (inclusive-inclusive):
      ``"64-79"``  → ``[64, 65, ..., 79]``
      ``"0-35"``   → ``[0, 1, ..., 35]``
      ``"64"``     → ``[64]``
    """
    if value is None:
        return None
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, str):
        parts = value.split("-")
        if len(parts) == 1:
            return [int(parts[0])]
        if len(parts) == 2:
            start, end = int(parts[0]), int(parts[1])
            if start > end:
                raise ValueError(
                    f"Invalid layer_indices range: {value!r}. "
                    f"Start ({start}) must be <= end ({end})."
                )
            return list(range(start, end + 1))
        raise ValueError(
            f"Invalid layer_indices string: {value!r}. "
            f'Use a single int ("5"), inclusive range ("5-11"), '
            f"or a YAML list ([5, 6, 7, 11])."
        )
    raise TypeError(
        f"layer_indices must be a list of ints, a range string, or None; "
        f"got {type(value).__name__}: {value!r}"
    )


# --------------------------------------------------------------------------- #
# Sub-configs
# --------------------------------------------------------------------------- #


@dataclass
class LoRAConfig:
    """LoRA adapters on the attention K/V projections."""

    # Set to True to skip LoRA entirely — only the frozen backbone +
    # CNN + MLP head will be used. Useful for ablation studies.
    disable: bool = False

    rank: int = 8
    alpha: int = 16
    # LoRA dropout applied to the *input* of the low-rank path only
    # (the frozen base path is untouched). Independent from the head
    # dropout below.
    dropout: float = 0.05
    # Which submodules to wrap with LoRA adapters.  Names are matched by
    # suffix (the last dot-separated segment), so they work regardless of
    # the full module path.
    #
    # Available targets for ESM-C (see lora.py docstring for details):
    #   "layernorm_qkv"  — fused QKV (default; K/V slices only)
    #   "ffn"            — fused SwiGLU MLP (both fc1 and fc2)
    #   "out_proj"       — attention output projection (nn.Linear)
    #
    # For standard HF models with separate K/V projections (e.g. BERT,
    # LLaMA), use "k_proj" and/or "v_proj" instead — but note that
    # ESM-C uses fused QKV and has no such separate modules.
    #
    # Check `list_linear_module_names(model)`,
    # `list_fused_qkv_module_names(model)` and
    # `list_fused_ffn_module_names(model)` in lora.py for your checkpoint.
    target_modules: List[str] = field(default_factory=lambda: ["layernorm_qkv"])
    # Restrict LoRA to a subset of transformer layers (0-indexed).
    # None = apply to every layer. With only 800 proteins, restricting
    # to the last N layers (e.g. list(range(30, 36)) for a 36-layer model)
    # is a reasonable way to cut trainable params further and reduce
    # overfitting risk.
    #
    # In YAML you can write either a list:
    #   layer_indices: [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
    #
    # …or a compact inclusive-inclusive range string:
    #   layer_indices: "64-79"
    #
    # For a single layer: "64" or [64].
    layer_indices: Optional[Union[List[int], str]] = None

    def __post_init__(self):
        if self.layer_indices is not None:
            self.layer_indices = _parse_layer_indices(self.layer_indices)


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

    # Precision for the CNN + MLP head (fp32 or fp64).
    # Use "fp64" (float64) when running on MPS (Apple Silicon) to avoid
    # NaN in validation metrics caused by numerical instability of float32
    # on that platform. Default is "fp32" for speed on CUDA / CPU.
    head_precision: str = "fp32"

    max_seq_length: int = 1024

    def __post_init__(self):
        # Allow plain dicts to be passed in (e.g. straight from yaml.safe_load).
        # Using getattr/setattr so Pylint sees the value as `Any` and can
        # narrow to `dict` after the isinstance check (avoids E1134).
        for attr, cls in (
            ("lora", LoRAConfig),
            ("cnn", CNNHeadConfig),
            ("mlp", MLPHeadConfig),
        ):
            val = getattr(self, attr)
            if isinstance(val, dict):
                setattr(self, attr, cls(**val))  # pylint: disable=not-a-mapping
        if self.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError(
                f"precision must be one of fp32/fp16/bf16, got {self.precision!r}"
            )
        if self.head_precision not in ("fp32", "fp64"):
            raise ValueError(
                f"head_precision must be one of fp32/fp64, got {self.head_precision!r}"
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

HEAD_PRECISION_TORCH_DTYPE_NAMES = {
    "fp32": "float32",
    "fp64": "float64",
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
        for attr, cls in (("data", DataConfig), ("scheduler", SchedulerConfig)):
            val = getattr(self, attr)
            if isinstance(val, dict):
                setattr(self, attr, cls(**val))  # pylint: disable=not-a-mapping
        if self.best_metric not in ("loss", "f1"):
            raise ValueError(
                f"best_metric must be 'loss' or 'f1', got {self.best_metric!r}"
            )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentConfig:
    """Top-level config: one YAML file with a `model:` section (as before)
    and a `training:` section, so a single file fully specifies a run.
    """

    model: ESMCFineTuneConfig = field(default_factory=ESMCFineTuneConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        for attr, cls in (("model", ESMCFineTuneConfig), ("training", TrainingConfig)):
            val = getattr(self, attr)
            if isinstance(val, dict):
                setattr(self, attr, cls(**val))  # pylint: disable=not-a-mapping

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
