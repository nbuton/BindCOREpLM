"""
Minimal LoRA implementation, deliberately dependency-free (no `peft`)
so it's easy to see exactly what's frozen vs trainable, and easy to
adapt if ESM-C's module names differ from a standard HF attention block.

Usage
-----
>>> from transformers import AutoModelForMaskedLM
>>> backbone = AutoModelForMaskedLM.from_pretrained("biohub/ESMC-6B")
>>> from config import LoRAConfig
>>> num_wrapped = inject_lora_adapters(backbone, LoRAConfig(rank=8, alpha=16))
>>> print(f"Wrapped {num_wrapped} linear layers with LoRA")
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn

from config import LoRAConfig


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank update.

    forward(x) = base(x) + scaling * dropout(x) @ A^T @ B^T

    A is (rank, in_features), B is (out_features, rank). B is zero-init
    so the adapter starts as a no-op (matches the original LoRA paper).
    """

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: int, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be a positive integer")

        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.rank = rank
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Keep adapter params in fp32 for stable optimization even if the
        # base model is loaded in bf16/fp16; autocast/mixed precision
        # training will still cast activations appropriately.
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_in = self.lora_dropout(x).to(self.lora_A.dtype)
        lora_out = (lora_in @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + self.scaling * lora_out.to(base_out.dtype)

    def extra_repr(self) -> str:
        return f"rank={self.rank}, scaling={self.scaling:.3f}"


def _layer_index_from_name(name: str) -> Optional[int]:
    """Best-effort extraction of a transformer layer index from a dotted
    module name such as '...layers.12.self_attn.k_proj' -> 12.
    Returns None if no integer segment is found.
    """
    for part in name.split("."):
        if part.isdigit():
            return int(part)
    return None


def list_linear_module_names(model: nn.Module) -> List[str]:
    """Debug helper: list all nn.Linear module names in a model, useful
    for figuring out the correct `target_modules` suffixes if ESM-C's
    attention projections aren't named k_proj/v_proj.
    """
    return [name for name, mod in model.named_modules() if isinstance(mod, nn.Linear)]


def inject_lora_adapters(model: nn.Module, lora_config: LoRAConfig) -> int:
    """Replaces target nn.Linear submodules in-place with LoRALinear wrappers.

    Matching rule: a module is wrapped if its dotted name's final
    component matches one of `lora_config.target_modules` exactly
    (e.g. "k_proj"), AND (if `layer_indices` is set) the layer index
    parsed from its name is in that list.

    Returns the number of linear layers wrapped.
    """
    targets = set(lora_config.target_modules)
    to_wrap = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf_name = name.split(".")[-1]
        if leaf_name not in targets:
            continue
        if lora_config.layer_indices is not None:
            idx = _layer_index_from_name(name)
            if idx is None or idx not in lora_config.layer_indices:
                continue
        to_wrap.append(name)

    if not to_wrap:
        available = list_linear_module_names(model)
        raise ValueError(
            "No linear modules matched target_modules="
            f"{lora_config.target_modules} (with layer_indices="
            f"{lora_config.layer_indices}). First 10 linear module names "
            f"found in the model: {available[:10]}. Use "
            "list_linear_module_names(model) to inspect all of them and "
            "adjust LoRAConfig.target_modules accordingly."
        )

    for name in to_wrap:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        base_linear = getattr(parent, child_name)
        wrapped = LoRALinear(
            base_linear,
            rank=lora_config.rank,
            alpha=lora_config.alpha,
            dropout=lora_config.dropout,
        )
        setattr(parent, child_name, wrapped)

    return len(to_wrap)


def count_trainable_parameters(model: nn.Module) -> "tuple[int, int]":
    """Returns (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
