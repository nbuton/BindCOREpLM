"""
Minimal LoRA implementation, deliberately dependency-free (no `peft`)
so it's easy to see exactly what's frozen vs trainable, and easy to
adapt if ESM-C's module names differ from a standard HF attention block.

For standard HF models with separate k_proj/v_proj Linear modules, use
target_modules=['k_proj', 'v_proj'].

For ESM-C (biohub/ESMC-*), the QKV projections are fused into a single
_PyTorchLayerNormLinear (LayerNorm + one weight of shape (3*d_model, d_model)).
This module handles that: it detects fused QKV modules and injects LoRA
specifically on the K and V slices, matching the standard LoRA paper
recommendations (Hu et al., 2021).
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from BindCOREpLM.config import LoRAConfig


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank update.

    forward(x) = base(x) + scaling * dropout(x) @ A^T @ B^T
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


class LoRAFusedQKV(nn.Module):
    """Wraps a frozen fused QKV module (LayerNorm + weight of shape (3*H, H))
    with LoRA adapters specifically on the K and V slices.

    The fused weight W has shape (3*hidden, hidden) where:
        W[0:hidden, :]     -> Q projection
        W[hidden:2*hidden, :] -> K projection  <-- LoRA applied here
        W[2*hidden:3*hidden, :] -> V projection  <-- LoRA applied here
    """

    def __init__(self, fused_qkv: nn.Module, rank: int, alpha: int, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be a positive integer")

        self.fused_qkv = fused_qkv
        for p in self.fused_qkv.parameters():
            p.requires_grad = False

        self.weight = fused_qkv.weight
        self.hidden_size = fused_qkv.weight.shape[1]
        if fused_qkv.weight.shape[0] != 3 * self.hidden_size:
            raise ValueError(
                f"Expected fused QKV weight shape (3*H, H), got "
                f"{fused_qkv.weight.shape}"
            )

        self.rank = rank
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # LoRA on K slice
        self.lora_A_k = nn.Parameter(
            torch.randn(rank, self.hidden_size, dtype=torch.float32)
        )
        self.lora_B_k = nn.Parameter(
            torch.zeros(self.hidden_size, rank, dtype=torch.float32)
        )

        # LoRA on V slice
        self.lora_A_v = nn.Parameter(
            torch.randn(rank, self.hidden_size, dtype=torch.float32)
        )
        self.lora_B_v = nn.Parameter(
            torch.zeros(self.hidden_size, rank, dtype=torch.float32)
        )

        nn.init.kaiming_uniform_(self.lora_A_k, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused QKV forward (includes LayerNorm internally)
        qkv = self.fused_qkv(x)  # (..., 3*H)

        H = self.hidden_size
        q, k, v = qkv[..., :H], qkv[..., H : 2 * H], qkv[..., 2 * H :]

        # Apply LoRA to K and V slices
        x_fp32 = x.to(torch.float32)
        lora_in = self.lora_dropout(x_fp32)

        delta_k = (lora_in @ self.lora_A_k.t()) @ self.lora_B_k.t()
        delta_k = (self.scaling * delta_k).to(k.dtype)

        delta_v = (lora_in @ self.lora_A_v.t()) @ self.lora_B_v.t()
        delta_v = (self.scaling * delta_v).to(v.dtype)

        k = k + delta_k
        v = v + delta_v

        return torch.cat([q, k, v], dim=-1)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, rank={self.rank}, "
            f"scaling={self.scaling:.3f}"
        )


def _layer_index_from_name(name: str) -> Optional[int]:
    for part in name.split("."):
        if part.isdigit():
            return int(part)
    return None


def list_linear_module_names(model: nn.Module) -> List[str]:
    return [name for name, mod in model.named_modules() if isinstance(mod, nn.Linear)]


def list_fused_qkv_module_names(model: nn.Module) -> List[str]:
    result = []
    for name, mod in model.named_modules():
        w = getattr(mod, "weight", None)
        if w is not None and w.ndim == 2 and w.shape[0] == 3 * w.shape[1]:
            result.append(name)
    return result


def inject_lora_adapters(model: nn.Module, lora_config: LoRAConfig) -> int:
    """Replaces target submodules in-place with LoRA wrappers.

    Handles two types of targets:
      - Standard nn.Linear modules (matched by suffix name, e.g. 'k_proj')
      - Fused QKV modules (matched by suffix name 'layernorm_qkv' or 'qkv_proj')

    Returns the number of modules wrapped.
    """
    if lora_config.disable:
        return 0

    targets = set(lora_config.target_modules)
    n_wrapped = 0

    # --- Phase 1: Wrap standard nn.Linear modules ---
    to_wrap_linear = []
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
        to_wrap_linear.append(name)

    for name in to_wrap_linear:
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
        n_wrapped += 1

    # --- Phase 2: Wrap fused QKV modules (not nn.Linear, but have 3*H, H weight) ---
    fused_names = list_fused_qkv_module_names(model)
    fused_targets = {t for t in targets if "layernorm_qkv" in t or "qkv" in t.lower()}

    fused_to_wrap = []
    for name in fused_names:
        leaf_name = name.split(".")[-1]
        if leaf_name in fused_targets:
            if lora_config.layer_indices is not None:
                idx = _layer_index_from_name(name)
                if idx is None or idx not in lora_config.layer_indices:
                    continue
            fused_to_wrap.append(name)

    for name in fused_to_wrap:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        fused_module = getattr(parent, child_name)
        wrapped = LoRAFusedQKV(
            fused_module,
            rank=lora_config.rank,
            alpha=lora_config.alpha,
            dropout=lora_config.dropout,
        )
        setattr(parent, child_name, wrapped)
        n_wrapped += 1

    if n_wrapped == 0 and targets:
        available_linear = list_linear_module_names(model)
        available_fused = list_fused_qkv_module_names(model)
        raise ValueError(
            f"No modules matched target_modules="
            f"{lora_config.target_modules} (with layer_indices="
            f"{lora_config.layer_indices}).\n"
            f"First 10 linear module names: {available_linear[:10]}\n"
            f"Fused QKV module names: {available_fused}\n"
            f"\nSuggestions:\n"
            f"  - For ESM-C: set target_modules=['layernorm_qkv']\n"
            f"  - For standard HF models: set target_modules=['k_proj', 'v_proj']\n"
            f"\n"
            f"Use list_linear_module_names(model) and "
            f"list_fused_qkv_module_names(model) to inspect."
        )

    return n_wrapped


def count_trainable_parameters(model: nn.Module) -> "tuple[int, int]":
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
