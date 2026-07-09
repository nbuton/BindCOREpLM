"""
Minimal LoRA implementation, deliberately dependency-free (no `peft`)
so it's easy to see exactly what's frozen vs trainable, and easy to
adapt if ESM-C's module names differ from a standard HF attention block.

Available target options for ``target_modules``:

``layernorm_qkv``
    Fused QKV projection (``_PyTorchLayerNormLinear``) — the primary
    LoRA target for ESM-C (K and V slices in the fused ``(3*H, H)``
    weight).

``ffn``
    Fused SwiGLU FFN (``_PyTorchLayerNormMLP``) — injects LoRA on
    **both** the ``fc1`` (gate + up) and ``fc2`` (down) projections.

``out_proj``
    Standard ``nn.Linear`` output projection in each attention head
    (already handled by the linear matching Phase 1).

``k_proj``, ``v_proj``
    For standard HF models with separate K/V linear projections
    (e.g. BERT, LLaMA).  **Not available in ESM-C** — it uses a
    fused ``layernorm_qkv`` instead.

Usage examples
--------------
Standard HF models with separate k_proj/v_proj Linear modules::

    target_modules=['k_proj', 'v_proj']

ESM-C (biohub/ESMC-*) with fused QKV — apply LoRA to K/V in attention::

    target_modules=['layernorm_qkv']

ESM-C — apply LoRA to K/V in attention AND both projections in the FFN::

    target_modules=['layernorm_qkv', 'ffn']

ESM-C — also add LoRA to the attention output projection::

    target_modules=['layernorm_qkv', 'ffn', 'out_proj']

Use ``list_linear_module_names(model)``,
``list_fused_qkv_module_names(model)`` and
``list_fused_ffn_module_names(model)`` to inspect available modules.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F  # pylint: disable=not-callable

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


class LoRAFusedMLP(nn.Module):
    """Wraps a frozen fused SwiGLU FFN (``_PyTorchLayerNormMLP``) with
    LoRA adapters on **both** the ``fc1`` (gate + up) and ``fc2`` (down)
    projections.

    The ``_PyTorchLayerNormMLP`` forward is::

        x = layer_norm(x)
        x = linear(x, fc1_weight)          # (H,) -> (2 * ffn_hidden_size,)
        x1, x2 = x.chunk(2, dim=-1)
        x = silu(x1) * x2                  # SwiGLU
        return linear(x, fc2_weight)        # (ffn_hidden_size,) -> (H,)

    LoRA is applied *after* the LayerNorm but *before* each linear,
    matching the standard ``output += scaling * dropout(x) @ A^T @ B^T``
    pattern.
    """

    def __init__(
        self,
        fused_mlp: nn.Module,
        rank: int,
        alpha: int,
        dropout: float,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be a positive integer")

        self.fused_mlp = fused_mlp
        for p in self.fused_mlp.parameters():
            p.requires_grad = False

        self.hidden_size = fused_mlp.hidden_size
        self.ffn_hidden_size = fused_mlp.ffn_hidden_size

        self.rank = rank
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # ---- LoRA on fc1: (2 * ffn_hidden_size, hidden_size) ----
        self.lora_A_fc1 = nn.Parameter(
            torch.randn(rank, self.hidden_size, dtype=torch.float32)
        )
        self.lora_B_fc1 = nn.Parameter(
            torch.zeros(2 * self.ffn_hidden_size, rank, dtype=torch.float32)
        )

        # ---- LoRA on fc2: (hidden_size, ffn_hidden_size) ----
        self.lora_A_fc2 = nn.Parameter(
            torch.randn(rank, self.ffn_hidden_size, dtype=torch.float32)
        )
        self.lora_B_fc2 = nn.Parameter(
            torch.zeros(self.hidden_size, rank, dtype=torch.float32)
        )

        nn.init.kaiming_uniform_(self.lora_A_fc1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_fc2, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ---- LayerNorm (same as fused_mlp.forward) ----
        x = F.layer_norm(
            x,
            (self.hidden_size,),
            self.fused_mlp.layer_norm_weight,
            self.fused_mlp.layer_norm_bias,
            self.fused_mlp.eps,
        )

        # ---- fc1 (gate + up projections, fused) ----
        x_fc1 = F.linear(x, self.fused_mlp.fc1_weight)

        # LoRA delta on fc1
        x_fp32 = x.to(torch.float32)
        lora_in = self.lora_dropout(x_fp32)
        delta_fc1 = (lora_in @ self.lora_A_fc1.t()) @ self.lora_B_fc1.t()
        x_fc1 = x_fc1 + self.scaling * delta_fc1.to(x_fc1.dtype)

        # ---- SwiGLU activation ----
        x1, x2 = x_fc1.chunk(2, dim=-1)
        x_act = F.silu(x1) * x2

        # ---- fc2 (down projection) ----
        x_out = F.linear(x_act, self.fused_mlp.fc2_weight)

        # LoRA delta on fc2
        act_fp32 = x_act.to(torch.float32)
        lora_in2 = self.lora_dropout(act_fp32)
        delta_fc2 = (lora_in2 @ self.lora_A_fc2.t()) @ self.lora_B_fc2.t()
        x_out = x_out + self.scaling * delta_fc2.to(x_out.dtype)

        return x_out

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"ffn_hidden_size={self.ffn_hidden_size}, "
            f"rank={self.rank}, scaling={self.scaling:.3f}"
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


def list_fused_ffn_module_names(model: nn.Module) -> List[str]:
    """Returns names of modules that have both ``fc1_weight`` and
    ``fc2_weight`` attributes (i.e. ``_PyTorchLayerNormMLP`` or similar
    fused SwiGLU FFN modules).

    This is a structural duck‑typing check — it matches any module whose
    set of named parameter keys includes ``fc1_weight`` and ``fc2_weight``.
    """
    result = []
    for name, mod in model.named_modules():
        param_keys = {k for k, _ in mod.named_parameters()}
        if "fc1_weight" in param_keys and "fc2_weight" in param_keys:
            result.append(name)
    return result


def inject_lora_adapters(model: nn.Module, lora_config: LoRAConfig) -> int:
    """Replaces target submodules in-place with LoRA wrappers.

    Handles three types of targets:

    **Phase 1 — Standard nn.Linear modules** (e.g. ``out_proj``, ``k_proj``,
    ``v_proj``, ``q_proj``).
    Matched by their leaf (last segment) name.

    **Phase 2 — Fused QKV modules** (``_PyTorchLayerNormLinear``, detected
    by a ``(3*H, H)`` weight).  Matched by leaf name ``layernorm_qkv`` or
    any name containing ``qkv``.

    **Phase 3 — Fused SwiGLU FFN modules** (``_PyTorchLayerNormMLP``,
    detected by the presence of ``fc1_weight`` and ``fc2_weight``
    parameters).  Matched by leaf name ``ffn``.

    Returns the number of modules wrapped.
    """
    if lora_config.disable:
        return 0

    targets = set(lora_config.target_modules)
    n_wrapped = 0

    # ---- Phase 1: Standard nn.Linear modules ----
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

    # ---- Phase 2: Fused QKV modules (3*H, H weight) ----
    fused_qkv_names = list_fused_qkv_module_names(model)
    fused_qkv_targets = {
        t for t in targets if "layernorm_qkv" in t or "qkv" in t.lower()
    }

    for name in fused_qkv_names:
        leaf_name = name.split(".")[-1]
        if leaf_name not in fused_qkv_targets:
            continue
        if lora_config.layer_indices is not None:
            idx = _layer_index_from_name(name)
            if idx is None or idx not in lora_config.layer_indices:
                continue
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

    # ---- Phase 3: Fused SwiGLU FFN modules (_PyTorchLayerNormMLP) ----
    fused_ffn_names = list_fused_ffn_module_names(model)
    fused_ffn_targets = {"ffn"}

    for name in fused_ffn_names:
        leaf_name = name.split(".")[-1]
        if leaf_name not in fused_ffn_targets:
            continue
        if "ffn" not in targets:
            continue
        if lora_config.layer_indices is not None:
            idx = _layer_index_from_name(name)
            if idx is None or idx not in lora_config.layer_indices:
                continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        fused_module = getattr(parent, child_name)
        wrapped = LoRAFusedMLP(
            fused_module,
            rank=lora_config.rank,
            alpha=lora_config.alpha,
            dropout=lora_config.dropout,
        )
        setattr(parent, child_name, wrapped)
        n_wrapped += 1

    # ---- Helpful error if nothing matched ----
    if n_wrapped == 0 and targets:
        available_linear = list_linear_module_names(model)
        available_fused_qkv = list_fused_qkv_module_names(model)
        available_fused_ffn = list_fused_ffn_module_names(model)
        raise ValueError(
            f"No modules matched target_modules="
            f"{lora_config.target_modules} (with layer_indices="
            f"{lora_config.layer_indices}).\n"
            f"First 10 linear module names: {available_linear[:10]}\n"
            f"Fused QKV module names: {available_fused_qkv}\n"
            f"Fused FFN module names: {available_fused_ffn}\n"
            f"\nSuggestions:\n"
            f"  - For ESM-C with fused QKV: target_modules=['layernorm_qkv']\n"
            f"  - For ESM-C with fused QKV + FFN: "
            f"target_modules=['layernorm_qkv', 'ffn']\n"
            f"  - For standard HF models: target_modules=['k_proj', 'v_proj']\n"
            f"\n"
            f"Use list_linear_module_names(model),\n"
            f"list_fused_qkv_module_names(model) and\n"
            f"list_fused_ffn_module_names(model) to inspect."
        )

    return n_wrapped


def count_trainable_parameters(model: nn.Module) -> "tuple[int, int]":
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
