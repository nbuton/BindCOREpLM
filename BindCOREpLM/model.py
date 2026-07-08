"""
ESM-C (LoRA fine-tuned) + multi-kernel 1D-CNN + MLP head for residue-level
binding-site prediction (LIP / MoRF style tasks).

Architecture
------------
    input_ids, attention_mask
        -> ESM-C backbone (LoRA adapters on K/V projections; rest frozen)
        -> per-residue hidden states (B, L, H)
        -> dropout
        -> N parallel Conv1d branches (different kernel sizes, 'same' padding)
        -> concat along channel dim -> (B, L, sum(out_channels))
        -> per-residue MLP (shared across positions) with high dropout
        -> per-residue logit(s) (B, L) for binary, or (B, L, num_labels)

Only the LoRA adapters + CNN branches + MLP head are trainable by default,
which matters a lot with an 800-protein fine-tuning set on top of a 6B backbone.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM

from config import (
    ESMCFineTuneConfig,
    PRECISION_TORCH_DTYPE_NAMES,
    HEAD_PRECISION_TORCH_DTYPE_NAMES,
)
from lora import inject_lora_adapters, count_trainable_parameters

_ACTIVATIONS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
}


def _resolve_dtype(precision: str) -> torch.dtype:
    return getattr(torch, PRECISION_TORCH_DTYPE_NAMES[precision])


class ESMCResidueBindingModel(nn.Module):
    def __init__(self, config: ESMCFineTuneConfig):
        super().__init__()
        self.config = config
        dtype = _resolve_dtype(config.precision)
        print(config.model_name_or_path)
        self.backbone = AutoModelForMaskedLM.from_pretrained(
            config.model_name_or_path,
            dtype=dtype,
        )

        if config.freeze_base_model:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Inject LoRA into the attention K/V projections. This is done
        # regardless of freeze_base_model -- freezing just means every
        # *other* parameter besides the LoRA A/B matrices stays fixed.
        transformer_root = self._get_transformer_root(self.backbone)
        n_wrapped = inject_lora_adapters(transformer_root, config.lora)
        self._n_lora_wrapped = n_wrapped

        if config.gradient_checkpointing:
            if hasattr(self.backbone, "gradient_checkpointing_enable"):
                try:
                    self.backbone.gradient_checkpointing_enable()
                except ValueError as e:
                    print(f"⚠️  Gradient checkpointing not supported by backbone: {e}")
                    print("   Training will proceed without gradient checkpointing.")

        hidden_size = self._infer_hidden_size(self.backbone)

        # ---- Multi-kernel CNN branches (residue-level: length preserving) ----
        self.input_dropout = nn.Dropout(config.cnn.input_dropout)
        self.cnn_branches = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=config.cnn.out_channels_per_branch,
                    kernel_size=k,
                    padding=k // 2,  # 'same' padding for odd kernel sizes
                )
                for k in config.cnn.kernel_sizes
            ]
        )
        cnn_out_dim = config.cnn.out_channels_per_branch * len(config.cnn.kernel_sizes)

        # ---- Per-residue MLP head, applied position-wise via nn.Linear ----
        act_cls = _ACTIVATIONS[config.mlp.activation]
        head_layers = []
        in_dim = cnn_out_dim
        for h in config.mlp.hidden_dims:
            head_layers += [
                nn.Linear(in_dim, h),
                act_cls(),
                nn.Dropout(config.mlp.dropout),
            ]
            in_dim = h
        head_layers.append(nn.Linear(in_dim, config.mlp.num_labels))
        self.head = nn.Sequential(*head_layers)

        # CNN branches + head are always trainable, run in fp32 for stability
        # even when the backbone is bf16/fp16.
        _head_dtype = _resolve_dtype(config.head_precision)
        self.cnn_branches.to(_head_dtype)
        self.head.to(_head_dtype)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_transformer_root(backbone: nn.Module) -> nn.Module:
        """Locate the underlying transformer stack inside the
        AutoModelForMaskedLM wrapper (name varies by architecture)."""
        for attr in ("esm", "model", "transformer", "encoder", "base_model"):
            if hasattr(backbone, attr):
                return getattr(backbone, attr)
        return backbone

    @staticmethod
    def _infer_hidden_size(backbone: nn.Module) -> int:
        cfg = backbone.config
        for attr in ("hidden_size", "d_model", "n_embd"):
            if hasattr(cfg, attr):
                return getattr(cfg, attr)
        raise ValueError(
            "Could not infer hidden size from backbone.config; "
            "inspect backbone.config and set it manually."
        )

    def trainable_parameter_summary(self) -> str:
        trainable, total = count_trainable_parameters(self)
        pct = 100 * trainable / total if total else 0.0
        return (
            f"LoRA adapters wrapped: {self._n_lora_wrapped} linear layers | "
            f"Trainable params: {trainable:,} / {total:,} ({pct:.3f}%)"
        )

    # ------------------------------------------------------------------ #
    # Threshold for binary predictions (optimised at end of training)
    # ------------------------------------------------------------------ #
    binary_threshold: float = 0.5

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]  # (B, L, H)

        # Run the head in fp32 regardless of backbone precision, for
        # numerically stable dropout/CNN/BCE-with-logits.
        head_dtype = _resolve_dtype(self.config.head_precision)
        hidden = hidden.to(head_dtype)
        hidden = self.input_dropout(hidden)

        x = hidden.transpose(1, 2)  # (B, H, L) for Conv1d
        branch_outputs = [branch(x) for branch in self.cnn_branches]  # each (B, C, L)
        cat = torch.cat(branch_outputs, dim=1)  # (B, C * num_branches, L)
        cat = cat.transpose(1, 2)  # (B, L, C * num_branches)

        logits = self.head(cat)  # (B, L, num_labels)
        if self.config.mlp.num_labels == 1:
            logits = logits.squeeze(-1)  # (B, L)
        return logits

    def compute_masked_bce_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Binary cross-entropy over residues, ignoring padding positions.

        labels: (B, L) float tensor of 0/1 (or -100 to also mask specific
                residues, e.g. unresolved/unlabeled positions).
        attention_mask: (B, L), 1 for real tokens, 0 for padding.
        pos_weight: optional scalar/tensor to upweight the positive
                (binding) class -- binding residues are typically a small
                minority, so this is worth tuning.
        """
        valid = (attention_mask.bool()) & (labels != -100)
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        labels_clamped = labels.clamp(min=0.0)
        per_residue_loss = loss_fct(logits, labels_clamped.float())
        per_residue_loss = per_residue_loss * valid.float()
        denom = valid.float().sum().clamp(min=1.0)
        return per_residue_loss.sum() / denom
