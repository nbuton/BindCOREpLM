"""
Merges (fuses) trained LoRA adapter weights into the base model, producing
a single set of full-rank weights.  After merging, the model no longer
depends on the LoRA adapter and can be used directly.

Usage
-----
    python -m BindCOREpLM.models.merge_lora --checkpoint outputs/run1/best.pt \
                         --config outputs/run1/best.config.yaml \
                         --output merged_model

The merged model is saved as:
    merged_model/
        pytorch_model.bin         (full state dict of ESMCResidueBindingModel)
        config.json               (copy of the model config YAML)
        merged_metadata.yaml      (notes about the merge)
"""

from __future__ import annotations

import argparse
import os
import shutil

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from BindCOREpLM.config import ExperimentConfig
from BindCOREpLM.models.lora import LoRALinear, LoRAFusedQKV
from BindCOREpLM.models.model import ESMCResidueBindingModel


def merge_lora_into_base(model: nn.Module) -> None:
    """Walk through *model* and fuse every LoRALinear / LoRAFusedQKV wrapper
    in-place, leaving a model with standard nn.Linear (or the original
    fused-QKV module) and the LoRA contributions baked into the weights.
    """
    # Collect all replaced names first so we don't mutate during iteration
    replacements = {}  # module_name -> new_module

    for name, module in model.named_modules():
        # ---- Standard LoRALinear ----
        if isinstance(module, LoRALinear):
            base_lin = module.base  # original nn.Linear
            # Compute merged weight: W' = W_base + scaling * (B @ A)
            merged_weight = base_lin.weight.data.to(torch.float32) + module.scaling * (
                module.lora_B @ module.lora_A
            ).to(torch.float32)

            new_linear = nn.Linear(
                in_features=base_lin.in_features,
                out_features=base_lin.out_features,
                bias=base_lin.bias is not None,
            )
            new_linear.weight.data = merged_weight.to(base_lin.weight.dtype)
            if base_lin.bias is not None:
                new_linear.bias.data = base_lin.bias.data.clone()

            replacements[name] = new_linear

        # ---- Fused QKV wrapper (ESM-C style) ----
        elif isinstance(module, LoRAFusedQKV):
            fused = (
                module.fused_qkv
            )  # original fused module (e.g. _PyTorchLayerNormLinear)
            H = module.hidden_size

            # Clone the original weight and add the K/V deltas
            updated_weight = fused.weight.data.to(torch.float32).clone()

            delta_k = module.scaling * (module.lora_B_k @ module.lora_A_k)
            delta_v = module.scaling * (module.lora_B_v @ module.lora_A_v)

            # K slice is rows H:2*H, V slice is rows 2*H:3*H
            updated_weight[H : 2 * H, :] += delta_k.to(torch.float32)
            updated_weight[2 * H : 3 * H, :] += delta_v.to(torch.float32)

            fused.weight.data = updated_weight.to(fused.weight.dtype)

            # The LoRA wrapper is no longer needed; replace it with the
            # original fused module (whose weight is now updated).
            replacements[name] = fused

    # Apply the replacements (walk from leaf to root to keep things safe)
    for name, new_module in replacements.items():
        parent_name, _, child_name = name.rpartition(".")
        if parent_name == "":
            parent = model
        else:
            parent = model.get_submodule(parent_name)
        setattr(parent, child_name, new_module)

    n_replaced = len(replacements)
    if n_replaced == 0:
        print("\u26a0\ufe0f  No LoRA wrappers found \u2013 nothing to merge.")
    else:
        print(f"\u2705 Merged {n_replaced} LoRA adapter(s) into the base model.")


def save_merged_model(model: nn.Module, output_dir: str) -> None:
    """Save the merged model state dict along with metadata."""
    os.makedirs(output_dir, exist_ok=True)

    # Full state dict (including the CNN/MLP head and backbone)
    torch.save(
        model.state_dict(),
        os.path.join(output_dir, "pytorch_model.bin"),
    )
    print(f"\U0001f4be Saved merged model to {output_dir}/pytorch_model.bin")


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters into the base model weights."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the LoRA checkpoint (.pt file from training)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the model config YAML file (e.g., best.config.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="merged_model",
        help="Output directory for the merged model (default: merged_model)",
    )
    args = parser.parse_args()

    device = "cpu"  # merging on CPU is safest (avoid dtype mismatches)

    # ------------------------------------------------------------------ #
    # Load the model (with LoRA wrappers) and the trained adapter weights
    # ------------------------------------------------------------------ #
    cfg = ExperimentConfig.from_yaml(args.config)
    print(f"Loading base model: {cfg.model.model_name_or_path}")
    model = ESMCResidueBindingModel(cfg.model).to(device)
    model.eval()

    # Load trainable state dict (LoRA A/B + CNN/MLP head)
    print(f"Loading checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location=device)

    # Extract and store the optimal binary threshold
    threshold = state.pop("_binary_threshold", None)

    own_state = {n: p for n, p in model.named_parameters() if p.requires_grad}
    loaded_keys = set()
    for name, param in state.items():
        if name in own_state:
            own_state[name].data.copy_(param)
            loaded_keys.add(name)
    missing = set(own_state.keys()) - loaded_keys
    if missing:
        print(
            f"\u26a0\ufe0f  {len(missing)} parameters not found in checkpoint: "
            f"{list(missing)[:5]}..."
        )

    if threshold is not None:
        model.binary_threshold = float(threshold.item())
        print(f"   Loaded binary_threshold = {model.binary_threshold}")
    else:
        print("   No binary_threshold in checkpoint \u2013 keeping default (0.5)")

    # ------------------------------------------------------------------ #
    # Merge LoRA into the base weights
    # ------------------------------------------------------------------ #
    merge_lora_into_base(model)

    # ------------------------------------------------------------------ #
    # Save the merged model
    # ------------------------------------------------------------------ #
    save_merged_model(model, args.output)

    # Also copy the config YAML for easy loading later
    shutil.copy2(
        args.config,
        os.path.join(args.output, "config.yaml"),
    )
    print(f"\U0001f4c4 Copied config to {args.output}/config.yaml")

    # Write a small metadata file
    with open(os.path.join(args.output, "merged_metadata.yaml"), "w") as f:
        f.write(f"checkpoint: {args.checkpoint}\n")
        f.write(f"base_model: {cfg.model.model_name_or_path}\n")
        f.write(f"binary_threshold: {model.binary_threshold}\n")
        f.write("LoRA adapters: merged\n")
    print(f"\U0001f4c4 Wrote metadata to {args.output}/merged_metadata.yaml")

    print("\n\u2705 Done. The merged model is stored in:", args.output)
    print("   To use it, load the state dict into an ESMCResidueBindingModel")
    print("   with LoRA injection disabled (or simply skip the LoRA step).")


if __name__ == "__main__":
    main()