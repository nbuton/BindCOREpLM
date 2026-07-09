"""
Inference script for BindCOREpLM.

Takes a trained model checkpoint, a LIP-format file, and produces a CSV
with per-residue probabilities and binary predictions.

The checkpoint can be a **self-contained** ``.pt`` file (with the model
config embedded inside it) — in that case ``--config`` is not needed.
For backward compatibility with checkpoints that do *not* contain an
embedded config, the ``--config`` argument is still supported.

Usage
-----
    # Self-contained checkpoint (config embedded)
    python -m BindCOREpLM.inference.predict --checkpoint outputs/run1/best.pt \
                            --input data/LIP_dataset/test.txt \
                            --output predictions.csv

    # Legacy mode (separate config file)
    python -m BindCOREpLM.inference.predict --checkpoint outputs/run1/best.pt \
                            --config outputs/run1/best.config.yaml \
                            --input data/LIP_dataset/test.txt \
                            --output predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from BindCOREpLM.config import ESMCFineTuneConfig
from BindCOREpLM.data.dataset import LIPDataset, LIPCollator
from BindCOREpLM.models.model import ESMCResidueBindingModel


def _load_config_from_checkpoint(checkpoint_path: str) -> ESMCFineTuneConfig:
    """Reconstruct model config from the embedded ``model_config_dict``."""
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_dict = raw.get("model_config_dict")
    if config_dict is None:
        raise ValueError(
            f"No embedded config found in {checkpoint_path}. "
            "Either use a checkpoint trained with the updated trainer, "
            "or provide --config manually."
        )
    return ESMCFineTuneConfig(**config_dict)


def _load_state_into_model(
    model: ESMCResidueBindingModel,
    checkpoint_path: str,
    device: torch.device,
) -> float | None:
    """Load trainable weights from checkpoint; return saved binary threshold.

    Handles both new-style checkpoints (nested dict with ``model_state_dict``)
    and legacy flat checkpoints (just a flat dict of parameter tensors).
    """
    state = torch.load(checkpoint_path, map_location="cpu")

    # New-style checkpoint (nested dict with "model_state_dict" key)
    if "model_state_dict" in state:
        weights = state["model_state_dict"]
        threshold_tensor = weights.pop("_binary_threshold", None)
    else:
        # Legacy flat checkpoint (just a dict of parameter tensors)
        weights = state
        threshold_tensor = weights.pop("_binary_threshold", None)

    own_state = {n: p for n, p in model.named_parameters() if p.requires_grad}
    for name, param in weights.items():
        if name in own_state:
            own_state[name].data.copy_(param)
        else:
            print(f"  Warning: parameter {name!r} not found in model, skipping.")

    if threshold_tensor is not None:
        return float(threshold_tensor.item())
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with a trained BindCOREpLM model."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint (.pt file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model config YAML file (optional if checkpoint has embedded config)",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input LIP-format file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Output CSV path (default: predictions.csv)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Inference batch size (default: 4)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Binary threshold (default: use the one saved in checkpoint)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cuda, mps, or cpu (default: auto)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Device
    # ------------------------------------------------------------------ #
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # ------------------------------------------------------------------ #
    # Config  (from checkpoint or separate YAML)
    # ------------------------------------------------------------------ #
    if args.config is not None:
        # Legacy mode: load config from separate YAML file
        from BindCOREpLM.config import ExperimentConfig

        cfg = ExperimentConfig.from_yaml(args.config)
        model_cfg = cfg.model
        print(f"Using config from: {args.config}")
    else:
        # Self-contained mode: config is embedded in the checkpoint
        model_cfg = _load_config_from_checkpoint(args.checkpoint)
        print(f"Using embedded config from checkpoint")

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    model = ESMCResidueBindingModel(model_cfg)
    saved_threshold = _load_state_into_model(model, args.checkpoint, device)
    model.to(device)
    model.eval()

    if args.threshold is not None:
        model.binary_threshold = args.threshold
        print(f"Using provided threshold: {args.threshold}")
    elif saved_threshold is not None:
        model.binary_threshold = saved_threshold
        print(f"Using checkpoint threshold: {model.binary_threshold:.4f}")
    else:
        print(f"Using default threshold: {model.binary_threshold}")

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name_or_path)
    dataset = LIPDataset([args.input])
    collator = LIPCollator(
        tokenizer=tokenizer,
        max_length=model_cfg.max_seq_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )
    print(f"Loaded {len(dataset)} proteins from {args.input}")

    # Inference
    results = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(input_ids, attention_mask=attention_mask)
            probabilities = torch.sigmoid(logits)
            binary = (probabilities > model.binary_threshold).long()

            for i, pid in enumerate(batch["ids"]):
                seq_len = attention_mask[i].sum().item()
                probs = probabilities[i, :seq_len].cpu().tolist()
                bins = binary[i, :seq_len].cpu().tolist()
                results.append(
                    {
                        "protein_id": pid,
                        "length": seq_len,
                        "probabilities": ",".join(f"{p:.6f}" for p in probs),
                        "binary_predictions": ",".join(str(b) for b in bins),
                    }
                )

    # Save CSV
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["protein_id", "length", "probabilities", "binary_predictions"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\u2705 Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
