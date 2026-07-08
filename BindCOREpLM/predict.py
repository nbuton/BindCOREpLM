"""
Prediction script for BindCOREpLM.

Takes a trained model checkpoint and a test dataset in LIP format
and outputs a CSV file with protein_id, length, predictions (comma-separated
sigmoid probabilities), and binary_predictions (comma-separated 0/1 using
the optimal threshold stored in the checkpoint).

Usage
-----
    python predict.py --checkpoint outputs/run1/best.pt \\
                      --config outputs/run1/best.config.yaml \\
                      --input data/LIP_dataset/test.txt \\
                      --output predictions.csv
"""

from __future__ import annotations

import argparse
import csv

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import ExperimentConfig
from dataset import LIPDataset, LIPCollator
from model import ESMCResidueBindingModel


def load_model(
    checkpoint_path: str, config_path: str, device: str
) -> ESMCResidueBindingModel:
    """Load the model from a checkpoint, restoring trainable params and the
    optimal binary threshold stored in the checkpoint."""
    cfg = ExperimentConfig.from_yaml(config_path)
    model = ESMCResidueBindingModel(cfg.model).to(device)
    model.eval()

    state = torch.load(checkpoint_path, map_location=device)
    # Extract the binary threshold if present
    threshold = state.pop("_binary_threshold", None)
    if threshold is not None:
        model.binary_threshold = float(threshold.item())
        print(f"Loaded binary_threshold = {model.binary_threshold:.4f} from checkpoint")
    else:
        print(f"Using default binary_threshold = {model.binary_threshold}")

    # Load trainable parameters
    own_state = {n: p for n, p in model.named_parameters() if p.requires_grad}
    loaded_keys = set()
    for name, param in state.items():
        if name in own_state:
            own_state[name].data.copy_(param)
            loaded_keys.add(name)
    missing = set(own_state.keys()) - loaded_keys
    if missing:
        print(
            f"Warning: {len(missing)} parameters not found in checkpoint: {list(missing)[:5]}..."
        )

    return model


@torch.no_grad()
def predict(
    model: ESMCResidueBindingModel,
    dataset: LIPDataset,
    tokenizer: AutoTokenizer,
    max_length: int,
    batch_size: int,
    device: str,
) -> list[dict]:
    """Run inference on a dataset and return per-protein results.

    Returns a list of dicts with keys:
      - protein_id: str
      - length: int (number of residues)
      - probabilities: list[float] (sigmoid outputs per residue)
      - binary_predictions: list[int] (0/1 using model.binary_threshold)
    """
    collator = LIPCollator(tokenizer=tokenizer, max_length=max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    results = []
    model.eval()

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        ids = batch["ids"]

        logits = model(input_ids=input_ids, attention_mask=attention_mask)  # (B, L)
        probs = torch.sigmoid(logits.float())  # (B, L)

        for i, protein_id in enumerate(ids):
            # Identify real residue positions (non-padding)
            valid_mask = attention_mask[i].bool()
            residue_mask = valid_mask
            # Also find the special_tokens_mask equivalent: for ESM models,
            # the first and last non-padding tokens are typically <cls> and <eos>.
            # We use the tokenizer to identify them. Since LIPCollator uses
            # return_special_tokens_mask, we can replicate by checking for
            # non-padding tokens and then skipping first/last.
            # Simpler approach: use the original sequence length from the dataset.
            seq = dataset.samples[ids.index(protein_id)][1]  # get original sequence
            seq_len = min(len(seq), max_length)

            # Get the per-residue probabilities: the model outputs logits for
            # every token position (including special tokens). We need to align.
            # LIPCollator puts labels for the exact residue positions; we can
            # approximate by taking the first `seq_len` valid logit positions
            # after the <cls> token.
            valid_indices = torch.where(residue_mask)[0]
            if len(valid_indices) < seq_len + 2:
                # Fallback: just take non-padding, non-special positions
                # The first valid token is <cls>, last is <eos>
                residue_indices = valid_indices[1 : 1 + seq_len]
            else:
                residue_indices = valid_indices[1 : 1 + seq_len]

            probs_i = probs[i, residue_indices].cpu().tolist()
            # Trim to actual sequence length (might be shorter if truncation)
            probs_i = probs_i[:seq_len]

            threshold = model.binary_threshold
            binary_i = [1 if p >= threshold else 0 for p in probs_i]

            results.append(
                {
                    "protein_id": protein_id,
                    "length": len(probs_i),
                    "probabilities": probs_i,
                    "binary_predictions": binary_i,
                }
            )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with a trained BindCOREpLM model."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the model config YAML file"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input test data file in LIP format (.txt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Output CSV file path (default: predictions.csv)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference (default: 4)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load config and tokenizer
    cfg = ExperimentConfig.from_yaml(args.config)
    print(f"Loading tokenizer: {cfg.model.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, args.config, device)
    print(
        f"Model loaded. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    print(f"Using binary threshold: {model.binary_threshold:.4f}")

    # Load dataset
    print(f"Loading input data from: {args.input}")
    dataset = LIPDataset([args.input])
    print(f"Loaded {len(dataset)} proteins")

    # Run prediction
    print("Running inference...")
    results = predict(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=cfg.model.max_seq_length,
        batch_size=args.batch_size,
        device=device,
    )

    # Write output CSV
    print(f"Writing predictions to: {args.output}")
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["protein_id", "length", "predictions", "binary_predictions"])
        for res in results:
            probs_str = ",".join(f"{p:.6f}" for p in res["probabilities"])
            binary_str = ",".join(str(b) for b in res["binary_predictions"])
            writer.writerow([res["protein_id"], res["length"], probs_str, binary_str])

    print(f"Done! Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
