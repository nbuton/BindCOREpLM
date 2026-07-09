"""
Evaluation script for BindCOREpLM predictions.

Takes a predictions CSV file (as produced by predict.py) and the true
labels file in LIP format, and produces a table of evaluation metrics:
MCC, F1, Precision, Recall, AUROC, AUPR, Average Precision.

Usage
-----
    python -m BindCOREpLM.inference.evaluate --predictions predictions.csv \\
                       --true_labels data/LIP_dataset/test.txt
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np
from sklearn.metrics import (
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
    precision_recall_curve,
    auc,
    average_precision_score,
)

from BindCOREpLM.data.dataset import parse_lip_file


def load_predictions(path: str) -> dict[str, dict]:
    """Load predictions CSV into a dict keyed by protein_id."""
    predictions = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["protein_id"]
            length = int(row["length"])
            probs = [float(x) for x in row["probabilities"].split(",")]
            binaries = [int(x) for x in row["binary_predictions"].split(",")]
            predictions[pid] = {
                "length": length,
                "probabilities": probs,
                "binary": binaries,
            }
    return predictions


def load_true_labels(lip_file: str) -> dict[str, list[int]]:
    """Load true labels from a LIP-format file.

    Returns a dict mapping protein_id -> list[int] of labels (0/1, ignoring '-')
    """
    samples = parse_lip_file(lip_file)
    true_labels = {}
    for pid, seq, labels in samples:
        true_labels[pid] = labels
    return true_labels


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate BindCOREpLM predictions against ground truth."
    )
    parser.add_argument(
        "--predictions", type=str, required=True, help="Path to predictions CSV file"
    )
    parser.add_argument(
        "--true_labels",
        type=str,
        required=True,
        help="Path to true labels file in LIP format",
    )
    args = parser.parse_args()

    # Load data
    print(f"Loading predictions from: {args.predictions}")
    preds = load_predictions(args.predictions)
    print(f"  Found {len(preds)} proteins")

    print(f"Loading true labels from: {args.true_labels}")
    true = load_true_labels(args.true_labels)
    print(f"  Found {len(true)} proteins")

    # Build aligned flat arrays, ignoring '-' positions in true labels
    all_probs = []
    all_binary = []
    all_true = []

    missing_ids = []
    for pid, true_labels in true.items():
        if pid not in preds:
            missing_ids.append(pid)
            continue
        p = preds[pid]
        seq_len = min(len(true_labels), len(p["probabilities"]))
        if seq_len == 0:
            continue
        for i in range(seq_len):
            t = true_labels[i]
            if t == 0 or t == 1:
                all_probs.append(p["probabilities"][i])
                all_binary.append(p["binary"][i])
                all_true.append(t)

    if missing_ids:
        print(
            f"Error: {len(missing_ids)} protein(s) in true labels not found in predictions: "
            f"{missing_ids[:5]}..."
        )
        print("Aborting evaluation — ensure all target proteins have predictions.")
        sys.exit(1)

    all_probs = np.array(all_probs)
    all_binary = np.array(all_binary)
    all_true = np.array(all_true)

    if len(all_true) == 0:
        print("Error: no valid residues to evaluate (all ignored?).")
        sys.exit(1)

    print(f"\nEvaluating on {len(all_true):,} residues across {len(preds)} proteins")
    print(
        f"  Positive residues: {all_true.sum():,} / {len(all_true):,} "
        f"({100 * all_true.mean():.2f}%)"
    )

    # --- Metrics ---
    # Threshold-based (using binary predictions from optimal threshold)
    mcc = matthews_corrcoef(all_true, all_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true, all_binary, average="binary", zero_division=0
    )

    # Threshold-free (using probabilities)
    auroc = roc_auc_score(all_true, all_probs)
    precision_pts, recall_pts, _ = precision_recall_curve(all_true, all_probs)
    aupr = auc(recall_pts, precision_pts)
    avg_precision = average_precision_score(all_true, all_probs)

    # --- Print results table ---
    print()
    print("=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  {'Metric':<25} {'Value':<12}")
    print("  " + "-" * 37)
    print(f"  {'MCC':<25} {mcc:<12.6f}")
    print(f"  {'F1 Score':<25} {f1:<12.6f}")
    print(f"  {'Precision':<25} {precision:<12.6f}")
    print(f"  {'Recall':<25} {recall:<12.6f}")
    print(f"  {'AUROC':<25} {auroc:<12.6f}")
    print(f"  {'AUPR (trapezoidal)':<25} {aupr:<12.6f}")
    print(f"  {'Average Precision (AP)':<25} {avg_precision:<12.6f}")
    print("=" * 60)

    # Also output as JSON for programmatic consumption
    import json

    results = {
        "mcc": float(mcc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "auroc": float(auroc),
        "aupr": float(aupr),
        "average_precision": float(avg_precision),
        "n_residues": int(len(all_true)),
        "n_positive": int(all_true.sum()),
    }
    print(f"\nJSON: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
