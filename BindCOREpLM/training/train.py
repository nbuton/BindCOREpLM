"""
Training loop for BindCOREpLM with MLflow logging, early stopping,
and best-checkpoint selection.

Usage
-----
    python -m BindCOREpLM.training.train --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from BindCOREpLM.config import ExperimentConfig
from BindCOREpLM.data.dataset import build_datasets, LIPCollator
from BindCOREpLM.models.model import ESMCResidueBindingModel


def _find_optimal_threshold(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Find best binary threshold via F1 on validation set."""
    thresholds = torch.linspace(0.05, 0.95, 91)
    best_f1 = 0.0
    best_thr = 0.5
    for thr in thresholds:
        preds = (probabilities > thr).float()
        tp = ((preds == 1) & (labels == 1)).sum().float()
        fp = ((preds == 1) & (labels == 0)).sum().float()
        fn = ((preds == 0) & (labels == 1)).sum().float()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr.item()
    return best_thr


def _compute_f1_and_threshold(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[float, float]:
    """Compute F1 at a given threshold."""
    preds = (probabilities > threshold).float()
    tp = ((preds == 1) & (labels == 1)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1.item(), precision.item(), recall.item()


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    metrics: dict,
    path: str,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    trainable_state = {
        n: p.data.cpu()
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    trainable_state["_binary_threshold"] = torch.tensor(model.binary_threshold)
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": trainable_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(trainable_state, path)


def main():
    parser = argparse.ArgumentParser(
        description="Train BindCOREpLM with LoRA fine-tuning."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: local ./mlruns)",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="BindCOREpLM",
        help="MLflow experiment name",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Config
    # ------------------------------------------------------------------ #
    cfg = ExperimentConfig.from_yaml(args.config)
    train_cfg = cfg.training
    model_cfg = cfg.model

    # Reproducibility
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)

    # ------------------------------------------------------------------ #
    # Device
    # ------------------------------------------------------------------ #
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name_or_path)
    train_ds, valid_ds, test_ds = build_datasets(
        data_dir=train_cfg.data.data_dir,
        train_file=train_cfg.data.train_file,
        valid_file=train_cfg.data.valid_file,
        test_file=train_cfg.data.test_file,
        combine_train_valid_for_final=train_cfg.data.combine_train_valid_for_final,
    )

    collator = LIPCollator(
        tokenizer=tokenizer,
        max_length=model_cfg.max_seq_length,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        collate_fn=collator,
        pin_memory=True if device.type == "cuda" else False,
    )

    if valid_ds is not None:
        valid_loader = DataLoader(
            valid_ds,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=train_cfg.num_workers,
            collate_fn=collator,
            pin_memory=True if device.type == "cuda" else False,
        )
    else:
        valid_loader = None

    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        collate_fn=collator,
        pin_memory=True if device.type == "cuda" else False,
    )

    print(f"Train samples: {len(train_ds)}")
    if valid_ds is not None:
        print(f"Valid samples: {len(valid_ds)}")
    print(f"Test samples:  {len(test_ds)}")

    # Compute pos_weight if not set
    if train_cfg.pos_weight is not None:
        pos_weight = torch.tensor(train_cfg.pos_weight, device=device)
    else:
        pw = train_ds.pos_weight()
        pos_weight = torch.tensor(pw, device=device)
        print(f"Computed pos_weight = {pw:.3f}")

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    model = ESMCResidueBindingModel(model_cfg).to(device)
    print(model.trainable_parameter_summary())

    # ------------------------------------------------------------------ #
    # Optimizer & Scheduler
    # ------------------------------------------------------------------ #
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    total_steps = len(train_loader) * train_cfg.num_epochs
    warmup_steps = train_cfg.scheduler.warmup_steps
    if warmup_steps is None:
        warmup_steps = int(train_cfg.scheduler.warmup_ratio * total_steps)

    if train_cfg.scheduler.name == "linear_warmup":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, step / max(1, warmup_steps)),
        )
    elif train_cfg.scheduler.name == "cosine_warmup":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps
        )
    elif train_cfg.scheduler.name == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: 1.0
        )
    elif train_cfg.scheduler.name == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=train_cfg.scheduler.factor,
            patience=train_cfg.scheduler.patience,
            min_lr=train_cfg.scheduler.min_lr,
        )
    else:
        scheduler = None

    # ------------------------------------------------------------------ #
    # MLflow
    # ------------------------------------------------------------------ #
    mlflow.set_tracking_uri(args.mlflow_tracking_uri or "file:./mlruns")
    mlflow.set_experiment(args.mlflow_experiment)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow run ID: {run_id}")

        # Log config
        mlflow.log_params(cfg.model.to_dict())
        mlflow.log_params(cfg.training.to_dict())
        mlflow.log_param("device", device.type)
        mlflow.log_param("pos_weight", pos_weight.item())

        # ------------------------------------------------------------------ #
        # Training loop
        # ------------------------------------------------------------------ #
        best_metric = -float("inf") if train_cfg.best_metric == "f1" else float("inf")
        best_epoch = 0
        steps_without_improvement = 0
        global_step = 0

        for epoch in range(1, train_cfg.num_epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_steps = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{train_cfg.num_epochs}")

            for batch in pbar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids, attention_mask=attention_mask)
                loss = model.compute_masked_bce_loss(
                    logits, labels, attention_mask, pos_weight=pos_weight
                )

                loss.backward()

                if (global_step + 1) % train_cfg.grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params, train_cfg.max_grad_norm
                    )
                    optimizer.step()
                    if scheduler is not None and train_cfg.scheduler.name != "reduce_on_plateau":
                        scheduler.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item()
                epoch_steps += 1
                global_step += 1

                pbar.set_postfix(loss=loss.item())

                if global_step % train_cfg.log_every_n_steps == 0:
                    mlflow.log_metric("train_loss", loss.item(), step=global_step)

            avg_train_loss = epoch_loss / max(1, epoch_steps)
            print(f"Epoch {epoch} average train loss: {avg_train_loss:.6f}")
            mlflow.log_metric("avg_train_loss", avg_train_loss, step=epoch)

            # ------------------------------------------------------------------ #
            # Validation
            # ------------------------------------------------------------------ #
            if valid_loader is not None and epoch % train_cfg.eval_every_n_epochs == 0:
                model.eval()
                val_loss = 0.0
                val_steps = 0
                all_probs = []
                all_labels = []

                with torch.no_grad():
                    for batch in tqdm(valid_loader, desc="Validating"):
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        labels = batch["labels"].to(device)

                        logits = model(input_ids, attention_mask=attention_mask)
                        loss = model.compute_masked_bce_loss(
                            logits, labels, attention_mask, pos_weight=pos_weight
                        )
                        val_loss += loss.item()
                        val_steps += 1

                        probs = torch.sigmoid(logits)
                        valid_mask = (attention_mask.bool()) & (labels != -100)
                        all_probs.append(probs[valid_mask].cpu())
                        all_labels.append(labels[valid_mask].cpu())

                avg_val_loss = val_loss / max(1, val_steps)
                all_probs = torch.cat(all_probs)
                all_labels = torch.cat(all_labels)

                # Find optimal threshold on validation set
                best_thr = _find_optimal_threshold(all_probs, all_labels)
                model.binary_threshold = best_thr
                val_f1, val_prec, val_rec = _compute_f1_and_threshold(
                    all_probs, all_labels, threshold=best_thr
                )

                print(f"  Val loss: {avg_val_loss:.6f}")
                print(f"  Val F1@{best_thr:.3f}: {val_f1:.4f} (P={val_prec:.4f} R={val_rec:.4f})")

                mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
                mlflow.log_metric("val_f1", val_f1, step=epoch)
                mlflow.log_metric("val_precision", val_prec, step=epoch)
                mlflow.log_metric("val_recall", val_rec, step=epoch)
                mlflow.log_metric("val_threshold", best_thr, step=epoch)

                # Checkpoint based on best_metric
                current = -avg_val_loss if train_cfg.best_metric == "loss" else val_f1
                is_best = current > best_metric

                if is_best:
                    best_metric = current
                    best_epoch = epoch
                    steps_without_improvement = 0

                    checkpoint_dir = os.path.join(train_cfg.output_dir, "best")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    _save_checkpoint(
                        model,
                        optimizer,
                        epoch,
                        global_step,
                        {
                            "val_loss": avg_val_loss,
                            "val_f1": val_f1,
                            "val_threshold": best_thr,
                        },
                        os.path.join(checkpoint_dir, "best.pt"),
                    )
                    cfg.model.to_yaml(os.path.join(checkpoint_dir, "best.config.yaml"))
                    print(f"  \u2b50 New best checkpoint (epoch {epoch})")
                else:
                    steps_without_improvement += 1

                # ReduceLROnPlateau steps on validation loss
                if (
                    scheduler is not None
                    and train_cfg.scheduler.name == "reduce_on_plateau"
                ):
                    scheduler.step(avg_val_loss)

                # Early stopping
                if (
                    train_cfg.early_stopping_patience is not None
                    and steps_without_improvement >= train_cfg.early_stopping_patience
                ):
                    print(f"\U0001f6a9 Early stopping after {epoch} epochs (no improvement for {steps_without_improvement} epochs)")
                    break

            else:
                # No validation set: just save periodic checkpoints
                if epoch % train_cfg.eval_every_n_epochs == 0:
                    ckpt_path = os.path.join(
                        train_cfg.output_dir, f"checkpoint_epoch_{epoch}.pt"
                    )
                    _save_checkpoint(
                        model,
                        optimizer,
                        epoch,
                        global_step,
                        {"train_loss": avg_train_loss},
                        ckpt_path,
                    )

        # ------------------------------------------------------------------ #
        # Final test evaluation
        # ------------------------------------------------------------------ #
        print("\nEvaluating on test set...")
        model.eval()
        test_loss = 0.0
        test_steps = 0
        all_test_probs = []
        all_test_labels = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids, attention_mask=attention_mask)
                loss = model.compute_masked_bce_loss(
                    logits, labels, attention_mask, pos_weight=pos_weight
                )
                test_loss += loss.item()
                test_steps += 1

                probs = torch.sigmoid(logits)
                valid_mask = (attention_mask.bool()) & (labels != -100)
                all_test_probs.append(probs[valid_mask].cpu())
                all_test_labels.append(labels[valid_mask].cpu())

        avg_test_loss = test_loss / max(1, test_steps)
        all_test_probs = torch.cat(all_test_probs)
        all_test_labels = torch.cat(all_test_labels)

        test_f1, test_prec, test_rec = _compute_f1_and_threshold(
            all_test_probs, all_test_labels, threshold=model.binary_threshold
        )

        print(f"  Test loss: {avg_test_loss:.6f}")
        print(f"  Test F1@{model.binary_threshold:.3f}: {test_f1:.4f} (P={test_prec:.4f} R={test_rec:.4f})")

        mlflow.log_metric("test_loss", avg_test_loss)
        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("test_precision", test_prec)
        mlflow.log_metric("test_recall", test_rec)

        # Save final checkpoint
        final_dir = os.path.join(train_cfg.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        _save_checkpoint(
            model,
            optimizer,
            epoch,
            global_step,
            {
                "test_loss": avg_test_loss,
                "test_f1": test_f1,
            },
            os.path.join(final_dir, "final.pt"),
        )
        cfg.model.to_yaml(os.path.join(final_dir, "final.config.yaml"))

        print(f"\n\u2705 Training complete. Run ID: {run_id}")
        print(f"Best epoch: {best_epoch}")
        print(f"Best val metric: {best_metric:.6f}")
        print(f"Test F1: {test_f1:.4f}")


if __name__ == "__main__":
    main()