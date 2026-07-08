"""
Training script for the ESM-C + LoRA + CNN/MLP residue-level binding
predictor, driven entirely by one YAML file (model hyperparameters under
`model:`, training hyperparameters under `training:` -- see
example_config.yaml).

Usage
-----
    python train.py --config example_config.yaml
    python train.py --config example_config.yaml --final   # combine train+valid

The --final flag is a convenience override for
training.data.combine_train_valid_for_final, so you can keep one config
file for both the model-selection run and the final run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Dict, Optional

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from config import ExperimentConfig, TrainingConfig
from dataset import LIPCollator, LABEL_IGNORE_INDEX, build_datasets
from model import ESMCResidueBindingModel

import yaml
from huggingface_hub import login as hf_login

from sklearn.metrics import (
    precision_recall_fscore_support,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
    average_precision_score,
)

from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_scheduler(optimizer, scheduler_cfg, num_optimizer_steps: int):
    name = scheduler_cfg.name
    if name == "none":
        return None
    if name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_cfg.factor,
            patience=scheduler_cfg.patience,
            min_lr=scheduler_cfg.min_lr,
        )

    warmup_steps = scheduler_cfg.warmup_steps
    if warmup_steps is None:
        warmup_steps = int(scheduler_cfg.warmup_ratio * num_optimizer_steps)

    if name == "linear_warmup":
        return get_linear_schedule_with_warmup(
            optimizer, warmup_steps, num_optimizer_steps
        )
    if name == "cosine_warmup":
        return get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, num_optimizer_steps
        )

    raise ValueError(f"Unknown scheduler name: {name}")


def compute_metrics(
    all_logits: torch.Tensor, all_labels: torch.Tensor
) -> Dict[str, float]:
    """Flattened, ignore-index-aware residue-level metrics.

    Returns standard threshold-based metrics (precision, recall, f1, mcc at
    threshold 0.5) plus threshold-free metrics (AUROC, AUPRC / average precision).
    """
    mask = all_labels != LABEL_IGNORE_INDEX
    probs = torch.sigmoid(all_logits[mask])
    preds = (probs >= 0.5).long()
    labels = all_labels[mask]

    metrics = {"n_residues": int(mask.sum().item())}
    if labels.numel() == 0:
        return metrics

    labels_np = labels.cpu().numpy()
    probs_np = probs.cpu().numpy()
    preds_np = preds.cpu().numpy()

    # --- Threshold-based metrics (at 0.5) ---
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_np, preds_np, average="binary", zero_division=0
    )
    metrics.update(precision=float(precision), recall=float(recall), f1=float(f1))
    try:
        metrics["mcc"] = float(matthews_corrcoef(labels_np, preds_np))
    except ValueError:
        metrics["mcc"] = float("nan")

    # --- Area under the ROC curve ---
    try:
        metrics["auroc"] = float(roc_auc_score(labels_np, probs_np))
    except ValueError:
        metrics["auroc"] = float("nan")

    # --- Area under the precision-recall curve (trapezoidal) ---
    try:
        precision_pts, recall_pts, _ = precision_recall_curve(labels_np, probs_np)
        metrics["aupr"] = float(auc(recall_pts, precision_pts))
    except ValueError:
        metrics["aupr"] = float("nan")

    # --- Average precision (AP) -- alternative to trapezoidal AUPR ---
    try:
        metrics["average_precision"] = float(
            average_precision_score(labels_np, probs_np)
        )
    except ValueError:
        metrics["average_precision"] = float("nan")

    return metrics


@torch.no_grad()
def find_best_threshold(all_logits: torch.Tensor, all_labels: torch.Tensor) -> float:
    """Find the threshold that maximises Matthews Correlation Coefficient (MCC)
    on the given logits/labels. Searches over 1000 evenly-spaced thresholds
    between 0.0 and 1.0."""
    mask = all_labels != LABEL_IGNORE_INDEX
    probs = torch.sigmoid(all_logits[mask]).cpu().numpy()
    labels = all_labels[mask].cpu().numpy()

    best_threshold = 0.5
    best_mcc = -1.0
    for thresh in np.linspace(0.01, 0.99, 1000):
        preds = (probs >= thresh).astype(int)
        try:
            mcc = matthews_corrcoef(labels, preds)
        except ValueError:
            mcc = -1.0
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = thresh
    return float(best_threshold)


@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    total_loss, total_residues = 0.0, 0
    all_logits, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = model.compute_masked_bce_loss(logits, labels, attention_mask)

        n_valid = int((labels != LABEL_IGNORE_INDEX).sum().item())
        total_loss += loss.item() * n_valid
        total_residues += n_valid

        all_logits.append(logits.detach().float().cpu())
        all_labels.append(labels.detach().cpu())

    all_logits = torch.cat([t.reshape(-1) for t in all_logits])
    all_labels = torch.cat([t.reshape(-1) for t in all_labels])
    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / max(total_residues, 1)
    return metrics


def save_checkpoint(model, cfg: ExperimentConfig, output_dir: str, tag: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    # Only save trainable params (LoRA + CNN + MLP head) -- the frozen
    # backbone is unchanged and re-loaded fresh from model_name_or_path.
    trainable_state = {
        n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad
    }
    # Also store the optimal binary threshold inside the checkpoint
    trainable_state["_binary_threshold"] = torch.tensor(model.binary_threshold)
    torch.save(trainable_state, os.path.join(output_dir, f"{tag}.pt"))
    cfg.to_yaml(os.path.join(output_dir, f"{tag}.config.yaml"))


def load_trainable_state(model: ESMCResidueBindingModel, checkpoint_path: str) -> None:
    """Load trainable state dict, including the binary threshold if present."""
    state = torch.load(checkpoint_path, map_location="cpu")
    # Separate the threshold from parameter tensors
    threshold = state.pop("_binary_threshold", None)
    if threshold is not None:
        model.binary_threshold = float(threshold.item())
    # Load remaining trainable parameters
    own_state = {n: p for n, p in model.named_parameters() if p.requires_grad}
    for name, param in state.items():
        if name in own_state:
            own_state[name].data.copy_(param)
        else:
            print(f"Warning: parameter {name} not found in model, skipping.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--final",
        action="store_true",
        help="Override training.data.combine_train_valid_for_final=True for this run.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show tqdm progress bars and per-batch info during training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "mps"],
        help="Device to use (overrides auto-detection). Use 'mps' for macOS Metal.",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.final:
        cfg.training.data.combine_train_valid_for_final = True

    train_cfg: TrainingConfig = cfg.training
    set_seed(train_cfg.seed)

    # ------------------------------------------------------------------ #
    # Hugging Face login using API key from data/hg_api_key.yaml
    # ------------------------------------------------------------------ #
    hg_api_key_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "hg_api_key.yaml"
    )
    if os.path.exists(hg_api_key_path):
        with open(hg_api_key_path, "r") as f:
            hg_api_data = yaml.safe_load(f)
        hf_token = hg_api_data.get("api_key")
        if hf_token:
            hf_login(token=hf_token)
            print("✅ Logged in to Hugging Face Hub.")
        else:
            print("⚠️  No 'api_key' found in data/hg_api_key.yaml")
    else:
        print("⚠️  data/hg_api_key.yaml not found – skipping HF login.")
    # ------------------------------------------------------------------ #

    device = (
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ------------------------------------------------------------------ #
    # MPS (Apple Silicon) warning
    # ------------------------------------------------------------------ #
    if device == "mps" and cfg.model.head_precision == "fp32":
        print(
            "⚠️  MPS device detected with head_precision='fp32'. "
            "Validation metrics may produce NaN values due to numerical "
            "instability of float32 on Apple Silicon.\n"
            "   Consider setting `head_precision: fp64` in your config YAML "
            "(model.head_precision) to use float64 for the CNN/MLP head.\n"
            "   See config.py docstring for details."
        )

    # ------------------------------------------------------------------ #
    # MLflow setup -- local file tracking
    # ------------------------------------------------------------------ #
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    tracking_uri = os.path.abspath(os.path.join(train_cfg.output_dir, "mlruns"))
    os.makedirs(tracking_uri, exist_ok=True)
    mlflow.set_tracking_uri("file:" + tracking_uri)
    experiment_name = f"BindCOREpLM_{os.path.basename(train_cfg.output_dir)}"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name, artifact_location=tracking_uri
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.start_run(experiment_id=experiment_id, run_name=train_cfg.output_dir)

    # Log config parameters to MLflow
    mlflow.log_params(
        {
            "model_name": cfg.model.model_name_or_path,
            "lora_rank": cfg.model.lora.rank,
            "lora_alpha": cfg.model.lora.alpha,
            "lora_dropout": cfg.model.lora.dropout,
            "cnn_kernel_sizes": str(cfg.model.cnn.kernel_sizes),
            "cnn_out_channels_per_branch": cfg.model.cnn.out_channels_per_branch,
            "mlp_hidden_dims": str(cfg.model.mlp.hidden_dims),
            "mlp_dropout": cfg.model.mlp.dropout,
            "batch_size": train_cfg.batch_size,
            "grad_accumulation_steps": train_cfg.grad_accumulation_steps,
            "lr": train_cfg.lr,
            "weight_decay": train_cfg.weight_decay,
            "num_epochs": train_cfg.num_epochs,
            "seed": train_cfg.seed,
            "scheduler": train_cfg.scheduler.name,
            "best_metric": train_cfg.best_metric,
        }
    )

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    train_ds, valid_ds, test_ds = build_datasets(
        data_dir=train_cfg.data.data_dir,
        train_file=train_cfg.data.train_file,
        valid_file=train_cfg.data.valid_file,
        test_file=train_cfg.data.test_file,
        combine_train_valid_for_final=train_cfg.data.combine_train_valid_for_final,
    )
    print(
        f"Train proteins: {len(train_ds)}"
        + (
            " (train+valid combined)"
            if train_cfg.data.combine_train_valid_for_final
            else ""
        )
    )
    if valid_ds is not None:
        print(f"Valid proteins: {len(valid_ds)}")
    print(f"Test proteins:  {len(test_ds)} (held out, not used during training)")

    pos_weight_value = train_cfg.pos_weight
    if pos_weight_value is None:
        pos_weight_value = train_ds.pos_weight()
    print(f"Using pos_weight={pos_weight_value:.3f} for BCEWithLogitsLoss")
    pos_weight_tensor = torch.tensor(pos_weight_value, device=device)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
    collator = LIPCollator(tokenizer=tokenizer, max_length=cfg.model.max_seq_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=train_cfg.num_workers,
    )
    valid_loader = None
    if valid_ds is not None:
        valid_loader = DataLoader(
            valid_ds,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=train_cfg.num_workers,
        )

    # ------------------------------------------------------------------ #
    # Model / optimizer / scheduler
    # ------------------------------------------------------------------ #
    model = ESMCResidueBindingModel(cfg.model).to(device)
    print(model.trainable_parameter_summary())

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )

    steps_per_epoch = math.ceil(len(train_loader) / train_cfg.grad_accumulation_steps)
    total_optimizer_steps = steps_per_epoch * train_cfg.num_epochs
    scheduler = build_scheduler(optimizer, train_cfg.scheduler, total_optimizer_steps)
    is_plateau_scheduler = train_cfg.scheduler.name == "reduce_on_plateau"

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    os.makedirs(train_cfg.output_dir, exist_ok=True)
    best_score = math.inf if train_cfg.best_metric == "loss" else -math.inf
    epochs_without_improvement = 0
    global_step = 0

    # Determine verbose from args (we'll store it after parse_args)
    verbose = getattr(args, "verbose", False)

    for epoch in range(1, train_cfg.num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0

        # Wrap train loader with tqdm if verbose
        train_iter = (
            tqdm(
                enumerate(train_loader, start=1),
                total=len(train_loader),
                desc=f"Epoch {epoch}",
                unit="batch",
            )
            if (verbose)
            else enumerate(train_loader, start=1)
        )

        for step, batch in train_iter:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = model.compute_masked_bce_loss(
                logits, labels, attention_mask, pos_weight=pos_weight_tensor
            )
            (loss / train_cfg.grad_accumulation_steps).backward()
            running_loss += loss.item()

            is_accumulation_boundary = step % train_cfg.grad_accumulation_steps == 0
            is_last_batch = step == len(train_loader)
            if is_accumulation_boundary or is_last_batch:
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, train_cfg.max_grad_norm
                )
                optimizer.step()
                if scheduler is not None and not is_plateau_scheduler:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % train_cfg.log_every_n_steps == 0:
                    lr_now = optimizer.param_groups[0]["lr"]
                    # Print always (but tqdm also shows progress)
                    if verbose:
                        # Update tqdm postfix with loss and lr
                        if isinstance(train_iter, tqdm):
                            train_iter.set_postfix(
                                loss=f"{running_loss / step:.4f}", lr=f"{lr_now:.2e}"
                            )
                    print(
                        f"epoch {epoch} step {global_step} "
                        f"loss={running_loss / step:.4f} lr={lr_now:.2e}"
                    )

            # If verbose, show per-batch info (only without tqdm)
            if verbose and (step % max(1, train_cfg.log_every_n_steps // 5) == 0):
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"  batch {step}/{len(train_loader)} loss={running_loss / step:.4f} lr={lr_now:.2e}"
                )

        avg_train_loss = running_loss / len(train_loader)
        print(f"== epoch {epoch} done -- avg train loss {avg_train_loss:.4f} ==")

        if valid_loader is not None and epoch % train_cfg.eval_every_n_epochs == 0:
            val_metrics = evaluate(model, valid_loader, device)
            print(f"   val: {val_metrics}")

            # Log validation metrics to MLflow
            mlflow.log_metrics(
                {
                    f"val_{k}": v
                    for k, v in val_metrics.items()
                    if isinstance(v, (int, float))
                },
                step=epoch,
            )

            if is_plateau_scheduler:
                scheduler.step(val_metrics["loss"])

            current_score = val_metrics.get(train_cfg.best_metric, val_metrics["loss"])
            improved = (
                current_score < best_score
                if train_cfg.best_metric == "loss"
                else current_score > best_score
            )
            if improved:
                best_score = current_score
                epochs_without_improvement = 0
                save_checkpoint(model, cfg, train_cfg.output_dir, tag="best")
                with open(
                    os.path.join(train_cfg.output_dir, "best_metrics.json"), "w"
                ) as f:
                    json.dump({"epoch": epoch, **val_metrics}, f, indent=2)
                print(
                    f"   -> new best ({train_cfg.best_metric}={current_score:.4f}), checkpoint saved"
                )
            else:
                epochs_without_improvement += 1
                if not train_cfg.save_best_only:
                    save_checkpoint(
                        model, cfg, train_cfg.output_dir, tag=f"epoch{epoch}"
                    )

            if (
                train_cfg.early_stopping_patience is not None
                and epochs_without_improvement >= train_cfg.early_stopping_patience
            ):
                print(
                    f"No improvement for {epochs_without_improvement} evals -- stopping early."
                )
                break
        else:
            # No held-out validation (e.g. combine_train_valid_for_final=True):
            # just checkpoint each epoch since there's no metric to select
            # a "best" epoch on.
            save_checkpoint(model, cfg, train_cfg.output_dir, tag="final")

    # ------------------------------------------------------------------ #
    # Final: find MCC-optimised threshold and store in the model checkpoint
    # ------------------------------------------------------------------ #
    print(
        "Computing optimal binary threshold that maximises MCC on the validation set..."
    )
    if valid_loader is not None:
        # Run evaluation to get all logits and labels
        model.eval()
        all_logits, all_labels = [], []
        for batch in valid_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(logits.detach().float().cpu())
            all_labels.append(labels.detach().cpu())
        all_logits = torch.cat([t.reshape(-1) for t in all_logits])
        all_labels = torch.cat([t.reshape(-1) for t in all_labels])

        best_threshold = find_best_threshold(all_logits, all_labels)
        print(f"Optimal threshold (max MCC on valid): {best_threshold:.4f}")

        # Store threshold in the model
        model.binary_threshold = best_threshold

        # Re-save best.pt with threshold included
        best_path = os.path.join(train_cfg.output_dir, "best.pt")
        if os.path.exists(best_path):
            save_checkpoint(model, cfg, train_cfg.output_dir, tag="best")
            print(f"Updated best.pt with binary_threshold={best_threshold:.4f}")
    else:
        model.binary_threshold = 0.5

    # Also update last.pt with threshold
    save_checkpoint(model, cfg, train_cfg.output_dir, tag="last")

    # Log final best threshold to MLflow
    mlflow.log_metric("best_binary_threshold", model.binary_threshold)

    mlflow.end_run()
    print(f"Training complete. Checkpoints in {train_cfg.output_dir}")


if __name__ == "__main__":
    main()
