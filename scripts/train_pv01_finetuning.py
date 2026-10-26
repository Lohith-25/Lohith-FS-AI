"""
SolarMap-India — Phase 3: Half U-Net Fine-Tuning Entry Point.

Executes controlled fine-tuning on high-resolution PV01 drone imagery while preserving
native Indian satellite imagery performance using a mixed-domain replay buffer.

Integrity & Safety Rules:
1. Baseline checkpoint (outputs/HalfUNet/models/halfunet_best.pth) is NEVER overwritten.
2. All experimental checkpoints are saved strictly under outputs/experiments/pv01_finetuning/models/.
3. Held-out test splits (PV01 test set and SolarMap-India locked test set) are NEVER loaded,
   used for gradients, model selection, or threshold tuning.
4. Validation split metrics alone dictate model checkpoint selection.
"""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.finetuning.checkpoint_manager import (
    BASELINE_CHECKPOINT_PATH,
    CheckpointSafetyManager,
    compute_file_sha256,
)
from ml.finetuning.dataset import (
    MixedDomainDataset,
    PV01Dataset,
    SolarMapReplayDataset,
    finetuning_collate_fn,
)
from ml.finetuning.losses import CompoundSegmentationLoss
from ml.finetuning.scheduler import build_warmup_cosine_scheduler, compute_epoch_learning_rate
from ml.inference.model import HalfUNet, get_device, load_halfunet_model

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "pv01_finetuning"
DEFAULT_PV01_SPLITS_DIR = DEFAULT_OUTPUT_DIR / "splits"
DEFAULT_SOLARMAP_SPLITS_DIR = PROJECT_ROOT / "splits"


def set_seed(seed: int = 42) -> None:
    """Sets deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.50,
    prefix: str = "val",
) -> Dict[str, Any]:
    """
    Evaluates model on a validation DataLoader using fixed 0.50 threshold.
    Returns micro-pixel confusion matrix and dataset-level metrics.
    """
    model.eval()
    tot_tp = 0
    tot_tn = 0
    tot_fp = 0
    tot_fn = 0

    category_stats = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0})

    with torch.no_grad():
        for images, masks, metas in loader:
            images = images.to(device)
            # Forward pass
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            # Channel 1 is solar foreground
            preds = (probs[:, 1] >= threshold).to(torch.int64)
            masks = masks.to(device)

            # Accumulate
            tp = int(((preds == 1) & (masks == 1)).sum().item())
            tn = int(((preds == 0) & (masks == 0)).sum().item())
            fp = int(((preds == 1) & (masks == 0)).sum().item())
            fn = int(((preds == 0) & (masks == 1)).sum().item())

            tot_tp += tp
            tot_tn += tn
            tot_fp += fp
            tot_fn += fn

            # Track per-category if available
            if isinstance(metas, list):
                for i, meta_item in enumerate(metas):
                    cat = meta_item.get("category", "default")
                    p_i = preds[i]
                    m_i = masks[i]
                    category_stats[cat]["tp"] += int(((p_i == 1) & (m_i == 1)).sum().item())
                    category_stats[cat]["tn"] += int(((p_i == 0) & (m_i == 0)).sum().item())
                    category_stats[cat]["fp"] += int(((p_i == 1) & (m_i == 0)).sum().item())
                    category_stats[cat]["fn"] += int(((p_i == 0) & (m_i == 1)).sum().item())
            elif isinstance(metas, dict) and "category" in metas:
                cats = metas["category"]
                for i, cat in enumerate(cats):
                    p_i = preds[i]
                    m_i = masks[i]
                    category_stats[cat]["tp"] += int(((p_i == 1) & (m_i == 1)).sum().item())
                    category_stats[cat]["tn"] += int(((p_i == 0) & (m_i == 0)).sum().item())
                    category_stats[cat]["fp"] += int(((p_i == 1) & (m_i == 0)).sum().item())
                    category_stats[cat]["fn"] += int(((p_i == 0) & (m_i == 1)).sum().item())

    eps = 1e-7
    dice = (2.0 * tot_tp) / (2.0 * tot_tp + tot_fp + tot_fn + eps)
    iou = float(tot_tp) / (tot_tp + tot_fp + tot_fn + eps)
    precision = float(tot_tp) / (tot_tp + tot_fp + eps) if (tot_tp + tot_fp) > 0 else 0.0
    recall = float(tot_tp) / (tot_tp + tot_fn + eps) if (tot_tp + tot_fn) > 0 else 0.0
    total_px = tot_tp + tot_tn + tot_fp + tot_fn
    accuracy = float(tot_tp + tot_tn) / max(total_px, 1)

    cat_metrics = {}
    for cat, s in category_stats.items():
        c_tp, c_fp, c_fn = s["tp"], s["fp"], s["fn"]
        c_dice = (2.0 * c_tp) / (2.0 * c_tp + c_fp + c_fn + eps)
        c_iou = float(c_tp) / (c_tp + c_fp + c_fn + eps)
        cat_metrics[cat] = {
            "dice": round(float(c_dice), 4),
            "iou": round(float(c_iou), 4),
            "tp": c_tp,
            "fp": c_fp,
            "fn": c_fn,
        }

    return {
        f"{prefix}_dice": round(float(dice), 4),
        f"{prefix}_iou": round(float(iou), 4),
        f"{prefix}_precision": round(float(precision), 4),
        f"{prefix}_recall": round(float(recall), 4),
        f"{prefix}_accuracy": round(float(accuracy), 4),
        f"{prefix}_tp": tot_tp,
        f"{prefix}_tn": tot_tn,
        f"{prefix}_fp": tot_fp,
        f"{prefix}_fn": tot_fn,
        f"{prefix}_categories": cat_metrics,
    }


def run_training(
    epochs: int = 15,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    min_lr: float = 1e-6,
    warmup_epochs: int = 2,
    weight_decay: float = 1e-4,
    patience: int = 6,
    use_amp: bool = True,
    smoke_test: bool = False,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    baseline_checkpoint: Path = BASELINE_CHECKPOINT_PATH,
    pv01_splits_dir: Path = DEFAULT_PV01_SPLITS_DIR,
    solarmap_splits_dir: Path = DEFAULT_SOLARMAP_SPLITS_DIR,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Main training execution function.
    """
    set_seed(seed)
    device = get_device()
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("SOLARMAP-INDIA: PHASE 3 HALF U-NET FINE-TUNING")
    print("=================================================================")
    print(f"Device:                  {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Baseline Checkpoint:     {baseline_checkpoint}")
    print(f"Output Directory:        {output_dir}")
    print(f"Batch Size:              {batch_size}")
    print(f"Base Learning Rate:      {learning_rate}")
    print(f"Warmup / Total Epochs:   {warmup_epochs} / {epochs}")
    print(f"Mixed Precision (AMP):   {use_amp and device.type == 'cuda'}")
    print(f"Smoke Test Mode:         {smoke_test}")
    print("=================================================================\n")

    # 1. Initialize Checkpoint Safety Manager & record initial baseline SHA256
    safety_mgr = CheckpointSafetyManager(
        baseline_path=baseline_checkpoint,
        experimental_dir=models_dir,
    )
    initial_sha = safety_mgr.initial_baseline_sha256
    print(f"Baseline checkpoint verified. Initial SHA-256: {initial_sha}")

    # 2. Load model from baseline checkpoint
    print("Loading Half U-Net weights from baseline checkpoint...")
    model = load_halfunet_model(baseline_checkpoint, device=device)
    print("Model initialized and verified (1,928,450 parameters).")

    # 3. Build Datasets
    pv01_train_csv = pv01_splits_dir / "pv01_train.csv"
    pv01_val_csv = pv01_splits_dir / "pv01_val.csv"
    if not pv01_train_csv.is_file() or not pv01_val_csv.is_file():
        raise FileNotFoundError(
            f"PV01 split CSVs missing in {pv01_splits_dir}. Run prepare_pv01_spatial_split.py first."
        )

    print("Initializing Datasets...")
    pv01_train_ds = PV01Dataset(pv01_train_csv, is_train=True)
    pv01_val_ds = PV01Dataset(pv01_val_csv, is_train=False)

    solarmap_train_ds = SolarMapReplayDataset(split="train", splits_dir=solarmap_splits_dir, is_train=True)
    solarmap_val_ds = SolarMapReplayDataset(split="val", splits_dir=solarmap_splits_dir, is_train=False)

    # Replay buffer combination
    mixed_train_ds = MixedDomainDataset(pv01_train_ds, solarmap_train_ds)

    print(f"  PV01 Train Set:        {len(pv01_train_ds)} images")
    print(f"  PV01 Validation Set:   {len(pv01_val_ds)} images")
    print(f"  SolarMap Train Set:    {len(solarmap_train_ds)} images (Replay Buffer)")
    print(f"  SolarMap Val Set:      {len(solarmap_val_ds)} images")
    print(f"  Mixed Domain Virtual:  {len(mixed_train_ds)} steps/epoch")

    # 4. DataLoaders
    # Keep num_workers=0 on Windows to avoid process spawn overhead
    train_loader = DataLoader(
        mixed_train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=finetuning_collate_fn,
    )
    pv01_val_loader = DataLoader(
        pv01_val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=finetuning_collate_fn,
    )
    solarmap_val_loader = DataLoader(
        solarmap_val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=finetuning_collate_fn,
    )

    # 5. Optimizer, Scheduler, Loss, AMP Scaler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=epochs,
        base_lr=learning_rate,
        min_lr=min_lr,
    )
    criterion = CompoundSegmentationLoss(weight_ce=1.0, weight_dice=1.0)

    scaler_enabled = use_amp and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

    # If smoke test, limit to 2 batches and 1 epoch
    num_epochs = 1 if smoke_test else epochs
    max_train_batches = 2 if smoke_test else None

    # Track metrics
    history: List[Dict[str, Any]] = []
    best_pv01_dice = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    print("\nStarting training loop...")
    start_total_time = time.time()

    for epoch_idx in range(num_epochs):
        epoch_num = epoch_idx + 1
        current_lr = optimizer.param_groups[0]["lr"]
        start_epoch_time = time.time()

        model.train()
        train_loss_accum = 0.0
        train_ce_accum = 0.0
        train_dice_accum = 0.0
        batch_count = 0

        for b_idx, (images, masks, _) in enumerate(train_loader):
            if max_train_batches and b_idx >= max_train_batches:
                break

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=scaler_enabled):
                logits = model(images)
                loss, loss_parts = criterion(logits, masks)

            if scaler_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss_accum += loss_parts["loss_total"]
            train_ce_accum += loss_parts["loss_ce"]
            train_dice_accum += loss_parts["loss_dice"]
            batch_count += 1

        scheduler.step()
        epoch_train_loss = train_loss_accum / max(batch_count, 1)

        # Validation on PV01
        pv01_val_res = evaluate_split(
            model=model,
            loader=pv01_val_loader,
            device=device,
            prefix="pv01_val",
        )

        # Validation on SolarMap (Native Domain Retention)
        solarmap_val_res = evaluate_split(
            model=model,
            loader=solarmap_val_loader,
            device=device,
            prefix="solarmap_val",
        )

        epoch_time = time.time() - start_epoch_time
        val_dice = pv01_val_res["pv01_val_dice"]
        sm_dice = solarmap_val_res["solarmap_val_dice"]

        print(
            f"Epoch {epoch_num:2d}/{num_epochs:2d} ({epoch_time:.1f}s) | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"PV01 Val Dice: {val_dice:.4f} (IoU: {pv01_val_res['pv01_val_iou']:.4f}) | "
            f"SolarMap Val Dice: {sm_dice:.4f}"
        )

        epoch_record = {
            "epoch": epoch_num,
            "lr": current_lr,
            "train_loss": round(epoch_train_loss, 4),
            "pv01_val_dice": val_dice,
            "pv01_val_iou": pv01_val_res["pv01_val_iou"],
            "pv01_val_precision": pv01_val_res["pv01_val_precision"],
            "pv01_val_recall": pv01_val_res["pv01_val_recall"],
            "pv01_val_accuracy": pv01_val_res["pv01_val_accuracy"],
            "solarmap_val_dice": sm_dice,
            "solarmap_val_iou": solarmap_val_res["solarmap_val_iou"],
            "epoch_time_sec": round(epoch_time, 2),
        }
        history.append(epoch_record)

        # Model selection: strictly based on validation metrics
        if not smoke_test and val_dice > best_pv01_dice:
            best_pv01_dice = val_dice
            best_epoch = epoch_num
            epochs_without_improvement = 0
            best_ckpt_path = safety_mgr.save_experimental_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch_num,
                val_metrics={"pv01_val": pv01_val_res, "solarmap_val": solarmap_val_res},
                filename="halfunet_pv01_best.pth",
            )
            print(f"  ⭐ New best checkpoint saved to: {best_ckpt_path.name} (PV01 Val Dice: {val_dice:.4f})")
        else:
            epochs_without_improvement += 1

        # Early stopping check
        if not smoke_test and epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without validation improvement.")
            break

    total_training_time = time.time() - start_total_time

    # Save latest checkpoint
    if not smoke_test:
        latest_ckpt_path = safety_mgr.save_experimental_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch_num,
            val_metrics={"pv01_val": pv01_val_res, "solarmap_val": solarmap_val_res},
            filename="halfunet_pv01_latest.pth",
        )
        print(f"Saved latest checkpoint to: {latest_ckpt_path.name}")

    # Final baseline check
    safety_mgr.assert_baseline_unmodified()
    final_sha = compute_file_sha256(baseline_checkpoint)
    assert final_sha == initial_sha, "CRITICAL ERROR: Baseline checkpoint SHA256 mismatch after training!"
    print(f"\nFinal baseline check: PASSED. SHA-256 untouched: {final_sha}")

    # Export history CSV
    df_hist = pd.DataFrame(history)
    hist_csv = output_dir / ("smoke_test_history.csv" if smoke_test else "training_history.csv")
    df_hist.to_csv(hist_csv, index=False)
    print(f"Saved training history to: {hist_csv}")

    # Export training configuration JSON
    config_summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "smoke_test": smoke_test,
        "epochs_trained": len(history),
        "total_training_time_sec": round(total_training_time, 2),
        "best_epoch": best_epoch,
        "best_pv01_val_dice": round(best_pv01_dice, 4),
        "hyperparameters": {
            "batch_size": batch_size,
            "base_lr": learning_rate,
            "min_lr": min_lr,
            "warmup_epochs": warmup_epochs,
            "total_epochs": epochs,
            "weight_decay": weight_decay,
            "patience": patience,
            "seed": seed,
            "amp_enabled": scaler_enabled,
        },
        "datasets": {
            "pv01_train_images": len(pv01_train_ds),
            "pv01_val_images": len(pv01_val_ds),
            "solarmap_train_images": len(solarmap_train_ds),
            "solarmap_val_images": len(solarmap_val_ds),
            "held_out_pv01_test_used": False,
            "locked_solarmap_test_used": False,
        },
        "checkpoint_preservation": {
            "baseline_checkpoint": str(baseline_checkpoint.resolve()),
            "baseline_sha256": final_sha,
            "immutability_verified": True,
        },
    }
    config_json = output_dir / ("smoke_test_config.json" if smoke_test else "training_config.json")
    with open(config_json, "w", encoding="utf-8") as f:
        json.dump(config_summary, f, indent=2)
    print(f"Saved training config to: {config_json}")

    print("\nTraining execution finished successfully!")
    return config_summary


def main():
    parser = argparse.ArgumentParser(description="SolarMap-India Phase 3 Fine-Tuning CLI")
    parser.add_argument("--epochs", type=int, default=15, help="Number of fine-tuning epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for RTX 2050 (default: 4)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Peak learning rate (default: 5e-5)")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Floor learning rate (default: 1e-6)")
    parser.add_argument("--warmup-epochs", type=int, default=2, help="Warmup epochs (default: 2)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay (default: 1e-4)")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience (default: 6)")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP mixed precision")
    parser.add_argument("--smoke-test", action="store_true", help="Run 2-batch GPU smoke test without full training")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--baseline", type=Path, default=BASELINE_CHECKPOINT_PATH, help="Path to baseline checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        min_lr=args.min_lr,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        patience=args.patience,
        use_amp=(not args.no_amp),
        smoke_test=args.smoke_test,
        output_dir=args.output_dir,
        baseline_checkpoint=args.baseline,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
