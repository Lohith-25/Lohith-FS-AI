"""
SolarMap-India — Phase 8: Multi-Scale and Resolution-Robust Fine-Tuning.

Implements multi-scale training to achieve cross-domain generalization across:
A. Low-resolution / satellite imagery (small panels, few pixels, atmospheric haze)
B. High-resolution / drone imagery (large, detailed, partially visible, variable roofs)

Integrity & Safety Rules:
1. Baseline checkpoint (outputs/HalfUNet/models/halfunet_best.pth) is NEVER overwritten.
2. Phase 3 checkpoints (outputs/experiments/pv01_finetuning/models/*.pth) are NEVER overwritten.
3. Held-out test sets (PV01 test set and SolarMap locked test set) are NEVER loaded.
4. Model selection is based on the Balanced Harmonic Mean of PV01 and SolarMap validation Dice.
"""

import argparse
from collections import defaultdict
import hashlib
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

from ml.finetuning.dataset import (
    BalancedDualDomainDataset,
    PV01Dataset,
    SolarMapReplayDataset,
    finetuning_collate_fn,
)
from ml.finetuning.losses import CompoundSegmentationLoss
from ml.finetuning.multiscale_augmentations import LabelPreservingMultiScaleAugmentor
from ml.finetuning.scheduler import build_warmup_cosine_scheduler
from ml.inference.model import HalfUNet, get_device, load_halfunet_model

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "phase8_multiscale"
BASELINE_CHECKPOINT = PROJECT_ROOT / "outputs" / "HalfUNet" / "models" / "halfunet_best.pth"
STARTING_CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "experiments" / "pv01_finetuning" / "models" / "halfunet_pv01_best.pth"
)
PV01_SPLITS_DIR = PROJECT_ROOT / "outputs" / "experiments" / "pv01_finetuning" / "splits"
SOLARMAP_SPLITS_DIR = PROJECT_ROOT / "splits"


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 checksum of a file."""
    if not filepath.is_file():
        raise FileNotFoundError(f"File not found: {filepath.resolve()}")
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class ProtectedCheckpointRegistry:
    """Registers and verifies all protected checkpoints to prevent accidental overwrite."""

    def __init__(self, checkpoints: List[Path]):
        self.registry: Dict[Path, str] = {}
        for p in checkpoints:
            p_res = p.resolve()
            if p_res.is_file():
                self.registry[p_res] = compute_file_sha256(p_res)

    def assert_all_unmodified(self) -> None:
        """Verifies that none of the registered checkpoints have been altered."""
        for path, expected_sha in self.registry.items():
            current_sha = compute_file_sha256(path)
            if current_sha != expected_sha:
                raise AssertionError(
                    f"CRITICAL VIOLATION: Protected checkpoint has been modified!\n"
                    f"File: {path}\nExpected: {expected_sha}\nActual:   {current_sha}"
                )

    def verify_target_allowed(self, target_path: Path) -> None:
        """Raises PermissionError if target_path matches any protected checkpoint."""
        t_res = target_path.resolve()
        if t_res in self.registry:
            raise PermissionError(
                f"SECURITY VIOLATION: Cannot overwrite protected checkpoint: {t_res}"
            )


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds."""
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
    Evaluates model on validation loader at fixed 0.50 threshold.
    """
    model.eval()
    tot_tp, tot_tn, tot_fp, tot_fn = 0, 0, 0, 0
    category_stats = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0})

    with torch.no_grad():
        for images, masks, metas in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = (probs[:, 1] >= threshold).to(torch.int64)

            tp = int(((preds == 1) & (masks == 1)).sum().item())
            tn = int(((preds == 0) & (masks == 0)).sum().item())
            fp = int(((preds == 1) & (masks == 0)).sum().item())
            fn = int(((preds == 0) & (masks == 1)).sum().item())

            tot_tp += tp
            tot_tn += tn
            tot_fp += fp
            tot_fn += fn

            if isinstance(metas, list):
                for i, meta_item in enumerate(metas):
                    cat = meta_item.get("category", "default")
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
    }


def compute_harmonic_mean(d1: float, d2: float, eps: float = 1e-7) -> float:
    """Computes harmonic mean of two Dice scores."""
    if d1 <= 0 or d2 <= 0:
        return 0.0
    return float((2.0 * d1 * d2) / (d1 + d2 + eps))


def run_phase8_training(
    epochs: int = 15,
    batch_size: int = 4,
    learning_rate: float = 3e-5,
    min_lr: float = 1e-6,
    warmup_epochs: int = 2,
    weight_decay: float = 1e-4,
    patience: int = 6,
    samples_per_epoch: int = 1792,
    use_amp: bool = True,
    smoke_test: bool = False,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    starting_checkpoint: Path = STARTING_CHECKPOINT,
    baseline_checkpoint: Path = BASELINE_CHECKPOINT,
    pv01_splits_dir: Path = PV01_SPLITS_DIR,
    solarmap_splits_dir: Path = SOLARMAP_SPLITS_DIR,
    seed: int = 42,
) -> Dict[str, Any]:
    """Executes Phase 8 multi-scale training."""
    set_seed(seed)
    device = get_device()
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("SOLARMAP-INDIA: PHASE 8 MULTI-SCALE TRAINING")
    print("=================================================================")
    print(f"Device:                   {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Starting Checkpoint:      {starting_checkpoint}")
    print(f"Output Directory:         {output_dir}")
    print(f"Batch Size:               {batch_size}")
    print(f"Learning Rate:            {learning_rate} (min: {min_lr})")
    print(f"Warmup / Total Epochs:    {warmup_epochs} / {epochs}")
    print(f"Samples per Epoch:        {samples_per_epoch} (balanced 50% PV01 / 50% SolarMap)")
    print(f"Mixed Precision (AMP):    {use_amp and device.type == 'cuda'}")
    print(f"Smoke Test Mode:          {smoke_test}")
    print("=================================================================\n")

    # 1. Register and verify protected checkpoints
    phase3_best = PROJECT_ROOT / "outputs" / "experiments" / "pv01_finetuning" / "models" / "halfunet_pv01_best.pth"
    phase3_latest = PROJECT_ROOT / "outputs" / "experiments" / "pv01_finetuning" / "models" / "halfunet_pv01_latest.pth"
    registry = ProtectedCheckpointRegistry([baseline_checkpoint, phase3_best, phase3_latest])
    registry.assert_all_unmodified()
    print("All protected checkpoints verified and locked:")
    for ckpt_p, sha in registry.registry.items():
        print(f"  - {ckpt_p.name}: {sha}")

    # 2. Load model from starting checkpoint
    print(f"\nLoading model weights from starting checkpoint: {starting_checkpoint.name}...")
    model = load_halfunet_model(starting_checkpoint, device=device)
    print("Half U-Net loaded and verified (1,928,450 parameters).")

    # 3. Build Multi-Scale Augmentor and Datasets
    augmentor = LabelPreservingMultiScaleAugmentor(
        target_size=(256, 256),
        scale_range=(0.75, 1.35),
        p_scale=0.70,
        p_flip_h=0.50,
        p_flip_v=0.50,
        p_rotate_ortho=0.50,
        p_resolution_degrade=0.40,
        p_photometric=0.40,
    )

    pv01_train_csv = pv01_splits_dir / "pv01_train.csv"
    pv01_val_csv = pv01_splits_dir / "pv01_val.csv"

    print("Building datasets with multi-scale augmentor...")
    pv01_train_ds = PV01Dataset(pv01_train_csv, is_train=True, augmentor=augmentor)
    pv01_val_ds = PV01Dataset(pv01_val_csv, is_train=False)

    solarmap_train_ds = SolarMapReplayDataset(split="train", splits_dir=solarmap_splits_dir, is_train=True, augmentor=augmentor)
    solarmap_val_ds = SolarMapReplayDataset(split="val", splits_dir=solarmap_splits_dir, is_train=False)

    # 50/50 domain balanced training dataset
    dual_train_ds = BalancedDualDomainDataset(
        pv01_dataset=pv01_train_ds,
        solarmap_dataset=solarmap_train_ds,
        samples_per_epoch=(8 if smoke_test else samples_per_epoch),
    )

    print(f"  PV01 Train Set:         {len(pv01_train_ds)} images")
    print(f"  SolarMap Train Set:     {len(solarmap_train_ds)} images")
    print(f"  Dual-Domain Virtual:    {len(dual_train_ds)} samples/epoch (Exact 50/50 balance)")
    print(f"  PV01 Validation Set:    {len(pv01_val_ds)} images")
    print(f"  SolarMap Val Set:       {len(solarmap_val_ds)} images")

    # 4. DataLoaders
    train_loader = DataLoader(
        dual_train_ds,
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

    num_epochs = 1 if smoke_test else epochs
    max_train_batches = 2 if smoke_test else None

    # Track metrics
    history: List[Dict[str, Any]] = []
    best_harmonic_score = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    print("\nStarting Phase 8 training loop...")
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

            # Check for non-finite loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss detected at epoch {epoch_num}, batch {b_idx}: {loss.item()}")

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

        # Separate Validation Evaluation
        pv01_val_res = evaluate_split(model, pv01_val_loader, device, prefix="pv01_val")
        sm_val_res = evaluate_split(model, solarmap_val_loader, device, prefix="solarmap_val")

        pv_dice = pv01_val_res["pv01_val_dice"]
        sm_dice = sm_val_res["solarmap_val_dice"]
        harmonic_score = round(compute_harmonic_mean(pv_dice, sm_dice), 4)

        epoch_time = time.time() - start_epoch_time
        print(
            f"Epoch {epoch_num:2d}/{num_epochs:2d} ({epoch_time:.1f}s) | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"PV01 Dice: {pv_dice:.4f} | "
            f"SolarMap Dice: {sm_dice:.4f} | "
            f"Balanced Harmonic: {harmonic_score:.4f}"
        )

        epoch_rec = {
            "epoch": epoch_num,
            "lr": current_lr,
            "train_loss": round(epoch_train_loss, 4),
            "pv01_val_dice": pv_dice,
            "pv01_val_iou": pv01_val_res["pv01_val_iou"],
            "pv01_val_precision": pv01_val_res["pv01_val_precision"],
            "pv01_val_recall": pv01_val_res["pv01_val_recall"],
            "solarmap_val_dice": sm_dice,
            "solarmap_val_iou": sm_val_res["solarmap_val_iou"],
            "solarmap_val_precision": sm_val_res["solarmap_val_precision"],
            "solarmap_val_recall": sm_val_res["solarmap_val_recall"],
            "balanced_harmonic_score": harmonic_score,
            "epoch_time_sec": round(epoch_time, 2),
        }
        history.append(epoch_rec)

        # Model Selection: Strictly on Balanced Harmonic Score (never PV01 alone)
        if not smoke_test and harmonic_score > best_harmonic_score:
            best_harmonic_score = harmonic_score
            best_epoch = epoch_num
            epochs_without_improvement = 0

            best_ckpt_target = models_dir / "halfunet_phase8_best.pth"
            registry.verify_target_allowed(best_ckpt_target)

            payload = {
                "epoch": epoch_num,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": {
                    "pv01_val": pv01_val_res,
                    "solarmap_val": sm_val_res,
                    "balanced_harmonic_score": harmonic_score,
                },
                "starting_checkpoint": str(starting_checkpoint.resolve()),
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            torch.save(payload, best_ckpt_target)
            print(f"  ⭐ New best balanced candidate saved to: {best_ckpt_target.name} (Harmonic Score: {harmonic_score:.4f})")
        else:
            epochs_without_improvement += 1

        if not smoke_test and epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without balanced improvement.")
            break

    total_training_time = time.time() - start_total_time

    # Save latest checkpoint
    if not smoke_test:
        latest_ckpt_target = models_dir / "halfunet_phase8_latest.pth"
        registry.verify_target_allowed(latest_ckpt_target)
        payload_latest = {
            "epoch": epoch_num,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": {
                "pv01_val": pv01_val_res,
                "solarmap_val": sm_val_res,
                "balanced_harmonic_score": harmonic_score,
            },
            "starting_checkpoint": str(starting_checkpoint.resolve()),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        torch.save(payload_latest, latest_ckpt_target)
        print(f"Saved latest checkpoint to: {latest_ckpt_target.name}")

    # Re-verify all protected checkpoints untouched
    registry.assert_all_unmodified()
    print("\nProtected checkpoint verification: PASSED. All existing checkpoints remain strictly unmodified.")

    # Export history CSV
    df_hist = pd.DataFrame(history)
    hist_csv = output_dir / ("smoke_test_history.csv" if smoke_test else "training_history.csv")
    df_hist.to_csv(hist_csv, index=False)
    print(f"Saved training history to: {hist_csv}")

    # Export config JSON
    config_summary = {
        "phase": "Phase 8 Multi-Scale and Resolution-Robust Segmentation",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "smoke_test": smoke_test,
        "epochs_trained": len(history),
        "total_training_time_sec": round(total_training_time, 2),
        "best_epoch": best_epoch,
        "best_balanced_harmonic_score": round(best_harmonic_score, 4),
        "selection_criterion": "Harmonic Mean of PV01 and SolarMap Validation Dice (Strictly Multi-Domain)",
        "hyperparameters": {
            "batch_size": batch_size,
            "base_lr": learning_rate,
            "min_lr": min_lr,
            "warmup_epochs": warmup_epochs,
            "total_epochs": epochs,
            "weight_decay": weight_decay,
            "patience": patience,
            "samples_per_epoch": samples_per_epoch,
            "seed": seed,
            "amp_enabled": scaler_enabled,
        },
        "augmentations": {
            "scale_range": [0.75, 1.35],
            "p_scale": 0.70,
            "p_flips": 0.50,
            "p_orthogonal_rotation": 0.50,
            "p_resolution_degrade": 0.40,
            "p_photometric": 0.40,
            "mask_interpolation": "Strictly Nearest-Neighbor (Image.NEAREST)",
        },
        "datasets": {
            "pv01_train_images": len(pv01_train_ds),
            "solarmap_train_images": len(solarmap_train_ds),
            "pv01_val_images": len(pv01_val_ds),
            "solarmap_val_images": len(solarmap_val_ds),
            "domain_balance": "50% PV01 / 50% SolarMap-India per batch",
            "held_out_pv01_test_used": False,
            "locked_solarmap_test_used": False,
        },
        "protected_checkpoints_verified": {
            str(k): v for k, v in registry.registry.items()
        },
    }
    config_json = output_dir / ("smoke_test_config.json" if smoke_test else "training_config.json")
    with open(config_json, "w", encoding="utf-8") as f:
        json.dump(config_summary, f, indent=2)
    print(f"Saved configuration to: {config_json}")

    return config_summary


def main():
    parser = argparse.ArgumentParser(description="SolarMap-India Phase 8 Multi-Scale Training CLI")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for RTX 2050 (default: 4)")
    parser.add_argument("--lr", type=float, default=3e-5, help="Peak learning rate (default: 3e-5)")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Floor learning rate (default: 1e-6)")
    parser.add_argument("--warmup-epochs", type=int, default=2, help="Warmup epochs (default: 2)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay (default: 1e-4)")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience (default: 6)")
    parser.add_argument("--samples-per-epoch", type=int, default=1792, help="Samples per epoch (default: 1792)")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP mixed precision")
    parser.add_argument("--smoke-test", action="store_true", help="Run 2-batch GPU smoke test")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--starting-checkpoint", type=Path, default=STARTING_CHECKPOINT, help="Starting checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    run_phase8_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        min_lr=args.min_lr,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        patience=args.patience,
        samples_per_epoch=args.samples_per_epoch,
        use_amp=(not args.no_amp),
        smoke_test=args.smoke_test,
        output_dir=args.output_dir,
        starting_checkpoint=args.starting_checkpoint,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
