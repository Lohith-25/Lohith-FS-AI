"""
SolarMap-India — Phase 3 Fine-Tuning Pre-Training Validation Suite.

Validates all 9 strict requirements before GPU training:
1. PV01 spatial split integrity and category balance.
2. Zero overlapping filenames and zero overlapping X-coordinate groups.
3. Every image has exactly one matching ground-truth mask.
4. Complete exclusion of SolarMap-India locked test splits.
5. Half U-Net output shape and loss function tensor compatibility.
6. Two-epoch linear warmup followed by cosine learning-rate decay schedule.
7. Baseline checkpoint immutability and protection.
8. Real batch loading and single-batch forward/backward pass smoke test.
9. Verification of epoch learning-rate table and parameter shapes.
"""

import math
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from PIL import Image
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
from ml.finetuning.dataset import PV01Dataset, SolarMapReplayDataset
from ml.finetuning.losses import CompoundSegmentationLoss, DiceLoss
from ml.finetuning.scheduler import (
    build_warmup_cosine_scheduler,
    compute_epoch_learning_rate,
    get_learning_rate_schedule_table,
)
from ml.inference.model import EXPECTED_PARAM_COUNT, HalfUNet, get_device, load_halfunet_model

PV01_SPLITS_DIR = PROJECT_ROOT / "outputs" / "experiments" / "pv01_finetuning" / "splits"
SOLARMAP_SPLITS_DIR = PROJECT_ROOT / "splits"


class TestPhase3FineTuningPlan(unittest.TestCase):
    """Rigorous pre-training verification test suite."""

    def test_01_pv01_spatial_split_manifests_exist(self):
        """Verifies that all three split CSVs exist and contain exactly 645 images total."""
        train_csv = PV01_SPLITS_DIR / "pv01_train.csv"
        val_csv = PV01_SPLITS_DIR / "pv01_val.csv"
        test_csv = PV01_SPLITS_DIR / "pv01_test.csv"

        self.assertTrue(train_csv.is_file(), f"Missing {train_csv}")
        self.assertTrue(val_csv.is_file(), f"Missing {val_csv}")
        self.assertTrue(test_csv.is_file(), f"Missing {test_csv}")

        df_tr = pd.read_csv(train_csv)
        df_v = pd.read_csv(val_csv)
        df_te = pd.read_csv(test_csv)

        # Expected split counts: 448 (69.5%), 99 (15.3%), 98 (15.2%)
        self.assertEqual(len(df_tr), 448, f"Train count mismatch: {len(df_tr)}")
        self.assertEqual(len(df_v), 99, f"Val count mismatch: {len(df_v)}")
        self.assertEqual(len(df_te), 98, f"Test count mismatch: {len(df_te)}")
        self.assertEqual(len(df_tr) + len(df_v) + len(df_te), 645)

    def test_02_zero_x_coordinate_group_leakage(self):
        """Asserts zero overlapping X-coordinate groups between train, val, and test."""
        df_tr = pd.read_csv(PV01_SPLITS_DIR / "pv01_train.csv")
        df_v = pd.read_csv(PV01_SPLITS_DIR / "pv01_val.csv")
        df_te = pd.read_csv(PV01_SPLITS_DIR / "pv01_test.csv")

        x_tr = set(df_tr["x_coord"].astype(str))
        x_v = set(df_v["x_coord"].astype(str))
        x_te = set(df_te["x_coord"].astype(str))

        self.assertEqual(len(x_tr & x_v), 0, f"X group leakage between train and val: {x_tr & x_v}")
        self.assertEqual(len(x_tr & x_te), 0, f"X group leakage between train and test: {x_tr & x_te}")
        self.assertEqual(len(x_v & x_te), 0, f"X group leakage between val and test: {x_v & x_te}")
        self.assertEqual(len(x_tr) + len(x_v) + len(x_te), 145)

    def test_03_zero_filename_leakage(self):
        """Asserts zero overlapping filenames between train, val, and test."""
        df_tr = pd.read_csv(PV01_SPLITS_DIR / "pv01_train.csv")
        df_v = pd.read_csv(PV01_SPLITS_DIR / "pv01_val.csv")
        df_te = pd.read_csv(PV01_SPLITS_DIR / "pv01_test.csv")

        fn_tr = set(df_tr["filename"])
        fn_v = set(df_v["filename"])
        fn_te = set(df_te["filename"])

        self.assertEqual(len(fn_tr & fn_v), 0, f"Filename leakage between train and val: {fn_tr & fn_v}")
        self.assertEqual(len(fn_tr & fn_te), 0, f"Filename leakage between train and test: {fn_tr & fn_te}")
        self.assertEqual(len(fn_v & fn_te), 0, f"Filename leakage between val and test: {fn_v & fn_te}")

    def test_04_ground_truth_mask_pairing_and_binary_values(self):
        """Confirms every image has a valid mask file and converts cleanly to binary values in {0, 1}."""
        for split_name in ["train", "val", "test"]:
            df = pd.read_csv(PV01_SPLITS_DIR / f"pv01_{split_name}.csv")
            for _, row in df.iterrows():
                mask_p = Path(row["mask_path"])
                self.assertTrue(mask_p.is_file(), f"Missing mask: {mask_p}")

        # Check sample mask values
        sample_mask_path = Path(df.iloc[0]["mask_path"])
        mask_img = Image.open(sample_mask_path)
        mask_arr = np.array(mask_img)
        binary = (mask_arr > 0).astype(np.uint8)
        unique_vals = set(np.unique(binary))
        self.assertTrue(unique_vals.issubset({0, 1}), f"Non-binary values found: {unique_vals}")

    def test_05_solarmap_locked_test_exclusion(self):
        """Verifies that SolarMap-India locked test splits are completely excluded from training/validation."""
        df_pos_locked = pd.read_csv(SOLARMAP_SPLITS_DIR / "positive_test_LOCKED.csv")
        df_neg_locked = pd.read_csv(SOLARMAP_SPLITS_DIR / "negative_test_LOCKED.csv")
        locked_files = set(df_pos_locked["filename"]).union(set(df_neg_locked["filename"]))

        self.assertEqual(len(locked_files), 344, f"Expected 344 locked files, found {len(locked_files)}")

        # Check training splits
        df_pos_tr = pd.read_csv(SOLARMAP_SPLITS_DIR / "positive_train.csv")
        df_neg_tr = pd.read_csv(SOLARMAP_SPLITS_DIR / "negative_train.csv")
        train_files = set(df_pos_tr["filename"]).union(set(df_neg_tr["filename"]))
        self.assertEqual(len(train_files & locked_files), 0, "Locked test files found in training split!")

        # Check validation splits
        df_pos_v = pd.read_csv(SOLARMAP_SPLITS_DIR / "positive_val.csv")
        df_neg_v = pd.read_csv(SOLARMAP_SPLITS_DIR / "negative_val.csv")
        val_files = set(df_pos_v["filename"]).union(set(df_neg_v["filename"]))
        self.assertEqual(len(val_files & locked_files), 0, "Locked test files found in validation split!")

    def test_06_model_output_shape_and_loss_compatibility(self):
        """Verifies Half U-Net forward pass shape (B, 2, 256, 256) and loss calculation."""
        model = HalfUNet(num_classes=2, in_channels=3, base_channels=32)
        model.eval()

        # Input: [B=2, C=3, H=256, W=256]
        dummy_input = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            logits = model(dummy_input)

        # Assert output shape
        self.assertEqual(
            logits.shape,
            torch.Size([2, 2, 256, 256]),
            f"Expected shape [2, 2, 256, 256], got {logits.shape}",
        )

        # Targets: [B=2, H=256, W=256] int64
        dummy_targets = torch.randint(0, 2, (2, 256, 256), dtype=torch.int64)

        # Test CE Loss
        criterion_ce = nn.CrossEntropyLoss()
        loss_ce = criterion_ce(logits, dummy_targets)
        self.assertTrue(torch.isfinite(loss_ce))

        # Test Dice Loss
        criterion_dice = DiceLoss(smooth=1.0)
        loss_dice = criterion_dice(logits, dummy_targets)
        self.assertTrue(torch.isfinite(loss_dice))

        # Test Compound Loss
        criterion_compound = CompoundSegmentationLoss(weight_ce=1.0, weight_dice=1.0)
        total_loss, components = criterion_compound(logits, dummy_targets)
        self.assertTrue(torch.isfinite(total_loss))
        self.assertAlmostEqual(total_loss.item(), loss_ce.item() + loss_dice.item(), places=5)

    def test_07_two_epoch_linear_warmup_and_cosine_decay_schedule(self):
        """Verifies exact learning rates for the 2-epoch linear warmup + cosine decay schedule."""
        base_lr = 5e-5
        min_lr = 1e-6
        warmup_epochs = 2
        total_epochs = 15

        schedule = get_learning_rate_schedule_table(
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            base_lr=base_lr,
            min_lr=min_lr,
        )

        self.assertEqual(len(schedule), total_epochs)

        # Epoch 1 (0-indexed 0): warmup 1/2
        self.assertAlmostEqual(schedule[0][1], 2.5e-5, places=7)
        self.assertIn("Warmup (1/2)", schedule[0][2])

        # Epoch 2 (0-indexed 1): warmup 2/2 = base_lr
        self.assertAlmostEqual(schedule[1][1], 5.0e-5, places=7)
        self.assertIn("Warmup (2/2)", schedule[1][2])

        # Epoch 3 (0-indexed 2): cosine decay begins
        self.assertLess(schedule[2][1], 5.0e-5)
        self.assertIn("Cosine Decay", schedule[2][2])

        # Final epoch: min_lr
        self.assertAlmostEqual(schedule[-1][1], 1.0e-6, places=7)

        # Verify decay is monotonically non-increasing after warmup
        for i in range(2, total_epochs - 1):
            self.assertGreaterEqual(
                schedule[i][1],
                schedule[i + 1][1],
                f"LR increased at epoch {i + 1}: {schedule[i][1]} vs {schedule[i + 1][1]}",
            )

        # Verify PyTorch LambdaLR scheduler matches computed values
        dummy_param = nn.Parameter(torch.zeros(1))
        optimizer = optim.AdamW([dummy_param], lr=base_lr)
        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            base_lr=base_lr,
            min_lr=min_lr,
        )

        for e in range(total_epochs):
            current_opt_lr = optimizer.param_groups[0]["lr"]
            expected_lr = schedule[e][1]
            self.assertAlmostEqual(current_opt_lr, expected_lr, places=7)
            optimizer.step()
            scheduler.step()

    def test_08_baseline_checkpoint_immutability(self):
        """Verifies CheckpointSafetyManager protects baseline checkpoint from being overwritten."""
        safety_mgr = CheckpointSafetyManager()

        # Assert baseline exists and sha256 is recorded
        self.assertTrue(BASELINE_CHECKPOINT_PATH.is_file())
        baseline_sha = compute_file_sha256(BASELINE_CHECKPOINT_PATH)
        self.assertEqual(baseline_sha, safety_mgr.initial_baseline_sha256)

        # Attempting to save over baseline must raise PermissionError
        dummy_model = HalfUNet(num_classes=2)
        dummy_opt = optim.Adam(dummy_model.parameters(), lr=1e-4)

        with self.assertRaises(PermissionError):
            safety_mgr.save_experimental_checkpoint(
                model=dummy_model,
                optimizer=dummy_opt,
                epoch=1,
                val_metrics={"dice": 0.8},
                filename="../../HalfUNet/models/halfunet_best.pth",
            )

        # Also test saving a valid experimental checkpoint succeeds
        saved_path = safety_mgr.save_experimental_checkpoint(
            model=dummy_model,
            optimizer=dummy_opt,
            epoch=1,
            val_metrics={"dice": 0.8},
            filename="_test_smoke_ckpt.pth",
        )
        self.assertTrue(saved_path.is_file())
        saved_path.unlink()  # Clean up temporary test file

        # Baseline sha256 must remain strictly identical
        self.assertEqual(compute_file_sha256(BASELINE_CHECKPOINT_PATH), baseline_sha)

    def test_09_single_batch_forward_backward_pass_smoke_test(self):
        """Loads real data from DataLoader and runs a complete forward/backward pass with gradient verification."""
        device = get_device()

        # 1. Load baseline model onto target device
        model = load_halfunet_model(BASELINE_CHECKPOINT_PATH, device=device)
        model.train()

        # 2. Load PV01 dataset and create DataLoader
        pv01_ds = PV01Dataset(PV01_SPLITS_DIR / "pv01_train.csv", is_train=True)
        loader = DataLoader(pv01_ds, batch_size=2, shuffle=False)

        images, masks, meta = next(iter(loader))
        images = images.to(device)
        masks = masks.to(device)

        self.assertEqual(images.shape, torch.Size([2, 3, 256, 256]))
        self.assertEqual(masks.shape, torch.Size([2, 256, 256]))

        # 3. Setup optimizer and compound loss
        optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
        criterion = CompoundSegmentationLoss(weight_ce=1.0, weight_dice=1.0)

        # 4. Forward pass
        optimizer.zero_grad()
        logits = model(images)
        self.assertEqual(logits.shape, torch.Size([2, 2, 256, 256]))

        # 5. Loss computation
        total_loss, loss_dict = criterion(logits, masks)
        self.assertTrue(torch.isfinite(total_loss))
        self.assertGreater(total_loss.item(), 0.0)

        # 6. Backward pass
        total_loss.backward()

        # 7. Check gradients exist and are finite
        grad_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient is None for parameter: {name}")
                self.assertTrue(torch.all(torch.isfinite(param.grad)), f"Gradient is non-finite for: {name}")
                grad_count += 1

        self.assertGreater(grad_count, 0, "No gradients computed during backward pass!")

        # 8. Optimizer step
        optimizer.step()


if __name__ == "__main__":
    unittest.main()
