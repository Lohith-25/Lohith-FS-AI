"""
SolarMap-India — Phase 8: Multi-Scale Plan and Pipeline Validation Suite.

Validates all 9 mandatory pre-training criteria:
1. Protected checkpoint integrity and immutability.
2. Image and mask dimension consistency under multi-scale transformations.
3. Strict binary mask label values (no interpolation artifacts).
4. Label preservation and small solar region survival during augmentation.
5. Deterministic, reproducible validation preprocessing.
6. Balanced dual-domain 50/50 sample counts and batch collation.
7. Complete isolation of held-out test sets.
8. Checkpoint registry protection against overwrites.
9. Harmonic mean multi-domain model selection criterion.
"""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.finetuning.dataset import (
    BalancedDualDomainDataset,
    PV01Dataset,
    SolarMapReplayDataset,
    finetuning_collate_fn,
)
from ml.finetuning.multiscale_augmentations import LabelPreservingMultiScaleAugmentor
from ml.inference.model import HalfUNet, load_halfunet_model
from scripts.train_phase8_multiscale import (
    BASELINE_CHECKPOINT,
    ProtectedCheckpointRegistry,
    STARTING_CHECKPOINT,
    compute_file_sha256,
    compute_harmonic_mean,
)

PHASE3_BEST = STARTING_CHECKPOINT
PHASE3_LATEST = (
    PROJECT_ROOT / "outputs" / "experiments" / "pv01_finetuning" / "models" / "halfunet_pv01_latest.pth"
)
PV01_SPLITS_DIR = PROJECT_ROOT / "outputs" / "experiments" / "pv01_finetuning" / "splits"
SOLARMAP_SPLITS_DIR = PROJECT_ROOT / "splits"


class TestPhase8MultiScalePlan(unittest.TestCase):
    """Validation test suite for Phase 8 Multi-Scale Pipeline."""

    def test_01_protected_checkpoints_integrity(self):
        """Verifies existence, non-empty size, and recorded SHA-256 for all protected checkpoints."""
        checkpoints = [BASELINE_CHECKPOINT, PHASE3_BEST, PHASE3_LATEST]
        for ckpt in checkpoints:
            self.assertTrue(ckpt.is_file(), f"Missing protected checkpoint: {ckpt}")
            self.assertGreater(ckpt.stat().st_size, 20_000_000, f"Checkpoint file suspiciously small: {ckpt}")

        # Baseline hash must match initial recorded value
        baseline_sha = compute_file_sha256(BASELINE_CHECKPOINT)
        self.assertEqual(
            baseline_sha,
            "7a297ade86f7ac8a8c7d6a55c9c0409671f9b5a6f3f0c5c29429610e7e74ac7e",
            "Baseline checkpoint SHA-256 altered!",
        )

    def test_02_image_and_mask_dimension_consistency(self):
        """Verifies that multi-scale augmented image and mask always match target_size exactly."""
        augmentor = LabelPreservingMultiScaleAugmentor(target_size=(256, 256), scale_range=(0.75, 1.35), p_scale=1.0)
        rng = np.random.RandomState(42)

        for _ in range(25):
            # Create synthetic image and mask
            img_arr = rng.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            mask_arr = (rng.rand(256, 256) > 0.8).astype(np.uint8)

            img_pil = Image.fromarray(img_arr, mode="RGB")
            mask_pil = Image.fromarray(mask_arr, mode="L")

            aug_img, aug_mask = augmentor(img_pil, mask_pil)

            self.assertEqual(aug_img.size, (256, 256))
            self.assertEqual(aug_mask.size, (256, 256))

    def test_03_correct_binary_mask_values(self):
        """Asserts that masks never contain intermediate fractional float artifacts after augmentation."""
        augmentor = LabelPreservingMultiScaleAugmentor(
            target_size=(256, 256),
            scale_range=(0.70, 1.40),
            p_scale=1.0,
            p_rotate_ortho=1.0,
        )

        mask_arr = np.zeros((256, 256), dtype=np.uint8)
        mask_arr[100:150, 100:150] = 1  # foreground square
        img_arr = np.full((256, 256, 3), 128, dtype=np.uint8)

        for _ in range(30):
            img_pil = Image.fromarray(img_arr, mode="RGB")
            mask_pil = Image.fromarray(mask_arr, mode="L")

            _, aug_mask = augmentor(img_pil, mask_pil)
            unique_vals = set(np.unique(np.array(aug_mask)))

            self.assertTrue(
                unique_vals.issubset({0, 1}),
                f"Corrupted mask labels detected: {unique_vals}",
            )

    def test_04_label_preservation_and_solar_region_survival(self):
        """Verifies that small solar panel regions survive scaling/cropping without being completely erased."""
        augmentor = LabelPreservingMultiScaleAugmentor(
            target_size=(256, 256),
            scale_range=(1.10, 1.30),  # zoom in
            p_scale=1.0,
        )

        mask_arr = np.zeros((256, 256), dtype=np.uint8)
        mask_arr[120:136, 120:136] = 1  # 16x16 solar panel patch
        img_arr = np.full((256, 256, 3), 100, dtype=np.uint8)

        survived_count = 0
        trials = 20
        for _ in range(trials):
            img_pil = Image.fromarray(img_arr, mode="RGB")
            mask_pil = Image.fromarray(mask_arr, mode="L")

            _, aug_mask = augmentor(img_pil, mask_pil)
            if np.array(aug_mask).sum() > 0:
                survived_count += 1

        # Because augmentor anchors around solar regions when zooming in, panel survival should be very high
        self.assertGreaterEqual(
            survived_count / trials,
            0.80,
            f"Solar panel survival rate too low: {survived_count}/{trials}",
        )

    def test_05_reproducible_validation_preprocessing(self):
        """Verifies deterministic, reproducible output for validation datasets."""
        pv01_val_ds = PV01Dataset(PV01_SPLITS_DIR / "pv01_val.csv", is_train=False)
        img1, mask1, _ = pv01_val_ds[0]
        img2, mask2, _ = pv01_val_ds[0]

        self.assertTrue(torch.equal(img1, img2), "PV01 validation preprocessing is non-deterministic!")
        self.assertTrue(torch.equal(mask1, mask2), "PV01 validation mask is non-deterministic!")

    def test_06_mixed_domain_sample_counts_and_domain_balance(self):
        """Verifies BalancedDualDomainDataset achieves exact 50/50 balance between PV01 and SolarMap."""
        pv01_ds = PV01Dataset(PV01_SPLITS_DIR / "pv01_train.csv", is_train=False)
        solarmap_ds = SolarMapReplayDataset(split="train", splits_dir=SOLARMAP_SPLITS_DIR, is_train=False)

        dual_ds = BalancedDualDomainDataset(pv01_ds, solarmap_ds, samples_per_epoch=20)
        self.assertEqual(len(dual_ds), 20)

        # Inspect domain tags of all 20 samples
        domains = [dual_ds[i][2]["dataset"] for i in range(20)]
        pv01_count = sum(1 for d in domains if d == "PV01")
        solarmap_count = sum(1 for d in domains if d == "SolarMap-India")

        self.assertEqual(pv01_count, 10, f"Expected 10 PV01 samples, got {pv01_count}")
        self.assertEqual(solarmap_count, 10, f"Expected 10 SolarMap samples, got {solarmap_count}")

    def test_07_test_set_isolation(self):
        """Verifies that no held-out test images are present in the Phase 8 training datasets."""
        # PV01 held-out test
        pv01_test_csv = PV01_SPLITS_DIR / "pv01_test.csv"
        df_pv_test = pd.read_csv(pv01_test_csv)
        pv01_test_files = set(df_pv_test["filename"])

        pv01_train_ds = PV01Dataset(PV01_SPLITS_DIR / "pv01_train.csv")
        pv01_train_files = set(pv01_train_ds.df["filename"])
        self.assertEqual(len(pv01_test_files & pv01_train_files), 0, "PV01 test leakage in train split!")

        # SolarMap locked test
        pos_locked_csv = SOLARMAP_SPLITS_DIR / "positive_test_LOCKED.csv"
        neg_locked_csv = SOLARMAP_SPLITS_DIR / "negative_test_LOCKED.csv"
        locked_files = set(pd.read_csv(pos_locked_csv)["filename"]).union(set(pd.read_csv(neg_locked_csv)["filename"]))

        solarmap_train_ds = SolarMapReplayDataset(split="train", splits_dir=SOLARMAP_SPLITS_DIR)
        train_sm_files = set(r["filename"] for r in solarmap_train_ds.records)
        self.assertEqual(len(locked_files & train_sm_files), 0, "SolarMap locked test leakage in train split!")

    def test_08_checkpoint_registry_protection(self):
        """Verifies ProtectedCheckpointRegistry prevents overwriting any protected checkpoint."""
        registry = ProtectedCheckpointRegistry([BASELINE_CHECKPOINT, PHASE3_BEST, PHASE3_LATEST])

        # Attempting to overwrite baseline must fail
        with self.assertRaises(PermissionError):
            registry.verify_target_allowed(BASELINE_CHECKPOINT)

        # Attempting to overwrite Phase 3 best must fail
        with self.assertRaises(PermissionError):
            registry.verify_target_allowed(PHASE3_BEST)

        # Attempting to overwrite Phase 3 latest must fail
        with self.assertRaises(PermissionError):
            registry.verify_target_allowed(PHASE3_LATEST)

        # Allowed new Phase 8 target should pass
        allowed_target = PROJECT_ROOT / "outputs" / "experiments" / "phase8_multiscale" / "models" / "halfunet_phase8_best.pth"
        registry.verify_target_allowed(allowed_target)

    def test_09_harmonic_mean_selection_criterion(self):
        """Verifies mathematical properties of harmonic mean for balanced model selection."""
        # Case A: Imbalanced model (high PV01, collapsed SolarMap)
        h_imbalanced = compute_harmonic_mean(0.96, 0.50)
        # Case B: Balanced model (strong across both domains)
        h_balanced = compute_harmonic_mean(0.92, 0.85)

        self.assertGreater(
            h_balanced,
            h_imbalanced,
            f"Harmonic mean failed to favor balanced model: {h_balanced:.4f} vs {h_imbalanced:.4f}",
        )


if __name__ == "__main__":
    unittest.main()
