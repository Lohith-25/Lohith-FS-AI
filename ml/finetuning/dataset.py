"""
SolarMap-India — Phase 3: Dataset and DataLoader Utilities for Fine-Tuning.

Implements:
1. PV01Dataset: Loads paired 256x256 BMP images and binary label masks from spatial split CSVs.
2. SolarMapReplayDataset: Loads native Indian training images (positive_train + negative_train)
   to prevent catastrophic forgetting. Strictly blocks locked test set images.
3. MixedDomainDataset: Merges PV01 and SolarMap training pools with balanced sampling.
"""

from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOLARMAP_IMAGES_BASE = PROJECT_ROOT / "dataset" / "Solar Images" / "Solar Images" / "images"
SOLARMAP_ANNOTATIONS_JSON = (
    PROJECT_ROOT / "dataset" / "Solar Images" / "Solar Images" / "annotations" / "merged_instances_default.json"
)
SOLARMAP_SPLITS_DIR = PROJECT_ROOT / "splits"


def rasterize_coco_polygons(segmentations: Any, height: int, width: int) -> np.ndarray:
    """Rasterize COCO polygon coordinates into a binary uint8 mask."""
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    if not segmentations:
        return np.array(mask_img, dtype=np.uint8)

    polys = []
    # If segmentations is a single flat list of floats: [x1, y1, x2, y2, ...]
    if isinstance(segmentations, list) and len(segmentations) > 0 and isinstance(segmentations[0], (int, float)):
        polys.append(segmentations)
    elif isinstance(segmentations, list):
        for item in segmentations:
            if isinstance(item, list) and len(item) > 0 and isinstance(item[0], (int, float)):
                polys.append(item)
            elif isinstance(item, list):
                for subitem in item:
                    if isinstance(subitem, list) and len(subitem) > 0 and isinstance(subitem[0], (int, float)):
                        polys.append(subitem)

    for poly in polys:
        if len(poly) < 6:
            continue
        xy = list(zip(poly[0::2], poly[1::2]))
        draw.polygon(xy, outline=1, fill=1)

    return np.array(mask_img, dtype=np.uint8)


class PV01Dataset(Dataset):
    """
    Dataset for PV01 imagery and binary label masks.
    
    Args:
        manifest_csv: Path to split CSV (pv01_train.csv, pv01_val.csv, or pv01_test.csv).
        is_train: If True, applies conservative geometric augmentations (HFlip, VFlip).
        img_size: Target (H, W) resolution (default: (256, 256)).
    """

    def __init__(
        self,
        manifest_csv: Union[str, Path],
        is_train: bool = False,
        img_size: Tuple[int, int] = (256, 256),
        augmentor: Optional[Any] = None,
    ):
        self.manifest_path = Path(manifest_csv)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest CSV not found: {self.manifest_path.resolve()}")

        self.df = pd.read_csv(self.manifest_path)
        self.is_train = is_train
        self.img_size = img_size
        self.augmentor = augmentor

        # Pre-verify all file paths exist
        for idx, row in self.df.iterrows():
            if not Path(row["image_path"]).is_file():
                raise FileNotFoundError(f"Image not found: {row['image_path']}")
            if not Path(row["mask_path"]).is_file():
                raise FileNotFoundError(f"Mask not found: {row['mask_path']}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        row = self.df.iloc[idx]

        # 1. Load image and mask
        image = Image.open(row["image_path"]).convert("RGB")
        mask_raw = Image.open(row["mask_path"])

        # 2. Convert mask to binary (pixels > 0 are solar foreground)
        mask_arr = (np.array(mask_raw) > 0).astype(np.uint8)
        mask = Image.fromarray(mask_arr)

        # 3. Apply augmentations
        if self.augmentor is not None:
            image, mask = self.augmentor(image, mask)
        else:
            # Resize if needed
            if image.size != self.img_size:
                image = image.resize(self.img_size, resample=Image.BILINEAR)
            if mask.size != self.img_size:
                mask = mask.resize(self.img_size, resample=Image.NEAREST)

            # Conservative augmentations during training
            if self.is_train:
                if random.random() > 0.5:
                    image = TF.hflip(image)
                    mask = TF.hflip(mask)
                if random.random() > 0.5:
                    image = TF.vflip(image)
                    mask = TF.vflip(mask)

        # 5. Convert to PyTorch tensors
        # Image: (3, H, W) float in [0.0, 1.0]
        image_tensor = TF.to_tensor(image)
        # Mask: (H, W) int64 in {0, 1}
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.int64))

        meta = {
            "dataset": "PV01",
            "filename": row["filename"],
            "category": row["category"],
            "x_coord": str(row["x_coord"]),
            "has_solar": 1,
        }
        return image_tensor, mask_tensor, meta


class SolarMapReplayDataset(Dataset):
    """
    Replay dataset of native SolarMap-India training samples (positive_train + negative_train).
    Strictly asserts that NO images from positive_test_LOCKED.csv or negative_test_LOCKED.csv
    are ever included.
    """

    def __init__(
        self,
        split: str = "train",
        splits_dir: Path = SOLARMAP_SPLITS_DIR,
        images_base: Path = SOLARMAP_IMAGES_BASE,
        annotations_json: Path = SOLARMAP_ANNOTATIONS_JSON,
        is_train: Optional[bool] = None,
        img_size: Tuple[int, int] = (256, 256),
        augmentor: Optional[Any] = None,
    ):
        if split not in ("train", "val"):
            raise ValueError(f"SolarMapReplayDataset only allows split='train' or 'val', got: {split}")

        self.split = split
        self.is_train = (split == "train") if is_train is None else is_train
        self.img_size = img_size
        self.images_base = images_base
        self.augmentor = augmentor

        # 1. Load locked test sets to build a strict exclusion filter
        pos_locked_csv = splits_dir / "positive_test_LOCKED.csv"
        neg_locked_csv = splits_dir / "negative_test_LOCKED.csv"
        df_pos_locked = pd.read_csv(pos_locked_csv)
        df_neg_locked = pd.read_csv(neg_locked_csv)
        self.locked_filenames = set(df_pos_locked["filename"]).union(set(df_neg_locked["filename"]))

        # 2. Load COCO annotations for positive polygons
        with open(annotations_json, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images_by_fn = {img["file_name"]: img for img in coco["images"]}
        self.anns_by_img_id = defaultdict(list)
        for ann in coco["annotations"]:
            self.anns_by_img_id[ann["image_id"]].append(ann)

        # 3. Load split positive and negative CSVs
        pos_csv = splits_dir / f"positive_{split}.csv"
        neg_csv = splits_dir / f"negative_{split}.csv"
        df_pos = pd.read_csv(pos_csv)
        df_neg = pd.read_csv(neg_csv)

        records = []
        # Positive samples
        for _, row in df_pos.iterrows():
            fn = row["filename"]
            assert fn not in self.locked_filenames, f"CRITICAL: Locked test image in positive_{split}: {fn}"
            img_path = self.images_base / "default" / fn
            if img_path.is_file():
                records.append({
                    "filename": fn,
                    "image_path": str(img_path),
                    "has_solar": 1,
                    "dataset_type": "positive",
                })

        # Negative samples
        for _, row in df_neg.iterrows():
            fn = row["filename"]
            assert fn not in self.locked_filenames, f"CRITICAL: Locked test image in negative_{split}: {fn}"
            img_path = self.images_base / "test" / fn
            if img_path.is_file():
                records.append({
                    "filename": fn,
                    "image_path": str(img_path),
                    "has_solar": 0,
                    "dataset_type": "negative",
                })

        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        item = self.records[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        w, h = image.size

        # Create mask
        if item["has_solar"] == 1:
            coco_img = self.images_by_fn.get(item["filename"])
            if coco_img:
                anns = self.anns_by_img_id.get(coco_img["id"], [])
                flat_segs = []
                for a in anns:
                    if "segmentation" in a and isinstance(a["segmentation"], list):
                        for s in a["segmentation"]:
                            flat_segs.append(s)
                mask_arr = rasterize_coco_polygons(flat_segs, h, w)
            else:
                mask_arr = np.zeros((h, w), dtype=np.uint8)
        else:
            mask_arr = np.zeros((h, w), dtype=np.uint8)

        mask = Image.fromarray(mask_arr)

        # Apply augmentations
        if self.augmentor is not None:
            image, mask = self.augmentor(image, mask)
        else:
            # Resize to target input resolution
            image = image.resize(self.img_size, resample=Image.BILINEAR)
            mask = mask.resize(self.img_size, resample=Image.NEAREST)

            # Conservative augmentations during training
            if self.is_train:
                if random.random() > 0.5:
                    image = TF.hflip(image)
                    mask = TF.hflip(mask)
                if random.random() > 0.5:
                    image = TF.vflip(image)
                    mask = TF.vflip(mask)

        image_tensor = TF.to_tensor(image)
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.int64))

        meta = {
            "dataset": "SolarMap-India",
            "filename": item["filename"],
            "category": item["dataset_type"],
            "x_coord": "N/A",
            "has_solar": item["has_solar"],
        }
        return image_tensor, mask_tensor, meta


class MixedDomainDataset(Dataset):
    """
    Combined dataset interleaving external PV01 drone imagery with
    native SolarMap-India satellite imagery to prevent catastrophic forgetting.
    """

    def __init__(
        self,
        pv01_dataset: PV01Dataset,
        solarmap_dataset: SolarMapReplayDataset,
        pv01_ratio: float = 0.5,
    ):
        self.pv01_ds = pv01_dataset
        self.solarmap_ds = solarmap_dataset
        self.pv01_ratio = pv01_ratio

        # Total virtual size per epoch
        self.length = len(self.pv01_ds) * 2

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        # Alternating or probabilistic interleaving
        if idx % 2 == 0:
            pv_idx = (idx // 2) % len(self.pv01_ds)
            return self.pv01_ds[pv_idx]
        else:
            sm_idx = (idx // 2) % len(self.solarmap_ds)
            return self.solarmap_ds[sm_idx]


class BalancedDualDomainDataset(Dataset):
    """
    Equally balanced dual-domain dataset pairing PV01 drone imagery and
    SolarMap-India satellite imagery.
    
    Guarantees:
    - 50% of sampled items are PV01 drone rooftop images.
    - 50% of sampled items are SolarMap-India satellite images.
    - Configurable epoch length (default: 2 * max(len(pv01), len(solarmap)) = 4,586).
    """

    def __init__(
        self,
        pv01_dataset: PV01Dataset,
        solarmap_dataset: SolarMapReplayDataset,
        samples_per_epoch: Optional[int] = None,
    ):
        self.pv01_ds = pv01_dataset
        self.solarmap_ds = solarmap_dataset
        self.length = (
            samples_per_epoch
            if samples_per_epoch is not None
            else 2 * max(len(self.pv01_ds), len(self.solarmap_ds))
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if idx % 2 == 0:
            pv_idx = (idx // 2) % len(self.pv01_ds)
            return self.pv01_ds[pv_idx]
        else:
            sm_idx = (idx // 2) % len(self.solarmap_ds)
            return self.solarmap_ds[sm_idx]


def finetuning_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]):
    """
    Standard batch collator that stacks image and mask tensors and aggregates metadata dictionaries.
    """
    images = torch.stack([item[0] for item in batch])
    masks = torch.stack([item[1] for item in batch])
    metas = [item[2] for item in batch]
    return images, masks, metas
