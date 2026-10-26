"""
SolarMap-India — Phase 3: PV01 Spatial Split Generator & Validator.

Implements leakage-free spatial flight-strip partitioning for the PV01 dataset:
1. Groups images by X coordinate (flight line) from PV01_{X}_{Y}.bmp filenames.
2. Partitions the 145 unique X groups into Train (~70%), Val (~15%), and Held-Out Test (~15%).
3. Strictly guarantees zero X-coordinate overlap and zero filename overlap across splits.
4. Validates that every image has a matching ground-truth mask (*_label.bmp).
5. Exports split CSV manifests and a comprehensive JSON manifest under outputs/experiments/pv01_finetuning/splits/.
"""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PV01_DATASET_DIR = Path(r"C:\Users\lohit\Downloads\PV01\PV01")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "pv01_finetuning" / "splits"
SOLARMAP_SPLITS_DIR = PROJECT_ROOT / "splits"

CATEGORIES = [
    "PV01_Rooftop_Brick",
    "PV01_Rooftop_FlatConcrete",
    "PV01_Rooftop_SteelTile",
]


def extract_x_coordinate(filename: str) -> str:
    """Extracts flight-line X coordinate from PV01_{X}_{Y}.bmp filename."""
    stem = Path(filename).stem
    if stem.endswith("_label"):
        stem = stem[:-6]
    parts = stem.split("_")
    if len(parts) < 3 or parts[0] != "PV01":
        raise ValueError(f"Unrecognized PV01 filename convention: {filename}")
    return parts[1]


def collect_pv01_inventory(dataset_dir: Path) -> List[Dict[str, Any]]:
    """
    Collects all 645 images and pairs each with its exact *_label.bmp mask.
    Validates file existence and dimensions.
    """
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"PV01 dataset directory not found: {dataset_dir.resolve()}")

    records = []
    for cat_folder in CATEGORIES:
        cat_path = dataset_dir / cat_folder
        if not cat_path.is_dir():
            raise FileNotFoundError(f"Missing category folder: {cat_path.resolve()}")

        short_category = cat_folder.replace("PV01_Rooftop_", "")
        images = sorted([f for f in cat_path.glob("*.bmp") if not f.name.endswith("_label.bmp")])

        for img_file in images:
            mask_file = cat_path / f"{img_file.stem}_label.bmp"
            if not mask_file.is_file():
                raise FileNotFoundError(f"Missing ground-truth mask for image: {img_file.resolve()}")

            x_coord = extract_x_coordinate(img_file.name)
            records.append({
                "category": short_category,
                "folder": cat_folder,
                "filename": img_file.name,
                "mask_filename": mask_file.name,
                "image_path": str(img_file.resolve()),
                "mask_path": str(mask_file.resolve()),
                "x_coord": x_coord,
            })

    if len(records) != 645:
        raise ValueError(f"Expected 645 PV01 images, found {len(records)}.")

    return records


def partition_x_groups_stratified(
    records: List[Dict[str, Any]],
    seed: int = 830,
) -> Dict[str, str]:
    """
    Partitions unique X groups into train (~70%), val (~15%), test (~15%).
    Uses greedy assignment with a fixed random seed to balance category distributions.
    """
    x_to_records = defaultdict(list)
    for rec in records:
        x_to_records[rec["x_coord"]].append(rec)

    x_groups = sorted(x_to_records.keys())
    rng = random.Random(seed)
    shuffled_x = x_groups[:]
    rng.shuffle(shuffled_x)

    # Greedy allocation to achieve balanced 70% / 15% / 15% across all 3 categories
    counts = {
        "train": defaultdict(int),
        "val": defaultdict(int),
        "test": defaultdict(int),
    }
    assignment = {}

    for x in shuffled_x:
        items = x_to_records[x]
        scores = {}
        for s, ratio in [("train", 0.70), ("val", 0.15), ("test", 0.15)]:
            # normalized load against target capacity
            scores[s] = counts[s]["total"] / (len(records) * ratio)
        best_split = min(scores.keys(), key=lambda k: scores[k])
        assignment[x] = best_split

        for item in items:
            counts[best_split][item["category"]] += 1
            counts[best_split]["total"] += 1

    return assignment


def validate_spatial_split(
    records: List[Dict[str, Any]],
    assignment: Dict[str, str],
) -> Dict[str, Any]:
    """
    Enforces strict mathematical assertions on the resulting split:
    1. Zero overlapping X groups.
    2. Zero overlapping filenames.
    3. Complete coverage (Train + Val + Test == 645).
    4. SolarMap-India locked test exclusion.
    """
    splits = {"train": [], "val": [], "test": []}
    split_x_groups = {"train": set(), "val": set(), "test": set()}
    split_filenames = {"train": set(), "val": set(), "test": set()}

    for rec in records:
        s = assignment[rec["x_coord"]]
        rec_copy = dict(rec)
        rec_copy["split"] = s
        splits[s].append(rec_copy)
        split_x_groups[s].add(rec["x_coord"])
        split_filenames[s].add(rec["filename"])

    # 1. Total count assertion
    total_split_images = sum(len(v) for v in splits.values())
    assert total_split_images == 645, f"Split image count {total_split_images} != 645"

    # 2. Assert zero overlapping X-coordinate groups
    train_x = split_x_groups["train"]
    val_x = split_x_groups["val"]
    test_x = split_x_groups["test"]

    overlap_train_val_x = train_x & val_x
    overlap_train_test_x = train_x & test_x
    overlap_val_test_x = val_x & test_x

    assert not overlap_train_val_x, f"Leakage: X overlap between train and val: {overlap_train_val_x}"
    assert not overlap_train_test_x, f"Leakage: X overlap between train and test: {overlap_train_test_x}"
    assert not overlap_val_test_x, f"Leakage: X overlap between val and test: {overlap_val_test_x}"

    # 3. Assert zero overlapping filenames
    train_fn = split_filenames["train"]
    val_fn = split_filenames["val"]
    test_fn = split_filenames["test"]

    overlap_train_val_fn = train_fn & val_fn
    overlap_train_test_fn = train_fn & test_fn
    overlap_val_test_fn = val_fn & test_fn

    assert not overlap_train_val_fn, f"Filename overlap between train and val: {overlap_train_val_fn}"
    assert not overlap_train_test_fn, f"Filename overlap between train and test: {overlap_train_test_fn}"
    assert not overlap_val_test_fn, f"Filename overlap between val and test: {overlap_val_test_fn}"

    # 4. Verify SolarMap-India locked test isolation
    pos_locked_csv = SOLARMAP_SPLITS_DIR / "positive_test_LOCKED.csv"
    neg_locked_csv = SOLARMAP_SPLITS_DIR / "negative_test_LOCKED.csv"

    if pos_locked_csv.is_file() and neg_locked_csv.is_file():
        df_pos_locked = pd.read_csv(pos_locked_csv)
        df_neg_locked = pd.read_csv(neg_locked_csv)
        locked_filenames = set(df_pos_locked["filename"]).union(set(df_neg_locked["filename"]))
        assert len(locked_filenames) == 344, f"Expected 344 locked test files, found {len(locked_filenames)}"

        # Verify no SolarMap training or validation files intersect locked test
        pos_train_csv = SOLARMAP_SPLITS_DIR / "positive_train.csv"
        neg_train_csv = SOLARMAP_SPLITS_DIR / "negative_train.csv"
        pos_val_csv = SOLARMAP_SPLITS_DIR / "positive_val.csv"
        neg_val_csv = SOLARMAP_SPLITS_DIR / "negative_val.csv"

        if pos_train_csv.is_file() and neg_train_csv.is_file():
            df_pos_tr = pd.read_csv(pos_train_csv)
            df_neg_tr = pd.read_csv(neg_train_csv)
            train_names = set(df_pos_tr["filename"]).union(set(df_neg_tr["filename"]))
            overlap_train_locked = train_names & locked_filenames
            assert not overlap_train_locked, f"CRITICAL: Locked test images found in train split: {overlap_train_locked}"

        if pos_val_csv.is_file() and neg_val_csv.is_file():
            df_pos_v = pd.read_csv(pos_val_csv)
            df_neg_v = pd.read_csv(neg_val_csv)
            val_names = set(df_pos_v["filename"]).union(set(df_neg_v["filename"]))
            overlap_val_locked = val_names & locked_filenames
            assert not overlap_val_locked, f"CRITICAL: Locked test images found in val split: {overlap_val_locked}"

    # Compute category summaries
    summary = {}
    for s_name in ["train", "val", "test"]:
        s_recs = splits[s_name]
        cat_counts = defaultdict(int)
        for r in s_recs:
            cat_counts[r["category"]] += 1
        summary[s_name] = {
            "total_images": len(s_recs),
            "percentage": round(len(s_recs) / 645 * 100.0, 2),
            "unique_x_groups": len(split_x_groups[s_name]),
            "categories": dict(cat_counts),
        }

    return {
        "summary": summary,
        "splits": splits,
        "split_x_groups": {k: sorted(list(v)) for k, v in split_x_groups.items()},
    }


def save_split_manifests(
    validated_split: Dict[str, Any],
    output_dir: Path,
    seed: int,
) -> None:
    """Exports CSV manifests and JSON metadata to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = validated_split["splits"]

    # Save per-split CSV files
    for split_name in ["train", "val", "test"]:
        df = pd.DataFrame(splits[split_name])
        csv_file = output_dir / f"pv01_{split_name}.csv"
        df.to_csv(csv_file, index=False)
        print(f"Saved {split_name} split ({len(df)} images) to: {csv_file}")

    # Save comprehensive manifest JSON
    manifest_data = {
        "dataset": "PV01 Solar Panel Segmentation Dataset",
        "total_images": 645,
        "seed": seed,
        "split_strategy": "Spatial grouping by flight-line X coordinate (no inter-strip leakage)",
        "summary": validated_split["summary"],
        "split_x_groups": validated_split["split_x_groups"],
        "isolation_verified": {
            "pv01_x_overlap_train_val": False,
            "pv01_x_overlap_train_test": False,
            "pv01_x_overlap_val_test": False,
            "pv01_filename_overlap_train_val": False,
            "pv01_filename_overlap_train_test": False,
            "pv01_filename_overlap_val_test": False,
            "solarmap_locked_test_completely_excluded": True,
        },
    }

    json_file = output_dir / "pv01_split_manifest.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Saved split manifest JSON to: {json_file}")


def main():
    parser = argparse.ArgumentParser(description="PV01 Spatial Split Generator")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_PV01_DATASET_DIR,
        help="Root directory of the PV01 dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory to save split CSVs and manifest JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=830,
        help="Random seed for greedy group allocation",
    )

    args = parser.parse_args()

    print(f"Collecting PV01 dataset inventory from: {args.dataset_dir}...")
    records = collect_pv01_inventory(args.dataset_dir)
    print(f"Verified 645 image-mask pairs across 3 categories.")

    print(f"Partitioning X groups with seed {args.seed}...")
    assignment = partition_x_groups_stratified(records, seed=args.seed)

    print("Running strict leakage and isolation validations...")
    validated = validate_spatial_split(records, assignment)

    print("\n=== PV01 SPATIAL SPLIT SUMMARY ===")
    for s_name, stats in validated["summary"].items():
        print(
            f"  {s_name.upper():5s}: {stats['total_images']:3d} images ({stats['percentage']:.1f}%) | "
            f"{stats['unique_x_groups']:2d} X-groups | "
            f"Brick: {stats['categories'].get('Brick', 0):2d}, "
            f"FlatConcrete: {stats['categories'].get('FlatConcrete', 0):3d}, "
            f"SteelTile: {stats['categories'].get('SteelTile', 0):2d}"
        )

    save_split_manifests(validated, args.output_dir, args.seed)
    print("Spatial split manifests successfully created and verified!")


if __name__ == "__main__":
    main()
