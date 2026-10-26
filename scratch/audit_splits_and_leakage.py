"""
SolarMap-India — Phase 1 & 3: Split Audit, Exact Duplicates, Perceptual Hashes, and Spatial Proximity.
Analyzes exact hashes, dHash distances, and GPS coordinates across Train, Val, and Test splits.
"""

import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

PROJECT_ROOT = Path("d:/SolarMap-India")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from PIL import Image

SPLIT_FILE = PROJECT_ROOT / "outputs/HalfUNet/metrics/split_image_ids.json"
ANNOTATIONS_FILE = PROJECT_ROOT / "dataset/Solar Images/Solar Images/annotations/merged_instances_default.json"
IMAGES_DIR = PROJECT_ROOT / "dataset/Solar Images/Solar Images/images/default"
GPS_CSV = PROJECT_ROOT / "dataset/EI_train_data(Sheet1).csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs/leakage_and_split_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def haversine_distance_m(lat1, lon1, lat2, lon2):
    """Calculates great-circle distance between two GPS coordinates in meters."""
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def compute_dhash(img: Image.Image, hash_size: int = 8) -> int:
    """Computes difference hash (dHash) of an image for perceptual similarity comparison."""
    # Resize to (hash_size + 1, hash_size)
    resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(resized, dtype=np.int32)
    # Difference between consecutive columns
    diff = pixels[:, 1:] > pixels[:, :-1]
    # Convert to 64-bit integer
    decimal_value = 0
    for val in diff.flatten():
        decimal_value = (decimal_value << 1) | int(val)
    return decimal_value


def hamming_distance(h1: int, h2: int) -> int:
    """Computes bitwise Hamming distance between two 64-bit integer hashes."""
    x = h1 ^ h2
    return bin(x).count("1")


def main():
    print("==================================================================")
    print("PHASE 1 & 3: DATASET SPLIT AUDIT & LEAKAGE INVESTIGATION")
    print("==================================================================")

    # 1. Load split IDs
    with open(SPLIT_FILE, "r") as f:
        splits = json.load(f)
    train_ids = set(splits["train"])
    val_ids = set(splits["val"])
    test_ids = set(splits["test"])

    print(f"Split counts: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    print(f"Train & Val overlap:  {len(train_ids & val_ids)}")
    print(f"Train & Test overlap: {len(train_ids & test_ids)}")
    print(f"Val & Test overlap:   {len(val_ids & test_ids)}")

    # 2. Load COCO images
    with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
        coco = json.load(f)
    coco_images = {x["id"]: x for x in coco["images"]}
    print(f"Total COCO images: {len(coco_images)}")

    # 3. Load GPS data
    gps_df = pd.read_csv(GPS_CSV)
    print(f"Loaded GPS metadata: {len(gps_df)} rows")
    gps_by_sampleid = {}
    for _, row in gps_df.iterrows():
        gps_by_sampleid[int(row["sampleid"])] = (float(row["latitude"]), float(row["longitude"]))

    # Parse sampleids from filenames and extract image hashes
    img_records = []
    sha_map = {}
    dhash_map = {}

    print("\nComputing SHA-256 and perceptual dHash for all 2,500 images...")
    t0 = time.time()
    for img_id, info in coco_images.items():
        fn = Path(info["file_name"]).name
        img_path = IMAGES_DIR / fn
        if not img_path.exists():
            print(f"Warning: image missing: {img_path}")
            continue

        with open(img_path, "rb") as f:
            file_bytes = f.read()
        sha256 = hashlib.sha256(file_bytes).hexdigest()

        with Image.open(img_path) as im:
            dhash = compute_dhash(im)

        # Parse sampleid
        m = re.match(r"(\d+)\.0_(\d+)\.0\.png", fn)
        if m:
            sid = int(m.group(1))
            label = int(m.group(2))
        else:
            # fallback for nonstandard filenames like 0960_1.png
            digits = re.findall(r"\d+", fn)
            sid = int(digits[0]) if digits else img_id
            label = 1

        split_name = "train" if img_id in train_ids else ("val" if img_id in val_ids else "test")
        lat, lon = gps_by_sampleid.get(sid, (None, None))

        rec = {
            "image_id": img_id,
            "file_name": fn,
            "sample_id": sid,
            "split": split_name,
            "sha256": sha256,
            "dhash": dhash,
            "latitude": lat,
            "longitude": lon,
        }
        img_records.append(rec)
        sha_map[img_id] = sha256
        dhash_map[img_id] = dhash

    print(f"Processed {len(img_records)} images in {time.time() - t0:.2f}s")
    df = pd.DataFrame(img_records)

    # -------------------------------------------------------------
    # 4. EXACT DUPLICATE CHECK (SHA-256)
    # -------------------------------------------------------------
    print("\n--- Exact File Duplication Analysis (SHA-256) ---")
    sha_counts = df["sha256"].value_counts()
    exact_duplicates = sha_counts[sha_counts > 1]
    print(f"Total unique SHA-256 hashes: {len(sha_counts)} / {len(df)}")
    print(f"Exact duplicate hashes found: {len(exact_duplicates)}")

    if len(exact_duplicates) > 0:
        for dup_sha, count in exact_duplicates.items():
            dup_rows = df[df["sha256"] == dup_sha]
            splits_involved = dup_rows["split"].tolist()
            files_involved = dup_rows["file_name"].tolist()
            print(f"  Hash {dup_sha[:12]}... repeated {count} times across splits: {splits_involved} (files: {files_involved})")
    else:
        print("  VERIFIED: Zero exact duplicate image files exist across the entire 2,500 dataset.")

    # -------------------------------------------------------------
    # 5. PERCEPTUAL DUPLICATE CHECK (dHash Hamming Distance <= 3)
    # -------------------------------------------------------------
    print("\n--- Perceptual Similarity Analysis (dHash Hamming Distance) ---")
    train_recs = [r for r in img_records if r["split"] == "train"]
    test_recs = [r for r in img_records if r["split"] == "test"]
    val_recs = [r for r in img_records if r["split"] == "val"]

    near_duplicates_train_test = []
    # Check pairwise dHash between Train and Test
    for te in test_recs:
        for tr in train_recs:
            h_dist = hamming_distance(te["dhash"], tr["dhash"])
            if h_dist <= 3:  # very high visual similarity (out of 64 bits)
                near_duplicates_train_test.append({
                    "test_id": te["image_id"],
                    "test_file": te["file_name"],
                    "train_id": tr["image_id"],
                    "train_file": tr["file_name"],
                    "hamming_distance": h_dist,
                })

    print(f"Near-duplicate pairs between Train and Test (dHash <= 3): {len(near_duplicates_train_test)}")
    for nd in near_duplicates_train_test[:10]:
        print(f"  Test {nd['test_file']} vs Train {nd['train_file']} (distance={nd['hamming_distance']})")

    # -------------------------------------------------------------
    # 6. SPATIAL PROXIMITY & LEAKAGE CHECK (GPS DISTANCE)
    # -------------------------------------------------------------
    print("\n--- Spatial Proximity & Scene Overlap Analysis ---")
    # For zoom-19 satellite imagery, 640x640 at ~0.25m/px covers ~160m x 160m.
    # Images with centers within 160m have overlapping geographic footprint!
    spatial_overlap_thresholds = [20.0, 50.0, 100.0, 150.0, 200.0]
    
    close_pairs_train_test = {thr: [] for thr in spatial_overlap_thresholds}
    
    t_gps_0 = time.time()
    for te in test_recs:
        lat1, lon1 = te["latitude"], te["longitude"]
        if lat1 is None or lon1 is None or math.isnan(lat1) or math.isnan(lon1):
            continue
        for tr in train_recs:
            lat2, lon2 = tr["latitude"], tr["longitude"]
            if lat2 is None or lon2 is None or math.isnan(lat2) or math.isnan(lon2):
                continue
            
            # Fast bounding box pre-filter (~1 degree lat ~ 111km, 0.002 deg ~ 220m)
            if abs(lat1 - lat2) > 0.003 or abs(lon1 - lon2) > 0.003:
                continue
            
            dist = haversine_distance_m(lat1, lon1, lat2, lon2)
            for thr in spatial_overlap_thresholds:
                if dist <= thr:
                    close_pairs_train_test[thr].append({
                        "test_id": te["image_id"],
                        "test_file": te["file_name"],
                        "train_id": tr["image_id"],
                        "train_file": tr["file_name"],
                        "distance_m": round(dist, 2),
                    })

    print(f"Spatial proximity check completed in {time.time() - t_gps_0:.2f}s")
    print("Pairs between Train and Test splits within spatial radii:")
    for thr in spatial_overlap_thresholds:
        count = len(close_pairs_train_test[thr])
        print(f"  Distance <= {thr:5.1f}m: {count:4d} image pairs")

    if len(close_pairs_train_test[100.0]) > 0:
        print("\nSample geographically overlapping pairs (distance <= 100m):")
        for p in close_pairs_train_test[100.0][:5]:
            print(f"  Test {p['test_file']} and Train {p['train_file']}: Distance = {p['distance_m']}m")

    # Save complete audit report JSON
    audit_summary = {
        "dataset_size": len(coco_images),
        "split_counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        "exact_duplicates": len(exact_duplicates),
        "perceptual_duplicates_dhash_le_3": len(near_duplicates_train_test),
        "spatial_leakage_train_test": {
            f"within_{int(thr)}m": len(close_pairs_train_test[thr])
            for thr in spatial_overlap_thresholds
        },
        "sample_spatial_overlaps": close_pairs_train_test[100.0][:10],
    }

    out_file = OUTPUT_DIR / "split_and_leakage_audit_summary.json"
    with open(out_file, "w") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"\nSaved audit summary to {out_file}")


if __name__ == "__main__":
    main()
