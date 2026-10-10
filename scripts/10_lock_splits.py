import os
import json
import zipfile
import hashlib

import numpy as np
import pandas as pd

BASE = r"C:\SolarMap-India"
ZIP_PATH = os.path.join(BASE, "dataset", "Solar Images.zip")
CSV_PATH = os.path.join(BASE, "dataset", "EI_train_data(Sheet1).csv")
BASELINE_TEST_CSV = os.path.join(BASE, "results", "baseline", "baseline_per_image_metrics.csv")
OUTPUT_DIR = os.path.join(BASE, "splits")
COCO_PATH_IN_ZIP = "Solar Images/annotations/merged_instances_default.json"

SEED = 42
POS_VAL = 249
NEG_TRAIN = 300
NEG_VAL = 75
NEG_TEST = 94


def parse_sample_id(filename):
    token = os.path.basename(filename).split("_")[0]
    return int(float(token))


def sha256_lines(values):
    payload = "\n".join(sorted(map(str, values))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


for path in [ZIP_PATH, CSV_PATH, BASELINE_TEST_CSV]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df["sampleid"] = df["sampleid"].astype(int)
df["has_solar"] = df["has_solar"].astype(int)

baseline_test = pd.read_csv(BASELINE_TEST_CSV)
test_filenames = baseline_test["filename"].astype(str).tolist()

if len(test_filenames) != 250 or len(set(test_filenames)) != 250:
    raise ValueError("Baseline test set is not exactly 250 unique images.")

with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    png_names = [n for n in zf.namelist() if n.lower().endswith(".png")]
    coco = json.loads(zf.read(COCO_PATH_IN_ZIP))

annotation_counts = {}
for ann in coco["annotations"]:
    annotation_counts[ann["image_id"]] = annotation_counts.get(ann["image_id"], 0) + 1

coco_rows = []
for img in coco["images"]:
    filename = os.path.basename(img["file_name"])
    coco_rows.append({
        "sampleid": parse_sample_id(filename),
        "filename": filename,
        "coco_image_id": img["id"],
        "width": img["width"],
        "height": img["height"],
        "annotation_count": annotation_counts.get(img["id"], 0),
        "has_solar": 1
    })

coco_df = pd.DataFrame(coco_rows)

if len(coco_df) != 2500:
    raise ValueError(f"Expected 2500 COCO images, found {len(coco_df)}")

test_rows = coco_df[coco_df["filename"].isin(test_filenames)].copy()

if len(test_rows) != 250 or (test_rows["annotation_count"] == 0).any():
    raise ValueError("Protected positive test set does not match 250 annotated COCO images.")

positive_annotated = coco_df[coco_df["annotation_count"] > 0].copy()
zero_annotation = coco_df[coco_df["annotation_count"] == 0].copy()

positive_trainval = positive_annotated[
    ~positive_annotated["filename"].isin(test_filenames)
].sort_values(["sampleid", "filename"]).reset_index(drop=True)

if len(positive_annotated) != 2492 or len(zero_annotation) != 8 or len(positive_trainval) != 2242:
    raise ValueError(
        f"Unexpected positive counts: annotated={len(positive_annotated)}, "
        f"zero={len(zero_annotation)}, trainval={len(positive_trainval)}"
    )

rng = np.random.RandomState(SEED)
perm = rng.permutation(len(positive_trainval))
positive_val = positive_trainval.iloc[perm[:POS_VAL]].copy()
positive_train = positive_trainval.iloc[perm[POS_VAL:]].copy()

zip_by_sample = {}
for n in png_names:
    f = os.path.basename(n)
    try:
        sid = parse_sample_id(f)
    except Exception:
        continue
    zip_by_sample.setdefault(sid, []).append(f)

negative_ids = sorted(df.loc[df["has_solar"] == 0, "sampleid"].tolist())
extra_positive_ids = sorted(
    df.loc[(df["has_solar"] == 1) & (df["sampleid"] >= 2501), "sampleid"].tolist()
)

if len(negative_ids) != 469 or len(extra_positive_ids) != 31:
    raise ValueError("CSV label counts do not match expected 469 negatives / 31 extra positives.")

negative_rows = []
for sid in negative_ids:
    files = zip_by_sample.get(sid, [])
    if len(files) != 1:
        raise ValueError(f"No-solar sample {sid} maps to {len(files)} PNGs: {files}")
    negative_rows.append({"sampleid": sid, "filename": files[0], "has_solar": 0})

negative_df = pd.DataFrame(negative_rows).sort_values("sampleid").reset_index(drop=True)

if NEG_TRAIN + NEG_VAL + NEG_TEST != len(negative_df):
    raise ValueError("Negative split counts do not sum to 469.")

rng = np.random.RandomState(SEED)
perm = rng.permutation(len(negative_df))
negative_train = negative_df.iloc[perm[:NEG_TRAIN]].copy()
negative_val = negative_df.iloc[perm[NEG_TRAIN:NEG_TRAIN + NEG_VAL]].copy()
negative_test = negative_df.iloc[perm[NEG_TRAIN + NEG_VAL:]].copy()

extra_positive_rows = []
for sid in extra_positive_ids:
    files = zip_by_sample.get(sid, [])
    if len(files) != 1:
        raise ValueError(f"Extra positive sample {sid} maps to {len(files)} PNGs: {files}")
    extra_positive_rows.append({"sampleid": sid, "filename": files[0], "has_solar": 1})

extra_positive = pd.DataFrame(extra_positive_rows).sort_values("sampleid").reset_index(drop=True)


def tag(frame, split, dataset_type):
    out = frame.copy()
    out["split"] = split
    out["dataset_type"] = dataset_type
    return out


positive_train = tag(positive_train, "train", "positive_annotated")
positive_val = tag(positive_val, "val", "positive_annotated")
test_rows = tag(test_rows, "test", "positive_annotated")
negative_train = tag(negative_train, "train", "negative")
negative_val = tag(negative_val, "val", "negative")
negative_test = tag(negative_test, "test", "negative")
zero_annotation = tag(zero_annotation, "excluded", "zero_annotation")
extra_positive = tag(extra_positive, "excluded", "positive_unannotated")

positive_train.to_csv(os.path.join(OUTPUT_DIR, "positive_train.csv"), index=False)
positive_val.to_csv(os.path.join(OUTPUT_DIR, "positive_val.csv"), index=False)
test_rows.to_csv(os.path.join(OUTPUT_DIR, "positive_test_LOCKED.csv"), index=False)
negative_train.to_csv(os.path.join(OUTPUT_DIR, "negative_train.csv"), index=False)
negative_val.to_csv(os.path.join(OUTPUT_DIR, "negative_val.csv"), index=False)
negative_test.to_csv(os.path.join(OUTPUT_DIR, "negative_test_LOCKED.csv"), index=False)
zero_annotation.to_csv(os.path.join(OUTPUT_DIR, "excluded_zero_annotation.csv"), index=False)
extra_positive.to_csv(os.path.join(OUTPUT_DIR, "excluded_positive_unannotated.csv"), index=False)

manifest = pd.concat([
    positive_train, positive_val, test_rows,
    negative_train, negative_val, negative_test,
    zero_annotation, extra_positive
], ignore_index=True, sort=False)

manifest.to_csv(os.path.join(OUTPUT_DIR, "dataset_manifest.csv"), index=False)

train_names = set(positive_train["filename"]) | set(negative_train["filename"])
val_names = set(positive_val["filename"]) | set(negative_val["filename"])
test_names = set(test_rows["filename"]) | set(negative_test["filename"])

if train_names & val_names or train_names & test_names or val_names & test_names:
    raise AssertionError("Split overlap detected.")

config = {
    "seed": SEED,
    "positive": {"train": len(positive_train), "val": len(positive_val), "test": len(test_rows)},
    "negative": {"train": len(negative_train), "val": len(negative_val), "test": len(negative_test)},
    "excluded": {"zero_annotation": len(zero_annotation), "positive_unannotated": len(extra_positive)},
    "protected_positive_test_sha256": sha256_lines(test_filenames),
    "policy": "Train proposed model only on positive_train + negative_train. Tune on positive_val + negative_val. Evaluate after lock on positive_test + negative_test."
}

with open(os.path.join(OUTPUT_DIR, "split_config.json"), "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

print("=" * 70)
print("SOLARMAP SPLIT LOCK COMPLETE")
print("=" * 70)
print(f"Positive: train={len(positive_train)}, val={len(positive_val)}, test={len(test_rows)}")
print(f"Negative: train={len(negative_train)}, val={len(negative_val)}, test={len(negative_test)}")
print(f"Excluded: zero-annotation={len(zero_annotation)}, positive-unannotated={len(extra_positive)}")
print(f"All split overlap checks passed.")
print(f"Saved to: {OUTPUT_DIR}")
