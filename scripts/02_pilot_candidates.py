import os
import io
import json
import zipfile
import random

import cv2
import torch
import numpy as np
import pandas as pd

from PIL import Image
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from scipy.optimize import linear_sum_assignment


# ============================================================
# PATHS
# ============================================================

BASE = r"C:\SolarMap-India"

ZIP_PATH = os.path.join(
    BASE,
    "dataset",
    "Solar Images.zip"
)

CHECKPOINT = os.path.join(
    BASE,
    "checkpoints",
    "maskrcnn_baseline_epoch_10.pth"
)

RESULTS_DIR = os.path.join(
    BASE,
    "results",
    "verification"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

OUTPUT_CSV = os.path.join(
    RESULTS_DIR,
    "pilot_candidates.csv"
)


# ============================================================
# SETTINGS
# ============================================================

PILOT_COUNT = 20

SCORE_THRESHOLD = 0.50
MASK_THRESHOLD = 0.50
IOU_THRESHOLD = 0.50

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("SOLARMAP STEP 4 — GPU PILOT")
print("=" * 70)

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Device:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0)
            .total_memory / 1024**3,
            2
        ),
        "GB"
    )


# ============================================================
# FILE CHECKS
# ============================================================

if not os.path.exists(ZIP_PATH):
    raise FileNotFoundError(ZIP_PATH)

if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(CHECKPOINT)

print("\nDataset: FOUND")
print("Checkpoint: FOUND")


# ============================================================
# LOAD POSITIVE TEST FILENAMES
# ============================================================

# We will use the baseline metrics already copied to Drive
# only if available locally. Otherwise we use the first 20
# annotated COCO images and explicitly exclude all test
# filenames known from the saved baseline metrics.

POSITIVE_METRICS_DRIVE = (
    r"C:\SolarMap-India\results\baseline"
    r"\baseline_per_image_metrics.csv"
)

test_filenames = set()

if os.path.exists(POSITIVE_METRICS_DRIVE):

    test_df = pd.read_csv(
        POSITIVE_METRICS_DRIVE
    )

    test_filenames = set(
        test_df["filename"]
        .astype(str)
        .tolist()
    )

    print(
        "Protected test images:",
        len(test_filenames)
    )

else:

    print(
        "\nWARNING: baseline_per_image_metrics.csv "
        "was not found locally."
    )

    print(
        "We will NOT assume arbitrary images are test."
    )


# ============================================================
# LOAD COCO
# ============================================================

print("\nReading COCO annotations...")

with zipfile.ZipFile(
    ZIP_PATH,
    "r"
) as zf:

    json_files = [
        n
        for n in zf.namelist()
        if n.endswith(
            "merged_instances_default.json"
        )
    ]

    if not json_files:
        raise RuntimeError(
            "COCO annotation JSON not found."
        )

    with zf.open(
        json_files[0]
    ) as f:

        coco = json.load(f)


print(
    "COCO images:",
    len(coco["images"])
)

print(
    "COCO annotations:",
    len(coco["annotations"])
)


# ============================================================
# LOOKUPS
# ============================================================

image_info = {
    img["file_name"]: img
    for img in coco["images"]
}

annotations_by_image = {}

for ann in coco["annotations"]:

    annotations_by_image.setdefault(
        ann["image_id"],
        []
    ).append(ann)


# ============================================================
# SELECT PILOT IMAGES
# ============================================================

eligible_images = []

for filename, info in image_info.items():

    # Only images that are not in protected test set
    if filename in test_filenames:
        continue

    annotations = annotations_by_image.get(
        info["id"],
        []
    )

    valid_annotations = [
        ann
        for ann in annotations
        if float(
            ann.get("area", 0)
        ) > 0
    ]

    if len(valid_annotations) > 0:

        eligible_images.append(
            filename
        )

if len(eligible_images) < PILOT_COUNT:

    raise RuntimeError(
        "Not enough eligible non-test images."
    )

pilot_images = random.sample(
    eligible_images,
    PILOT_COUNT
)

print(
    "\nPilot images selected:",
    len(pilot_images)
)

print(
    "Test overlap:",
    len(
        set(pilot_images)
        &
        test_filenames
    )
)

assert (
    len(
        set(pilot_images)
        &
        test_filenames
    )
    == 0
)


# ============================================================
# POLYGON -> MASK
# ============================================================

def annotation_to_mask(
    annotation,
    height=640,
    width=640
):

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    segmentation = annotation.get(
        "segmentation",
        []
    )

    if not isinstance(
        segmentation,
        list
    ):
        return mask

    for polygon in segmentation:

        if len(polygon) < 6:
            continue

        points = np.asarray(
            polygon,
            dtype=np.float32
        ).reshape(-1, 2)

        points[:, 0] = np.clip(
            points[:, 0],
            0,
            width - 1
        )

        points[:, 1] = np.clip(
            points[:, 1],
            0,
            height - 1
        )

        points = np.round(
            points
        ).astype(np.int32)

        cv2.fillPoly(
            mask,
            [points],
            1
        )

    return mask


# ============================================================
# GT MASKS
# ============================================================

def get_gt_masks(filename):

    info = image_info.get(
        filename
    )

    if info is None:
        return []

    annotations = annotations_by_image.get(
        info["id"],
        []
    )

    masks = []

    for ann in annotations:

        if float(
            ann.get("area", 0)
        ) <= 0:
            continue

        mask = annotation_to_mask(
            ann
        )

        if mask.sum() > 0:
            masks.append(
                mask
            )

    return masks


# ============================================================
# IoU
# ============================================================

def mask_iou(a, b):

    a = a.astype(bool)
    b = b.astype(bool)

    intersection = np.logical_and(
        a,
        b
    ).sum()

    union = np.logical_or(
        a,
        b
    ).sum()

    if union == 0:
        return 0.0

    return float(
        intersection / union
    )


# ============================================================
# MATCH
# ============================================================

def match_predictions(
    pred_masks,
    gt_masks
):

    if len(pred_masks) == 0:
        return set()

    if len(gt_masks) == 0:
        return set()

    matrix = np.zeros(
        (
            len(pred_masks),
            len(gt_masks)
        ),
        dtype=np.float32
    )

    for i, pm in enumerate(
        pred_masks
    ):

        for j, gm in enumerate(
            gt_masks
        ):

            matrix[i, j] = mask_iou(
                pm,
                gm
            )

    rows, cols = linear_sum_assignment(
        -matrix
    )

    matched = set()

    for r, c in zip(rows, cols):

        if matrix[r, c] >= IOU_THRESHOLD:

            matched.add(r)

    return matched


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Mask R-CNN...")

model = maskrcnn_resnet50_fpn(
    weights=None,
    weights_backbone=None,
    num_classes=2
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False
)

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    elif (
        "model" in checkpoint
        and isinstance(
            checkpoint["model"],
            dict
        )
    ):

        state_dict = checkpoint[
            "model"
        ]

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):
        key = key[7:]

    clean_state_dict[key] = value


model.load_state_dict(
    clean_state_dict,
    strict=True
)

model.to(device)
model.eval()

print(
    "Checkpoint loaded successfully."
)


# ============================================================
# TRANSFORM
# ============================================================

to_tensor = transforms.ToTensor()


# ============================================================
# FIND ZIP MEMBER
# ============================================================

with zipfile.ZipFile(
    ZIP_PATH,
    "r"
) as zf:

    zip_names = set(
        zf.namelist()
    )

    def find_member(filename):

        matches = [
            n
            for n in zip_names
            if n.endswith(
                "/" + filename
            )
            or n == filename
        ]

        return (
            matches[0]
            if matches
            else None
        )

    # ========================================================
    # PROCESS
    # ========================================================

    rows = []

    candidate_id = 0

    for idx, filename in enumerate(
        pilot_images,
        start=1
    ):

        print(
            f"\n[{idx}/{PILOT_COUNT}] "
            f"{filename}"
        )

        member = find_member(
            filename
        )

        if member is None:

            print(
                "WARNING: image not found."
            )

            continue

        with zf.open(member) as f:

            image = Image.open(f).convert(
                "RGB"
            )

        tensor = to_tensor(
            image
        ).to(device)

        # GPU timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with torch.inference_mode():

            output = model(
                [tensor]
            )[0]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        scores = (
            output["scores"]
            .detach()
            .cpu()
            .numpy()
        )

        boxes = (
            output["boxes"]
            .detach()
            .cpu()
            .numpy()
        )

        masks = (
            output["masks"]
            .detach()
            .cpu()
            .numpy()[:, 0]
            >= MASK_THRESHOLD
        )

        keep = (
            scores >= SCORE_THRESHOLD
        )

        scores = scores[keep]
        boxes = boxes[keep]
        masks = masks[keep]

        print(
            "Candidates:",
            len(scores)
        )

        gt_masks = get_gt_masks(
            filename
        )

        matched = match_predictions(
            masks,
            gt_masks
        )

        for i in range(
            len(scores)
        ):

            candidate_id += 1

            label = (
                "TP"
                if i in matched
                else "FP"
            )

            x1, y1, x2, y2 = (
                boxes[i]
            )

            mask_area = float(
                masks[i].sum()
            )

            bbox_width = float(
                max(
                    x2 - x1,
                    1e-6
                )
            )

            bbox_height = float(
                max(
                    y2 - y1,
                    1e-6
                )
            )

            bbox_area = (
                bbox_width *
                bbox_height
            )

            aspect_ratio = (
                max(
                    bbox_width,
                    bbox_height
                )
                /
                min(
                    bbox_width,
                    bbox_height
                )
            )

            rectangularity = (
                mask_area /
                bbox_area
            )

            rows.append({

    "candidate_id":
        candidate_id,

    "filename":
        filename,

    "label":
        label,

    "score":
        float(scores[i]),

    # Bounding-box coordinates
    "bbox_x1":
        float(x1),

    "bbox_y1":
        float(y1),

    "bbox_x2":
        float(x2),

    "bbox_y2":
        float(y2),

    "mask_area_px":
        mask_area,

    "bbox_width":
        bbox_width,

    "bbox_height":
        bbox_height,

    "bbox_area_px":
        bbox_area,

    "aspect_ratio":
        aspect_ratio,

    "rectangularity":
        rectangularity
})


# ============================================================
# SAVE
# ============================================================

pilot_df = pd.DataFrame(
    rows
)

pilot_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 — PILOT COMPLETE")
print("=" * 70)

print(
    "Images processed:",
    len(pilot_images)
)

print(
    "Total candidates:",
    len(pilot_df)
)

if len(pilot_df) > 0:

    print("\nLabels:")

    print(
        pilot_df[
            "label"
        ].value_counts()
    )

    print("\nPreview:")

    print(
        pilot_df.head(10)
    )

print(
    "\nSaved:",
    OUTPUT_CSV
)

print("\n✅ STEP 4 COMPLETE")