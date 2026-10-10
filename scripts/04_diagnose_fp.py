import os
import zipfile
import json

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
    BASE, "dataset", "Solar Images.zip"
)

CHECKPOINT = os.path.join(
    BASE,
    "checkpoints",
    "maskrcnn_baseline_epoch_10.pth"
)

PILOT_CSV = os.path.join(
    BASE,
    "results",
    "verification",
    "pilot_candidates.csv"
)

OUTPUT_CSV = os.path.join(
    BASE,
    "results",
    "verification",
    "fp_diagnosis.csv"
)


# ============================================================
# SETTINGS
# ============================================================

IOU_MATCH_THRESHOLD = 0.50

# Diagnostic thresholds only.
# These are NOT final research labels.
DUPLICATE_IOU = 0.30
BACKGROUND_IOU = 0.10


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("SOLARMAP STEP 5 — FP DIAGNOSIS")
print("=" * 70)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD PILOT CSV
# ============================================================

pilot_df = pd.read_csv(PILOT_CSV)

pilot_images = sorted(
    pilot_df["filename"].unique()
)

print(
    "Pilot images:",
    len(pilot_images)
)


# ============================================================
# LOAD COCO
# ============================================================

with zipfile.ZipFile(
    ZIP_PATH,
    "r"
) as zf:

    json_files = [
        n for n in zf.namelist()
        if n.endswith(
            "merged_instances_default.json"
        )
    ]

    with zf.open(json_files[0]) as f:
        coco = json.load(f)


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

        pts = np.asarray(
            polygon,
            dtype=np.float32
        ).reshape(-1, 2)

        pts[:, 0] = np.clip(
            pts[:, 0],
            0,
            width - 1
        )

        pts[:, 1] = np.clip(
            pts[:, 1],
            0,
            height - 1
        )

        pts = np.round(
            pts
        ).astype(np.int32)

        cv2.fillPoly(
            mask,
            [pts],
            1
        )

    return mask


def get_gt_masks(filename):

    info = image_info[filename]

    anns = annotations_by_image.get(
        info["id"],
        []
    )

    masks = []

    for ann in anns:

        if float(
            ann.get("area", 0)
        ) <= 0:
            continue

        mask = annotation_to_mask(
            ann
        )

        if mask.sum() > 0:
            masks.append(mask)

    return masks


# ============================================================
# MASK IoU
# ============================================================

def mask_iou(a, b):

    a = a.astype(bool)
    b = b.astype(bool)

    intersection = np.logical_and(
        a, b
    ).sum()

    union = np.logical_or(
        a, b
    ).sum()

    if union == 0:
        return 0.0

    return float(
        intersection / union
    )


# ============================================================
# MATCH
# ============================================================

def get_matches(
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

    for i, pm in enumerate(pred_masks):

        for j, gm in enumerate(gt_masks):

            matrix[i, j] = mask_iou(
                pm,
                gm
            )

    rows, cols = linear_sum_assignment(
        -matrix
    )

    matched = set()

    for r, c in zip(rows, cols):

        if matrix[r, c] >= IOU_MATCH_THRESHOLD:
            matched.add(r)

    return matched


# ============================================================
# LOAD MASK R-CNN
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

if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
elif (
    "model" in checkpoint
    and isinstance(checkpoint["model"], dict)
):
    state_dict = checkpoint["model"]
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

print("Checkpoint loaded.")


# ============================================================
# OPEN ZIP
# ============================================================

zf = zipfile.ZipFile(
    ZIP_PATH,
    "r"
)

zip_names = set(
    zf.namelist()
)


def find_member(filename):

    matches = [
        n for n in zip_names
        if n.endswith(
            "/" + filename
        )
        or n == filename
    ]

    return matches[0] if matches else None


to_tensor = transforms.ToTensor()


# ============================================================
# DIAGNOSE EACH PILOT IMAGE
# ============================================================

results = []

for image_number, filename in enumerate(
    pilot_images,
    start=1
):

    print(
        f"[{image_number}/{len(pilot_images)}] "
        f"{filename}"
    )

    member = find_member(
        filename
    )

    if member is None:
        continue

    with zf.open(member) as f:
        image = Image.open(f).convert(
            "RGB"
        )

    tensor = to_tensor(
        image
    ).to(device)

    with torch.inference_mode():

        output = model(
            [tensor]
        )[0]

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
        >= 0.50
    )

    keep = scores >= 0.50

    scores = scores[keep]
    boxes = boxes[keep]
    masks = masks[keep]

    gt_masks = get_gt_masks(
        filename
    )

    matched = get_matches(
        masks,
        gt_masks
    )

    fp_indices = [
        i
        for i in range(len(masks))
        if i not in matched
    ]

    # --------------------------------------------------------
    # For each unmatched prediction:
    #
    # 1. Max IoU with any GT
    # 2. Max IoU with a matched prediction
    # --------------------------------------------------------

    for fp_idx in fp_indices:

        fp_mask = masks[fp_idx]

        max_gt_iou = 0.0

        for gt_mask in gt_masks:

            max_gt_iou = max(
                max_gt_iou,
                mask_iou(
                    fp_mask,
                    gt_mask
                )
            )

        max_matched_pred_iou = 0.0

        for matched_idx in matched:

            max_matched_pred_iou = max(
                max_matched_pred_iou,
                mask_iou(
                    fp_mask,
                    masks[matched_idx]
                )
            )

        # ----------------------------------------------------
        # Diagnostic category
        # ----------------------------------------------------

        if (
            max_matched_pred_iou
            >= DUPLICATE_IOU
        ):

            category = (
                "DUPLICATE_LIKE"
            )

        elif (
            max_gt_iou
            < BACKGROUND_IOU
        ):

            category = (
                "BACKGROUND_LIKE"
            )

        else:

            category = (
                "AMBIGUOUS"
            )

        x1, y1, x2, y2 = boxes[fp_idx]

        results.append({

            "filename":
                filename,

            "prediction_index":
                fp_idx,

            "score":
                float(scores[fp_idx]),

            "bbox_x1":
                float(x1),

            "bbox_y1":
                float(y1),

            "bbox_x2":
                float(x2),

            "bbox_y2":
                float(y2),

            "max_iou_with_GT":
                max_gt_iou,

            "max_iou_with_matched_prediction":
                max_matched_pred_iou,

            "diagnostic_category":
                category
        })


zf.close()


# ============================================================
# SAVE
# ============================================================

diagnosis_df = pd.DataFrame(
    results
)

diagnosis_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FP DIAGNOSIS COMPLETE")
print("=" * 70)

print(
    "Unmatched candidates analyzed:",
    len(diagnosis_df)
)

print("\nDiagnostic categories:")

print(
    diagnosis_df[
        "diagnostic_category"
    ].value_counts()
)

print(
    "\nAverage IoU values by category:"
)

display(
    diagnosis_df.groupby(
        "diagnostic_category"
    )[
        [
            "score",
            "max_iou_with_GT",
            "max_iou_with_matched_prediction"
        ]
    ].mean()
)

print(
    "\nSaved:",
    OUTPUT_CSV
)