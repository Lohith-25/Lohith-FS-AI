# ============================================================
# SOLARMAP — STEP 6
# HUMAN VERIFICATION OF UNMATCHED PREDICTIONS
# ============================================================
#
# Labels:
#   1 = TRUE_BACKGROUND
#   2 = DUPLICATE
#   3 = OVER_SEGMENTATION
#   4 = POSSIBLE_UNANNOTATED_PANEL
#   5 = AMBIGUOUS
#
# Controls inside the OpenCV window:
#   1-5 = label candidate
#   S   = skip candidate
#   Q   = quit and save
#   ESC = quit and save
#
# The script saves after every label.
# It can be stopped and resumed later.
#
# IMPORTANT:
# This script analyzes the 63 unmatched candidates from the
# non-test pilot. It does NOT train anything.
# ============================================================

import os
import io
import json
import zipfile
import warnings

import cv2
import torch
import numpy as np
import pandas as pd

from PIL import Image
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn


warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
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

DIAGNOSIS_CSV = os.path.join(
    BASE,
    "results",
    "verification",
    "fp_diagnosis.csv"
)

OUTPUT_CSV = os.path.join(
    BASE,
    "results",
    "verification",
    "fp_human_labels.csv"
)


# ============================================================
# 2. LABEL DEFINITIONS
# ============================================================

LABELS = {
    "1": "TRUE_BACKGROUND",
    "2": "DUPLICATE",
    "3": "OVER_SEGMENTATION",
    "4": "POSSIBLE_UNANNOTATED_PANEL",
    "5": "AMBIGUOUS"
}


# ============================================================
# 3. FILE CHECKS
# ============================================================

print("=" * 80)
print("SOLARMAP — STEP 6: HUMAN FP VERIFICATION")
print("=" * 80)

required_files = [
    ZIP_PATH,
    CHECKPOINT,
    DIAGNOSIS_CSV
]

for path in required_files:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )

print("\nRequired files found.")


# ============================================================
# 4. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nDevice:", device)

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
# 5. LOAD DIAGNOSIS CSV
# ============================================================

df = pd.read_csv(
    DIAGNOSIS_CSV
)

required_columns = [
    "filename",
    "prediction_index",
    "score",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "max_iou_with_GT",
    "max_iou_with_matched_prediction",
    "diagnostic_category"
]

missing_columns = [
    c
    for c in required_columns
    if c not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing columns in fp_diagnosis.csv:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# 6. RESUME SUPPORT
# ============================================================

if os.path.exists(OUTPUT_CSV):

    print(
        "\nExisting human-label file found."
    )

    labels_df = pd.read_csv(
        OUTPUT_CSV
    )

    # Make sure required columns exist
    if "human_label" not in labels_df.columns:

        labels_df["human_label"] = ""

    if "human_label_name" not in labels_df.columns:

        labels_df["human_label_name"] = ""

else:

    labels_df = df.copy()

    labels_df["human_label"] = ""

    labels_df["human_label_name"] = ""


# ============================================================
# 7. STABLE ROW ID
# ============================================================

if "verification_id" not in labels_df.columns:

    labels_df.insert(
        0,
        "verification_id",
        np.arange(
            len(labels_df)
        )
    )


# ============================================================
# 8. SORT BY CONFIDENCE
# ============================================================

labels_df = labels_df.sort_values(
    "score",
    ascending=False
).reset_index(
    drop=True
)


# ============================================================
# 9. LOAD COCO
# ============================================================

print("\nLoading COCO annotations...")

with zipfile.ZipFile(
    ZIP_PATH,
    "r"
) as zf:

    json_files = [
        name
        for name in zf.namelist()
        if name.endswith(
            "merged_instances_default.json"
        )
    ]

    if not json_files:

        raise RuntimeError(
            "Could not find merged_instances_default.json"
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
# 10. COCO LOOKUPS
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
# 11. POLYGON -> MASK
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
        ).reshape(
            -1,
            2
        )

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
        ).astype(
            np.int32
        )

        cv2.fillPoly(
            mask,
            [points],
            1
        )

    return mask


# ============================================================
# 12. GET GT MASKS
# ============================================================

def get_gt_masks(filename):

    if filename not in image_info:

        return []

    image_id = image_info[
        filename
    ]["id"]

    annotations = annotations_by_image.get(
        image_id,
        []
    )

    masks = []

    for ann in annotations:

        # Ignore invalid annotations
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
# 13. MASK IoU
# ============================================================

def mask_iou(
    mask_a,
    mask_b
):

    a = mask_a.astype(bool)
    b = mask_b.astype(bool)

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
# 14. LOAD MASK R-CNN
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

if isinstance(
    checkpoint,
    dict
):

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


# ------------------------------------------------------------
# Remove DataParallel prefix
# ------------------------------------------------------------

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
# 15. TRANSFORM
# ============================================================

to_tensor = transforms.ToTensor()


# ============================================================
# 16. ZIP LOOKUP
# ============================================================

print(
    "\nOpening dataset ZIP..."
)

zf = zipfile.ZipFile(
    ZIP_PATH,
    "r"
)

zip_names = set(
    zf.namelist()
)


def find_member(filename):

    matches = [
        name
        for name in zip_names
        if name.endswith(
            "/" + filename
        )
        or name == filename
    ]

    if not matches:

        return None

    return matches[0]


# ============================================================
# 17. IMAGE CACHE
# ============================================================
#
# There are only 20 pilot images, so cache them after loading.
# This avoids repeatedly reading the ZIP.
# ============================================================

image_cache = {}


def load_image(filename):

    if filename in image_cache:

        return image_cache[
            filename
        ]

    member = find_member(
        filename
    )

    if member is None:

        raise FileNotFoundError(
            f"Image not found in ZIP: {filename}"
        )

    with zf.open(member) as f:

        image = Image.open(f).convert(
            "RGB"
        )

    image_np = np.asarray(
        image
    ).copy()

    image_cache[
        filename
    ] = image_np

    return image_np


# ============================================================
# 18. INFERENCE CACHE
# ============================================================
#
# One model inference per image, not per candidate.
# ============================================================

prediction_cache = {}


def get_predictions(filename):

    if filename in prediction_cache:

        return prediction_cache[
            filename
        ]

    image_np = load_image(
        filename
    )

    image_pil = Image.fromarray(
        image_np
    )

    tensor = to_tensor(
        image_pil
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

    scores = scores[
        keep
    ]

    boxes = boxes[
        keep
    ]

    masks = masks[
        keep
    ]

    prediction_cache[
        filename
    ] = {
        "scores": scores,
        "boxes": boxes,
        "masks": masks
    }

    return prediction_cache[
        filename
    ]


# ============================================================
# 19. FIND CLOSEST CURRENT PREDICTION
# ============================================================
#
# The stored prediction index normally remains identical,
# but matching by bounding-box coordinates makes the script
# robust to small ordering differences.
# ============================================================

def find_candidate_index(
    row,
    boxes,
    scores
):

    target_box = np.array([
        float(row["bbox_x1"]),
        float(row["bbox_y1"]),
        float(row["bbox_x2"]),
        float(row["bbox_y2"])
    ])

    target_score = float(
        row["score"]
    )

    if len(boxes) == 0:

        return None

    # Normalized box-distance
    distances = []

    for i, box in enumerate(
        boxes
    ):

        box = np.asarray(
            box,
            dtype=np.float32
        )

        box_distance = np.mean(
            np.abs(
                box -
                target_box
            )
        )

        score_distance = abs(
            float(scores[i])
            -
            target_score
        )

        # Give box geometry more weight
        combined = (
            box_distance
            +
            50.0 *
            score_distance
        )

        distances.append(
            combined
        )

    return int(
        np.argmin(
            distances
        )
    )


# ============================================================
# 20. DRAW GT CONTOURS
# ============================================================

def draw_gt_contours(
    image,
    gt_masks
):

    result = image.copy()

    for mask in gt_masks:

        mask_uint8 = (
            mask.astype(
                np.uint8
            )
            *
            255
        )

        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            result,
            contours,
            -1,
            (0, 255, 0),
            2
        )

    return result


# ============================================================
# 21. DRAW ALL PREDICTION BOXES
# ============================================================

def draw_prediction_boxes(
    image,
    boxes,
    scores,
    candidate_index
):

    result = image.copy()

    for i, box in enumerate(
        boxes
    ):

        x1, y1, x2, y2 = [
            int(round(v))
            for v in box
        ]

        x1 = max(
            0,
            min(x1, image.shape[1] - 1)
        )

        y1 = max(
            0,
            min(y1, image.shape[0] - 1)
        )

        x2 = max(
            x1 + 1,
            min(x2, image.shape[1] - 1)
        )

        y2 = max(
            y1 + 1,
            min(y2, image.shape[0] - 1)
        )

        if i == candidate_index:

            # RED = candidate being reviewed
            color = (
                0,
                0,
                255
            )

            thickness = 4

        else:

            # BLUE = other model predictions
            color = (
                255,
                0,
                0
            )

            thickness = 1

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )

        # Candidate label
        if i == candidate_index:

            cv2.putText(
                result,
                f"CANDIDATE {i}",
                (
                    x1,
                    max(
                        20,
                        y1 - 7
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA
            )

    return result


# ============================================================
# 22. CREATE CONTEXT CROP
# ============================================================

def create_context_crop(
    image,
    box
):

    h, w = image.shape[:2]

    x1, y1, x2, y2 = [
        int(round(v))
        for v in box
    ]

    x1 = max(
        0,
        min(x1, w - 1)
    )

    y1 = max(
        0,
        min(y1, h - 1)
    )

    x2 = max(
        x1 + 1,
        min(x2, w)
    )

    y2 = max(
        y1 + 1,
        min(y2, h)
    )

    bw = max(
        x2 - x1,
        1
    )

    bh = max(
        y2 - y1,
        1
    )

    pad_x = int(
        bw * 2.5
    )

    pad_y = int(
        bh * 2.5
    )

    cx1 = max(
        0,
        x1 - pad_x
    )

    cy1 = max(
        0,
        y1 - pad_y
    )

    cx2 = min(
        w,
        x2 + pad_x
    )

    cy2 = min(
        h,
        y2 + pad_y
    )

    crop = image[
        cy1:cy2,
        cx1:cx2
    ].copy()

    # Draw candidate box on context crop
    rx1 = x1 - cx1
    ry1 = y1 - cy1
    rx2 = x2 - cx1
    ry2 = y2 - cy1

    cv2.rectangle(
        crop,
        (rx1, ry1),
        (rx2, ry2),
        (0, 0, 255),
        3
    )

    return crop


# ============================================================
# 23. RESIZE KEEPING ASPECT RATIO
# ============================================================

def fit_image(
    image,
    max_width,
    max_height
):

    h, w = image.shape[:2]

    scale = min(
        max_width / max(w, 1),
        max_height / max(h, 1)
    )

    new_w = max(
        1,
        int(round(w * scale))
    )

    new_h = max(
        1,
        int(round(h * scale))
    )

    return cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# 24. BUILD REVIEW WINDOW
# ============================================================

def build_review_image(
    row,
    position,
    total
):

    filename = row["filename"]

    image = load_image(
        filename
    )

    prediction = get_predictions(
        filename
    )

    scores = prediction[
        "scores"
    ]

    boxes = prediction[
        "boxes"
    ]

    masks = prediction[
        "masks"
    ]

    candidate_index = (
        find_candidate_index(
            row,
            boxes,
            scores
        )
    )

    if candidate_index is None:

        raise RuntimeError(
            f"Could not locate candidate for {filename}"
        )

    gt_masks = get_gt_masks(
        filename
    )

    # --------------------------------------------------------
    # Full image with GT and predictions
    # --------------------------------------------------------

    full_view = (
        draw_gt_contours(
            image,
            gt_masks
        )
    )

    full_view = (
        draw_prediction_boxes(
            full_view,
            boxes,
            scores,
            candidate_index
        )
    )

    # --------------------------------------------------------
    # Context crop
    # --------------------------------------------------------

    context_view = (
        create_context_crop(
            image,
            boxes[candidate_index]
        )
    )

    # --------------------------------------------------------
    # Add information banners
    # --------------------------------------------------------

    full_view = cv2.copyMakeBorder(
        full_view,
        90,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(30, 30, 30)
    )

    context_view = cv2.copyMakeBorder(
        context_view,
        90,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(30, 30, 30)
    )

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    candidate_score = float(
        scores[candidate_index]
    )

    gt_iou = float(
        row["max_iou_with_GT"]
    )

    matched_iou = float(
        row["max_iou_with_matched_prediction"]
    )

    diagnosis = str(
        row["diagnostic_category"]
    )

    # --------------------------------------------------------
    # Information text
    # --------------------------------------------------------

    header_lines = [
        f"Candidate {position}/{total}  |  {filename}",
        f"Confidence: {candidate_score:.4f}   "
        f"GT IoU: {gt_iou:.3f}   "
        f"Matched-pred IoU: {matched_iou:.3f}",
        f"Automatic diagnosis: {diagnosis}",
    ]

    for i, text in enumerate(
        header_lines
    ):

        cv2.putText(
            full_view,
            text,
            (
                10,
                25 + i * 24
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            context_view,
            text,
            (
                10,
                25 + i * 24
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    cv2.putText(
        full_view,
        "GREEN=GT  BLUE=other predictions  RED=candidate",
        (
            10,
            78
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Fit panels
    # --------------------------------------------------------

    panel_h = 620
    panel_w = 620

    full_view = fit_image(
        full_view,
        panel_w,
        panel_h
    )

    context_view = fit_image(
        context_view,
        panel_w,
        panel_h
    )

    # Make identical height
    target_h = max(
        full_view.shape[0],
        context_view.shape[0]
    )

    def pad_to_height(
        img,
        height
    ):

        if img.shape[0] == height:
            return img

        top = (
            height -
            img.shape[0]
        ) // 2

        bottom = (
            height -
            img.shape[0]
            -
            top
        )

        return cv2.copyMakeBorder(
            img,
            top,
            bottom,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(25, 25, 25)
        )

    full_view = pad_to_height(
        full_view,
        target_h
    )

    context_view = pad_to_height(
        context_view,
        target_h
    )

    combined = np.hstack(
        [
            full_view,
            context_view
        ]
    )

    # --------------------------------------------------------
    # Instructions at bottom
    # --------------------------------------------------------

    instruction_height = 80

    instruction = np.zeros(
        (
            instruction_height,
            combined.shape[1],
            3
        ),
        dtype=np.uint8
    )

    cv2.putText(
        instruction,
        "1=BACKGROUND  2=DUPLICATE  3=OVER-SEGMENTATION",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        instruction,
        "4=POSSIBLE UNANNOTATED PANEL  5=AMBIGUOUS  S=SKIP  Q/ESC=QUIT",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    combined = np.vstack(
        [
            combined,
            instruction
        ]
    )

    return combined


# ============================================================
# 25. PRINT LABEL DEFINITIONS
# ============================================================

print("\n" + "=" * 80)
print("LABEL DEFINITIONS")
print("=" * 80)

print(
    "1 = TRUE_BACKGROUND"
)
print(
    "    Clearly non-solar object/structure."
)

print(
    "2 = DUPLICATE"
)
print(
    "    Another prediction already represents the same physical panel."
)

print(
    "3 = OVER_SEGMENTATION"
)
print(
    "    One physical panel/structure is split into multiple predicted parts."
)

print(
    "4 = POSSIBLE_UNANNOTATED_PANEL"
)
print(
    "    Looks genuinely solar, but there is no corresponding GT annotation."
)

print(
    "5 = AMBIGUOUS"
)
print(
    "    Cannot confidently determine the correct category."
)

print(
    "\nUse the RED box as the candidate you are judging."
)

print(
    "GREEN outlines = ground-truth annotations."
)

print(
    "BLUE boxes = other predictions."
)


# ============================================================
# 26. RESUME INFORMATION
# ============================================================

completed_mask = (
    labels_df["human_label"]
    .isin(
        LABELS.values()
    )
)

completed_count = int(
    completed_mask.sum()
)

print(
    "\nAlready labeled:",
    completed_count,
    "/",
    len(labels_df)
)

print(
    "Remaining:",
    len(labels_df) -
    completed_count
)


# ============================================================
# 27. HUMAN REVIEW LOOP
# ============================================================

WINDOW_NAME = (
    "SolarMap — Human FP Verification"
)

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    1400,
    850
)

try:

    total = len(labels_df)

    for idx in range(total):

        existing = str(
            labels_df.at[
                idx,
                "human_label"
            ]
        ).strip()

        # Resume support
        if existing in LABELS.values():

            continue

        row = labels_df.loc[
            idx
        ]

        position = idx + 1

        print("\n" + "-" * 80)

        print(
            f"Candidate {position}/{total}"
        )

        print(
            f"Filename: {row['filename']}"
        )

        print(
            f"Confidence: {float(row['score']):.4f}"
        )

        print(
            f"Automatic diagnosis: "
            f"{row['diagnostic_category']}"
        )

        try:

            review_image = (
                build_review_image(
                    row,
                    position,
                    total
                )
            )

        except Exception as e:

            print(
                "Could not build review image:"
            )

            print(
                str(e)
            )

            print(
                "Press S to skip this candidate."
            )

            continue

        cv2.imshow(
            WINDOW_NAME,
            review_image
        )

        print(
            "Look at the OpenCV window and press 1-5, S, or Q."
        )

        # ----------------------------------------------------
        # Wait for key
        # ----------------------------------------------------

        while True:

            key = cv2.waitKey(
                0
            ) & 0xFF

            # ------------------------------------------------
            # Quit
            # ------------------------------------------------

            if key in [
                ord("q"),
                ord("Q"),
                27
            ]:

                labels_df.to_csv(
                    OUTPUT_CSV,
                    index=False
                )

                print(
                    "\nProgress saved."
                )

                print(
                    "Saved:",
                    OUTPUT_CSV
                )

                raise SystemExit(0)

            # ------------------------------------------------
            # Skip
            # ------------------------------------------------

            if key in [
                ord("s"),
                ord("S")
            ]:

                print(
                    "Skipped."
                )

                break

            # ------------------------------------------------
            # Valid labels
            # ------------------------------------------------

            key_char = chr(key)

            if key_char in LABELS:

                label_name = (
                    LABELS[key_char]
                )

                labels_df.at[
                    idx,
                    "human_label"
                ] = label_name

                labels_df.at[
                    idx,
                    "human_label_name"
                ] = label_name

                # Save immediately
                labels_df.to_csv(
                    OUTPUT_CSV,
                    index=False
                )

                print(
                    "Saved label:",
                    label_name
                )

                break

    # ========================================================
    # FINAL SAVE
    # ========================================================

    labels_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

finally:

    cv2.destroyAllWindows()

    zf.close()


# ============================================================
# 28. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("HUMAN VERIFICATION COMPLETE")
print("=" * 80)

completed = (
    labels_df["human_label"]
    .isin(
        LABELS.values()
    )
)

print(
    "Completed:",
    int(completed.sum()),
    "/",
    len(labels_df)
)

print("\nLabel counts:")

if completed.any():

    print(
        labels_df.loc[
            completed,
            "human_label"
        ].value_counts()
    )

else:

    print(
        "No labels completed."
    )

print(
    "\nSaved:",
    OUTPUT_CSV
)

print(
    "\n✅ STEP 6 COMPLETE"
)