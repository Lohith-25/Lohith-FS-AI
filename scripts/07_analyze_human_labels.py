# ============================================================
# SOLARMAP — STEP 7
# HUMAN LABEL + FEATURE ANALYSIS
# ============================================================
#
# This version DOES NOT require:
#   candidate_verification_features.csv
#
# It calculates the required features directly for the
# 63 human-reviewed candidates using the local checkpoint.
#
# ============================================================

import os
import io
import json
import zipfile
import warnings

import cv2
import numpy as np
import pandas as pd
import torch

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

HUMAN_LABELS = os.path.join(
    BASE,
    "results",
    "verification",
    "fp_human_labels.csv"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "results",
    "verification",
    "human_label_analysis"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. CHECK FILES
# ============================================================

print("=" * 80)
print("SOLARMAP — STEP 7: HUMAN LABEL FEATURE ANALYSIS")
print("=" * 80)

required_files = [
    ZIP_PATH,
    CHECKPOINT,
    HUMAN_LABELS
]

for path in required_files:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )

print("\nAll required local files found.")


# ============================================================
# 3. LOAD HUMAN LABELS
# ============================================================

human = pd.read_csv(
    HUMAN_LABELS
)

VALID_LABELS = [
    "TRUE_BACKGROUND",
    "DUPLICATE",
    "OVER_SEGMENTATION",
    "POSSIBLE_UNANNOTATED_PANEL",
    "AMBIGUOUS"
]

human = human[
    human["human_label"].isin(
        VALID_LABELS
    )
].copy()

print(
    "\nHuman-labeled candidates:",
    len(human)
)


# ============================================================
# 4. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    device
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0)
            .total_memory
            / 1024**3,
            2
        ),
        "GB"
    )


# ============================================================
# 5. LOAD MASK R-CNN
# ============================================================

print(
    "\nLoading Mask R-CNN..."
)

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
# 6. OPEN ZIP
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
# 7. LOAD IMAGE
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
            f"Image not found in ZIP:\n{filename}"
        )

    with zf.open(member) as f:

        image = Image.open(
            f
        ).convert(
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
# 8. IMAGE TRANSFORM
# ============================================================

to_tensor = transforms.ToTensor()


# ============================================================
# 9. FEATURE FUNCTIONS
# ============================================================

def calculate_features(
    mask,
    box,
    image_height,
    image_width,
    all_boxes
):

    mask_uint8 = (
        mask.astype(
            np.uint8
        )
        *
        255
    )

    mask_area = float(
        np.sum(mask)
    )

    x1, y1, x2, y2 = box

    bbox_width = max(
        1.0,
        x2 - x1
    )

    bbox_height = max(
        1.0,
        y2 - y1
    )

    bbox_area = (
        bbox_width
        *
        bbox_height
    )

    aspect_ratio = (
        bbox_width
        /
        bbox_height
    )

    # --------------------------------------------------------
    # Contours
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 0:

        largest = max(
            contours,
            key=cv2.contourArea
        )

        contour_area = float(
            cv2.contourArea(
                largest
            )
        )

        perimeter = float(
            cv2.arcLength(
                largest,
                True
            )
        )

        hull = cv2.convexHull(
            largest
        )

        hull_area = float(
            cv2.contourArea(
                hull
            )
        )

    else:

        contour_area = 0.0
        perimeter = 0.0
        hull_area = 0.0

    # --------------------------------------------------------
    # Rectangularity
    # --------------------------------------------------------

    if bbox_area > 0:

        rectangularity = (
            mask_area
            /
            bbox_area
        )

    else:

        rectangularity = 0.0

    # --------------------------------------------------------
    # Solidity
    # --------------------------------------------------------

    if hull_area > 0:

        solidity = (
            contour_area
            /
            hull_area
        )

    else:

        solidity = 0.0

    # --------------------------------------------------------
    # Compactness
    # --------------------------------------------------------

    if perimeter > 0:

        compactness = (
            4
            *
            np.pi
            *
            contour_area
            /
            (
                perimeter
                *
                perimeter
            )
        )

    else:

        compactness = 0.0

    # --------------------------------------------------------
    # Relative area
    # --------------------------------------------------------

    image_area = (
        image_height
        *
        image_width
    )

    relative_mask_area = (
        mask_area
        /
        image_area
    )

    # --------------------------------------------------------
    # Center
    # --------------------------------------------------------

    center_x = (
        x1 + x2
    ) / 2.0

    center_y = (
        y1 + y2
    ) / 2.0

    center_x_norm = (
        center_x
        /
        image_width
    )

    center_y_norm = (
        center_y
        /
        image_height
    )

    # --------------------------------------------------------
    # Edge distance
    # --------------------------------------------------------

    edge_distance_norm = min(
        center_x_norm,
        1.0 - center_x_norm,
        center_y_norm,
        1.0 - center_y_norm
    )

    # --------------------------------------------------------
    # Neighbors within 100 px
    # --------------------------------------------------------

    neighbor_count = 0

    for other_box in all_boxes:

        ox1, oy1, ox2, oy2 = other_box

        other_center_x = (
            ox1 + ox2
        ) / 2.0

        other_center_y = (
            oy1 + oy2
        ) / 2.0

        distance = np.sqrt(
            (
                center_x
                -
                other_center_x
            ) ** 2
            +
            (
                center_y
                -
                other_center_y
            ) ** 2
        )

        if (
            distance <= 100
            and
            distance > 0
        ):

            neighbor_count += 1

    return {
        "mask_area_px": mask_area,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_area_px": bbox_area,
        "aspect_ratio": aspect_ratio,
        "rectangularity": rectangularity,
        "solidity": solidity,
        "compactness": compactness,
        "relative_mask_area": relative_mask_area,
        "center_x_norm": center_x_norm,
        "center_y_norm": center_y_norm,
        "edge_distance_norm": edge_distance_norm,
        "neighbor_count_100px": neighbor_count
    }


# ============================================================
# 10. UNIQUE IMAGES
# ============================================================

unique_filenames = (
    human["filename"]
    .dropna()
    .unique()
)

print(
    "\nUnique images to process:",
    len(unique_filenames)
)


# ============================================================
# 11. RUN MODEL ON UNIQUE IMAGES
# ============================================================

prediction_cache = {}

for image_number, filename in enumerate(
    unique_filenames,
    start=1
):

    print(
        f"\nProcessing image "
        f"{image_number}/{len(unique_filenames)}: "
        f"{filename}"
    )

    image_np = load_image(
        filename
    )

    image_tensor = (
        to_tensor(
            Image.fromarray(
                image_np
            )
        )
        .to(device)
    )

    with torch.inference_mode():

        output = model(
            [image_tensor]
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

    print(
        "Predictions >= 0.50:",
        len(scores)
    )


# ============================================================
# 12. MATCH HUMAN CANDIDATE TO MODEL PREDICTION
# ============================================================

def find_prediction(
    row,
    predictions
):

    scores = predictions[
        "scores"
    ]

    boxes = predictions[
        "boxes"
    ]

    if len(scores) == 0:

        return None

    target_score = float(
        row["score"]
    )

    target_box = np.array(
        [
            float(row["bbox_x1"]),
            float(row["bbox_y1"]),
            float(row["bbox_x2"]),
            float(row["bbox_y2"])
        ],
        dtype=np.float32
    )

    best_index = None
    best_distance = float(
        "inf"
    )

    for i in range(
        len(scores)
    ):

        current_box = (
            boxes[i]
        )

        box_distance = np.mean(
            np.abs(
                current_box
                -
                target_box
            )
        )

        score_distance = abs(
            float(scores[i])
            -
            target_score
        )

        distance = (
            box_distance
            +
            50.0
            *
            score_distance
        )

        if distance < best_distance:

            best_distance = distance
            best_index = i

    return best_index


# ============================================================
# 13. BUILD 63-CANDIDATE FEATURE TABLE
# ============================================================

records = []

print(
    "\n" + "=" * 80
)

print(
    "CALCULATING FEATURES FOR HUMAN-LABELED CANDIDATES"
)

print(
    "=" * 80
)

for row_number, row in human.iterrows():

    filename = row[
        "filename"
    ]

    predictions = prediction_cache[
        filename
    ]

    prediction_index = find_prediction(
        row,
        predictions
    )

    if prediction_index is None:

        print(
            "WARNING: could not match:",
            filename,
            row["score"]
        )

        continue

    boxes = predictions[
        "boxes"
    ]

    masks = predictions[
        "masks"
    ]

    scores = predictions[
        "scores"
    ]

    image_np = load_image(
        filename
    )

    height, width = (
        image_np.shape[:2]
    )

    candidate_box = boxes[
        prediction_index
    ]

    candidate_mask = masks[
        prediction_index
    ]

    feature_values = calculate_features(
        candidate_mask,
        candidate_box,
        height,
        width,
        boxes
    )

    record = {

        "human_row": row_number,

        "filename": filename,

        "human_label": row[
            "human_label"
        ],

        "score": float(
            scores[
                prediction_index
            ]
        ),

        "max_iou_with_GT": float(
            row[
                "max_iou_with_GT"
            ]
        ),

        "max_iou_with_matched_prediction": float(
            row[
                "max_iou_with_matched_prediction"
            ]
        ),

        "diagnostic_category": row[
            "diagnostic_category"
        ],

        "prediction_index": int(
            prediction_index
        )
    }

    record.update(
        feature_values
    )

    records.append(
        record
    )


analysis_df = pd.DataFrame(
    records
)


# ============================================================
# 14. SAVE RAW FEATURE TABLE
# ============================================================

raw_output = os.path.join(
    OUTPUT_DIR,
    "human_labeled_candidates_with_features.csv"
)

analysis_df.to_csv(
    raw_output,
    index=False
)

print(
    "\nSaved:",
    raw_output
)

print(
    "Matched candidates:",
    len(analysis_df),
    "/",
    len(human)
)


# ============================================================
# 15. LABEL COUNTS
# ============================================================

label_counts = (
    analysis_df[
        "human_label"
    ]
    .value_counts()
    .rename_axis(
        "human_label"
    )
    .reset_index(
        name="count"
    )
)

label_counts[
    "percentage"
] = (
    label_counts[
        "count"
    ]
    /
    len(analysis_df)
    *
    100
)

print(
    "\n" + "=" * 80
)

print(
    "HUMAN LABEL DISTRIBUTION"
)

print(
    "=" * 80
)

print(
    label_counts.to_string(
        index=False
    )
)

label_counts.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "human_label_counts.csv"
    ),
    index=False
)


# ============================================================
# 16. FEATURE SUMMARY
# ============================================================

feature_columns = [
    "score",
    "mask_area_px",
    "bbox_area_px",
    "bbox_width",
    "bbox_height",
    "aspect_ratio",
    "rectangularity",
    "solidity",
    "compactness",
    "relative_mask_area",
    "center_x_norm",
    "center_y_norm",
    "edge_distance_norm",
    "neighbor_count_100px"
]

mean_summary = (
    analysis_df
    .groupby(
        "human_label"
    )[
        feature_columns
    ]
    .mean()
    .round(4)
)

median_summary = (
    analysis_df
    .groupby(
        "human_label"
    )[
        feature_columns
    ]
    .median()
    .round(4)
)

print(
    "\n" + "=" * 80
)

print(
    "MEAN FEATURES BY HUMAN LABEL"
)

print(
    "=" * 80
)

print(
    mean_summary.to_string()
)

print(
    "\n" + "=" * 80
)

print(
    "MEDIAN FEATURES BY HUMAN LABEL"
)

print(
    "=" * 80
)

print(
    median_summary.to_string()
)

mean_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "mean_features_by_human_label.csv"
    )
)

median_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "median_features_by_human_label.csv"
    )
)


# ============================================================
# 17. CONFIDENCE SUMMARY
# ============================================================

confidence_summary = (
    analysis_df
    .groupby(
        "human_label"
    )[
        "score"
    ]
    .agg(
        [
            "count",
            "min",
            "max",
            "mean",
            "median",
            "std"
        ]
    )
    .round(4)
)

print(
    "\n" + "=" * 80
)

print(
    "CONFIDENCE BY HUMAN LABEL"
)

print(
    "=" * 80
)

print(
    confidence_summary.to_string()
)

confidence_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "confidence_by_human_label.csv"
    )
)


# ============================================================
# 18. AUC ANALYSIS
# ============================================================

try:

    from sklearn.metrics import roc_auc_score

    auc_records = []

    targets = [
        "TRUE_BACKGROUND",
        "DUPLICATE",
        "OVER_SEGMENTATION",
        "POSSIBLE_UNANNOTATED_PANEL"
    ]

    for target in targets:

        subset = analysis_df[
            analysis_df[
                "human_label"
            ]
            !=
            "AMBIGUOUS"
        ].copy()

        y = (
            subset[
                "human_label"
            ]
            ==
            target
        ).astype(int)

        if y.nunique() < 2:

            continue

        for feature in feature_columns:

            x = pd.to_numeric(
                subset[
                    feature
                ],
                errors="coerce"
            )

            valid = x.notna()

            x_valid = x[
                valid
            ]

            y_valid = y[
                valid
            ]

            if y_valid.nunique() < 2:

                continue

            auc = roc_auc_score(
                y_valid,
                x_valid
            )

            separability = max(
                auc,
                1.0 - auc
            )

            auc_records.append(
                {
                    "target_label": target,
                    "feature": feature,
                    "auc": round(
                        auc,
                        4
                    ),
                    "separability": round(
                        separability,
                        4
                    )
                }
            )

    auc_df = pd.DataFrame(
        auc_records
    )

    if len(auc_df) > 0:

        auc_df = auc_df.sort_values(
            [
                "target_label",
                "separability"
            ],
            ascending=[
                True,
                False
            ]
        )

        print(
            "\n" + "=" * 80
        )

        print(
            "PILOT FEATURE SEPARABILITY"
        )

        print(
            "=" * 80
        )

        for target in targets:

            print(
                f"\n--- {target} ---"
            )

            print(
                auc_df[
                    auc_df[
                        "target_label"
                    ]
                    ==
                    target
                ]
                .head(8)
                .to_string(
                    index=False
                )
            )

        auc_df.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "feature_separability_auc.csv"
            ),
            index=False
        )

except Exception as e:

    print(
        "\nAUC analysis skipped:"
    )

    print(
        str(e)
    )


# ============================================================
# 19. CLEAN UP
# ============================================================

zf.close()

if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# 20. FINAL
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "STEP 7 COMPLETE"
)

print(
    "=" * 80
)

print(
    "\nResults saved in:"
)

print(
    OUTPUT_DIR
)

print(
    "\n✅ No candidate_verification_features.csv required."
)

print(
    "✅ Only the 63 human-reviewed candidates were analyzed."
)# ============================================================
# SOLARMAP — STEP 7
# HUMAN LABEL + FEATURE ANALYSIS
# ============================================================
#
# This version DOES NOT require:
#   candidate_verification_features.csv
#
# It calculates the required features directly for the
# 63 human-reviewed candidates using the local checkpoint.
#
# ============================================================

import os
import io
import json
import zipfile
import warnings

import cv2
import numpy as np
import pandas as pd
import torch

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

HUMAN_LABELS = os.path.join(
    BASE,
    "results",
    "verification",
    "fp_human_labels.csv"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "results",
    "verification",
    "human_label_analysis"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. CHECK FILES
# ============================================================

print("=" * 80)
print("SOLARMAP — STEP 7: HUMAN LABEL FEATURE ANALYSIS")
print("=" * 80)

required_files = [
    ZIP_PATH,
    CHECKPOINT,
    HUMAN_LABELS
]

for path in required_files:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )

print("\nAll required local files found.")


# ============================================================
# 3. LOAD HUMAN LABELS
# ============================================================

human = pd.read_csv(
    HUMAN_LABELS
)

VALID_LABELS = [
    "TRUE_BACKGROUND",
    "DUPLICATE",
    "OVER_SEGMENTATION",
    "POSSIBLE_UNANNOTATED_PANEL",
    "AMBIGUOUS"
]

human = human[
    human["human_label"].isin(
        VALID_LABELS
    )
].copy()

print(
    "\nHuman-labeled candidates:",
    len(human)
)


# ============================================================
# 4. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    device
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0)
            .total_memory
            / 1024**3,
            2
        ),
        "GB"
    )


# ============================================================
# 5. LOAD MASK R-CNN
# ============================================================

print(
    "\nLoading Mask R-CNN..."
)

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
# 6. OPEN ZIP
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
# 7. LOAD IMAGE
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
            f"Image not found in ZIP:\n{filename}"
        )

    with zf.open(member) as f:

        image = Image.open(
            f
        ).convert(
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
# 8. IMAGE TRANSFORM
# ============================================================

to_tensor = transforms.ToTensor()


# ============================================================
# 9. FEATURE FUNCTIONS
# ============================================================

def calculate_features(
    mask,
    box,
    image_height,
    image_width,
    all_boxes
):

    mask_uint8 = (
        mask.astype(
            np.uint8
        )
        *
        255
    )

    mask_area = float(
        np.sum(mask)
    )

    x1, y1, x2, y2 = box

    bbox_width = max(
        1.0,
        x2 - x1
    )

    bbox_height = max(
        1.0,
        y2 - y1
    )

    bbox_area = (
        bbox_width
        *
        bbox_height
    )

    aspect_ratio = (
        bbox_width
        /
        bbox_height
    )

    # --------------------------------------------------------
    # Contours
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 0:

        largest = max(
            contours,
            key=cv2.contourArea
        )

        contour_area = float(
            cv2.contourArea(
                largest
            )
        )

        perimeter = float(
            cv2.arcLength(
                largest,
                True
            )
        )

        hull = cv2.convexHull(
            largest
        )

        hull_area = float(
            cv2.contourArea(
                hull
            )
        )

    else:

        contour_area = 0.0
        perimeter = 0.0
        hull_area = 0.0

    # --------------------------------------------------------
    # Rectangularity
    # --------------------------------------------------------

    if bbox_area > 0:

        rectangularity = (
            mask_area
            /
            bbox_area
        )

    else:

        rectangularity = 0.0

    # --------------------------------------------------------
    # Solidity
    # --------------------------------------------------------

    if hull_area > 0:

        solidity = (
            contour_area
            /
            hull_area
        )

    else:

        solidity = 0.0

    # --------------------------------------------------------
    # Compactness
    # --------------------------------------------------------

    if perimeter > 0:

        compactness = (
            4
            *
            np.pi
            *
            contour_area
            /
            (
                perimeter
                *
                perimeter
            )
        )

    else:

        compactness = 0.0

    # --------------------------------------------------------
    # Relative area
    # --------------------------------------------------------

    image_area = (
        image_height
        *
        image_width
    )

    relative_mask_area = (
        mask_area
        /
        image_area
    )

    # --------------------------------------------------------
    # Center
    # --------------------------------------------------------

    center_x = (
        x1 + x2
    ) / 2.0

    center_y = (
        y1 + y2
    ) / 2.0

    center_x_norm = (
        center_x
        /
        image_width
    )

    center_y_norm = (
        center_y
        /
        image_height
    )

    # --------------------------------------------------------
    # Edge distance
    # --------------------------------------------------------

    edge_distance_norm = min(
        center_x_norm,
        1.0 - center_x_norm,
        center_y_norm,
        1.0 - center_y_norm
    )

    # --------------------------------------------------------
    # Neighbors within 100 px
    # --------------------------------------------------------

    neighbor_count = 0

    for other_box in all_boxes:

        ox1, oy1, ox2, oy2 = other_box

        other_center_x = (
            ox1 + ox2
        ) / 2.0

        other_center_y = (
            oy1 + oy2
        ) / 2.0

        distance = np.sqrt(
            (
                center_x
                -
                other_center_x
            ) ** 2
            +
            (
                center_y
                -
                other_center_y
            ) ** 2
        )

        if (
            distance <= 100
            and
            distance > 0
        ):

            neighbor_count += 1

    return {
        "mask_area_px": mask_area,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_area_px": bbox_area,
        "aspect_ratio": aspect_ratio,
        "rectangularity": rectangularity,
        "solidity": solidity,
        "compactness": compactness,
        "relative_mask_area": relative_mask_area,
        "center_x_norm": center_x_norm,
        "center_y_norm": center_y_norm,
        "edge_distance_norm": edge_distance_norm,
        "neighbor_count_100px": neighbor_count
    }


# ============================================================
# 10. UNIQUE IMAGES
# ============================================================

unique_filenames = (
    human["filename"]
    .dropna()
    .unique()
)

print(
    "\nUnique images to process:",
    len(unique_filenames)
)


# ============================================================
# 11. RUN MODEL ON UNIQUE IMAGES
# ============================================================

prediction_cache = {}

for image_number, filename in enumerate(
    unique_filenames,
    start=1
):

    print(
        f"\nProcessing image "
        f"{image_number}/{len(unique_filenames)}: "
        f"{filename}"
    )

    image_np = load_image(
        filename
    )

    image_tensor = (
        to_tensor(
            Image.fromarray(
                image_np
            )
        )
        .to(device)
    )

    with torch.inference_mode():

        output = model(
            [image_tensor]
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

    print(
        "Predictions >= 0.50:",
        len(scores)
    )


# ============================================================
# 12. MATCH HUMAN CANDIDATE TO MODEL PREDICTION
# ============================================================

def find_prediction(
    row,
    predictions
):

    scores = predictions[
        "scores"
    ]

    boxes = predictions[
        "boxes"
    ]

    if len(scores) == 0:

        return None

    target_score = float(
        row["score"]
    )

    target_box = np.array(
        [
            float(row["bbox_x1"]),
            float(row["bbox_y1"]),
            float(row["bbox_x2"]),
            float(row["bbox_y2"])
        ],
        dtype=np.float32
    )

    best_index = None
    best_distance = float(
        "inf"
    )

    for i in range(
        len(scores)
    ):

        current_box = (
            boxes[i]
        )

        box_distance = np.mean(
            np.abs(
                current_box
                -
                target_box
            )
        )

        score_distance = abs(
            float(scores[i])
            -
            target_score
        )

        distance = (
            box_distance
            +
            50.0
            *
            score_distance
        )

        if distance < best_distance:

            best_distance = distance
            best_index = i

    return best_index


# ============================================================
# 13. BUILD 63-CANDIDATE FEATURE TABLE
# ============================================================

records = []

print(
    "\n" + "=" * 80
)

print(
    "CALCULATING FEATURES FOR HUMAN-LABELED CANDIDATES"
)

print(
    "=" * 80
)

for row_number, row in human.iterrows():

    filename = row[
        "filename"
    ]

    predictions = prediction_cache[
        filename
    ]

    prediction_index = find_prediction(
        row,
        predictions
    )

    if prediction_index is None:

        print(
            "WARNING: could not match:",
            filename,
            row["score"]
        )

        continue

    boxes = predictions[
        "boxes"
    ]

    masks = predictions[
        "masks"
    ]

    scores = predictions[
        "scores"
    ]

    image_np = load_image(
        filename
    )

    height, width = (
        image_np.shape[:2]
    )

    candidate_box = boxes[
        prediction_index
    ]

    candidate_mask = masks[
        prediction_index
    ]

    feature_values = calculate_features(
        candidate_mask,
        candidate_box,
        height,
        width,
        boxes
    )

    record = {

        "human_row": row_number,

        "filename": filename,

        "human_label": row[
            "human_label"
        ],

        "score": float(
            scores[
                prediction_index
            ]
        ),

        "max_iou_with_GT": float(
            row[
                "max_iou_with_GT"
            ]
        ),

        "max_iou_with_matched_prediction": float(
            row[
                "max_iou_with_matched_prediction"
            ]
        ),

        "diagnostic_category": row[
            "diagnostic_category"
        ],

        "prediction_index": int(
            prediction_index
        )
    }

    record.update(
        feature_values
    )

    records.append(
        record
    )


analysis_df = pd.DataFrame(
    records
)


# ============================================================
# 14. SAVE RAW FEATURE TABLE
# ============================================================

raw_output = os.path.join(
    OUTPUT_DIR,
    "human_labeled_candidates_with_features.csv"
)

analysis_df.to_csv(
    raw_output,
    index=False
)

print(
    "\nSaved:",
    raw_output
)

print(
    "Matched candidates:",
    len(analysis_df),
    "/",
    len(human)
)


# ============================================================
# 15. LABEL COUNTS
# ============================================================

label_counts = (
    analysis_df[
        "human_label"
    ]
    .value_counts()
    .rename_axis(
        "human_label"
    )
    .reset_index(
        name="count"
    )
)

label_counts[
    "percentage"
] = (
    label_counts[
        "count"
    ]
    /
    len(analysis_df)
    *
    100
)

print(
    "\n" + "=" * 80
)

print(
    "HUMAN LABEL DISTRIBUTION"
)

print(
    "=" * 80
)

print(
    label_counts.to_string(
        index=False
    )
)

label_counts.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "human_label_counts.csv"
    ),
    index=False
)


# ============================================================
# 16. FEATURE SUMMARY
# ============================================================

feature_columns = [
    "score",
    "mask_area_px",
    "bbox_area_px",
    "bbox_width",
    "bbox_height",
    "aspect_ratio",
    "rectangularity",
    "solidity",
    "compactness",
    "relative_mask_area",
    "center_x_norm",
    "center_y_norm",
    "edge_distance_norm",
    "neighbor_count_100px"
]

mean_summary = (
    analysis_df
    .groupby(
        "human_label"
    )[
        feature_columns
    ]
    .mean()
    .round(4)
)

median_summary = (
    analysis_df
    .groupby(
        "human_label"
    )[
        feature_columns
    ]
    .median()
    .round(4)
)

print(
    "\n" + "=" * 80
)

print(
    "MEAN FEATURES BY HUMAN LABEL"
)

print(
    "=" * 80
)

print(
    mean_summary.to_string()
)

print(
    "\n" + "=" * 80
)

print(
    "MEDIAN FEATURES BY HUMAN LABEL"
)

print(
    "=" * 80
)

print(
    median_summary.to_string()
)

mean_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "mean_features_by_human_label.csv"
    )
)

median_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "median_features_by_human_label.csv"
    )
)


# ============================================================
# 17. CONFIDENCE SUMMARY
# ============================================================

confidence_summary = (
    analysis_df
    .groupby(
        "human_label"
    )[
        "score"
    ]
    .agg(
        [
            "count",
            "min",
            "max",
            "mean",
            "median",
            "std"
        ]
    )
    .round(4)
)

print(
    "\n" + "=" * 80
)

print(
    "CONFIDENCE BY HUMAN LABEL"
)

print(
    "=" * 80
)

print(
    confidence_summary.to_string()
)

confidence_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "confidence_by_human_label.csv"
    )
)


# ============================================================
# 18. AUC ANALYSIS
# ============================================================

try:

    from sklearn.metrics import roc_auc_score

    auc_records = []

    targets = [
        "TRUE_BACKGROUND",
        "DUPLICATE",
        "OVER_SEGMENTATION",
        "POSSIBLE_UNANNOTATED_PANEL"
    ]

    for target in targets:

        subset = analysis_df[
            analysis_df[
                "human_label"
            ]
            !=
            "AMBIGUOUS"
        ].copy()

        y = (
            subset[
                "human_label"
            ]
            ==
            target
        ).astype(int)

        if y.nunique() < 2:

            continue

        for feature in feature_columns:

            x = pd.to_numeric(
                subset[
                    feature
                ],
                errors="coerce"
            )

            valid = x.notna()

            x_valid = x[
                valid
            ]

            y_valid = y[
                valid
            ]

            if y_valid.nunique() < 2:

                continue

            auc = roc_auc_score(
                y_valid,
                x_valid
            )

            separability = max(
                auc,
                1.0 - auc
            )

            auc_records.append(
                {
                    "target_label": target,
                    "feature": feature,
                    "auc": round(
                        auc,
                        4
                    ),
                    "separability": round(
                        separability,
                        4
                    )
                }
            )

    auc_df = pd.DataFrame(
        auc_records
    )

    if len(auc_df) > 0:

        auc_df = auc_df.sort_values(
            [
                "target_label",
                "separability"
            ],
            ascending=[
                True,
                False
            ]
        )

        print(
            "\n" + "=" * 80
        )

        print(
            "PILOT FEATURE SEPARABILITY"
        )

        print(
            "=" * 80
        )

        for target in targets:

            print(
                f"\n--- {target} ---"
            )

            print(
                auc_df[
                    auc_df[
                        "target_label"
                    ]
                    ==
                    target
                ]
                .head(8)
                .to_string(
                    index=False
                )
            )

        auc_df.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "feature_separability_auc.csv"
            ),
            index=False
        )

except Exception as e:

    print(
        "\nAUC analysis skipped:"
    )

    print(
        str(e)
    )


# ============================================================
# 19. CLEAN UP
# ============================================================

zf.close()

if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# 20. FINAL
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "STEP 7 COMPLETE"
)

print(
    "=" * 80
)

print(
    "\nResults saved in:"
)

print(
    OUTPUT_DIR
)

print(
    "\n✅ No candidate_verification_features.csv required."
)

print(
    "✅ Only the 63 human-reviewed candidates were analyzed."
)