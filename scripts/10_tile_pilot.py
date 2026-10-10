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
# SOLARMAP — STEP 10
# ADAPTIVE GLOBAL + LOCAL TILE INFERENCE PILOT
# ============================================================

BASE = r"C:\SolarMap-India"

ZIP_PATH = os.path.join(
    BASE, "dataset", "Solar Images.zip"
)

CHECKPOINT = os.path.join(
    BASE, "checkpoints", "maskrcnn_baseline_epoch_10.pth"
)

BASELINE_METRICS = os.path.join(
    BASE, "results", "baseline", "baseline_per_image_metrics.csv"
)

OUTPUT_DIR = os.path.join(
    BASE, "results", "verification", "tile_pilot"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

SCORE_THRESHOLDS = [0.50, 0.75]
TILE_SIZE = 448
TILE_OVERLAP = 0.25
MASK_NMS_IOU = 0.50
MATCH_IOU = 0.50


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 80)
print("SOLARMAP — STEP 10: GLOBAL + LOCAL TILE PILOT")
print("=" * 80)

for path in [
    ZIP_PATH,
    CHECKPOINT,
    BASELINE_METRICS
]:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )

print("\nAll required files found.")


# ============================================================
# DEVICE
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


# ============================================================
# LOAD BASELINE METRICS
# ============================================================

baseline_df = pd.read_csv(
    BASELINE_METRICS
)

required_baseline_cols = [
    "filename",
    "ground_truth_count",
    "predicted_count",
    "count_error",
    "TP",
    "FP",
    "FN"
]

missing = [
    c
    for c in required_baseline_cols
    if c not in baseline_df.columns
]

if missing:

    raise ValueError(
        "Missing columns in baseline metrics:\n"
        +
        "\n".join(missing)
    )

baseline_df = baseline_df.copy()

baseline_df["filename"] = (
    baseline_df["filename"]
    .astype(str)
)

test_filenames = (
    baseline_df["filename"]
    .tolist()
)

print(
    "\nPositive test images:",
    len(test_filenames)
)

if len(test_filenames) != 250:

    print(
        "WARNING: expected 250 test images, "
        f"found {len(test_filenames)}."
    )


# ============================================================
# LOAD COCO FROM ZIP
# ============================================================

print(
    "\nOpening dataset ZIP..."
)

zf = zipfile.ZipFile(
    ZIP_PATH,
    "r"
)

json_member = None

for name in zf.namelist():

    if name.endswith(
        "merged_instances_default.json"
    ):

        json_member = name
        break

if json_member is None:

    raise FileNotFoundError(
        "merged_instances_default.json not found in ZIP."
    )

with zf.open(
    json_member
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
# COCO LOOKUPS
# ============================================================

image_by_filename = {
    str(img["file_name"]): img
    for img in coco["images"]
}

annotations_by_image = {}

for ann in coco["annotations"]:

    image_id = ann["image_id"]

    annotations_by_image.setdefault(
        image_id,
        []
    ).append(
        ann
    )


# ============================================================
# ZIP IMAGE LOOKUP
# ============================================================

zip_names = zf.namelist()

image_cache = {}


def find_zip_member(filename):

    for name in zip_names:

        if (
            name == filename
            or
            name.endswith(
                "/" + filename
            )
        ):

            return name

    return None


def load_image(filename):

    if filename in image_cache:

        return image_cache[
            filename
        ]

    member = find_zip_member(
        filename
    )

    if member is None:

        raise FileNotFoundError(
            f"Image not found in ZIP: {filename}"
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
# POLYGON -> MASK
# ============================================================

def annotation_to_mask(
    annotation,
    height,
    width
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

        if not isinstance(
            polygon,
            list
        ):

            continue

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


def get_gt_masks(filename):

    if filename not in image_by_filename:

        raise KeyError(
            f"Filename not found in COCO: {filename}"
        )

    image_info = image_by_filename[
        filename
    ]

    image_id = image_info[
        "id"
    ]

    height = image_info.get(
        "height",
        640
    )

    width = image_info.get(
        "width",
        640
    )

    annotations = (
        annotations_by_image.get(
            image_id,
            []
        )
    )

    masks = []

    for ann in annotations:

        if float(
            ann.get(
                "area",
                0
            )
        ) <= 0:

            continue

        mask = annotation_to_mask(
            ann,
            height,
            width
        )

        if mask.sum() > 0:

            masks.append(
                mask
            )

    return masks


# ============================================================
# LOAD BASELINE MASK R-CNN
# ============================================================

print(
    "\nLoading baseline Mask R-CNN..."
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

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    elif "state_dict" in checkpoint:

        state_dict = (
            checkpoint[
                "state_dict"
            ]
        )

    elif (
        "model" in checkpoint
        and
        isinstance(
            checkpoint["model"],
            dict
        )
    ):

        state_dict = (
            checkpoint[
                "model"
            ]
        )

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith(
        "module."
    ):

        key = key[7:]

    clean_state_dict[
        key
    ] = value


model.load_state_dict(
    clean_state_dict,
    strict=True
)

model.to(
    device
)

model.eval()

print(
    "Baseline Mask R-CNN loaded."
)


# ============================================================
# TRANSFORM
# ============================================================

to_tensor = transforms.ToTensor()


# ============================================================
# SINGLE IMAGE INFERENCE
# ============================================================

def infer_image(
    image_np
):

    tensor = to_tensor(
        Image.fromarray(
            image_np
        )
    ).to(
        device
    )

    with torch.inference_mode():

        output = model(
            [tensor]
        )[0]

    scores = (
        output[
            "scores"
        ]
        .detach()
        .cpu()
        .numpy()
    )

    boxes = (
        output[
            "boxes"
        ]
        .detach()
        .cpu()
        .numpy()
    )

    masks = (
        output[
            "masks"
        ]
        .detach()
        .cpu()
        .numpy()[:, 0]
        >= 0.50
    )

    keep = (
        scores
        >=
        min(
            SCORE_THRESHOLDS
        )
    )

    return {
        "scores":
            scores[keep],

        "boxes":
            boxes[keep],

        "masks":
            masks[keep]
    }


# ============================================================
# GENERATE OVERLAPPING TILES
# ============================================================

def generate_tiles(
    height,
    width,
    tile_size=TILE_SIZE,
    overlap=TILE_OVERLAP
):

    stride = int(
        tile_size
        *
        (
            1.0
            -
            overlap
        )
    )

    y_positions = [0]

    while True:

        next_y = (
            y_positions[-1]
            +
            stride
        )

        if (
            next_y
            +
            tile_size
            >=
            height
        ):

            break

        y_positions.append(
            next_y
        )

    final_y = max(
        0,
        height - tile_size
    )

    if y_positions[-1] != final_y:

        y_positions.append(
            final_y
        )


    x_positions = [0]

    while True:

        next_x = (
            x_positions[-1]
            +
            stride
        )

        if (
            next_x
            +
            tile_size
            >=
            width
        ):

            break

        x_positions.append(
            next_x
        )

    final_x = max(
        0,
        width - tile_size
    )

    if x_positions[-1] != final_x:

        x_positions.append(
            final_x
        )


    tiles = []

    for y in sorted(
        set(y_positions)
    ):

        for x in sorted(
            set(x_positions)
        ):

            x2 = min(
                width,
                x + tile_size
            )

            y2 = min(
                height,
                y + tile_size
            )

            tiles.append(
                (
                    x,
                    y,
                    x2,
                    y2
                )
            )

    return tiles


# ============================================================
# MASK IoU
# ============================================================

def mask_iou(
    mask_a,
    mask_b
):

    a = mask_a.astype(
        bool
    )

    b = mask_b.astype(
        bool
    )

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
        intersection
        /
        union
    )


# ============================================================
# MASK NMS
# ============================================================

def mask_nms(
    scores,
    boxes,
    masks,
    iou_threshold=MASK_NMS_IOU
):

    if len(scores) == 0:

        return (
            scores,
            boxes,
            masks
        )

    order = np.argsort(
        -scores
    )

    keep_indices = []

    for index in order:

        suppress = False

        for kept_index in keep_indices:

            iou = mask_iou(
                masks[index],
                masks[kept_index]
            )

            if (
                iou
                >=
                iou_threshold
            ):

                suppress = True

                break

        if not suppress:

            keep_indices.append(
                index
            )

    keep_indices = np.asarray(
        keep_indices,
        dtype=np.int64
    )

    return (
        scores[
            keep_indices
        ],
        boxes[
            keep_indices
        ],
        masks[
            keep_indices
        ]
    )


# ============================================================
# GLOBAL + LOCAL INFERENCE
# ============================================================

def global_local_inference(
    image_np
):

    height, width = image_np.shape[:2]

    all_scores = []

    all_boxes = []

    all_masks = []


    # --------------------------------------------------------
    # GLOBAL IMAGE
    # --------------------------------------------------------

    global_output = infer_image(
        image_np
    )

    for score, box, mask in zip(
        global_output["scores"],
        global_output["boxes"],
        global_output["masks"]
    ):

        all_scores.append(
            float(score)
        )

        all_boxes.append(
            box.astype(
                np.float32
            )
        )

        all_masks.append(
            mask
        )


    # --------------------------------------------------------
    # LOCAL TILES
    # --------------------------------------------------------

    tiles = generate_tiles(
        height,
        width
    )

    for (
        tx1,
        ty1,
        tx2,
        ty2
    ) in tiles:

        tile = image_np[
            ty1:ty2,
            tx1:tx2
        ]

        tile_output = infer_image(
            tile
        )

        for (
            score,
            box,
            mask
        ) in zip(
            tile_output["scores"],
            tile_output["boxes"],
            tile_output["masks"]
        ):

            bx1, by1, bx2, by2 = box

            global_box = np.array(
                [
                    bx1 + tx1,
                    by1 + ty1,
                    bx2 + tx1,
                    by2 + ty1
                ],
                dtype=np.float32
            )

            global_mask = np.zeros(
                (height, width),
                dtype=bool
            )

            tile_height = min(
                ty2 - ty1,
                mask.shape[0]
            )

            tile_width = min(
                tx2 - tx1,
                mask.shape[1]
            )

            global_mask[
                ty1:ty1 + tile_height,
                tx1:tx1 + tile_width
            ] = mask[
                :tile_height,
                :tile_width
            ]

            all_scores.append(
                float(score)
            )

            all_boxes.append(
                global_box
            )

            all_masks.append(
                global_mask
            )


    # --------------------------------------------------------
    # NOTHING DETECTED
    # --------------------------------------------------------

    if len(all_scores) == 0:

        return {
            "scores":
                np.array(
                    []
                ),

            "boxes":
                np.empty(
                    (0, 4),
                    dtype=np.float32
                ),

            "masks":
                np.empty(
                    (
                        0,
                        height,
                        width
                    ),
                    dtype=bool
                )
        }


    scores = np.asarray(
        all_scores,
        dtype=np.float32
    )

    boxes = np.asarray(
        all_boxes,
        dtype=np.float32
    )

    masks = np.asarray(
        all_masks,
        dtype=bool
    )


    # --------------------------------------------------------
    # REMOVE DUPLICATES BETWEEN TILES
    # --------------------------------------------------------

    scores, boxes, masks = mask_nms(
        scores,
        boxes,
        masks,
        MASK_NMS_IOU
    )


    return {
        "scores":
            scores,

        "boxes":
            boxes,

        "masks":
            masks
    }


# ============================================================
# EVALUATE PREDICTIONS
# ============================================================

def evaluate_predictions(
    predictions,
    gt_masks,
    threshold
):

    scores = predictions[
        "scores"
    ]

    masks = predictions[
        "masks"
    ]

    keep = (
        scores
        >=
        threshold
    )

    scores = scores[
        keep
    ]

    masks = masks[
        keep
    ]

    if len(scores) == 0:

        return {
            "TP": 0,
            "FP": 0,
            "FN": len(gt_masks),
            "predicted_count": 0,
            "gt_count": len(gt_masks)
        }


    # Highest confidence first
    order = np.argsort(
        -scores
    )

    masks = masks[
        order
    ]

    matched_gt = set()

    tp = 0

    fp = 0


    for pred_mask in masks:

        best_iou = 0.0

        best_gt = None


        for gt_index, gt_mask in enumerate(
            gt_masks
        ):

            if gt_index in matched_gt:

                continue

            current_iou = mask_iou(
                pred_mask,
                gt_mask
            )

            if current_iou > best_iou:

                best_iou = current_iou

                best_gt = gt_index


        if (
            best_gt is not None
            and
            best_iou >= MATCH_IOU
        ):

            tp += 1

            matched_gt.add(
                best_gt
            )

        else:

            fp += 1


    fn = (
        len(gt_masks)
        -
        len(matched_gt)
    )


    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "predicted_count":
            len(masks),
        "gt_count":
            len(gt_masks)
    }


# ============================================================
# AGGREGATE STORAGE
# ============================================================

aggregate = {
    threshold: {
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "count_errors": []
    }
    for threshold in SCORE_THRESHOLDS
}

per_image_rows = []


# ============================================================
# RUN EXPERIMENT
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "RUNNING GLOBAL + LOCAL TILE INFERENCE"
)

print(
    "=" * 80
)

print(
    f"Tile size: {TILE_SIZE}"
)

print(
    f"Tile overlap: {TILE_OVERLAP}"
)

print(
    f"Mask NMS IoU: {MASK_NMS_IOU}"
)

print(
    f"Match IoU: {MATCH_IOU}"
)


for image_number, filename in enumerate(
    test_filenames,
    start=1
):

    print(
        f"\n[{image_number}/{len(test_filenames)}] "
        f"{filename}"
    )

    image = load_image(
        filename
    )

    gt_masks = get_gt_masks(
        filename
    )

    predictions = global_local_inference(
        image
    )

    candidate_count = len(
        predictions[
            "scores"
        ]
    )

    print(
        "Candidates after tile NMS:",
        candidate_count
    )

    row = {
        "filename":
            filename,

        "ground_truth_count":
            len(gt_masks),

        "tile_nms_candidate_count":
            candidate_count
    }


    for threshold in SCORE_THRESHOLDS:

        metrics = evaluate_predictions(
            predictions,
            gt_masks,
            threshold
        )

        prefix = (
            f"thr_{threshold:.2f}_"
        )

        row[
            prefix + "TP"
        ] = metrics["TP"]

        row[
            prefix + "FP"
        ] = metrics["FP"]

        row[
            prefix + "FN"
        ] = metrics["FN"]

        row[
            prefix + "predicted_count"
        ] = metrics["predicted_count"]

        count_error = (
            metrics["predicted_count"]
            -
            metrics["gt_count"]
        )

        row[
            prefix + "count_error"
        ] = count_error


        aggregate[
            threshold
        ]["TP"] += metrics["TP"]

        aggregate[
            threshold
        ]["FP"] += metrics["FP"]

        aggregate[
            threshold
        ]["FN"] += metrics["FN"]

        aggregate[
            threshold
        ]["count_errors"].append(
            count_error
        )


    per_image_rows.append(
        row
    )


# ============================================================
# SAVE PER-IMAGE RESULTS
# ============================================================

per_image_df = pd.DataFrame(
    per_image_rows
)

per_image_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "tile_pilot_per_image_metrics.csv"
    ),
    index=False
)


# ============================================================
# BASELINE AGGREGATE METRICS
# ============================================================

baseline_tp = int(
    baseline_df["TP"].sum()
)

baseline_fp = int(
    baseline_df["FP"].sum()
)

baseline_fn = int(
    baseline_df["FN"].sum()
)


baseline_precision = (
    baseline_tp
    /
    max(
        1,
        baseline_tp + baseline_fp
    )
)

baseline_recall = (
    baseline_tp
    /
    max(
        1,
        baseline_tp + baseline_fn
    )
)

baseline_f1 = (
    2
    *
    baseline_precision
    *
    baseline_recall
    /
    max(
        1e-12,
        baseline_precision
        +
        baseline_recall
    )
)

baseline_count_mae = (
    baseline_df[
        "count_error"
    ]
    .abs()
    .mean()
)

baseline_count_rmse = np.sqrt(
    np.mean(
        baseline_df[
            "count_error"
        ] ** 2
    )
)


result_rows = []


result_rows.append(
    {
        "method":
            "BASELINE_FULL_IMAGE",

        "threshold":
            0.50,

        "TP":
            baseline_tp,

        "FP":
            baseline_fp,

        "FN":
            baseline_fn,

        "precision":
            baseline_precision,

        "recall":
            baseline_recall,

        "F1":
            baseline_f1,

        "count_MAE":
            baseline_count_mae,

        "count_RMSE":
            baseline_count_rmse
    }
)


# ============================================================
# TILE AGGREGATE RESULTS
# ============================================================

for threshold in SCORE_THRESHOLDS:

    values = aggregate[
        threshold
    ]

    tp = values["TP"]

    fp = values["FP"]

    fn = values["FN"]


    precision = (
        tp
        /
        max(
            1,
            tp + fp
        )
    )

    recall = (
        tp
        /
        max(
            1,
            tp + fn
        )
    )

    f1 = (
        2
        *
        precision
        *
        recall
        /
        max(
            1e-12,
            precision
            +
            recall
        )
    )


    count_errors = np.asarray(
        values[
            "count_errors"
        ],
        dtype=np.float64
    )


    count_mae = np.mean(
        np.abs(
            count_errors
        )
    )

    count_rmse = np.sqrt(
        np.mean(
            count_errors ** 2
        )
    )


    result_rows.append(
        {
            "method":
                "GLOBAL_PLUS_LOCAL_TILES",

            "threshold":
                threshold,

            "TP":
                tp,

            "FP":
                fp,

            "FN":
                fn,

            "precision":
                precision,

            "recall":
                recall,

            "F1":
                f1,

            "count_MAE":
                count_mae,

            "count_RMSE":
                count_rmse
        }
    )


results_df = pd.DataFrame(
    result_rows
)


# ============================================================
# SAVE COMPARISON
# ============================================================

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "tile_pilot_comparison.csv"
    ),
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "GLOBAL + LOCAL TILE PILOT RESULTS"
)

print(
    "=" * 80
)

print(
    results_df.round(
        4
    ).to_string(
        index=False
    )
)


# ============================================================
# BEST TILE RESULT
# ============================================================

tile_results = results_df[
    results_df["method"]
    ==
    "GLOBAL_PLUS_LOCAL_TILES"
].copy()


if len(tile_results) > 0:

    best_tile = (
        tile_results
        .sort_values(
            "F1",
            ascending=False
        )
        .iloc[0]
    )

    print(
        "\nBest tile threshold:",
        best_tile[
            "threshold"
        ]
    )

    print(
        f"Tile F1: "
        f"{best_tile['F1']:.4f}"
    )

    print(
        f"Tile precision: "
        f"{best_tile['precision']:.4f}"
    )

    print(
        f"Tile recall: "
        f"{best_tile['recall']:.4f}"
    )

    print(
        f"Tile count MAE: "
        f"{best_tile['count_MAE']:.4f}"
    )


# ============================================================
# BASELINE COMPARISON
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "BASELINE REFERENCE"
)

print(
    "=" * 80
)

print(
    f"Baseline F1: "
    f"{baseline_f1:.4f}"
)

print(
    f"Baseline precision: "
    f"{baseline_precision:.4f}"
)

print(
    f"Baseline recall: "
    f"{baseline_recall:.4f}"
)

print(
    f"Baseline count MAE: "
    f"{baseline_count_mae:.4f}"
)

if len(tile_results) > 0:

    improvement = (
        best_tile["F1"]
        -
        baseline_f1
    )

    print(
        f"\nBest tile F1 change: "
        f"{improvement:+.4f}"
    )


# ============================================================
# CLEANUP
# ============================================================

zf.close()

if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "STEP 10 COMPLETE"
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
    "\n✅ Global + local tile inference pilot finished."
)