# ============================================================
# SOLARMAP — STEP 9
# VISUAL CANDIDATE VERIFIER
# ============================================================
#
# Purpose:
#   Test whether actual image appearance contains useful
#   information for distinguishing human-verified failure modes.
#
# Classes:
#   TRUE_BACKGROUND
#   DUPLICATE
#   OVER_SEGMENTATION
#   POSSIBLE_UNANNOTATED_PANEL
#
# AMBIGUOUS candidates are excluded.
#
# IMPORTANT:
#   - Does NOT retrain Mask R-CNN
#   - Does NOT use the final test set
#   - Uses only the 63 human-reviewed pilot candidates
#   - 6 AMBIGUOUS candidates are excluded
#   - Uses prediction_index + score because Step 7 CSV
#     does not contain bbox_x1/y1/x2/y2
#
# Visual method:
#   Mask R-CNN candidate
#       ↓
#   Context crop
#       ↓
#   Pretrained ResNet-18
#       ↓
#   512-dimensional visual embedding
#       ↓
#   Logistic Regression
#       ↓
#   Grouped cross-validation by image
#
# ============================================================

import os
import zipfile
import warnings

import numpy as np
import pandas as pd
import torch

from PIL import Image

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

from sklearn.model_selection import (
    StratifiedGroupKFold
)

from sklearn.pipeline import (
    Pipeline
)

from sklearn.impute import (
    SimpleImputer
)

from sklearn.preprocessing import (
    StandardScaler
)

from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)


warnings.filterwarnings(
    "ignore"
)


# ============================================================
# 1. PATHS
# ============================================================

BASE = r"C:\SolarMap-India"

ZIP_PATH = os.path.join(
    BASE,
    "dataset",
    "Solar Images.zip"
)

INPUT_CSV = os.path.join(
    BASE,
    "results",
    "verification",
    "human_label_analysis",
    "human_labeled_candidates_with_features.csv"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "results",
    "verification",
    "visual_verifier"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. START
# ============================================================

print("=" * 80)
print("SOLARMAP — STEP 9: VISUAL CANDIDATE VERIFIER")
print("=" * 80)


# ============================================================
# 3. CHECK INPUT FILES
# ============================================================

for path in [
    ZIP_PATH,
    INPUT_CSV
]:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )

print(
    "\nAll required files found."
)


# ============================================================
# 4. LOAD HUMAN DATA
# ============================================================

df = pd.read_csv(
    INPUT_CSV
)

print(
    "\nTotal rows in human-label file:",
    len(df)
)


# ============================================================
# 5. REQUIRED COLUMNS
# ============================================================

required_columns = [
    "filename",
    "human_label",
    "score",
    "prediction_index"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns:\n"
        +
        "\n".join(
            missing_columns
        )
    )


# ============================================================
# 6. REMOVE AMBIGUOUS
# ============================================================

VALID_LABELS = [
    "TRUE_BACKGROUND",
    "DUPLICATE",
    "OVER_SEGMENTATION",
    "POSSIBLE_UNANNOTATED_PANEL"
]

before = len(df)

df = df[
    df["human_label"].isin(
        VALID_LABELS
    )
].copy()

df.reset_index(
    drop=True,
    inplace=True
)

print(
    "\nAMBIGUOUS candidates excluded:",
    before - len(df)
)

print(
    "Candidates used:",
    len(df)
)


# ============================================================
# 7. LABEL DISTRIBUTION
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "CLASS DISTRIBUTION"
)

print(
    "=" * 80
)

print(
    df["human_label"]
    .value_counts()
    .to_string()
)


# ============================================================
# 8. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "\nDevice:",
    device
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    total_vram = (
        torch.cuda
        .get_device_properties(0)
        .total_memory
        /
        (1024 ** 3)
    )

    print(
        f"VRAM: {total_vram:.2f} GB"
    )


# ============================================================
# 9. LOAD PRETRAINED RESNET-18
# ============================================================

print(
    "\nLoading pretrained ResNet-18..."
)

weights = ResNet18_Weights.DEFAULT

resnet = resnet18(
    weights=weights
)

# Remove final ImageNet classifier
feature_extractor = torch.nn.Sequential(
    *list(
        resnet.children()
    )[:-1]
)

feature_extractor.to(
    device
)

feature_extractor.eval()

preprocess = weights.transforms()

print(
    "ResNet-18 loaded."
)

print(
    "Embedding dimension: 512"
)


# ============================================================
# 10. OPEN DATASET ZIP
# ============================================================

print(
    "\nOpening SolarMap ZIP..."
)

zf = zipfile.ZipFile(
    ZIP_PATH,
    "r"
)

zip_names = zf.namelist()


# ============================================================
# 11. IMAGE CACHE
# ============================================================

image_cache = {}


def find_member(
    filename
):

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


def load_image(
    filename
):

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

    with zf.open(member) as file:

        image = Image.open(
            file
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
# 12. MASK R-CNN
# ============================================================
#
# We need the SAME baseline prediction ordering so that
# prediction_index from Step 7 can identify the candidate.
#
# Rather than loading the checkpoint in this script, we use
# the candidate's saved prediction_index and recover the box
# by running the baseline model.
#
# ============================================================

CHECKPOINT = os.path.join(
    BASE,
    "checkpoints",
    "maskrcnn_baseline_epoch_10.pth"
)

if not os.path.exists(
    CHECKPOINT
):

    raise FileNotFoundError(
        f"Baseline checkpoint not found:\n{CHECKPOINT}"
    )

print(
    "\nLoading baseline Mask R-CNN..."
)

from torchvision.models.detection import (
    maskrcnn_resnet50_fpn
)

baseline_model = maskrcnn_resnet50_fpn(
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
        and
        isinstance(
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

    if key.startswith(
        "module."
    ):

        key = key[
            len("module.") :
        ]

    clean_state_dict[
        key
    ] = value


baseline_model.load_state_dict(
    clean_state_dict,
    strict=True
)

baseline_model.to(
    device
)

baseline_model.eval()

print(
    "Baseline Mask R-CNN loaded."
)


# ============================================================
# 13. CANDIDATE PREDICTION CACHE
# ============================================================

prediction_cache = {}


def get_predictions(
    filename
):

    if filename in prediction_cache:

        return prediction_cache[
            filename
        ]

    image_np = load_image(
        filename
    )

    pil_image = Image.fromarray(
        image_np
    )

    image_tensor = (
        preprocess_image_tensor(
            pil_image
        )
    )

    with torch.inference_mode():

        output = baseline_model(
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

    # SAME 0.50 confidence filtering used previously
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

    result = {
        "scores": scores,
        "boxes": boxes,
        "masks": masks
    }

    prediction_cache[
        filename
    ] = result

    return result


# ============================================================
# 14. TENSOR TRANSFORM FOR MASK R-CNN
# ============================================================

from torchvision import transforms

image_to_tensor = transforms.ToTensor()


def preprocess_image_tensor(
    pil_image
):

    return image_to_tensor(
        pil_image
    ).to(
        device
    )


# ============================================================
# 15. FIND CANDIDATE
# ============================================================
#
# Primary method:
#   prediction_index
#
# Validation:
#   stored score vs current score
#
# Fallback:
#   find prediction with closest confidence score.
#
# ============================================================

def find_candidate(
    row,
    predictions
):

    scores = predictions[
        "scores"
    ]

    boxes = predictions[
        "boxes"
    ]

    masks = predictions[
        "masks"
    ]

    if len(scores) == 0:

        return None

    target_index = int(
        row["prediction_index"]
    )

    stored_score = float(
        row["score"]
    )

    # --------------------------------------------------------
    # Try prediction_index directly
    # --------------------------------------------------------

    if (
        0 <= target_index
        <
        len(scores)
    ):

        current_score = float(
            scores[
                target_index
            ]
        )

        if abs(
            current_score
            -
            stored_score
        ) <= 0.002:

            return {
                "index": target_index,
                "score": current_score,
                "box": boxes[
                    target_index
                ],
                "mask": masks[
                    target_index
                ]
            }

    # --------------------------------------------------------
    # Fallback: closest confidence
    # --------------------------------------------------------

    score_distances = np.abs(
        scores
        -
        stored_score
    )

    best_index = int(
        np.argmin(
            score_distances
        )
    )

    best_distance = float(
        score_distances[
            best_index
        ]
    )

    if best_distance <= 0.01:

        return {
            "index": best_index,
            "score": float(
                scores[
                    best_index
                ]
            ),
            "box": boxes[
                best_index
            ],
            "mask": masks[
                best_index
            ]
        }

    return None


# ============================================================
# 16. CREATE CONTEXT CROP
# ============================================================

def make_context_crop(
    image,
    box
):

    h, w = image.shape[:2]

    x1, y1, x2, y2 = [
        int(
            round(
                value
            )
        )
        for value in box
    ]

    x1 = max(
        0,
        min(
            x1,
            w - 1
        )
    )

    y1 = max(
        0,
        min(
            y1,
            h - 1
        )
    )

    x2 = max(
        x1 + 1,
        min(
            x2,
            w
        )
    )

    y2 = max(
        y1 + 1,
        min(
            y2,
            h
        )
    )

    box_width = max(
        1,
        x2 - x1
    )

    box_height = max(
        1,
        y2 - y1
    )

    # Include broad context
    pad_x = int(
        2.5 *
        box_width
    )

    pad_y = int(
        2.5 *
        box_height
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
    ]

    if crop.size == 0:

        return None

    return crop


# ============================================================
# 17. EXTRACT RESNET EMBEDDING
# ============================================================

def extract_embedding(
    crop
):

    if crop is None:

        return None

    pil_crop = Image.fromarray(
        crop
    )

    tensor = preprocess(
        pil_crop
    ).unsqueeze(
        0
    ).to(
        device
    )

    with torch.inference_mode():

        embedding = (
            feature_extractor(
                tensor
            )
            .flatten(
                1
            )
            .squeeze(
                0
            )
            .cpu()
            .numpy()
        )

    return embedding.astype(
        np.float32
    )


# ============================================================
# 18. EXTRACT VISUAL EMBEDDINGS
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "EXTRACTING VISUAL EMBEDDINGS"
)

print(
    "=" * 80
)


embeddings = []

metadata = []

failed_candidates = []


for i, row in df.iterrows():

    filename = row[
        "filename"
    ]

    print(
        f"\nCandidate "
        f"{i + 1}/{len(df)}: "
        f"{filename}"
    )

    print(
        f"Stored confidence: "
        f"{float(row['score']):.4f}"
    )

    print(
        f"Prediction index: "
        f"{int(row['prediction_index'])}"
    )

    try:

        image = load_image(
            filename
        )

        predictions = get_predictions(
            filename
        )

        candidate = find_candidate(
            row,
            predictions
        )

        if candidate is None:

            print(
                "WARNING: candidate "
                "could not be matched."
            )

            failed_candidates.append(
                {
                    "row": i,
                    "filename": filename,
                    "score":
                        row["score"],
                    "prediction_index":
                        row[
                            "prediction_index"
                        ]
                }
            )

            continue

        print(
            f"Matched confidence: "
            f"{candidate['score']:.4f}"
        )

        crop = make_context_crop(
            image,
            candidate["box"]
        )

        if crop is None:

            print(
                "WARNING: empty crop."
            )

            failed_candidates.append(
                {
                    "row": i,
                    "filename": filename,
                    "score":
                        row["score"],
                    "prediction_index":
                        row[
                            "prediction_index"
                        ]
                }
            )

            continue

        embedding = extract_embedding(
            crop
        )

        if embedding is None:

            print(
                "WARNING: embedding failed."
            )

            failed_candidates.append(
                {
                    "row": i,
                    "filename": filename,
                    "score":
                        row["score"],
                    "prediction_index":
                        row[
                            "prediction_index"
                        ]
                }
            )

            continue

        embeddings.append(
            embedding
        )

        metadata.append(
            {
                "source_index": i,
                "filename": filename,
                "human_label":
                    row[
                        "human_label"
                    ]
            }
        )

    except Exception as error:

        print(
            "ERROR:",
            str(error)
        )

        failed_candidates.append(
            {
                "row": i,
                "filename": filename,
                "score":
                    row["score"],
                "prediction_index":
                    row[
                        "prediction_index"
                    ]
            }
        )


# ============================================================
# 19. CHECK EMBEDDINGS
# ============================================================

if len(embeddings) < 8:

    raise RuntimeError(
        "Too few successful visual embeddings."
    )


X = np.vstack(
    embeddings
)

meta_df = pd.DataFrame(
    metadata
)

y = meta_df[
    "human_label"
].to_numpy()

groups = meta_df[
    "filename"
].to_numpy()


print(
    "\n" + "=" * 80
)

print(
    "EMBEDDING SUMMARY"
)

print(
    "=" * 80
)

print(
    "Embedding shape:",
    X.shape
)

print(
    "Successful candidates:",
    len(X)
)

print(
    "Failed candidates:",
    len(
        failed_candidates
    )
)

print(
    "Unique image groups:",
    len(
        np.unique(
            groups
        )
    )
)


# ============================================================
# 20. SAVE EMBEDDINGS
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "visual_embeddings.npy"
    ),
    X
)

meta_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "visual_embedding_metadata.csv"
    ),
    index=False
)

if failed_candidates:

    pd.DataFrame(
        failed_candidates
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "failed_candidates.csv"
        ),
        index=False
    )


# ============================================================
# 21. CLASS COUNTS
# ============================================================

class_counts = (
    pd.Series(y)
    .value_counts()
)

print(
    "\nClass distribution after embedding:"
)

print(
    class_counts.to_string()
)


# ============================================================
# 22. GROUPED CROSS-VALIDATION
# ============================================================
#
# Candidates from the same image are kept together.
#
# This prevents image-level leakage.
#
# ============================================================

n_groups = len(
    np.unique(
        groups
    )
)

min_class_count = int(
    class_counts.min()
)

n_splits = min(
    4,
    min_class_count,
    n_groups
)

if n_splits < 2:

    raise RuntimeError(
        "Not enough class/group diversity for CV."
    )

print(
    "\nGrouped CV folds:",
    n_splits
)


cv = StratifiedGroupKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=42
)


# ============================================================
# 23. CLASSIFIER FACTORY
# ============================================================

def create_classifier():

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            ),

            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]
    )


# ============================================================
# 24. CROSS-VALIDATION
# ============================================================

y_true_all = []

y_pred_all = []

fold_results = []

prediction_rows = []


print(
    "\n" + "=" * 80
)

print(
    "GROUPED CROSS-VALIDATION"
)

print(
    "=" * 80
)


for fold_number, (
    train_idx,
    validation_idx
) in enumerate(
    cv.split(
        X,
        y,
        groups
    ),
    start=1
):

    print(
        "\n" + "-" * 70
    )

    print(
        f"Fold {fold_number}"
    )

    print(
        "Training candidates:",
        len(train_idx)
    )

    print(
        "Validation candidates:",
        len(validation_idx)
    )

    train_classes = np.unique(
        y[
            train_idx
        ]
    )

    print(
        "Training classes:",
        list(
            train_classes
        )
    )

    if len(
        train_classes
    ) < 2:

        print(
            "Fold skipped."
        )

        continue

    model = create_classifier()

    model.fit(
        X[
            train_idx
        ],
        y[
            train_idx
        ]
    )

    predictions = model.predict(
        X[
            validation_idx
        ]
    )

    y_validation = y[
        validation_idx
    ]

    accuracy = accuracy_score(
        y_validation,
        predictions
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_validation,
            predictions
        )
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_validation,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    fold_results.append(
        {
            "fold":
                fold_number,

            "accuracy":
                accuracy,

            "balanced_accuracy":
                balanced_accuracy,

            "macro_precision":
                precision,

            "macro_recall":
                recall,

            "macro_f1":
                f1,

            "validation_candidates":
                len(
                    validation_idx
                ),

            "validation_images":
                len(
                    np.unique(
                        groups[
                            validation_idx
                        ]
                    )
                )
        }
    )

    y_true_all.extend(
        y_validation.tolist()
    )

    y_pred_all.extend(
        predictions.tolist()
    )

    for position, global_index in enumerate(
        validation_idx
    ):

        prediction_rows.append(
            {
                "fold":
                    fold_number,

                "filename":
                    groups[
                        global_index
                    ],

                "true_label":
                    y_validation[
                        position
                    ],

                "predicted_label":
                    predictions[
                        position
                    ]
            }
        )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy:.4f}"
    )

    print(
        f"Macro F1: {f1:.4f}"
    )


# ============================================================
# 25. CHECK CV RESULTS
# ============================================================

if len(
    y_true_all
) == 0:

    raise RuntimeError(
        "No valid cross-validation results."
    )


# ============================================================
# 26. OVERALL METRICS
# ============================================================

overall_accuracy = accuracy_score(
    y_true_all,
    y_pred_all
)

overall_balanced_accuracy = (
    balanced_accuracy_score(
        y_true_all,
        y_pred_all
    )
)

overall_precision, overall_recall, overall_f1, _ = (
    precision_recall_fscore_support(
        y_true_all,
        y_pred_all,
        average="macro",
        zero_division=0
    )
)


# ============================================================
# 27. MAIN RESULTS
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "VISUAL VERIFIER — GROUPED CROSS-VALIDATION"
)

print(
    "=" * 80
)

print(
    f"Accuracy: {overall_accuracy:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{overall_balanced_accuracy:.4f}"
)

print(
    f"Macro Precision: "
    f"{overall_precision:.4f}"
)

print(
    f"Macro Recall: "
    f"{overall_recall:.4f}"
)

print(
    f"Macro F1: "
    f"{overall_f1:.4f}"
)


# ============================================================
# 28. FOLD TABLE
# ============================================================

fold_df = pd.DataFrame(
    fold_results
)

print(
    "\nFold results:"
)

print(
    fold_df.to_string(
        index=False
    )
)


# ============================================================
# 29. CLASSIFICATION REPORT
# ============================================================

labels = sorted(
    np.unique(
        np.concatenate(
            [
                np.asarray(
                    y_true_all
                ),
                np.asarray(
                    y_pred_all
                )
            ]
        )
    )
)

report = classification_report(
    y_true_all,
    y_pred_all,
    labels=labels,
    zero_division=0
)

print(
    "\nClassification Report:"
)

print(
    report
)


# ============================================================
# 30. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true_all,
    y_pred_all,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print(
    "Confusion Matrix:"
)

print(
    cm_df.to_string()
)


# ============================================================
# 31. SAVE RESULTS
# ============================================================

fold_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "grouped_fold_metrics.csv"
    ),
    index=False
)

cm_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.csv"
    )
)

with open(
    os.path.join(
        OUTPUT_DIR,
        "classification_report.txt"
    ),
    "w",
    encoding="utf-8"
) as file:

    file.write(
        report
    )

prediction_df = pd.DataFrame(
    prediction_rows
)

prediction_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "grouped_out_of_fold_predictions.csv"
    ),
    index=False
)


# ============================================================
# 32. SUMMARY CSV
# ============================================================

results_df = pd.DataFrame(
    [
        {
            "method":
                "ResNet18_visual_embedding_"
                "LogisticRegression",

            "accuracy":
                overall_accuracy,

            "balanced_accuracy":
                overall_balanced_accuracy,

            "macro_precision":
                overall_precision,

            "macro_recall":
                overall_recall,

            "macro_f1":
                overall_f1,

            "candidates_evaluated":
                len(y_true_all),

            "unique_images":
                len(
                    np.unique(
                        groups
                    )
                ),

            "cv_folds":
                n_splits
        }
    ]
)

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "visual_verifier_results.csv"
    ),
    index=False
)


# ============================================================
# 33. CLEANUP
# ============================================================

zf.close()

if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# 34. FINAL
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "STEP 9 COMPLETE"
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
    "\n✅ Visual verifier experiment finished."
)