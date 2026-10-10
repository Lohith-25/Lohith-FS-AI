# ============================================================
# SOLARMAP — STEP 8
# PILOT CANDIDATE VERIFIER
# ============================================================
#
# Purpose:
#   Test whether the human-verified candidate features can
#   distinguish different failure modes.
#
# Classes:
#   TRUE_BACKGROUND
#   DUPLICATE
#   OVER_SEGMENTATION
#   POSSIBLE_UNANNOTATED_PANEL
#
# AMBIGUOUS samples are excluded.
#
# IMPORTANT:
#   This is a pilot experiment only.
#   It does NOT modify the Mask R-CNN model.
#   It does NOT use the final test set.
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
# ============================================================

BASE = r"C:\SolarMap-India"

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
    "pilot_verifier"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. IMPORT SCIKIT-LEARN
# ============================================================

try:

    from sklearn.base import clone

    from sklearn.model_selection import StratifiedKFold

    from sklearn.pipeline import Pipeline

    from sklearn.impute import SimpleImputer

    from sklearn.preprocessing import StandardScaler

    from sklearn.linear_model import LogisticRegression

    from sklearn.ensemble import RandomForestClassifier

    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        precision_recall_fscore_support,
        classification_report,
        confusion_matrix
    )

except ImportError:

    raise ImportError(
        "scikit-learn is not installed in this environment."
    )


# ============================================================
# 3. START
# ============================================================

print("=" * 80)
print("SOLARMAP — STEP 8: PILOT CANDIDATE VERIFIER")
print("=" * 80)


# ============================================================
# 4. CHECK INPUT FILE
# ============================================================

if not os.path.exists(INPUT_CSV):

    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_CSV}"
    )

print("\nInput file found:")
print(INPUT_CSV)


# ============================================================
# 5. LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_CSV
)

print(
    "\nTotal rows loaded:",
    len(df)
)


# ============================================================
# 6. VALID LABELS
# ============================================================

VALID_LABELS = [
    "TRUE_BACKGROUND",
    "DUPLICATE",
    "OVER_SEGMENTATION",
    "POSSIBLE_UNANNOTATED_PANEL"
]


# ============================================================
# 7. REMOVE AMBIGUOUS
# ============================================================

before_count = len(df)

df = df[
    df["human_label"].isin(
        VALID_LABELS
    )
].copy()

after_count = len(df)

print(
    "\nRemoved AMBIGUOUS rows:",
    before_count - after_count
)

print(
    "Rows used for pilot verifier:",
    after_count
)


# ============================================================
# 8. LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("LABEL DISTRIBUTION")
print("=" * 80)

label_counts = (
    df["human_label"]
    .value_counts()
)

print(
    label_counts.to_string()
)

label_counts.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "label_distribution.csv"
    ),
    header=True
)


# ============================================================
# 9. FEATURES
# ============================================================

FEATURES = [
    "score",
    "mask_area_px",
    "bbox_width",
    "bbox_height",
    "bbox_area_px",
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


# ============================================================
# 10. CHECK FEATURES
# ============================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:

    raise ValueError(
        "Missing feature columns:\n"
        +
        "\n".join(
            missing_features
        )
    )


# ============================================================
# 11. X AND y
# ============================================================

X = df[
    FEATURES
].copy()

y = df[
    "human_label"
].copy()


# ============================================================
# 12. CONVERT TO NUMERIC
# ============================================================

for feature in FEATURES:

    X[feature] = pd.to_numeric(
        X[feature],
        errors="coerce"
    )


# ============================================================
# 13. CHECK CLASS COUNTS
# ============================================================

print("\nClass counts used for training:")

for label, count in y.value_counts().items():

    print(
        f"  {label}: {count}"
    )


# ============================================================
# 14. NUMBER OF FOLDS
# ============================================================

minimum_class_count = int(
    y.value_counts().min()
)

n_splits = min(
    5,
    minimum_class_count
)

if n_splits < 2:

    raise ValueError(
        "Not enough samples per class for cross-validation."
    )

print(
    "\nCross-validation folds:",
    n_splits
)


# ============================================================
# 15. MODELS
# ============================================================

logistic_model = Pipeline(
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
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


random_forest_model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


models = {
    "LogisticRegression": logistic_model,
    "RandomForest": random_forest_model
}


# ============================================================
# 16. CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=42
)


all_results = []

all_oof_predictions = []


# ============================================================
# 17. TRAIN EACH MODEL
# ============================================================

for model_name, base_model in models.items():

    print("\n" + "=" * 80)
    print(model_name)
    print("=" * 80)

    y_true_all = []
    y_pred_all = []

    fold_results = []

    for fold_number, (
        train_indices,
        validation_indices
    ) in enumerate(
        cv.split(X, y),
        start=1
    ):

        X_train = X.iloc[
            train_indices
        ]

        X_validation = X.iloc[
            validation_indices
        ]

        y_train = y.iloc[
            train_indices
        ]

        y_validation = y.iloc[
            validation_indices
        ]

        model = clone(
            base_model
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_validation
        )

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
                "model": model_name,
                "fold": fold_number,
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "macro_precision": precision,
                "macro_recall": recall,
                "macro_f1": f1
            }
        )

        y_true_all.extend(
            y_validation.tolist()
        )

        y_pred_all.extend(
            predictions.tolist()
        )

        print(
            f"Fold {fold_number}: "
            f"Accuracy={accuracy:.4f} | "
            f"Balanced Accuracy={balanced_accuracy:.4f} | "
            f"Macro F1={f1:.4f}"
        )

    # --------------------------------------------------------
    # Overall out-of-fold metrics
    # --------------------------------------------------------

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

    all_results.append(
        {
            "model": model_name,
            "accuracy": overall_accuracy,
            "balanced_accuracy": overall_balanced_accuracy,
            "macro_precision": overall_precision,
            "macro_recall": overall_recall,
            "macro_f1": overall_f1
        }
    )

    # --------------------------------------------------------
    # Save fold metrics
    # --------------------------------------------------------

    fold_df = pd.DataFrame(
        fold_results
    )

    fold_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{model_name}_fold_metrics.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_true_all,
        y_pred_all,
        zero_division=0
    )

    print(
        "\nClassification Report:"
    )

    print(
        report
    )

    with open(
        os.path.join(
            OUTPUT_DIR,
            f"{model_name}_classification_report.txt"
        ),
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            report
        )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    labels = sorted(
        y.unique()
    )

    matrix = confusion_matrix(
        y_true_all,
        y_pred_all,
        labels=labels
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels
    )

    print(
        "Confusion Matrix:"
    )

    print(
        matrix_df.to_string()
    )

    matrix_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{model_name}_confusion_matrix.csv"
        )
    )

    # --------------------------------------------------------
    # Save out-of-fold predictions
    # --------------------------------------------------------

    for true_label, predicted_label in zip(
        y_true_all,
        y_pred_all
    ):

        all_oof_predictions.append(
            {
                "model": model_name,
                "true_label": true_label,
                "predicted_label": predicted_label
            }
        )


# ============================================================
# 18. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_df = results_df.sort_values(
    "macro_f1",
    ascending=False
).reset_index(
    drop=True
)


print("\n" + "=" * 80)
print("OVERALL CROSS-VALIDATION RESULTS")
print("=" * 80)

print(
    results_df.to_string(
        index=False
    )
)


results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_comparison.csv"
    ),
    index=False
)


# ============================================================
# 19. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 80)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 80)

rf_model = clone(
    random_forest_model
)

rf_model.fit(
    X,
    y
)

rf_classifier = (
    rf_model.named_steps[
        "classifier"
    ]
)

importance_df = pd.DataFrame(
    {
        "feature": FEATURES,
        "importance":
            rf_classifier.feature_importances_
    }
)

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
).reset_index(
    drop=True
)

print(
    importance_df.to_string(
        index=False
    )
)

importance_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "random_forest_feature_importance.csv"
    ),
    index=False
)


# ============================================================
# 20. LOGISTIC REGRESSION COEFFICIENTS
# ============================================================

print("\n" + "=" * 80)
print("LOGISTIC REGRESSION COEFFICIENTS")
print("=" * 80)

lr_model = clone(
    logistic_model
)

lr_model.fit(
    X,
    y
)

lr_classifier = (
    lr_model.named_steps[
        "classifier"
    ]
)

coefficients = np.asarray(
    lr_classifier.coef_
)

classes = list(
    lr_classifier.classes_
)

coefficient_rows = []

for class_index, class_name in enumerate(
    classes
):

    for feature_index, feature_name in enumerate(
        FEATURES
    ):

        coefficient = coefficients[
            class_index,
            feature_index
        ]

        coefficient_rows.append(
            {
                "class": class_name,
                "feature": feature_name,
                "coefficient": coefficient,
                "absolute_coefficient":
                    abs(coefficient)
            }
        )


coefficient_df = pd.DataFrame(
    coefficient_rows
)

coefficient_df = coefficient_df.sort_values(
    [
        "class",
        "absolute_coefficient"
    ],
    ascending=[
        True,
        False
    ]
)

print(
    coefficient_df.to_string(
        index=False
    )
)

coefficient_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "logistic_regression_coefficients.csv"
    ),
    index=False
)


# ============================================================
# 21. OUT-OF-FOLD PREDICTIONS
# ============================================================

oof_df = pd.DataFrame(
    all_oof_predictions
)

oof_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "out_of_fold_predictions.csv"
    ),
    index=False
)


# ============================================================
# 22. BEST MODEL
# ============================================================

best_model = results_df.iloc[
    0
]


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("PILOT VERIFIER CONCLUSION")
print("=" * 80)

print(
    f"Best model: {best_model['model']}"
)

print(
    f"Accuracy: {best_model['accuracy']:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{best_model['balanced_accuracy']:.4f}"
)

print(
    f"Macro Precision: "
    f"{best_model['macro_precision']:.4f}"
)

print(
    f"Macro Recall: "
    f"{best_model['macro_recall']:.4f}"
)

print(
    f"Macro F1: "
    f"{best_model['macro_f1']:.4f}"
)

print()
print(
    "IMPORTANT: These are exploratory cross-validation "
    "results on the 63 human-reviewed pilot candidates."
)

print(
    "They are NOT final model performance numbers."
)


# ============================================================
# 24. SAVE SIMPLE TEXT SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "pilot_verifier_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "SOLARMAP PILOT VERIFIER SUMMARY\n"
    )

    file.write(
        "=" * 60
        + "\n\n"
    )

    file.write(
        f"Rows used: {len(df)}\n"
    )

    file.write(
        f"Cross-validation folds: {n_splits}\n\n"
    )

    file.write(
        "MODEL RESULTS\n"
    )

    file.write(
        results_df.to_string(
            index=False
        )
    )

    file.write(
        "\n\n"
    )

    file.write(
        f"Best model: {best_model['model']}\n"
    )

    file.write(
        f"Best Macro F1: "
        f"{best_model['macro_f1']:.4f}\n"
    )


# ============================================================
# 25. FINISH
# ============================================================

print("\n" + "=" * 80)
print("STEP 8 COMPLETE")
print("=" * 80)

print()
print(
    "Results saved in:"
)

print(
    OUTPUT_DIR
)

print()
print(
    "Main files:"
)

print(
    "  model_comparison.csv"
)

print(
    "  random_forest_feature_importance.csv"
)

print(
    "  logistic_regression_coefficients.csv"
)

print(
    "  out_of_fold_predictions.csv"
)

print(
    "  pilot_verifier_summary.txt"
)

print()
print(
    "✅ Pilot verifier training and evaluation finished."
)