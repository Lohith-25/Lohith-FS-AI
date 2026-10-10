import io
import json
import os
import zipfile

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


# ============================================================
# SOLARMAP — STEP 13
# HARD-NEGATIVE MINING ON TRAINING NEGATIVES ONLY
#
# IMPORTANT:
#   This script NEVER touches the locked test set.
#
# Goal:
#   Run the original baseline model on the 300 negative TRAIN
#   images and identify which "no-solar" backgrounds the baseline
#   incorrectly predicts as solar panels.
#
# Primary hard-negative threshold:
#   confidence >= 0.75
#
# Output:
#   C:\SolarMap-India\results\hard_negative_mining\
# ============================================================


BASE = r"C:\SolarMap-India"

ZIP_PATH = os.path.join(
    BASE,
    "dataset",
    "Solar Images.zip",
)

SPLIT_DIR = os.path.join(
    BASE,
    "splits",
)

BASELINE_CHECKPOINT = os.path.join(
    BASE,
    "checkpoints",
    "maskrcnn_baseline_epoch_10.pth",
)

OUTPUT_DIR = os.path.join(
    BASE,
    "results",
    "hard_negative_mining",
)

NUM_CLASSES = 2
SCORE_THRESHOLD = 0.75


def build_model():
    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
    )

    box_features = (
        model.roi_heads.box_predictor.cls_score.in_features
    )

    model.roi_heads.box_predictor = FastRCNNPredictor(
        box_features,
        NUM_CLASSES,
    )

    mask_features = (
        model.roi_heads.mask_predictor.conv5_mask.in_channels
    )

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        mask_features,
        256,
        NUM_CLASSES,
    )

    return model


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    state = checkpoint

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]

    cleaned = {}

    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value

    missing, unexpected = model.load_state_dict(
        cleaned,
        strict=False,
    )

    if missing:
        raise RuntimeError(
            "Missing checkpoint parameters:\n"
            + "\n".join(missing[:30])
        )

    if unexpected:
        print(
            f"WARNING: {len(unexpected)} unexpected checkpoint keys."
        )

    return checkpoint


class ZipStore:
    def __init__(self, zip_path):
        self.zf = zipfile.ZipFile(
            zip_path,
            "r",
        )

        self.members = {}

        for name in self.zf.namelist():
            if name.lower().endswith(".png"):
                self.members[
                    os.path.basename(name)
                ] = name

    def read(self, filename):
        if filename not in self.members:
            raise FileNotFoundError(
                f"Image not found in ZIP: {filename}"
            )

        data = self.zf.read(
            self.members[filename]
        )

        return Image.open(
            io.BytesIO(data)
        ).convert("RGB")


def predict_one(model, store, filename, device):
    image = store.read(filename)

    image_np = np.array(
        image,
        copy=True,
    )

    image_tensor = (
        torch.from_numpy(
            image_np.transpose(2, 0, 1)
        ).float()
        / 255.0
    )

    with torch.inference_mode():
        output = model(
            [image_tensor.to(device)]
        )[0]

    scores = (
        output["scores"]
        .detach()
        .cpu()
        .numpy()
    )

    return scores


def main():
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    negative_train_path = os.path.join(
        SPLIT_DIR,
        "negative_train.csv",
    )

    required = [
        ZIP_PATH,
        BASELINE_CHECKPOINT,
        negative_train_path,
    ]

    for path in required:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print("SOLARMAP — HARD-NEGATIVE MINING")
    print("=" * 80)

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    negative_df = pd.read_csv(
        negative_train_path
    )

    if len(negative_df) != 300:
        raise RuntimeError(
            f"Expected exactly 300 negative training images; "
            f"found {len(negative_df)}"
        )

    filenames = (
        negative_df["filename"]
        .astype(str)
        .tolist()
    )

    if len(set(filenames)) != 300:
        raise RuntimeError(
            "Duplicate filenames found in negative_train.csv"
        )

    print(
        f"Negative training images: {len(filenames)}"
    )

    print("\nBuilding baseline model...")
    model = build_model().to(device)

    print("Loading baseline checkpoint...")
    checkpoint = load_checkpoint(
        model,
        BASELINE_CHECKPOINT,
        device,
    )

    model.eval()

    if isinstance(checkpoint, dict):
        if checkpoint.get("epoch") is not None:
            print(
                f"Checkpoint epoch: {checkpoint['epoch']}"
            )

    store = ZipStore(
        ZIP_PATH
    )

    rows = []

    total_predictions = 0
    hard_negative_count = 0

    for idx, filename in enumerate(
        filenames,
        start=1,
    ):
        scores = predict_one(
            model,
            store,
            filename,
            device,
        )

        high_scores = scores[
            scores >= SCORE_THRESHOLD
        ]

        total_count = len(scores)
        high_count = len(high_scores)

        max_conf = (
            float(scores.max())
            if total_count > 0
            else 0.0
        )

        mean_conf_all = (
            float(scores.mean())
            if total_count > 0
            else 0.0
        )

        mean_conf_high = (
            float(high_scores.mean())
            if high_count > 0
            else 0.0
        )

        # More false detections + higher confidence => higher priority.
        priority_score = (
            high_count * mean_conf_high
        )

        is_hard_negative = (
            high_count > 0
        )

        if is_hard_negative:
            hard_negative_count += 1

        total_predictions += high_count

        rows.append({
            "filename": filename,
            "sampleid": negative_df.iloc[idx - 1].get(
                "sampleid",
                np.nan,
            ),
            "score_threshold": SCORE_THRESHOLD,
            "raw_prediction_count": total_count,
            "high_confidence_prediction_count": high_count,
            "max_confidence": max_conf,
            "mean_confidence_all_predictions": mean_conf_all,
            "mean_confidence_high_predictions": mean_conf_high,
            "priority_score": priority_score,
            "is_hard_negative": is_hard_negative,
        })

        print(
            f"\rProcessed {idx}/{len(filenames)}",
            end="",
            flush=True,
        )

    print()

    result = pd.DataFrame(rows)

    result = result.sort_values(
        [
            "is_hard_negative",
            "priority_score",
            "max_confidence",
            "high_confidence_prediction_count",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    all_path = os.path.join(
        OUTPUT_DIR,
        "negative_train_mining_results.csv",
    )

    hard_only = result[
        result["is_hard_negative"]
    ].copy()

    hard_path = os.path.join(
        OUTPUT_DIR,
        "hard_negative_train.csv",
    )

    result.to_csv(
        all_path,
        index=False,
    )

    hard_only.to_csv(
        hard_path,
        index=False,
    )

    summary = {
        "baseline_checkpoint": BASELINE_CHECKPOINT,
        "source_split": negative_train_path,
        "source_image_count": 300,
        "hard_negative_threshold": SCORE_THRESHOLD,
        "hard_negative_image_count": int(
            hard_negative_count
        ),
        "hard_negative_rate_percent": float(
            hard_negative_count / 300 * 100.0
        ),
        "high_confidence_prediction_count": int(
            total_predictions
        ),
        "mean_high_confidence_predictions_per_image": float(
            total_predictions / 300
        ),
        "test_set_used": False,
        "locked_test_protected": True,
    }

    summary_path = os.path.join(
        OUTPUT_DIR,
        "hard_negative_mining_summary.json",
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("HARD-NEGATIVE MINING COMPLETE")
    print("=" * 80)

    print(
        f"Negative training images: {len(filenames)}"
    )
    print(
        f"Hard negatives (>= {SCORE_THRESHOLD:.2f}): "
        f"{hard_negative_count}"
    )
    print(
        f"Hard-negative rate: "
        f"{hard_negative_count / 300 * 100:.2f}%"
    )
    print(
        f"High-confidence false predictions: "
        f"{total_predictions}"
    )

    print("\nTop 20 hard negatives:")
    print(
        hard_only[
            [
                "filename",
                "high_confidence_prediction_count",
                "max_confidence",
                "mean_confidence_high_predictions",
                "priority_score",
            ]
        ].head(20).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nSaved:")
    print(all_path)
    print(hard_path)
    print(summary_path)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
