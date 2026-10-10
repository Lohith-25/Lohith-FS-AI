import io
import json
import os
import zipfile

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
)
from torchvision.models.detection.mask_rcnn import (
    MaskRCNNPredictor,
)
from pycocotools import mask as mask_utils


BASE = r"C:\SolarMap-India"

ZIP_PATH = os.path.join(
    BASE, "dataset", "Solar Images.zip"
)

SPLIT_DIR = os.path.join(
    BASE, "splits"
)

OUT_DIR = os.path.join(
    BASE, "results", "validation_threshold_selection"
)

MODELS = {
    "baseline": os.path.join(
        BASE,
        "checkpoints",
        "maskrcnn_baseline_epoch_10.pth",
    ),
    "background_aware": os.path.join(
        BASE,
        "checkpoints",
        "background_aware",
        "background_aware_epoch_10.pth",
    ),
    "hard_negative_replay": os.path.join(
        BASE,
        "checkpoints",
        "hard_negative_replay",
        "hard_negative_replay_epoch_10.pth",
    ),
}

COCO_MEMBER = (
    "Solar Images/annotations/"
    "merged_instances_default.json"
)

IOU_THRESHOLD = 0.50

THRESHOLDS = [
    round(x, 2)
    for x in np.arange(
        0.50,
        1.001,
        0.01,
    )
]


def build_model():
    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
    )

    box_in = (
        model.roi_heads
        .box_predictor
        .cls_score
        .in_features
    )

    model.roi_heads.box_predictor = (
        FastRCNNPredictor(
            box_in,
            2,
        )
    )

    mask_in = (
        model.roi_heads
        .mask_predictor
        .conv5_mask
        .in_channels
    )

    model.roi_heads.mask_predictor = (
        MaskRCNNPredictor(
            mask_in,
            256,
            2,
        )
    )

    return model


def load_checkpoint(
    model,
    path,
    device,
):
    checkpoint = torch.load(
        path,
        map_location=device,
    )

    state = checkpoint

    if isinstance(
        checkpoint,
        dict,
    ):
        if "model_state_dict" in checkpoint:
            state = checkpoint[
                "model_state_dict"
            ]
        elif "state_dict" in checkpoint:
            state = checkpoint[
                "state_dict"
            ]

    cleaned = {}

    for key, value in state.items():
        if key.startswith("module."):
            key = key[7:]

        cleaned[key] = value

    missing, unexpected = (
        model.load_state_dict(
            cleaned,
            strict=False,
        )
    )

    if missing:
        raise RuntimeError(
            "Missing checkpoint keys:\n"
            + "\n".join(
                missing[:20]
            )
        )

    if unexpected:
        print(
            f"WARNING: {len(unexpected)} "
            "unexpected checkpoint keys"
        )

    return checkpoint


class ZipStore:
    def __init__(
        self,
        path,
    ):
        self.zf = zipfile.ZipFile(
            path,
            "r",
        )

        self.members = {
            os.path.basename(name): name
            for name in self.zf.namelist()
            if name.lower().endswith(
                ".png"
            )
        }

    def read(
        self,
        filename,
    ):
        data = self.zf.read(
            self.members[filename]
        )

        return Image.open(
            io.BytesIO(data)
        ).convert("RGB")


def load_coco():
    with zipfile.ZipFile(
        ZIP_PATH,
        "r",
    ) as zf:
        return json.loads(
            zf.read(COCO_MEMBER)
        )


def build_maps(coco):
    image_map = {}
    ann_map = {}

    for image in coco["images"]:
        image_map[
            os.path.basename(
                image["file_name"]
            )
        ] = image

    for ann in coco["annotations"]:
        ann_map.setdefault(
            ann["image_id"],
            [],
        ).append(ann)

    return image_map, ann_map


def segmentation_to_mask(
    segmentation,
    h,
    w,
):
    if isinstance(
        segmentation,
        list,
    ):
        polygons = [
            p
            for p in segmentation
            if isinstance(
                p,
                (list, tuple),
            )
            and len(p) >= 6
        ]

        if not polygons:
            return np.zeros(
                (h, w),
                dtype=np.uint8,
            )

        rles = mask_utils.frPyObjects(
            polygons,
            h,
            w,
        )

        decoded = mask_utils.decode(
            mask_utils.merge(rles)
        )

    elif isinstance(
        segmentation,
        dict,
    ):
        decoded = mask_utils.decode(
            segmentation
        )

    else:
        return np.zeros(
            (h, w),
            dtype=np.uint8,
        )

    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]

    return decoded.astype(
        np.uint8
    )


def get_gt(
    filename,
    image_map,
    ann_map,
    h,
    w,
):
    image_record = image_map.get(
        filename
    )

    if image_record is None:
        return []

    masks = []

    for ann in ann_map.get(
        image_record["id"],
        [],
    ):
        mask = segmentation_to_mask(
            ann.get(
                "segmentation",
                [],
            ),
            h,
            w,
        )

        if int(mask.sum()) > 0:
            masks.append(mask)

    return masks


def encode_masks(
    masks,
):
    if len(masks) == 0:
        return []

    return [
        mask_utils.encode(
            np.asfortranarray(
                mask.astype(
                    np.uint8
                )
            )
        )
        for mask in masks
    ]


def match_masks(
    pred_masks,
    pred_scores,
    gt_masks,
):
    n_pred = len(
        pred_masks
    )

    n_gt = len(
        gt_masks
    )

    if n_pred == 0:
        return 0, 0, n_gt

    if n_gt == 0:
        return 0, n_pred, 0

    iou_matrix = mask_utils.iou(
        encode_masks(
            pred_masks
        ),
        encode_masks(
            gt_masks
        ),
        [0] * n_gt,
    )

    order = np.argsort(
        -np.asarray(
            pred_scores
        )
    )

    used_gt = set()

    tp = 0
    fp = 0

    for pred_idx in order:
        candidates = [
            gt_idx
            for gt_idx in range(
                n_gt
            )
            if gt_idx not in used_gt
        ]

        if len(candidates) == 0:
            fp += 1
            continue

        best_gt = max(
            candidates,
            key=lambda gt_idx: float(
                iou_matrix[
                    pred_idx,
                    gt_idx
                ]
            ),
        )

        best_iou = float(
            iou_matrix[
                pred_idx,
                best_gt
            ]
        )

        if best_iou >= IOU_THRESHOLD:
            tp += 1
            used_gt.add(
                best_gt
            )
        else:
            fp += 1

    fn = n_gt - tp

    return tp, fp, fn


def predict(
    model,
    store,
    filename,
    device,
):
    image = store.read(
        filename
    )

    array = np.array(
        image,
        copy=True,
    )

    tensor = (
        torch.from_numpy(
            array.transpose(
                2,
                0,
                1,
            )
        ).float()
        / 255.0
    )

    with torch.inference_mode():
        output = model(
            [tensor.to(device)]
        )[0]

    scores = (
        output["scores"]
        .detach()
        .cpu()
        .numpy()
    )

    masks = (
        output["masks"][:, 0]
        .detach()
        .cpu()
        .numpy()
        >= 0.5
    )

    return (
        image,
        scores,
        masks,
    )


def evaluate_thresholds(
    positive_cache,
    negative_cache,
    positive_files,
    negative_files,
    model_name,
):
    rows = []

    for threshold in THRESHOLDS:
        tp_total = 0
        fp_total = 0
        fn_total = 0

        count_errors = []

        for filename in positive_files:
            scores, masks, gt = (
                positive_cache[
                    filename
                ]
            )

            keep = (
                scores >= threshold
            )

            pred_scores = (
                scores[keep]
            )

            pred_masks = (
                masks[keep]
            )

            tp, fp, fn = (
                match_masks(
                    pred_masks,
                    pred_scores,
                    gt,
                )
            )

            tp_total += tp
            fp_total += fp
            fn_total += fn

            count_errors.append(
                len(pred_masks)
                - len(gt)
            )

        precision = (
            tp_total
            / (
                tp_total
                + fp_total
            )
            if tp_total + fp_total
            else 0.0
        )

        recall = (
            tp_total
            / (
                tp_total
                + fn_total
            )
            if tp_total + fn_total
            else 0.0
        )

        f1 = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
            if precision + recall
            else 0.0
        )

        count_mae = float(
            np.mean(
                np.abs(
                    count_errors
                )
            )
        )

        negative_fp_images = 0
        negative_predictions = 0

        for filename in negative_files:
            scores, masks = (
                negative_cache[
                    filename
                ]
            )

            keep = (
                scores >= threshold
            )

            n_pred = int(
                keep.sum()
            )

            if n_pred > 0:
                negative_fp_images += 1
                negative_predictions += n_pred

        negative_fp_rate = (
            negative_fp_images
            / len(negative_files)
            * 100.0
        )

        rows.append({
            "model": model_name,
            "threshold": threshold,
            "TP": tp_total,
            "FP": fp_total,
            "FN": fn_total,
            "precision": precision,
            "recall": recall,
            "F1": f1,
            "count_MAE": count_mae,
            "negative_FP_images": negative_fp_images,
            "negative_total_images": len(
                negative_files
            ),
            "negative_FP_rate_percent": (
                negative_fp_rate
            ),
            "negative_total_predictions": (
                negative_predictions
            ),
        })

    return rows


def main():
    os.makedirs(
        OUT_DIR,
        exist_ok=True,
    )

    required_paths = [
        ZIP_PATH,
        *MODELS.values(),
        os.path.join(
            SPLIT_DIR,
            "positive_val.csv",
        ),
        os.path.join(
            SPLIT_DIR,
            "negative_val.csv",
        ),
    ]

    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(
                path
            )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"PyTorch: {torch.__version__}"
    )

    print(
        f"TorchVision: "
        f"{torchvision.__version__}"
    )

    print(
        f"CUDA: "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    positive_val = pd.read_csv(
        os.path.join(
            SPLIT_DIR,
            "positive_val.csv",
        )
    )

    negative_val = pd.read_csv(
        os.path.join(
            SPLIT_DIR,
            "negative_val.csv",
        )
    )

    positive_files = (
        positive_val[
            "filename"
        ]
        .astype(str)
        .tolist()
    )

    negative_files = (
        negative_val[
            "filename"
        ]
        .astype(str)
        .tolist()
    )

    if len(positive_files) != 249:
        raise RuntimeError(
            "Expected exactly 249 "
            "positive validation images."
        )

    if len(negative_files) != 75:
        raise RuntimeError(
            "Expected exactly 75 "
            "negative validation images."
        )

    print(
        "\nVALIDATION ONLY:"
    )

    print(
        "  Positive:",
        len(positive_files),
    )

    print(
        "  Negative:",
        len(negative_files),
    )

    print(
        "  Locked test set: NOT USED"
    )

    coco = load_coco()

    image_map, ann_map = build_maps(
        coco
    )

    store = ZipStore(
        ZIP_PATH
    )

    all_rows = []
    selected_rows = []

    for model_name, checkpoint in MODELS.items():
        print(
            "\n" + "=" * 80
        )

        print(
            "MODEL:",
            model_name,
        )

        model = build_model().to(
            device
        )

        load_checkpoint(
            model,
            checkpoint,
            device,
        )

        model.eval()

        positive_cache = {}
        negative_cache = {}

        for i, filename in enumerate(
            positive_files,
            1,
        ):
            image, scores, masks = (
                predict(
                    model,
                    store,
                    filename,
                    device,
                )
            )

            gt = get_gt(
                filename,
                image_map,
                ann_map,
                image.height,
                image.width,
            )

            positive_cache[
                filename
            ] = (
                scores,
                masks,
                gt,
            )

            print(
                f"\rPositive "
                f"{i}/"
                f"{len(positive_files)}",
                end="",
                flush=True,
            )

        print()

        for i, filename in enumerate(
            negative_files,
            1,
        ):
            _, scores, masks = (
                predict(
                    model,
                    store,
                    filename,
                    device,
                )
            )

            negative_cache[
                filename
            ] = (
                scores,
                masks,
            )

            print(
                f"\rNegative "
                f"{i}/"
                f"{len(negative_files)}",
                end="",
                flush=True,
            )

        print()

        model_rows = evaluate_thresholds(
            positive_cache,
            negative_cache,
            positive_files,
            negative_files,
            model_name,
        )

        model_df = pd.DataFrame(
            model_rows
        )

        all_rows.extend(
            model_rows
        )

        model_df.to_csv(
            os.path.join(
                OUT_DIR,
                f"{model_name}_"
                "validation_thresholds.csv",
            ),
            index=False,
        )

        ranked = model_df.sort_values(
            by=[
                "F1",
                "negative_FP_rate_percent",
                "recall",
            ],
            ascending=[
                False,
                True,
                False,
            ],
        )

        best = ranked.iloc[0]

        selected_rows.append({
            "model": model_name,
            "selected_threshold": float(
                best["threshold"]
            ),
            "validation_F1": float(
                best["F1"]
            ),
            "validation_precision": float(
                best["precision"]
            ),
            "validation_recall": float(
                best["recall"]
            ),
            "validation_count_MAE": float(
                best["count_MAE"]
            ),
            "validation_negative_FP_rate_percent": float(
                best[
                    "negative_FP_rate_percent"
                ]
            ),
        })

        print(
            "\nSelected threshold:",
            f"{best['threshold']:.2f}",
        )

        print(
            "Validation F1:",
            f"{best['F1']:.4f}",
        )

        print(
            "Validation precision:",
            f"{best['precision']:.4f}",
        )

        print(
            "Validation recall:",
            f"{best['recall']:.4f}",
        )

        print(
            "Validation negative FP rate:",
            f"{best['negative_FP_rate_percent']:.4f}%",
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_df = pd.DataFrame(
        all_rows
    )

    all_df.to_csv(
        os.path.join(
            OUT_DIR,
            "all_validation_threshold_results.csv",
        ),
        index=False,
    )

    selected_df = pd.DataFrame(
        selected_rows
    )

    selected_path = os.path.join(
        OUT_DIR,
        "selected_validation_thresholds.csv",
    )

    selected_df.to_csv(
        selected_path,
        index=False,
    )

    print(
        "\n" + "=" * 90
    )

    print(
        "FINAL VALIDATION-BASED "
        "THRESHOLD SELECTION"
    )

    print(
        "=" * 90
    )

    print(
        selected_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        "\nSaved to:"
    )

    print(
        OUT_DIR
    )


if __name__ == "__main__":
    main()
