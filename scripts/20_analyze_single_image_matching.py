import io
import json
import os
import zipfile

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from pycocotools import mask as mask_utils


BASE = r"C:\SolarMap-India"
ZIP_PATH = os.path.join(BASE, "dataset", "Solar Images.zip")
CHECKPOINT = os.path.join(
    BASE, "checkpoints", "maskrcnn_baseline_epoch_10.pth"
)
COCO_MEMBER = "Solar Images/annotations/merged_instances_default.json"
OUTPUT_DIR = os.path.join(
    BASE, "results", "single_image_error_analysis"
)

FILENAME = "100.0_1.0.png"
THRESHOLD = 0.77
IOU_THRESHOLD = 0.50


def build_model():
    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
    )

    box_in = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_in, 2)

    mask_in = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        mask_in, 256, 2
    )

    return model


def load_model(device):
    model = build_model().to(device)

    ckpt = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    state = ckpt
    if isinstance(ckpt, dict):
        state = ckpt.get(
            "model_state_dict",
            ckpt.get("state_dict", ckpt),
        )

    state = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state.items()
    }

    missing, _ = model.load_state_dict(
        state,
        strict=False,
    )

    if missing:
        raise RuntimeError(
            "Missing checkpoint parameters."
        )

    model.eval()
    return model


def load_image_and_coco():
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        member = next(
            (
                n for n in zf.namelist()
                if n.lower().endswith(".png")
                and os.path.basename(n) == FILENAME
            ),
            None,
        )

        if member is None:
            raise FileNotFoundError(
                f"{FILENAME} not found in ZIP."
            )

        image = Image.open(
            io.BytesIO(zf.read(member))
        ).convert("RGB")

        coco = json.loads(
            zf.read(COCO_MEMBER)
        )

    return image, coco


def polygon_to_mask(segmentation, h, w):
    if not isinstance(segmentation, list):
        return np.zeros((h, w), dtype=np.uint8)

    polygons = [
        p for p in segmentation
        if isinstance(p, (list, tuple)) and len(p) >= 6
    ]

    if not polygons:
        return np.zeros((h, w), dtype=np.uint8)

    rles = mask_utils.frPyObjects(
        polygons,
        h,
        w,
    )

    mask = mask_utils.decode(
        mask_utils.merge(rles)
    )

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    return mask.astype(np.uint8)


def get_gt(coco, h, w):
    image_record = None

    for image in coco["images"]:
        if os.path.basename(image["file_name"]) == FILENAME:
            image_record = image
            break

    if image_record is None:
        raise RuntimeError(
            f"No COCO image record found for {FILENAME}"
        )

    gt_masks = []

    for ann in coco["annotations"]:
        if ann["image_id"] != image_record["id"]:
            continue

        mask = polygon_to_mask(
            ann.get("segmentation", []),
            h,
            w,
        )

        if int(mask.sum()) > 0:
            gt_masks.append(mask)

    return gt_masks


def encode_masks(masks):
    return [
        mask_utils.encode(
            np.asfortranarray(m.astype(np.uint8))
        )
        for m in masks
    ]


def predict(model, image, device):
    arr = np.array(image, copy=True)

    tensor = (
        torch.from_numpy(
            arr.transpose(2, 0, 1)
        ).float()
        / 255.0
    )

    with torch.inference_mode():
        output = model(
            [tensor.to(device)]
        )[0]

    scores = output["scores"].detach().cpu().numpy()
    boxes = output["boxes"].detach().cpu().numpy()
    masks = (
        output["masks"][:, 0].detach().cpu().numpy()
        >= 0.5
    )

    keep = scores >= THRESHOLD

    return (
        boxes[keep],
        scores[keep],
        masks[keep],
    )


def match_predictions(pred_masks, pred_scores, gt_masks):
    if len(pred_masks) == 0:
        return [], set(range(len(gt_masks)))

    if len(gt_masks) == 0:
        return [
            {
                "pred_index": i + 1,
                "gt_index": None,
                "iou": 0.0,
                "type": "FP",
            }
            for i in range(len(pred_masks))
        ], set()

    matrix = mask_utils.iou(
        encode_masks(pred_masks),
        encode_masks(gt_masks),
        [0] * len(gt_masks),
    )

    order = np.argsort(
        -np.asarray(pred_scores)
    )

    used_gt = set()
    matches = []

    for pred_idx in order:
        candidates = [
            gt_idx
            for gt_idx in range(len(gt_masks))
            if gt_idx not in used_gt
        ]

        if len(candidates) == 0:
            matches.append({
                "pred_index": int(pred_idx) + 1,
                "gt_index": None,
                "iou": 0.0,
                "type": "FP",
            })
            continue

        best_gt = max(
            candidates,
            key=lambda gt_idx: float(
                matrix[pred_idx, gt_idx]
            ),
        )

        best_iou = float(
            matrix[pred_idx, best_gt]
        )

        if best_iou >= IOU_THRESHOLD:
            used_gt.add(best_gt)
            match_type = "TP"
        else:
            match_type = "FP"

        matches.append({
            "pred_index": int(pred_idx) + 1,
            "gt_index": int(best_gt) + 1,
            "iou": best_iou,
            "type": match_type,
        })

    missed_gt = (
        set(range(len(gt_masks)))
        - used_gt
    )

    return matches, missed_gt


def mask_bbox(mask):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
    )


def color_mask(base, mask, color, alpha=90):
    overlay = np.zeros(
        (mask.shape[0], mask.shape[1], 4),
        dtype=np.uint8,
    )

    overlay[mask, 0] = color[0]
    overlay[mask, 1] = color[1]
    overlay[mask, 2] = color[2]
    overlay[mask, 3] = alpha

    return Image.alpha_composite(
        base,
        Image.fromarray(
            overlay,
            mode="RGBA",
        ),
    )


def main():
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print("SOLARMAP — SINGLE IMAGE ERROR ANALYSIS")
    print("=" * 80)

    print("Image:", FILENAME)
    print("Threshold:", THRESHOLD)
    print("IoU match threshold:", IOU_THRESHOLD)
    print("Device:", device)

    image, coco = load_image_and_coco()

    print("Image size:", image.size)

    gt_masks = get_gt(
        coco,
        image.height,
        image.width,
    )

    print(
        "Ground-truth instances:",
        len(gt_masks),
    )

    model = load_model(device)

    pred_boxes, pred_scores, pred_masks = predict(
        model,
        image,
        device,
    )

    print(
        "Predicted instances:",
        len(pred_masks),
    )

    matches, missed_gt = match_predictions(
        pred_masks,
        pred_scores,
        gt_masks,
    )

    tp = sum(
        1 for x in matches
        if x["type"] == "TP"
    )

    fp = sum(
        1 for x in matches
        if x["type"] == "FP"
    )

    fn = len(missed_gt)

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    print("\nMATCHING RESULT")
    print("-" * 80)
    print("TP:", tp)
    print("FP:", fp)
    print("FN:", fn)
    print("Precision:", f"{precision:.4f}")
    print("Recall:", f"{recall:.4f}")
    print("F1:", f"{f1:.4f}")

    rows = []

    for match in matches:
        pidx = match["pred_index"] - 1
        gt_index = match["gt_index"]

        area = int(
            pred_masks[pidx].sum()
        )

        x1, y1, x2, y2 = (
            pred_boxes[pidx].tolist()
        )

        rows.append({
            "prediction_index": match["pred_index"],
            "confidence": float(
                pred_scores[pidx]
            ),
            "best_gt_index": gt_index,
            "mask_iou": match["iou"],
            "classification": match["type"],
            "mask_area_pixels": area,
            "box_x1": x1,
            "box_y1": y1,
            "box_x2": x2,
            "box_y2": y2,
        })

    details = pd.DataFrame(rows)

    details_path = os.path.join(
        OUTPUT_DIR,
        "100_0_prediction_matching.csv",
    )

    details.to_csv(
        details_path,
        index=False,
    )

    missed_rows = []

    for gt_index_zero in sorted(
        missed_gt
    ):
        area = int(
            gt_masks[gt_index_zero].sum()
        )

        missed_rows.append({
            "gt_index": gt_index_zero + 1,
            "mask_area_pixels": area,
            "bbox": mask_bbox(
                gt_masks[gt_index_zero]
            ),
        })

    missed_path = os.path.join(
        OUTPUT_DIR,
        "100_0_missed_ground_truth.csv",
    )

    pd.DataFrame(
        missed_rows
    ).to_csv(
        missed_path,
        index=False,
    )

    summary = pd.DataFrame([{
        "filename": FILENAME,
        "threshold": THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "ground_truth_instances": len(gt_masks),
        "predicted_instances": len(pred_masks),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "count_error": len(pred_masks) - len(gt_masks),
    }])

    summary_path = os.path.join(
        OUTPUT_DIR,
        "100_0_summary.csv",
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # Build a clear diagnostic visualization:
    # Green = all ground truth.
    # Blue  = true-positive prediction.
    # Red   = false-positive prediction.
    # Yellow outline = missed GT.
    canvas = image.convert("RGBA")

    for mask in gt_masks:
        canvas = color_mask(
            canvas,
            mask.astype(bool),
            (0, 255, 0),
            alpha=55,
        )

    for match in matches:
        pidx = match["pred_index"] - 1

        if match["type"] == "TP":
            canvas = color_mask(
                canvas,
                pred_masks[pidx],
                (0, 120, 255),
                alpha=90,
            )
        else:
            canvas = color_mask(
                canvas,
                pred_masks[pidx],
                (255, 0, 0),
                alpha=100,
            )

    draw = ImageDraw.Draw(canvas)

    for idx, mask in enumerate(
        gt_masks,
        start=1,
    ):
        bbox = mask_bbox(mask)

        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox

        if (idx - 1) in missed_gt:
            outline = (255, 255, 0, 255)
            width = 4
            label = f"FN GT#{idx}"
        else:
            outline = (0, 255, 0, 255)
            width = 2
            label = f"GT#{idx}"

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=outline,
            width=width,
        )

        draw.text(
            [x1 + 2, y1 + 2],
            label,
            fill=outline,
        )

    for match in matches:
        pidx = match["pred_index"] - 1
        x1, y1, x2, y2 = (
            pred_boxes[pidx].tolist()
        )

        if match["type"] == "TP":
            outline = (0, 120, 255, 255)
            label = (
                f"TP P#{match['pred_index']} "
                f"IoU={match['iou']:.2f}"
            )
        else:
            outline = (255, 0, 0, 255)
            label = (
                f"FP P#{match['pred_index']} "
                f"{pred_scores[pidx]:.2f}"
            )

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=outline,
            width=3,
        )

        draw.text(
            [x1 + 2, max(0, y1 - 15)],
            label,
            fill=outline,
        )

    # Legend.
    draw.rectangle(
        [5, 5, 230, 65],
        fill=(0, 0, 0, 190),
    )

    draw.text(
        [10, 10],
        "GREEN = Ground Truth",
        fill=(0, 255, 0, 255),
    )

    draw.text(
        [10, 27],
        "BLUE = True Positive",
        fill=(0, 120, 255, 255),
    )

    draw.text(
        [10, 44],
        "RED = False Positive / YELLOW = Missed GT",
        fill=(255, 255, 255, 255),
    )

    visual_path = os.path.join(
        OUTPUT_DIR,
        "100_0_error_analysis.png",
    )

    canvas.convert("RGB").save(
        visual_path
    )

    print("\nSaved:")
    print(details_path)
    print(missed_path)
    print(summary_path)
    print(visual_path)


if __name__ == "__main__":
    main()
