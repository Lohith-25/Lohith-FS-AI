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

try:
    from pycocotools import mask as mask_utils
except ImportError as exc:
    raise ImportError(
        "pycocotools is required. Run: pip install pycocotools"
    ) from exc


BASE = r"C:\SolarMap-India"
ZIP_PATH = os.path.join(BASE, "dataset", "Solar Images.zip")
SPLIT_DIR = os.path.join(BASE, "splits")

BASELINE_CHECKPOINT = os.path.join(
    BASE, "checkpoints", "maskrcnn_baseline_epoch_10.pth"
)
BACKGROUND_AWARE_CHECKPOINT = os.path.join(
    BASE, "checkpoints", "background_aware",
    "background_aware_epoch_10.pth"
)

OUTPUT_DIR = os.path.join(BASE, "results", "final_comparison")
COCO_PATH = "Solar Images/annotations/merged_instances_default.json"

NUM_CLASSES = 2
PRIMARY_THRESHOLD = 0.75
THRESHOLDS = [0.50, 0.75, 0.90, 0.98]


def build_model():
    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
    )

    box_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        box_features, NUM_CLASSES
    )

    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        mask_features, 256, NUM_CLASSES
    )

    return model


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]

    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[7:]
        cleaned[key] = value

    missing, unexpected = model.load_state_dict(
        cleaned, strict=False
    )

    if missing:
        raise RuntimeError(
            "Checkpoint missing parameters:\n" +
            "\n".join(missing[:30])
        )

    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected checkpoint keys")

    return ckpt


def polygon_to_mask(segmentation, h, w):
    if isinstance(segmentation, list):
        polys = [
            p for p in segmentation
            if isinstance(p, (list, tuple)) and len(p) >= 6
        ]
        if not polys:
            return np.zeros((h, w), dtype=np.uint8)

        rles = mask_utils.frPyObjects(polys, h, w)
        decoded = mask_utils.decode(mask_utils.merge(rles))
    elif isinstance(segmentation, dict):
        decoded = mask_utils.decode(segmentation)
    else:
        return np.zeros((h, w), dtype=np.uint8)

    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]

    return decoded.astype(np.uint8)


class ZipStore:
    def __init__(self, path):
        self.zf = zipfile.ZipFile(path, "r")
        self.members = {
            os.path.basename(name): name
            for name in self.zf.namelist()
            if name.lower().endswith(".png")
        }

    def read(self, filename):
        if filename not in self.members:
            raise FileNotFoundError(
                f"Image not found in ZIP: {filename}"
            )

        data = self.zf.read(self.members[filename])
        return Image.open(io.BytesIO(data)).convert("RGB")


def load_coco():
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        return json.loads(zf.read(COCO_PATH))


def build_coco_maps(coco):
    image_map = {}
    ann_map = {}

    for image in coco["images"]:
        image_map[os.path.basename(image["file_name"])] = image

    for ann in coco["annotations"]:
        ann_map.setdefault(ann["image_id"], []).append(ann)

    return image_map, ann_map


def ground_truth(filename, image_map, ann_map, h, w):
    record = image_map.get(filename)
    if record is None:
        return []

    result = []
    for ann in ann_map.get(record["id"], []):
        m = polygon_to_mask(ann.get("segmentation", []), h, w)
        if int(m.sum()) > 0:
            result.append(m)
    return result


def encode_masks(masks):
    return [
        mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
        for m in masks
    ]


def match_masks(pred_masks, pred_scores, gt_masks, iou_threshold=0.50):
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)

    if n_pred == 0:
        return 0, 0, n_gt, []

    if n_gt == 0:
        return 0, n_pred, 0, []

    ious = mask_utils.iou(
        encode_masks(pred_masks),
        encode_masks(gt_masks),
        [0] * n_gt,
    )

    order = np.argsort(-np.asarray(pred_scores))
    used_gt = set()
    tp = fp = 0
    matched = []

    for pi in order:
        candidates = [
            gi for gi in range(n_gt)
            if gi not in used_gt
        ]

        if not candidates:
            fp += 1
            continue

        best_gt = max(
            candidates,
            key=lambda gi: float(ious[pi, gi])
        )
        best_iou = float(ious[pi, best_gt])

        if best_iou >= iou_threshold:
            tp += 1
            used_gt.add(best_gt)
            matched.append(best_iou)
        else:
            fp += 1

    fn = n_gt - tp
    return tp, fp, fn, matched


def predict(model, store, filename, device):
    image = store.read(filename)
    arr = np.array(image, copy=True)

    tensor = (
        torch.from_numpy(arr.transpose(2, 0, 1)).float() / 255.0
    )

    with torch.inference_mode():
        out = model([tensor.to(device)])[0]

    scores = out["scores"].detach().cpu().numpy()
    masks = (
        out["masks"][:, 0].detach().cpu().numpy() >= 0.5
    )

    return image, scores, masks


def evaluate_model(
    name,
    checkpoint,
    positive_files,
    negative_files,
    store,
    image_map,
    ann_map,
    device,
):
    print("\n" + "=" * 80)
    print(f"EVALUATING {name}")
    print("=" * 80)

    model = build_model().to(device)
    ckpt = load_checkpoint(model, checkpoint, device)
    model.eval()

    if isinstance(ckpt, dict) and "epoch" in ckpt:
        print(f"Checkpoint epoch: {ckpt['epoch']}")

    positive_cache = {}
    negative_cache = {}

    for i, filename in enumerate(positive_files, 1):
        image, scores, masks = predict(
            model, store, filename, device
        )
        gt = ground_truth(
            filename,
            image_map,
            ann_map,
            image.height,
            image.width,
        )
        positive_cache[filename] = (scores, masks, gt)
        print(
            f"\rPositive {i}/{len(positive_files)}",
            end="",
            flush=True,
        )
    print()

    for i, filename in enumerate(negative_files, 1):
        _, scores, masks = predict(
            model, store, filename, device
        )
        negative_cache[filename] = (scores, masks)
        print(
            f"\rNo-solar {i}/{len(negative_files)}",
            end="",
            flush=True,
        )
    print()

    summaries = []

    for threshold in THRESHOLDS:
        tp_total = fp_total = fn_total = 0
        count_errors = []
        area_errors_pct = []
        matched_ious = []
        positive_rows = []
        negative_rows = []

        for filename in positive_files:
            scores, masks, gt = positive_cache[filename]
            keep = scores >= threshold
            ps = scores[keep]
            pm = masks[keep]

            tp, fp, fn, ious = match_masks(
                pm, ps, gt, 0.50
            )

            gt_count = len(gt)
            pred_count = len(pm)
            gt_area = sum(int(m.sum()) for m in gt)
            pred_area = int(pm.sum()) if len(pm) else 0

            count_errors.append(pred_count - gt_count)
            if gt_area > 0:
                area_errors_pct.append(
                    abs(pred_area - gt_area) / gt_area * 100.0
                )

            matched_ious.extend(ious)
            tp_total += tp
            fp_total += fp
            fn_total += fn

            positive_rows.append({
                "model": name,
                "filename": filename,
                "score_threshold": threshold,
                "gt_count": gt_count,
                "pred_count": pred_count,
                "count_error": pred_count - gt_count,
                "gt_area": gt_area,
                "pred_area": pred_area,
                "area_error": pred_area - gt_area,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "mean_matched_iou": (
                    float(np.mean(ious)) if ious else 0.0
                ),
            })

        precision = (
            tp_total / (tp_total + fp_total)
            if tp_total + fp_total else 0.0
        )
        recall = (
            tp_total / (tp_total + fn_total)
            if tp_total + fn_total else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        count_mae = (
            float(np.mean(np.abs(count_errors)))
            if count_errors else 0.0
        )
        count_rmse = (
            float(np.sqrt(np.mean(np.square(count_errors))))
            if count_errors else 0.0
        )
        mean_iou = (
            float(np.mean(matched_ious))
            if matched_ious else 0.0
        )
        area_mape = (
            float(np.mean(area_errors_pct))
            if area_errors_pct else 0.0
        )

        negative_fp_images = 0
        negative_predictions = 0
        negative_scores = []

        for filename in negative_files:
            scores, masks = negative_cache[filename]
            keep = scores >= threshold
            ps = scores[keep]
            n_pred = len(ps)

            if n_pred:
                negative_fp_images += 1
                negative_predictions += n_pred
                negative_scores.extend(ps.tolist())

            negative_rows.append({
                "model": name,
                "filename": filename,
                "score_threshold": threshold,
                "pred_count": n_pred,
                "max_confidence": (
                    float(ps.max()) if n_pred else 0.0
                ),
            })

        neg_rate = (
            negative_fp_images / len(negative_files) * 100.0
        )
        neg_avg = (
            negative_predictions / len(negative_files)
        )
        neg_mean_conf = (
            float(np.mean(negative_scores))
            if negative_scores else 0.0
        )
        neg_max_conf = (
            float(np.max(negative_scores))
            if negative_scores else 0.0
        )

        summaries.append({
            "model": name,
            "checkpoint": checkpoint,
            "score_threshold": threshold,
            "positive_TP": tp_total,
            "positive_FP": fp_total,
            "positive_FN": fn_total,
            "precision": precision,
            "recall": recall,
            "F1": f1,
            "count_MAE": count_mae,
            "count_RMSE": count_rmse,
            "mean_matched_mask_IoU": mean_iou,
            "area_MAPE_percent": area_mape,
            "negative_FP_images": negative_fp_images,
            "negative_total_images": len(negative_files),
            "negative_FP_rate_percent": neg_rate,
            "negative_total_predictions": negative_predictions,
            "negative_avg_predictions_per_image": neg_avg,
            "negative_mean_FP_confidence": neg_mean_conf,
            "negative_max_FP_confidence": neg_max_conf,
        })

        pd.DataFrame(positive_rows).to_csv(
            os.path.join(
                OUTPUT_DIR,
                f"{name}_positive_threshold_{threshold:.2f}.csv",
            ),
            index=False,
        )

        pd.DataFrame(negative_rows).to_csv(
            os.path.join(
                OUTPUT_DIR,
                f"{name}_negative_threshold_{threshold:.2f}.csv",
            ),
            index=False,
        )

    return summaries


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    required = [
        ZIP_PATH,
        BASELINE_CHECKPOINT,
        BACKGROUND_AWARE_CHECKPOINT,
        os.path.join(SPLIT_DIR, "positive_test_LOCKED.csv"),
        os.path.join(SPLIT_DIR, "negative_test_LOCKED.csv"),
    ]

    for path in required:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    pos_df = pd.read_csv(
        os.path.join(SPLIT_DIR, "positive_test_LOCKED.csv")
    )
    neg_df = pd.read_csv(
        os.path.join(SPLIT_DIR, "negative_test_LOCKED.csv")
    )

    positive_files = pos_df["filename"].tolist()
    negative_files = neg_df["filename"].tolist()

    if len(positive_files) != 250:
        raise RuntimeError(
            f"Expected 250 positive test images; got {len(positive_files)}"
        )
    if len(negative_files) != 94:
        raise RuntimeError(
            f"Expected 94 negative test images; got {len(negative_files)}"
        )

    if set(positive_files) & set(negative_files):
        raise RuntimeError("Positive/negative test overlap detected.")

    coco = load_coco()
    image_map, ann_map = build_coco_maps(coco)
    store = ZipStore(ZIP_PATH)

    rows = []

    rows.extend(
        evaluate_model(
            "baseline",
            BASELINE_CHECKPOINT,
            positive_files,
            negative_files,
            store,
            image_map,
            ann_map,
            device,
        )
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rows.extend(
        evaluate_model(
            "background_aware_epoch_10",
            BACKGROUND_AWARE_CHECKPOINT,
            positive_files,
            negative_files,
            store,
            image_map,
            ann_map,
            device,
        )
    )

    summary = pd.DataFrame(rows)

    summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "final_model_comparison.csv",
        ),
        index=False,
    )

    primary = summary[
        summary["score_threshold"] == PRIMARY_THRESHOLD
    ].copy()

    primary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "primary_threshold_075_comparison.csv",
        ),
        index=False,
    )

    print("\n" + "=" * 80)
    print("PRIMARY COMPARISON — THRESHOLD 0.75")
    print("=" * 80)
    print(
        primary[
            [
                "model",
                "precision",
                "recall",
                "F1",
                "count_MAE",
                "mean_matched_mask_IoU",
                "area_MAPE_percent",
                "negative_FP_rate_percent",
                "negative_avg_predictions_per_image",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n" + "=" * 80)
    print("ALL THRESHOLDS")
    print("=" * 80)
    print(
        summary[
            [
                "model",
                "score_threshold",
                "precision",
                "recall",
                "F1",
                "count_MAE",
                "negative_FP_rate_percent",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nResults saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
