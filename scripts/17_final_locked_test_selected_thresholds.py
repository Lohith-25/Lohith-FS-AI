import io
import json
import os
import zipfile

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from pycocotools import mask as mask_utils


BASE = r"C:\SolarMap-India"
ZIP_PATH = os.path.join(BASE, "dataset", "Solar Images.zip")
SPLIT_DIR = os.path.join(BASE, "splits")
OUT_DIR = os.path.join(BASE, "results", "final_selected_threshold_test")

MODELS = {
    "baseline": (
        os.path.join(BASE, "checkpoints", "maskrcnn_baseline_epoch_10.pth"),
        0.77,
    ),
    "background_aware": (
        os.path.join(BASE, "checkpoints", "background_aware", "background_aware_epoch_10.pth"),
        0.66,
    ),
    "hard_negative_replay": (
        os.path.join(BASE, "checkpoints", "hard_negative_replay", "hard_negative_replay_epoch_10.pth"),
        0.72,
    ),
}

COCO_MEMBER = "Solar Images/annotations/merged_instances_default.json"
IOU_THRESHOLD = 0.50


def build_model():
    m = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)

    box_in = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(box_in, 2)

    mask_in = m.roi_heads.mask_predictor.conv5_mask.in_channels
    m.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in, 256, 2)
    return m


def load_weights(model, path, device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]

    state = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state.items()
    }

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError("Missing checkpoint keys:\n" + "\n".join(missing[:20]))
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected keys")
    return ckpt


class ZipStore:
    def __init__(self, path):
        self.zf = zipfile.ZipFile(path, "r")
        self.members = {
            os.path.basename(n): n
            for n in self.zf.namelist()
            if n.lower().endswith(".png")
        }

    def read(self, filename):
        data = self.zf.read(self.members[filename])
        return Image.open(io.BytesIO(data)).convert("RGB")


def load_coco():
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        return json.loads(zf.read(COCO_MEMBER))


def coco_maps(coco):
    image_map = {}
    ann_map = {}
    for x in coco["images"]:
        image_map[os.path.basename(x["file_name"])] = x
    for a in coco["annotations"]:
        ann_map.setdefault(a["image_id"], []).append(a)
    return image_map, ann_map


def ann_to_mask(seg, h, w):
    if isinstance(seg, list):
        polys = [p for p in seg if isinstance(p, (list, tuple)) and len(p) >= 6]
        if not polys:
            return np.zeros((h, w), np.uint8)
        rles = mask_utils.frPyObjects(polys, h, w)
        x = mask_utils.decode(mask_utils.merge(rles))
    elif isinstance(seg, dict):
        x = mask_utils.decode(seg)
    else:
        return np.zeros((h, w), np.uint8)

    if x.ndim == 3:
        x = x[:, :, 0]
    return x.astype(np.uint8)


def gt_masks(filename, image_map, ann_map, h, w):
    rec = image_map.get(filename)
    if rec is None:
        return []

    out = []
    for a in ann_map.get(rec["id"], []):
        m = ann_to_mask(a.get("segmentation", []), h, w)
        if int(m.sum()) > 0:
            out.append(m)
    return out


def encode_masks(ms):
    return [mask_utils.encode(np.asfortranarray(m.astype(np.uint8))) for m in ms]


def match(pred_masks, pred_scores, gt):
    npred, ngt = len(pred_masks), len(gt)
    if npred == 0:
        return 0, 0, ngt, []
    if ngt == 0:
        return 0, npred, 0, []

    mat = mask_utils.iou(encode_masks(pred_masks), encode_masks(gt), [0] * ngt)
    order = np.argsort(-np.asarray(pred_scores))
    used = set()
    tp = fp = 0
    miou = []

    for pi in order:
        candidates = [gi for gi in range(ngt) if gi not in used]
        if not candidates:
            fp += 1
            continue
        gi = max(candidates, key=lambda j: float(mat[pi, j]))
        val = float(mat[pi, gi])
        if val >= IOU_THRESHOLD:
            tp += 1
            used.add(gi)
            miou.append(val)
        else:
            fp += 1

    return tp, fp, ngt - tp, miou


def predict(model, store, filename, device):
    img = store.read(filename)
    arr = np.array(img, copy=True)
    x = torch.from_numpy(arr.transpose(2, 0, 1)).float() / 255.0
    with torch.inference_mode():
        out = model([x.to(device)])[0]

    scores = out["scores"].detach().cpu().numpy()
    masks = (out["masks"][:, 0].detach().cpu().numpy() >= 0.5)
    return img, scores, masks


def evaluate_model(name, ckpt_path, threshold, pos_files, neg_files, store, image_map, ann_map, device):
    print("\n" + "=" * 80)
    print(f"EVALUATING: {name}")
    print("=" * 80)

    model = build_model().to(device)
    ckpt = load_weights(model, ckpt_path, device)
    model.eval()
    if isinstance(ckpt, dict):
        print("Checkpoint epoch:", ckpt.get("epoch", "unknown"))

    pos_cache = {}
    neg_cache = {}

    for i, f in enumerate(pos_files, 1):
        img, s, m = predict(model, store, f, device)
        gt = gt_masks(f, image_map, ann_map, img.height, img.width)
        pos_cache[f] = (s, m, gt)
        print(f"\rPositive {i}/{len(pos_files)}", end="", flush=True)
    print()

    for i, f in enumerate(neg_files, 1):
        _, s, m = predict(model, store, f, device)
        neg_cache[f] = (s, m)
        print(f"\rNo-solar {i}/{len(neg_files)}", end="", flush=True)
    print()

    th = threshold
    TP = FP = FN = 0
    count_errors = []
    area_pct = []
    ious = []

    for f in pos_files:
        s, m, gt = pos_cache[f]
        keep = s >= th
        ps, pm = s[keep], m[keep]

        tp, fp, fn, miou = match(pm, ps, gt)
        TP += tp; FP += fp; FN += fn
        ious.extend(miou)

        gt_count = len(gt)
        pred_count = len(pm)
        count_errors.append(pred_count - gt_count)

        gt_area = sum(int(x.sum()) for x in gt)
        pred_area = int(pm.sum()) if len(pm) else 0
        if gt_area > 0:
            area_pct.append(abs(pred_area - gt_area) / gt_area * 100.0)

    precision = TP / (TP + FP) if TP + FP else 0.0
    recall = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    neg_images = 0
    neg_preds = 0
    neg_scores = []

    for f in neg_files:
        s, _ = neg_cache[f]
        ps = s[s >= th]
        if len(ps):
            neg_images += 1
            neg_preds += len(ps)
            neg_scores.extend(ps.tolist())

    row = {
        "model": name,
        "threshold": th,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "count_MAE": float(np.mean(np.abs(count_errors))),
        "count_RMSE": float(np.sqrt(np.mean(np.square(count_errors)))),
        "mean_matched_mask_IoU": float(np.mean(ious)) if ious else 0.0,
        "area_MAPE_percent": float(np.mean(area_pct)) if area_pct else 0.0,
        "negative_FP_images": neg_images,
        "negative_FP_rate_percent": neg_images / len(neg_files) * 100.0,
        "negative_total_predictions": neg_preds,
        "negative_avg_predictions_per_image": neg_preds / len(neg_files),
        "negative_mean_FP_confidence": float(np.mean(neg_scores)) if neg_scores else 0.0,
        "negative_max_FP_confidence": float(np.max(neg_scores)) if neg_scores else 0.0,
    }

    return [row]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for path in [
        ZIP_PATH,
        *[
            model_info[0]
            for model_info in MODELS.values()
        ],
        os.path.join(SPLIT_DIR, "positive_test_LOCKED.csv"),
        os.path.join(SPLIT_DIR, "negative_test_LOCKED.csv"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PyTorch:", torch.__version__)
    print("TorchVision:", torchvision.__version__)
    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    pos = pd.read_csv(os.path.join(SPLIT_DIR, "positive_test_LOCKED.csv"))
    neg = pd.read_csv(os.path.join(SPLIT_DIR, "negative_test_LOCKED.csv"))

    pos_files = pos["filename"].astype(str).tolist()
    neg_files = neg["filename"].astype(str).tolist()

    if len(pos_files) != 250 or len(neg_files) != 94:
        raise RuntimeError("Locked test sizes changed; stopping.")
    if set(pos_files) & set(neg_files):
        raise RuntimeError("Positive/negative test overlap detected.")

    print("\nLOCKED TEST SET: 250 positive + 94 negative")

    coco = load_coco()
    image_map, ann_map = coco_maps(coco)
    store = ZipStore(ZIP_PATH)

    all_rows = []
    for name, (ckpt, threshold) in MODELS.items():
        all_rows.extend(
            evaluate_model(
                name, ckpt, threshold, pos_files, neg_files,
                store, image_map, ann_map, device
            )
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)
    result_path = os.path.join(OUT_DIR, "FINAL_LOCKED_TEST_SELECTED_THRESHOLDS.csv")
    df.to_csv(result_path, index=False)

    print("\n" + "=" * 105)
    print("FINAL LOCKED TEST RESULTS — THRESHOLDS SELECTED FROM VALIDATION")
    print("=" * 105)
    print(
        df[
            [
                "model", "threshold", "precision", "recall", "F1",
                "count_MAE", "mean_matched_mask_IoU",
                "area_MAPE_percent",
                "negative_FP_rate_percent",
                "negative_total_predictions",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nSaved:")
    print(result_path)


if __name__ == "__main__":
    main()
import io
import json
import os
import zipfile

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from pycocotools import mask as mask_utils


BASE = r"C:\SolarMap-India"
ZIP_PATH = os.path.join(BASE, "dataset", "Solar Images.zip")
SPLIT_DIR = os.path.join(BASE, "splits")
OUT_DIR = os.path.join(BASE, "results", "final_selected_threshold_test")

MODELS = {
    "baseline": (
        os.path.join(BASE, "checkpoints", "maskrcnn_baseline_epoch_10.pth"),
        0.77,
    ),
    "background_aware": (
        os.path.join(BASE, "checkpoints", "background_aware", "background_aware_epoch_10.pth"),
        0.66,
    ),
    "hard_negative_replay": (
        os.path.join(BASE, "checkpoints", "hard_negative_replay", "hard_negative_replay_epoch_10.pth"),
        0.72,
    ),
}

COCO_MEMBER = "Solar Images/annotations/merged_instances_default.json"
IOU_THRESHOLD = 0.50


def build_model():
    m = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)

    box_in = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(box_in, 2)

    mask_in = m.roi_heads.mask_predictor.conv5_mask.in_channels
    m.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in, 256, 2)
    return m


def load_weights(model, path, device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]

    state = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state.items()
    }

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError("Missing checkpoint keys:\n" + "\n".join(missing[:20]))
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected keys")
    return ckpt


class ZipStore:
    def __init__(self, path):
        self.zf = zipfile.ZipFile(path, "r")
        self.members = {
            os.path.basename(n): n
            for n in self.zf.namelist()
            if n.lower().endswith(".png")
        }

    def read(self, filename):
        data = self.zf.read(self.members[filename])
        return Image.open(io.BytesIO(data)).convert("RGB")


def load_coco():
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        return json.loads(zf.read(COCO_MEMBER))


def coco_maps(coco):
    image_map = {}
    ann_map = {}
    for x in coco["images"]:
        image_map[os.path.basename(x["file_name"])] = x
    for a in coco["annotations"]:
        ann_map.setdefault(a["image_id"], []).append(a)
    return image_map, ann_map


def ann_to_mask(seg, h, w):
    if isinstance(seg, list):
        polys = [p for p in seg if isinstance(p, (list, tuple)) and len(p) >= 6]
        if not polys:
            return np.zeros((h, w), np.uint8)
        rles = mask_utils.frPyObjects(polys, h, w)
        x = mask_utils.decode(mask_utils.merge(rles))
    elif isinstance(seg, dict):
        x = mask_utils.decode(seg)
    else:
        return np.zeros((h, w), np.uint8)

    if x.ndim == 3:
        x = x[:, :, 0]
    return x.astype(np.uint8)


def gt_masks(filename, image_map, ann_map, h, w):
    rec = image_map.get(filename)
    if rec is None:
        return []

    out = []
    for a in ann_map.get(rec["id"], []):
        m = ann_to_mask(a.get("segmentation", []), h, w)
        if int(m.sum()) > 0:
            out.append(m)
    return out


def encode_masks(ms):
    return [mask_utils.encode(np.asfortranarray(m.astype(np.uint8))) for m in ms]


def match(pred_masks, pred_scores, gt):
    npred, ngt = len(pred_masks), len(gt)
    if npred == 0:
        return 0, 0, ngt, []
    if ngt == 0:
        return 0, npred, 0, []

    mat = mask_utils.iou(encode_masks(pred_masks), encode_masks(gt), [0] * ngt)
    order = np.argsort(-np.asarray(pred_scores))
    used = set()
    tp = fp = 0
    miou = []

    for pi in order:
        candidates = [gi for gi in range(ngt) if gi not in used]
        if not candidates:
            fp += 1
            continue
        gi = max(candidates, key=lambda j: float(mat[pi, j]))
        val = float(mat[pi, gi])
        if val >= IOU_THRESHOLD:
            tp += 1
            used.add(gi)
            miou.append(val)
        else:
            fp += 1

    return tp, fp, ngt - tp, miou


def predict(model, store, filename, device):
    img = store.read(filename)
    arr = np.array(img, copy=True)
    x = torch.from_numpy(arr.transpose(2, 0, 1)).float() / 255.0
    with torch.inference_mode():
        out = model([x.to(device)])[0]

    scores = out["scores"].detach().cpu().numpy()
    masks = (out["masks"][:, 0].detach().cpu().numpy() >= 0.5)
    return img, scores, masks


def evaluate_model(name, ckpt_path, threshold, pos_files, neg_files, store, image_map, ann_map, device):
    print("\n" + "=" * 80)
    print(f"EVALUATING: {name}")
    print("=" * 80)

    model = build_model().to(device)
    ckpt = load_weights(model, ckpt_path, device)
    model.eval()
    if isinstance(ckpt, dict):
        print("Checkpoint epoch:", ckpt.get("epoch", "unknown"))

    pos_cache = {}
    neg_cache = {}

    for i, f in enumerate(pos_files, 1):
        img, s, m = predict(model, store, f, device)
        gt = gt_masks(f, image_map, ann_map, img.height, img.width)
        pos_cache[f] = (s, m, gt)
        print(f"\rPositive {i}/{len(pos_files)}", end="", flush=True)
    print()

    for i, f in enumerate(neg_files, 1):
        _, s, m = predict(model, store, f, device)
        neg_cache[f] = (s, m)
        print(f"\rNo-solar {i}/{len(neg_files)}", end="", flush=True)
    print()

    th = threshold
    TP = FP = FN = 0
    count_errors = []
    area_pct = []
    ious = []

    for f in pos_files:
        s, m, gt = pos_cache[f]
        keep = s >= th
        ps, pm = s[keep], m[keep]

        tp, fp, fn, miou = match(pm, ps, gt)
        TP += tp; FP += fp; FN += fn
        ious.extend(miou)

        gt_count = len(gt)
        pred_count = len(pm)
        count_errors.append(pred_count - gt_count)

        gt_area = sum(int(x.sum()) for x in gt)
        pred_area = int(pm.sum()) if len(pm) else 0
        if gt_area > 0:
            area_pct.append(abs(pred_area - gt_area) / gt_area * 100.0)

    precision = TP / (TP + FP) if TP + FP else 0.0
    recall = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    neg_images = 0
    neg_preds = 0
    neg_scores = []

    for f in neg_files:
        s, _ = neg_cache[f]
        ps = s[s >= th]
        if len(ps):
            neg_images += 1
            neg_preds += len(ps)
            neg_scores.extend(ps.tolist())

    row = {
        "model": name,
        "threshold": th,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "count_MAE": float(np.mean(np.abs(count_errors))),
        "count_RMSE": float(np.sqrt(np.mean(np.square(count_errors)))),
        "mean_matched_mask_IoU": float(np.mean(ious)) if ious else 0.0,
        "area_MAPE_percent": float(np.mean(area_pct)) if area_pct else 0.0,
        "negative_FP_images": neg_images,
        "negative_FP_rate_percent": neg_images / len(neg_files) * 100.0,
        "negative_total_predictions": neg_preds,
        "negative_avg_predictions_per_image": neg_preds / len(neg_files),
        "negative_mean_FP_confidence": float(np.mean(neg_scores)) if neg_scores else 0.0,
        "negative_max_FP_confidence": float(np.max(neg_scores)) if neg_scores else 0.0,
    }

    return [row]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for path in [
        ZIP_PATH,
        *[
            model_info[0]
            for model_info in MODELS.values()
        ],
        os.path.join(SPLIT_DIR, "positive_test_LOCKED.csv"),
        os.path.join(SPLIT_DIR, "negative_test_LOCKED.csv"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PyTorch:", torch.__version__)
    print("TorchVision:", torchvision.__version__)
    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    pos = pd.read_csv(os.path.join(SPLIT_DIR, "positive_test_LOCKED.csv"))
    neg = pd.read_csv(os.path.join(SPLIT_DIR, "negative_test_LOCKED.csv"))

    pos_files = pos["filename"].astype(str).tolist()
    neg_files = neg["filename"].astype(str).tolist()

    if len(pos_files) != 250 or len(neg_files) != 94:
        raise RuntimeError("Locked test sizes changed; stopping.")
    if set(pos_files) & set(neg_files):
        raise RuntimeError("Positive/negative test overlap detected.")

    print("\nLOCKED TEST SET: 250 positive + 94 negative")

    coco = load_coco()
    image_map, ann_map = coco_maps(coco)
    store = ZipStore(ZIP_PATH)

    all_rows = []
    for name, (ckpt, threshold) in MODELS.items():
        all_rows.extend(
            evaluate_model(
                name, ckpt, threshold, pos_files, neg_files,
                store, image_map, ann_map, device
            )
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)
    result_path = os.path.join(OUT_DIR, "FINAL_LOCKED_TEST_SELECTED_THRESHOLDS.csv")
    df.to_csv(result_path, index=False)

    print("\n" + "=" * 105)
    print("FINAL LOCKED TEST RESULTS — THRESHOLDS SELECTED FROM VALIDATION")
    print("=" * 105)
    print(
        df[
            [
                "model", "threshold", "precision", "recall", "F1",
                "count_MAE", "mean_matched_mask_IoU",
                "area_MAPE_percent",
                "negative_FP_rate_percent",
                "negative_total_predictions",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nSaved:")
    print(result_path)


if __name__ == "__main__":
    main()
