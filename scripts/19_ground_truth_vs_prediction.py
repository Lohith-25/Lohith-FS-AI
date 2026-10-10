import io
import json
import os
import zipfile

import numpy as np
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
OUTPUT_DIR = os.path.join(BASE, "results", "gt_vs_prediction")

DEFAULT_FILENAME = "100.0_1.0.png"
DEFAULT_THRESHOLD = 0.77


def build_model():
    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
    )

    box_in = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        box_in, 2
    )

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


def load_coco():
    with zipfile.ZipFile(
        ZIP_PATH,
        "r",
    ) as zf:
        return json.loads(
            zf.read(COCO_MEMBER)
        )


def read_image(filename):
    with zipfile.ZipFile(
        ZIP_PATH,
        "r",
    ) as zf:
        member = next(
            (
                name
                for name in zf.namelist()
                if name.lower().endswith(".png")
                and os.path.basename(name) == filename
            ),
            None,
        )

        if member is None:
            raise FileNotFoundError(
                f"{filename} not found in ZIP."
            )

        data = zf.read(member)

    return Image.open(
        io.BytesIO(data)
    ).convert("RGB")


def polygon_mask(segmentation, h, w):
    if not isinstance(segmentation, list):
        return np.zeros((h, w), dtype=np.uint8)

    polygons = [
        p
        for p in segmentation
        if isinstance(p, (list, tuple))
        and len(p) >= 6
    ]

    if not polygons:
        return np.zeros((h, w), dtype=np.uint8)

    rles = mask_utils.frPyObjects(
        polygons,
        h,
        w,
    )

    m = mask_utils.decode(
        mask_utils.merge(rles)
    )

    if m.ndim == 3:
        m = m[:, :, 0]

    return m.astype(np.uint8)


def get_ground_truth(coco, filename, h, w):
    image_record = None

    for image in coco["images"]:
        if os.path.basename(image["file_name"]) == filename:
            image_record = image
            break

    if image_record is None:
        raise RuntimeError(
            f"No COCO image record found for {filename}"
        )

    masks = []

    for ann in coco["annotations"]:
        if ann["image_id"] != image_record["id"]:
            continue

        m = polygon_mask(
            ann.get("segmentation", []),
            h,
            w,
        )

        if int(m.sum()) > 0:
            masks.append(m)

    return masks


def predict(model, image, device, threshold):
    arr = np.array(image, copy=True)

    tensor = (
        torch.from_numpy(
            arr.transpose(2, 0, 1)
        ).float()
        / 255.0
    )

    with torch.inference_mode():
        out = model(
            [tensor.to(device)]
        )[0]

    scores = out["scores"].detach().cpu().numpy()
    boxes = out["boxes"].detach().cpu().numpy()
    masks = out["masks"][:, 0].detach().cpu().numpy() >= 0.5

    keep = scores >= threshold

    return (
        boxes[keep],
        scores[keep],
        masks[keep],
    )


def add_mask(
    canvas,
    mask,
    fill,
):
    """
    fill = (R,G,B,A)
    """
    arr = np.zeros(
        (mask.shape[0], mask.shape[1], 4),
        dtype=np.uint8,
    )

    arr[mask, 0] = fill[0]
    arr[mask, 1] = fill[1]
    arr[mask, 2] = fill[2]
    arr[mask, 3] = fill[3]

    overlay = Image.fromarray(
        arr,
        mode="RGBA",
    )

    return Image.alpha_composite(
        canvas,
        overlay,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--filename",
        default=DEFAULT_FILENAME,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
    )

    args = parser.parse_args()

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 80)
    print("SOLARMAP — GROUND TRUTH VS PREDICTION")
    print("=" * 80)

    print("Image:", args.filename)
    print("Threshold:", args.threshold)
    print("Device:", device)

    image = read_image(
        args.filename
    )

    coco = load_coco()

    gt_masks = get_ground_truth(
        coco,
        args.filename,
        image.height,
        image.width,
    )

    model = load_model(
        device
    )

    pred_boxes, pred_scores, pred_masks = predict(
        model,
        image,
        device,
        args.threshold,
    )

    print("\nGround-truth instances:", len(gt_masks))
    print("Predicted instances:", len(pred_masks))

    # Base image.
    canvas = image.convert("RGBA")

    # Ground truth = green.
    for mask in gt_masks:
        canvas = add_mask(
            canvas,
            mask.astype(bool),
            (0, 255, 0, 75),
        )

    # Predictions = red.
    for mask in pred_masks:
        canvas = add_mask(
            canvas,
            mask.astype(bool),
            (255, 0, 0, 80),
        )

    draw = ImageDraw.Draw(canvas)

    # Ground-truth boundaries + IDs.
    for idx, mask in enumerate(
        gt_masks,
        start=1,
    ):
        ys, xs = np.where(mask > 0)

        if len(xs) == 0:
            continue

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=(0, 255, 0, 255),
            width=2,
        )

        draw.text(
            [x1 + 2, y1 + 2],
            f"GT#{idx}",
            fill=(0, 255, 0, 255),
        )

    # Prediction boundaries + confidence.
    for idx, (box, score) in enumerate(
        zip(
            pred_boxes,
            pred_scores,
        ),
        start=1,
    ):
        x1, y1, x2, y2 = box

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=(255, 0, 0, 255),
            width=2,
        )

        draw.text(
            [x1 + 2, max(0, y1 + 2)],
            f"P#{idx} {score:.2f}",
            fill=(255, 0, 0, 255),
        )

    # Small legend box.
    draw.rectangle(
        [5, 5, 190, 55],
        fill=(0, 0, 0, 180),
    )

    draw.text(
        [10, 10],
        "GREEN = Ground Truth",
        fill=(0, 255, 0, 255),
    )

    draw.text(
        [10, 30],
        "RED = Prediction",
        fill=(255, 0, 0, 255),
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        (
            os.path.splitext(args.filename)[0]
            + "_GT_vs_prediction.png"
        ),
    )

    canvas.convert("RGB").save(
        output_path
    )

    print("\nSaved:")
    print(output_path)


if __name__ == "__main__":
    main()
