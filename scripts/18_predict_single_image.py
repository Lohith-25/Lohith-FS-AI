import argparse
import io
import os
import zipfile

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


BASE = r"C:\SolarMap-India"

ZIP_PATH = os.path.join(
    BASE,
    "dataset",
    "Solar Images.zip",
)

CHECKPOINT = os.path.join(
    BASE,
    "checkpoints",
    "maskrcnn_baseline_epoch_10.pth",
)

OUTPUT_DIR = os.path.join(
    BASE,
    "results",
    "predictions",
)

DEFAULT_THRESHOLD = 0.77


def build_model():
    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
    )

    box_in = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        box_in,
        2,
    )

    mask_in = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        mask_in,
        256,
        2,
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

    state = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state.items()
    }

    missing, unexpected = model.load_state_dict(
        state,
        strict=False,
    )

    if missing:
        raise RuntimeError(
            "Missing checkpoint keys:\n"
            + "\n".join(missing[:20])
        )

    if unexpected:
        print(
            f"WARNING: {len(unexpected)} unexpected checkpoint keys"
        )


def read_zip_image(filename):
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        matches = [
            name
            for name in zf.namelist()
            if name.lower().endswith(".png")
            and os.path.basename(name) == filename
        ]

        if not matches:
            raise FileNotFoundError(
                f"Could not find {filename} inside {ZIP_PATH}"
            )

        data = zf.read(matches[0])

    return Image.open(
        io.BytesIO(data)
    ).convert("RGB")


def read_disk_image(path):
    return Image.open(path).convert("RGB")


def run_prediction(model, image, device, threshold):
    array = np.array(
        image,
        copy=True,
    )

    tensor = (
        torch.from_numpy(
            array.transpose(2, 0, 1)
        ).float()
        / 255.0
    )

    with torch.inference_mode():
        output = model(
            [tensor.to(device)]
        )[0]

    scores = output["scores"].detach().cpu().numpy()
    labels = output["labels"].detach().cpu().numpy()
    boxes = output["boxes"].detach().cpu().numpy()
    masks = output["masks"][:, 0].detach().cpu().numpy()

    keep = scores >= threshold

    return {
        "scores": scores[keep],
        "labels": labels[keep],
        "boxes": boxes[keep],
        "masks": masks[keep] >= 0.5,
        "raw_count": len(scores),
        "kept_count": int(keep.sum()),
    }


def draw_predictions(image, prediction):
    canvas = image.convert("RGBA")
    overlay = Image.new(
        "RGBA",
        canvas.size,
        (255, 0, 0, 0),
    )

    overlay_array = np.zeros(
        (image.height, image.width, 4),
        dtype=np.uint8,
    )

    for mask in prediction["masks"]:
        overlay_array[mask, 0] = 255
        overlay_array[mask, 1] = 30
        overlay_array[mask, 2] = 30
        overlay_array[mask, 3] = 90

    overlay = Image.fromarray(
        overlay_array,
        mode="RGBA",
    )

    canvas = Image.alpha_composite(
        canvas,
        overlay,
    )

    draw = ImageDraw.Draw(canvas)

    for idx, (box, score) in enumerate(
        zip(
            prediction["boxes"],
            prediction["scores"],
        ),
        start=1,
    ):
        x1, y1, x2, y2 = box.tolist()

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=(255, 0, 0, 255),
            width=2,
        )

        text = f"#{idx} {score:.3f}"

        draw.rectangle(
            [x1, max(0, y1 - 18), x1 + 90, y1],
            fill=(255, 0, 0, 220),
        )

        draw.text(
            [x1 + 2, max(0, y1 - 17)],
            text,
            fill=(255, 255, 255, 255),
        )

    return canvas.convert("RGB")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--filename",
        type=str,
        default="100.0_1.0.png",
        help="Filename inside Solar Images.zip",
    )

    parser.add_argument(
        "--image",
        type=str,
        default="",
        help="Optional direct image path. Overrides --filename.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Confidence threshold. Final baseline operating point is 0.77.",
    )

    args = parser.parse_args()

    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(CHECKPOINT)

    if not os.path.exists(ZIP_PATH) and not args.image:
        raise FileNotFoundError(ZIP_PATH)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 80)
    print("SOLARMAP — SINGLE IMAGE PREDICTION")
    print("=" * 80)

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print(
        f"Confidence threshold: {args.threshold:.2f}"
    )

    if args.image:
        image = read_disk_image(
            args.image
        )
        source_name = os.path.basename(
            args.image
        )
    else:
        image = read_zip_image(
            args.filename
        )
        source_name = args.filename

    print("Image:", source_name)
    print("Size:", image.size)

    print("\nLoading baseline model...")

    model = build_model().to(
        device
    )

    load_checkpoint(
        model,
        CHECKPOINT,
        device,
    )

    model.eval()

    prediction = run_prediction(
        model,
        image,
        device,
        args.threshold,
    )

    print("\nPrediction result")
    print("-" * 80)
    print(
        "Raw detections:",
        prediction["raw_count"],
    )
    print(
        "Detections after threshold:",
        prediction["kept_count"],
    )

    if prediction["kept_count"] == 0:
        print(
            "FINAL DECISION: No solar-panel detections"
        )
    else:
        print(
            "FINAL DECISION:",
            prediction["kept_count"],
            "solar-panel instance(s) detected",
        )

        print("\nDetected instances:")

        for idx, score in enumerate(
            prediction["scores"],
            start=1,
        ):
            x1, y1, x2, y2 = (
                prediction["boxes"][idx - 1]
            )

            area = int(
                prediction["masks"][idx - 1].sum()
            )

            print(
                f"  #{idx:02d} | "
                f"confidence={score:.4f} | "
                f"box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) | "
                f"mask_area={area}px"
            )

    result_image = draw_predictions(
        image,
        prediction,
    )

    stem = os.path.splitext(
        source_name
    )[0]

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{stem}_prediction.png",
    )

    result_image.save(
        output_path
    )

    print(
        "\nSaved visual prediction:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()
