import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
)
from torchvision.models.detection.mask_rcnn import (
    MaskRCNNPredictor,
)


BASE = r"C:\SolarMap-India"

CHECKPOINT = os.path.join(
    BASE,
    "checkpoints",
    "maskrcnn_baseline_epoch_10.pth",
)

OUTPUT_DIR = os.path.join(
    BASE,
    "results",
    "external_test",
)

DEFAULT_THRESHOLD = 0.77


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


def load_model(device):
    model = build_model().to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    state = checkpoint

    if isinstance(checkpoint, dict):
        state = checkpoint.get(
            "model_state_dict",
            checkpoint.get(
                "state_dict",
                checkpoint,
            ),
        )

    state = {
        (
            key[7:]
            if key.startswith("module.")
            else key
        ): value
        for key, value in state.items()
    }

    missing, unexpected = (
        model.load_state_dict(
            state,
            strict=False,
        )
    )

    if missing:
        raise RuntimeError(
            "Missing checkpoint parameters:\n"
            + "\n".join(missing[:20])
        )

    if unexpected:
        print(
            f"WARNING: {len(unexpected)} "
            "unexpected checkpoint keys"
        )

    model.eval()

    return model


def predict(
    model,
    image,
    device,
    threshold,
):
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

    boxes = (
        output["boxes"]
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

    keep = scores >= threshold

    return (
        boxes[keep],
        scores[keep],
        masks[keep],
        len(scores),
    )


def draw_result(
    image,
    boxes,
    scores,
    masks,
):
    canvas = image.convert("RGBA")

    overlay_array = np.zeros(
        (
            image.height,
            image.width,
            4,
        ),
        dtype=np.uint8,
    )

    for mask in masks:
        overlay_array[
            mask,
            0,
        ] = 255

        overlay_array[
            mask,
            1,
        ] = 0

        overlay_array[
            mask,
            2,
        ] = 0

        overlay_array[
            mask,
            3,
        ] = 80

    overlay = Image.fromarray(
        overlay_array,
        mode="RGBA",
    )

    canvas = Image.alpha_composite(
        canvas,
        overlay,
    )

    draw = ImageDraw.Draw(
        canvas
    )

    for idx, (
        box,
        score,
    ) in enumerate(
        zip(
            boxes,
            scores,
        ),
        start=1,
    ):
        x1, y1, x2, y2 = (
            box.tolist()
        )

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=(
                255,
                0,
                0,
                255,
            ),
            width=3,
        )

        label = (
            f"#{idx} {score:.3f}"
        )

        text_y = max(
            0,
            int(y1) - 18,
        )

        draw.rectangle(
            [
                x1,
                text_y,
                x1 + 100,
                text_y + 18,
            ],
            fill=(
                255,
                0,
                0,
                210,
            ),
        )

        draw.text(
            [
                x1 + 2,
                text_y + 1,
            ],
            label,
            fill=(
                255,
                255,
                255,
                255,
            ),
        )

    # Legend
    draw.rectangle(
        [5, 5, 230, 42],
        fill=(
            0,
            0,
            0,
            190,
        ),
    )

    draw.text(
        [10, 10],
        (
            "RED = predicted solar-panel "
            "instance"
        ),
        fill=(
            255,
            255,
            255,
            255,
        ),
    )

    return canvas.convert("RGB")


def process_image(
    model,
    image_path,
    device,
    threshold,
):
    image = Image.open(
        image_path
    ).convert("RGB")

    boxes, scores, masks, raw_count = predict(
        model,
        image,
        device,
        threshold,
    )

    print("\n" + "-" * 80)
    print(
        "Image:",
        image_path,
    )
    print(
        "Original size:",
        image.size,
    )
    print(
        "Raw detections:",
        raw_count,
    )
    print(
        "Accepted detections:",
        len(scores),
    )

    if len(scores) == 0:
        print(
            "FINAL DECISION: "
            "No solar-panel detections"
        )
    else:
        print(
            "FINAL DECISION:",
            len(scores),
            "solar-panel instance(s)"
        )

        for idx, (
            box,
            score,
        ) in enumerate(
            zip(
                boxes,
                scores,
            ),
            start=1,
        ):
            area = int(
                masks[idx - 1].sum()
            )

            print(
                f"  #{idx:02d} | "
                f"confidence={score:.4f} | "
                f"mask_area={area}px | "
                f"box=("
                f"{box[0]:.1f},"
                f"{box[1]:.1f},"
                f"{box[2]:.1f},"
                f"{box[3]:.1f})"
            )

    result = draw_result(
        image,
        boxes,
        scores,
        masks,
    )

    filename = os.path.basename(
        image_path
    )

    stem = os.path.splitext(
        filename
    )[0]

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{stem}_prediction.png",
    )

    result.save(
        output_path
    )

    print(
        "Saved:",
        output_path,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "images",
        nargs="+",
        help="One or more image file paths.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
    )

    args = parser.parse_args()

    if not os.path.exists(
        CHECKPOINT
    ):
        raise FileNotFoundError(
            CHECKPOINT
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    for image_path in args.images:
        if not os.path.exists(
            image_path
        ):
            raise FileNotFoundError(
                image_path
            )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print(
        "SOLARMAP — EXTERNAL REAL-WORLD TEST"
    )
    print("=" * 80)

    print(
        "Device:",
        device,
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print(
        "Threshold:",
        args.threshold,
    )

    print(
        "\nLoading baseline model..."
    )

    model = load_model(
        device
    )

    for image_path in args.images:
        process_image(
            model,
            image_path,
            device,
            args.threshold,
        )

    print(
        "\nExternal testing complete."
    )


if __name__ == "__main__":
    main()
