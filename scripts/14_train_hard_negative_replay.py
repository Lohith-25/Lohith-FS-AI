import io
import json
import os
import random
import time
import zipfile

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


# ============================================================
# SOLARMAP — STEP 14
# EXPERIMENT 2: HARD-NEGATIVE REPLAY FINE-TUNING
#
# Starting point:
#   Original baseline Mask R-CNN epoch 10
#
# Training data:
#   1993 positive TRAIN images
#   300 negative TRAIN images
#   + one replay of the 54 mined hard negatives
#
# Hard negative definition:
#   Baseline confidence >= 0.75 on a known no-solar TRAIN image
#
# Therefore:
#   Total training occurrences per epoch = 2347
#   (1993 positive + 300 negative + 54 hard-negative replay)
#
# Protected:
#   250 positive test images
#   94 negative test images
#
# This is Experiment 2, NOT a continuation of Experiment 1.
# It starts again from the original baseline checkpoint so the
# scientific comparison remains clean.
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

HARD_NEGATIVE_CSV = os.path.join(
    BASE,
    "results",
    "hard_negative_mining",
    "hard_negative_train.csv",
)

OUTPUT_DIR = os.path.join(
    BASE,
    "checkpoints",
    "hard_negative_replay",
)

LOG_PATH = os.path.join(
    OUTPUT_DIR,
    "training_log.csv",
)

MANIFEST_PATH = os.path.join(
    OUTPUT_DIR,
    "experiment_manifest.csv",
)

CONFIG_PATH = os.path.join(
    OUTPUT_DIR,
    "experiment_config.json",
)

COCO_PATH = "Solar Images/annotations/merged_instances_default.json"

NUM_CLASSES = 2

BATCH_SIZE = 1
NUM_EPOCHS = 10

LEARNING_RATE = 1e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

STEP_SIZE = 7
GAMMA = 0.1

SEED = 42

HARD_NEGATIVE_REPLAY_FACTOR = 2


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    return tuple(zip(*batch))


def get_amp_components(device):
    use_amp = device.type == "cuda"

    if hasattr(torch, "amp"):
        try:
            scaler = torch.amp.GradScaler(
                "cuda",
                enabled=use_amp,
            )
            return use_amp, scaler
        except (TypeError, AttributeError):
            pass

    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp,
    )

    return use_amp, scaler


def autocast_context(use_amp):
    if hasattr(torch, "amp"):
        try:
            return torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            )
        except (TypeError, AttributeError):
            pass

    return torch.cuda.amp.autocast(
        enabled=use_amp,
    )


def parse_sample_id(filename):
    token = os.path.basename(
        filename
    ).split("_")[0]

    return int(float(token))


def polygon_to_mask(
    segmentation,
    height,
    width,
):
    mask = Image.new(
        "L",
        (width, height),
        0,
    )

    draw = ImageDraw.Draw(mask)

    if not isinstance(
        segmentation,
        list,
    ):
        return np.zeros(
            (height, width),
            dtype=np.uint8,
        )

    for polygon in segmentation:
        if (
            not isinstance(
                polygon,
                (list, tuple),
            )
            or len(polygon) < 6
        ):
            continue

        points = []

        for i in range(
            0,
            len(polygon) - 1,
            2,
        ):
            x = float(polygon[i])
            y = float(polygon[i + 1])

            points.append(
                (x, y)
            )

        if len(points) >= 3:
            draw.polygon(
                points,
                outline=1,
                fill=1,
            )

    return np.array(
        mask,
        dtype=np.uint8,
    )


def build_model():
    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
    )

    box_features = (
        model.roi_heads
        .box_predictor
        .cls_score.in_features
    )

    model.roi_heads.box_predictor = (
        FastRCNNPredictor(
            box_features,
            NUM_CLASSES,
        )
    )

    mask_features = (
        model.roi_heads
        .mask_predictor
        .conv5_mask.in_channels
    )

    model.roi_heads.mask_predictor = (
        MaskRCNNPredictor(
            mask_features,
            256,
            NUM_CLASSES,
        )
    )

    return model


def load_checkpoint(
    model,
    checkpoint_path,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
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

    if not isinstance(
        state,
        dict,
    ):
        raise ValueError(
            "Unsupported checkpoint format."
        )

    cleaned = {}

    for key, value in state.items():
        if key.startswith("module."):
            key = key[
                len("module.") :
            ]

        cleaned[key] = value

    missing, unexpected = (
        model.load_state_dict(
            cleaned,
            strict=False,
        )
    )

    if missing:
        raise RuntimeError(
            "Missing checkpoint parameters:\n"
            + "\n".join(
                missing[:30]
            )
        )

    if unexpected:
        print(
            f"WARNING: {len(unexpected)} "
            "unexpected checkpoint keys."
        )

    return checkpoint


class SolarMapZipDataset(
    Dataset
):
    def __init__(
        self,
        filenames,
        zip_path,
        coco_data,
    ):
        self.filenames = list(
            filenames
        )

        self.zf = zipfile.ZipFile(
            zip_path,
            "r",
        )

        self.member_lookup = {}

        for name in self.zf.namelist():
            if name.lower().endswith(
                ".png"
            ):
                self.member_lookup[
                    os.path.basename(name)
                ] = name

        self.image_by_filename = {}
        self.annotations_by_image_id = {}

        for image in coco_data[
            "images"
        ]:
            filename = os.path.basename(
                image["file_name"]
            )

            self.image_by_filename[
                filename
            ] = image

        for ann in coco_data[
            "annotations"
        ]:
            self.annotations_by_image_id.setdefault(
                ann["image_id"],
                [],
            ).append(ann)

        missing = [
            f
            for f in self.filenames
            if f not in self.member_lookup
        ]

        if missing:
            raise FileNotFoundError(
                "Images missing from ZIP:\n"
                + "\n".join(
                    missing[:20]
                )
            )

    def __len__(self):
        return len(
            self.filenames
        )

    def __getitem__(
        self,
        index,
    ):
        filename = self.filenames[
            index
        ]

        image_bytes = self.zf.read(
            self.member_lookup[
                filename
            ]
        )

        image = Image.open(
            io.BytesIO(
                image_bytes
            )
        ).convert("RGB")

        image_np = np.array(
            image,
            copy=True,
        )

        height, width = (
            image_np.shape[:2]
        )

        image_record = (
            self.image_by_filename.get(
                filename
            )
        )

        annotations = []

        if image_record is not None:
            annotations = (
                self.annotations_by_image_id.get(
                    image_record["id"],
                    [],
                )
            )

        masks = []
        boxes = []

        for ann in annotations:
            mask = polygon_to_mask(
                ann.get(
                    "segmentation",
                    [],
                ),
                height,
                width,
            )

            if int(mask.sum()) <= 0:
                continue

            ys, xs = np.where(
                mask > 0
            )

            if len(xs) == 0:
                continue

            x1 = float(
                xs.min()
            )
            y1 = float(
                ys.min()
            )
            x2 = float(
                xs.max() + 1
            )
            y2 = float(
                ys.max() + 1
            )

            if x2 <= x1 or y2 <= y1:
                continue

            masks.append(mask)
            boxes.append(
                [
                    x1,
                    y1,
                    x2,
                    y2,
                ]
            )

        image_tensor = (
            torch.from_numpy(
                image_np.transpose(
                    2,
                    0,
                    1,
                )
            ).float()
            / 255.0
        )

        if boxes:
            boxes_tensor = torch.tensor(
                boxes,
                dtype=torch.float32,
            )

            labels_tensor = torch.ones(
                (
                    len(boxes),
                ),
                dtype=torch.int64,
            )

            masks_tensor = (
                torch.from_numpy(
                    np.stack(
                        masks
                    )
                ).to(
                    torch.uint8
                )
            )

            areas_tensor = (
                masks_tensor
                .flatten(1)
                .sum(dim=1)
                .float()
            )

            iscrowd_tensor = (
                torch.zeros(
                    (
                        len(boxes),
                    ),
                    dtype=torch.int64,
                )
            )

        else:
            boxes_tensor = torch.zeros(
                (0, 4),
                dtype=torch.float32,
            )

            labels_tensor = torch.zeros(
                (0,),
                dtype=torch.int64,
            )

            masks_tensor = torch.zeros(
                (
                    0,
                    height,
                    width,
                ),
                dtype=torch.uint8,
            )

            areas_tensor = torch.zeros(
                (0,),
                dtype=torch.float32,
            )

            iscrowd_tensor = torch.zeros(
                (0,),
                dtype=torch.int64,
            )

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "masks": masks_tensor,
            "image_id": torch.tensor(
                [parse_sample_id(filename)],
                dtype=torch.int64,
            ),
            "area": areas_tensor,
            "iscrowd": iscrowd_tensor,
        }

        return (
            image_tensor,
            target,
        )


def load_training_manifest():
    positive_train = pd.read_csv(
        os.path.join(
            SPLIT_DIR,
            "positive_train.csv",
        )
    )

    negative_train = pd.read_csv(
        os.path.join(
            SPLIT_DIR,
            "negative_train.csv",
        )
    )

    hard_negative = pd.read_csv(
        HARD_NEGATIVE_CSV
    )

    if len(positive_train) != 1993:
        raise RuntimeError(
            f"Expected 1993 positive train images; "
            f"found {len(positive_train)}"
        )

    if len(negative_train) != 300:
        raise RuntimeError(
            f"Expected 300 negative train images; "
            f"found {len(negative_train)}"
        )

    if "filename" not in hard_negative.columns:
        raise ValueError(
            "hard_negative_train.csv "
            "has no filename column."
        )

    if "is_hard_negative" in hard_negative.columns:
        hard_negative = hard_negative[
            hard_negative[
                "is_hard_negative"
            ].astype(bool)
        ].copy()

    hard_filenames = (
        hard_negative[
            "filename"
        ]
        .astype(str)
        .tolist()
    )

    negative_filenames = set(
        negative_train[
            "filename"
        ]
        .astype(str)
    )

    positive_filenames = set(
        positive_train[
            "filename"
        ].astype(str)
    )

    negative_test = pd.read_csv(
        os.path.join(
            SPLIT_DIR,
            "negative_test_LOCKED.csv",
        )
    )

    positive_test = pd.read_csv(
        os.path.join(
            SPLIT_DIR,
            "positive_test_LOCKED.csv",
        )
    )

    negative_test_filenames = set(
        negative_test[
            "filename"
        ].astype(str)
    )

    positive_test_filenames = set(
        positive_test[
            "filename"
        ].astype(str)
    )

    for filename in hard_filenames:
        if filename not in negative_filenames:
            raise RuntimeError(
                f"Hard negative is not in negative_train: "
                f"{filename}"
            )

        if (
            filename in negative_test_filenames
            or filename in positive_test_filenames
        ):
            raise RuntimeError(
                f"TEST LEAKAGE DETECTED: {filename}"
            )

    if len(set(hard_filenames)) != len(
        hard_filenames
    ):
        raise RuntimeError(
            "Duplicate hard-negative filenames found."
        )

    if len(hard_filenames) != 54:
        raise RuntimeError(
            f"Expected 54 mined hard negatives; "
            f"found {len(hard_filenames)}"
        )

    # Original train occurrences:
    #   1993 positive + 300 negative = 2293
    #
    # Replay:
    #   one additional occurrence of every hard negative
    #
    # Total:
    #   1993 + 300 + 54 = 2347
    positive_rows = pd.DataFrame({
        "filename": positive_train[
            "filename"
        ].astype(str),
        "source": "positive",
        "replay": 0,
    })

    negative_rows = pd.DataFrame({
        "filename": negative_train[
            "filename"
        ].astype(str),
        "source": "negative",
        "replay": 0,
    })

    replay_rows = pd.DataFrame({
        "filename": hard_filenames,
        "source": "hard_negative_replay",
        "replay": 1,
    })

    train_df = pd.concat(
        [
            positive_rows,
            negative_rows,
            replay_rows,
        ],
        ignore_index=True,
    )

    train_df = train_df.sample(
        frac=1.0,
        random_state=SEED,
    ).reset_index(
        drop=True
    )

    return (
        train_df,
        positive_rows,
        negative_rows,
        replay_rows,
    )


def read_coco():
    with zipfile.ZipFile(
        ZIP_PATH,
        "r",
    ) as zf:
        return json.loads(
            zf.read(COCO_PATH)
        )


def dry_run(device):
    (
        train_df,
        _,
        _,
        replay_rows,
    ) = load_training_manifest()

    coco = read_coco()

    positive_filename = train_df[
        train_df["source"] == "positive"
    ]["filename"].iloc[0]

    negative_filename = train_df[
        train_df["source"] == "negative"
    ]["filename"].iloc[0]

    replay_filename = replay_rows[
        "filename"
    ].iloc[0]

    dataset = SolarMapZipDataset(
        [
            positive_filename,
            negative_filename,
            replay_filename,
        ],
        ZIP_PATH,
        coco,
    )

    print("=" * 80)
    print(
        "SOLARMAP — HARD-NEGATIVE REPLAY DRY RUN"
    )
    print("=" * 80)

    print(
        f"Total training occurrences: "
        f"{len(train_df)}"
    )

    print(
        f"Hard-negative replay occurrences: "
        f"{len(replay_rows)}"
    )

    model = build_model().to(device)
    model.train()

    for idx in range(
        len(dataset)
    ):
        image, target = dataset[
            idx
        ]

        print(
            f"\nSample {idx + 1}"
        )

        print(
            f"  Filename: "
            f"{dataset.filenames[idx]}"
        )

        print(
            f"  Image: "
            f"{tuple(image.shape)}"
        )

        print(
            f"  Target instances: "
            f"{target['labels'].shape[0]}"
        )

        print(
            f"  Boxes: "
            f"{tuple(target['boxes'].shape)}"
        )

        print(
            f"  Masks: "
            f"{tuple(target['masks'].shape)}"
        )

        image_device = image.to(
            device
        )

        target_device = {
            key: value.to(device)
            if torch.is_tensor(value)
            else value
            for key, value in target.items()
        }

        with torch.enable_grad():
            losses = model(
                [image_device],
                [target_device],
            )

        total_loss = sum(
            value
            for value in losses.values()
        )

        print(
            "  Total loss:",
            float(
                total_loss.detach().cpu()
            ),
        )

        model.zero_grad(
            set_to_none=True
        )

    print(
        "\nDRY RUN PASSED."
    )


def train(device):
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    (
        train_df,
        positive_rows,
        negative_rows,
        replay_rows,
    ) = load_training_manifest()

    coco = read_coco()

    dataset = SolarMapZipDataset(
        train_df[
            "filename"
        ].tolist(),
        ZIP_PATH,
        coco,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    print("=" * 80)
    print(
        "SOLARMAP — EXPERIMENT 2: "
        "HARD-NEGATIVE REPLAY"
    )
    print("=" * 80)

    print(
        f"PyTorch: {torch.__version__}"
    )

    print(
        f"TorchVision: {torchvision.__version__}"
    )

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Positive occurrences: "
        f"{len(positive_rows)}"
    )

    print(
        f"Original negative occurrences: "
        f"{len(negative_rows)}"
    )

    print(
        f"Hard-negative replay occurrences: "
        f"{len(replay_rows)}"
    )

    print(
        f"Total training occurrences/epoch: "
        f"{len(train_df)}"
    )

    print("\nBuilding model...")

    model = build_model().to(
        device
    )

    print(
        "Loading original baseline checkpoint..."
    )

    baseline_checkpoint = (
        load_checkpoint(
            model,
            BASELINE_CHECKPOINT,
            device,
        )
    )

    if isinstance(
        baseline_checkpoint,
        dict,
    ):
        if baseline_checkpoint.get(
            "epoch"
        ) is not None:
            print(
                "Baseline checkpoint epoch: "
                f"{baseline_checkpoint['epoch']}"
            )

    print(
        "Baseline checkpoint loaded."
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=STEP_SIZE,
            gamma=GAMMA,
        )
    )

    use_amp, scaler = (
        get_amp_components(
            device
        )
    )

    print(
        f"Mixed precision (AMP): "
        f"{use_amp}"
    )

    # Save the exact run manifest/config before training.
    train_df.to_csv(
        MANIFEST_PATH,
        index=False,
    )

    config = {
        "experiment": (
            "Hard-Negative Replay Fine-Tuning"
        ),
        "starting_checkpoint": (
            BASELINE_CHECKPOINT
        ),
        "positive_train_images": 1993,
        "negative_train_images": 300,
        "hard_negative_images": 54,
        "hard_negative_replay_factor": (
            HARD_NEGATIVE_REPLAY_FACTOR
        ),
        "total_training_occurrences": (
            len(train_df)
        ),
        "learning_rate": LEARNING_RATE,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "epochs": NUM_EPOCHS,
        "scheduler": (
            "StepLR(step_size=7,gamma=0.1)"
        ),
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "amp": use_amp,
        "locked_positive_test": 250,
        "locked_negative_test": 94,
        "test_set_used_for_training": False,
    }

    with open(
        CONFIG_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    if not os.path.exists(
        LOG_PATH
    ):
        pd.DataFrame(
            columns=[
                "epoch",
                "learning_rate",
                "train_loss",
                "elapsed_seconds",
            ]
        ).to_csv(
            LOG_PATH,
            index=False,
        )

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):
        model.train()

        running_loss = 0.0
        batches = 0

        epoch_start = time.time()

        for batch_idx, (
            images,
            targets,
        ) in enumerate(
            loader,
            start=1,
        ):
            images = [
                image.to(
                    device,
                    non_blocking=True,
                )
                for image in images
            ]

            targets = [
                {
                    key: (
                        value.to(
                            device,
                            non_blocking=True,
                        )
                        if torch.is_tensor(value)
                        else value
                    )
                    for key, value in target.items()
                }
                for target in targets
            ]

            optimizer.zero_grad(
                set_to_none=True
            )

            with autocast_context(
                use_amp
            ):
                loss_dict = model(
                    images,
                    targets,
                )

                losses = sum(
                    value
                    for value in loss_dict.values()
                )

            if not torch.isfinite(
                losses
            ):
                raise RuntimeError(
                    f"Non-finite loss at "
                    f"epoch {epoch}, "
                    f"batch {batch_idx}: "
                    f"{float(losses.detach().cpu())}"
                )

            if use_amp:
                scaler.scale(
                    losses
                ).backward()

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

            else:
                losses.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                optimizer.step()

            running_loss += float(
                losses.detach().cpu()
            )

            batches += 1

            if (
                batch_idx == 1
                or batch_idx % 100 == 0
            ):
                print(
                    f"Epoch {epoch}/{NUM_EPOCHS} | "
                    f"Batch {batch_idx}/"
                    f"{len(loader)} | "
                    f"Loss "
                    f"{float(losses.detach().cpu()):.5f}"
                )

        scheduler.step()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        mean_loss = (
            running_loss
            / max(
                batches,
                1,
            )
        )

        elapsed = (
            time.time()
            - epoch_start
        )

        lr = (
            optimizer.param_groups[
                0
            ]["lr"]
        )

        pd.DataFrame(
            [{
                "epoch": epoch,
                "learning_rate": lr,
                "train_loss": mean_loss,
                "elapsed_seconds": elapsed,
            }]
        ).to_csv(
            LOG_PATH,
            mode="a",
            header=False,
            index=False,
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "scheduler_state_dict": (
                scheduler.state_dict()
            ),
            "train_loss": mean_loss,
            "learning_rate": lr,
            "seed": SEED,
            "model_name": (
                "Hard_Negative_Replay_"
                "Mask_R-CNN_ResNet50_FPN"
            ),
            "experiment": (
                "Experiment 2 — Hard-Negative Replay"
            ),
            "positive_train_images": 1993,
            "negative_train_images": 300,
            "hard_negative_images": 54,
            "hard_negative_replay_factor": (
                HARD_NEGATIVE_REPLAY_FACTOR
            ),
            "total_training_occurrences": (
                len(train_df)
            ),
            "baseline_checkpoint": (
                BASELINE_CHECKPOINT
            ),
            "locked_test_protected": True,
        }

        epoch_path = os.path.join(
            OUTPUT_DIR,
            f"hard_negative_replay_epoch_{epoch}.pth",
        )

        latest_path = os.path.join(
            OUTPUT_DIR,
            "latest.pth",
        )

        torch.save(
            checkpoint,
            epoch_path,
        )

        torch.save(
            checkpoint,
            latest_path,
        )

        print(
            "\n"
            + "-" * 80
        )

        print(
            f"Epoch {epoch} complete | "
            f"Mean loss: "
            f"{mean_loss:.6f} | "
            f"LR: {lr:.7f} | "
            f"Time: {elapsed:.1f}s"
        )

        print(
            f"Saved: {epoch_path}"
        )

        print(
            "-" * 80
        )

    print(
        "\nTRAINING COMPLETE"
    )

    print(
        f"Checkpoints: {OUTPUT_DIR}"
    )

    print(
        f"Training log: {LOG_PATH}"
    )

    print(
        f"Manifest: {MANIFEST_PATH}"
    )


def main():
    if not os.path.exists(
        ZIP_PATH
    ):
        raise FileNotFoundError(
            ZIP_PATH
        )

    if not os.path.exists(
        BASELINE_CHECKPOINT
    ):
        raise FileNotFoundError(
            BASELINE_CHECKPOINT
        )

    if not os.path.exists(
        HARD_NEGATIVE_CSV
    ):
        raise FileNotFoundError(
            HARD_NEGATIVE_CSV
        )

    if not os.path.exists(
        SPLIT_DIR
    ):
        raise FileNotFoundError(
            SPLIT_DIR
        )

    seed_everything(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if "--dry-run" in os.sys.argv:
        dry_run(device)
    else:
        train(device)


if __name__ == "__main__":
    main()
