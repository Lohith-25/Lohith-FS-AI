import argparse
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
# SOLARMAP — STEP 11
# BACKGROUND-AWARE MASK R-CNN TRAINING
#
# Purpose:
#   Fine-tune the verified baseline Mask R-CNN using:
#       1993 annotated positive images
#       + 300 real no-solar images
#
# Protected:
#   250 positive test images
#   94 no-solar test images
#
# Model:
#   Mask R-CNN + ResNet-50-FPN
#
# IMPORTANT:
#   Run --dry-run first.
# ============================================================


BASE = r"C:\SolarMap-India"

ZIP_PATH = os.path.join(BASE, "dataset", "Solar Images.zip")
CSV_PATH = os.path.join(BASE, "dataset", "EI_train_data(Sheet1).csv")
SPLIT_DIR = os.path.join(BASE, "splits")
CHECKPOINT_IN = os.path.join(
    BASE, "checkpoints", "maskrcnn_baseline_epoch_10.pth"
)

OUTPUT_DIR = os.path.join(
    BASE, "checkpoints", "background_aware"
)

LOG_PATH = os.path.join(
    OUTPUT_DIR, "training_log.csv"
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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def collate_fn(batch):
    return tuple(zip(*batch))


def parse_sample_id(filename):
    token = os.path.basename(filename).split("_")[0]
    return int(float(token))


def polygon_to_mask(segmentation, height, width):
    """
    Rasterize COCO polygon segmentation into a binary mask.
    The audited SolarMap COCO file uses polygon lists.
    """
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    if not isinstance(segmentation, list):
        return np.zeros((height, width), dtype=np.uint8)

    for polygon in segmentation:
        if not isinstance(polygon, (list, tuple)) or len(polygon) < 6:
            continue

        points = []
        for i in range(0, len(polygon) - 1, 2):
            x = float(polygon[i])
            y = float(polygon[i + 1])
            points.append((x, y))

        if len(points) >= 3:
            draw.polygon(points, outline=1, fill=1)

    return np.asarray(mask, dtype=np.uint8)


def build_model():
    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
    )

    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features_box,
        NUM_CLASSES,
    )

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        NUM_CLASSES,
    )

    return model


def load_checkpoint(model, path, device):
    checkpoint = torch.load(
        path,
        map_location=device,
    )

    state = checkpoint

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]

    if not isinstance(state, dict):
        raise ValueError(
            "Unsupported checkpoint format. Expected a state_dict "
            "or a dict containing model_state_dict/state_dict."
        )

    # Handle DataParallel-style keys.
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
            "Checkpoint is missing model parameters:\n"
            + "\n".join(missing[:30])
        )

    if unexpected:
        print(
            "WARNING: unexpected checkpoint keys:",
            len(unexpected),
        )

    return checkpoint


class SolarMapZipDataset(Dataset):
    def __init__(self, filenames, zip_path, coco_data):
        self.filenames = list(filenames)

        self.zf = zipfile.ZipFile(zip_path, "r")

        self.member_lookup = {}
        for name in self.zf.namelist():
            if name.lower().endswith(".png"):
                self.member_lookup[os.path.basename(name)] = name

        self.image_by_filename = {}
        self.annotations_by_image_id = {}

        for image in coco_data["images"]:
            filename = os.path.basename(image["file_name"])
            self.image_by_filename[filename] = image

        for ann in coco_data["annotations"]:
            self.annotations_by_image_id.setdefault(
                ann["image_id"], []
            ).append(ann)

        missing = [
            f for f in self.filenames
            if f not in self.member_lookup
        ]

        if missing:
            raise FileNotFoundError(
                "Images missing from ZIP:\n"
                + "\n".join(missing[:20])
            )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]

        member = self.member_lookup[filename]

        image_bytes = self.zf.read(member)
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        image_np = np.array(image, copy=True)
        height, width = image_np.shape[:2]

        image_record = self.image_by_filename.get(filename)

        annotations = []
        if image_record is not None:
            annotations = self.annotations_by_image_id.get(
                image_record["id"],
                []
            )

        masks = []
        boxes = []

        for ann in annotations:
            segmentation = ann.get("segmentation", [])

            mask = polygon_to_mask(
                segmentation,
                height,
                width,
            )

            area = int(mask.sum())

            if area <= 0:
                continue

            ys, xs = np.where(mask > 0)

            if len(xs) == 0 or len(ys) == 0:
                continue

            x1 = float(xs.min())
            y1 = float(ys.min())
            x2 = float(xs.max() + 1)
            y2 = float(ys.max() + 1)

            if x2 <= x1 or y2 <= y1:
                continue

            masks.append(mask)
            boxes.append([x1, y1, x2, y2])

        image_tensor = (
            torch.from_numpy(
                image_np.transpose(2, 0, 1)
            ).float() / 255.0
        )

        if boxes:
            boxes_tensor = torch.tensor(
                boxes,
                dtype=torch.float32,
            )
            labels_tensor = torch.ones(
                (len(boxes),),
                dtype=torch.int64,
            )
            masks_tensor = torch.from_numpy(
                np.stack(masks)
            ).to(torch.uint8)

            areas_tensor = masks_tensor.flatten(1).sum(
                dim=1
            ).float()

            iscrowd_tensor = torch.zeros(
                (len(boxes),),
                dtype=torch.int64,
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
                (0, height, width),
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

        sample_id = parse_sample_id(filename)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "masks": masks_tensor,
            "image_id": torch.tensor(
                [sample_id],
                dtype=torch.int64,
            ),
            "area": areas_tensor,
            "iscrowd": iscrowd_tensor,
        }

        return image_tensor, target


def load_manifests():
    pos_train = pd.read_csv(
        os.path.join(SPLIT_DIR, "positive_train.csv")
    )

    neg_train = pd.read_csv(
        os.path.join(SPLIT_DIR, "negative_train.csv")
    )

    pos_val = pd.read_csv(
        os.path.join(SPLIT_DIR, "positive_val.csv")
    )

    neg_val = pd.read_csv(
        os.path.join(SPLIT_DIR, "negative_val.csv")
    )

    required = [
        ("positive_train", pos_train, 1993),
        ("negative_train", neg_train, 300),
        ("positive_val", pos_val, 249),
        ("negative_val", neg_val, 75),
    ]

    for name, frame, expected in required:
        if "filename" not in frame.columns:
            raise ValueError(
                f"{name}.csv has no filename column."
            )
        if len(frame) != expected:
            raise ValueError(
                f"{name}: expected {expected} rows, found {len(frame)}"
            )

    train_df = pd.concat(
        [
            pos_train.assign(source="positive"),
            neg_train.assign(source="negative"),
        ],
        ignore_index=True,
    )

    val_df = pd.concat(
        [
            pos_val.assign(source="positive"),
            neg_val.assign(source="negative"),
        ],
        ignore_index=True,
    )

    # Deterministic shuffle.
    train_df = train_df.sample(
        frac=1.0,
        random_state=SEED,
    ).reset_index(drop=True)

    val_df = val_df.sample(
        frac=1.0,
        random_state=SEED,
    ).reset_index(drop=True)

    return train_df, val_df


def read_coco():
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        return json.loads(
            zf.read(COCO_PATH)
        )


def dry_run(device):
    print("=" * 80)
    print("SOLARMAP — BACKGROUND-AWARE MODEL DRY RUN")
    print("=" * 80)

    train_df, _ = load_manifests()
    coco = read_coco()

    positive_names = train_df[
        train_df["source"] == "positive"
    ]["filename"].tolist()

    negative_names = train_df[
        train_df["source"] == "negative"
    ]["filename"].tolist()

    if not positive_names or not negative_names:
        raise RuntimeError(
            "Training manifest does not contain both positive and negative images."
        )

    dry_run_names = positive_names[:1] + negative_names[:1]
    dry_run_sources = ["positive", "negative"]

    dataset = SolarMapZipDataset(
        dry_run_names,
        ZIP_PATH,
        coco,
    )

    model = build_model().to(device)
    model.train()

    for idx in range(len(dataset)):
        image, target = dataset[idx]

        n_instances = int(
            target["labels"].shape[0]
        )

        print(
            f"\nSample {idx + 1}: {dry_run_sources[idx]}"
        )
        print(
            f"  Filename: {dataset.filenames[idx]}"
        )
        print(
            f"  Image: {tuple(image.shape)}"
        )
        print(
            f"  Target instances: {n_instances}"
        )
        print(
            f"  Boxes shape: {tuple(target['boxes'].shape)}"
        )
        print(
            f"  Masks shape: {tuple(target['masks'].shape)}"
        )

        image_device = image.to(device)
        target_device = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in target.items()
        }

        with torch.enable_grad():
            losses = model(
                [image_device],
                [target_device],
            )

        total_loss = sum(
            value for value in losses.values()
        )

        print(
            "  Losses:",
            {
                k: float(v.detach().cpu())
                for k, v in losses.items()
            }
        )
        print(
            f"  Total loss: {float(total_loss.detach().cpu()):.6f}"
        )

        model.zero_grad(set_to_none=True)

    print("\nDRY RUN PASSED.")
    print("Positive target + empty no-solar target both work.")


def train(device):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df, val_df = load_manifests()
    coco = read_coco()

    train_dataset = SolarMapZipDataset(
        train_df["filename"].tolist(),
        ZIP_PATH,
        coco,
    )

    val_dataset = SolarMapZipDataset(
        val_df["filename"].tolist(),
        ZIP_PATH,
        coco,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    print("=" * 80)
    print("SOLARMAP — BACKGROUND-AWARE MASK R-CNN")
    print("=" * 80)

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Positive train images: {len(train_df[train_df.source == 'positive'])}")
    print(f"Negative train images: {len(train_df[train_df.source == 'negative'])}")
    print(f"Positive val images: {len(val_df[val_df.source == 'positive'])}")
    print(f"Negative val images: {len(val_df[val_df.source == 'negative'])}")

    print("\nBuilding model...")
    model = build_model().to(device)

    print("Loading baseline checkpoint...")
    baseline_checkpoint = load_checkpoint(
        model,
        CHECKPOINT_IN,
        device,
    )

    print("Baseline checkpoint loaded.")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=STEP_SIZE,
        gamma=GAMMA,
    )

    use_amp, scaler = get_amp_components(device)
    print(f"Mixed precision (AMP): {use_amp}")

    start_epoch = 1

    resume_path = os.path.join(
        OUTPUT_DIR,
        "latest.pth",
    )

    if os.path.exists(resume_path):
        print("\nResuming from latest background-aware checkpoint...")
        checkpoint = torch.load(
            resume_path,
            map_location=device,
        )

        load_checkpoint(
            model,
            resume_path,
            device,
        )

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        start_epoch = int(
            checkpoint.get("epoch", 0)
        ) + 1

        print(
            f"Resuming at epoch {start_epoch}"
        )

    # Write/create the training log.
    if not os.path.exists(LOG_PATH):
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
        start_epoch,
        NUM_EPOCHS + 1,
    ):
        model.train()

        running_loss = 0.0
        batches = 0

        epoch_start = time.time()

        for batch_idx, (images, targets) in enumerate(
            train_loader,
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

            with autocast_context(use_amp):
                loss_dict = model(
                    images,
                    targets,
                )

                losses = sum(
                    loss for loss in loss_dict.values()
                )

            if not torch.isfinite(losses):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch}, "
                    f"batch {batch_idx}: {float(losses.detach().cpu())}"
                )

            if use_amp:
                scaler.scale(losses).backward()
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                scaler.step(optimizer)
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

            if batch_idx % 100 == 0 or batch_idx == 1:
                print(
                    f"Epoch {epoch}/{NUM_EPOCHS} | "
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss {float(losses.detach().cpu()):.5f}"
                )

        scheduler.step()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        mean_loss = running_loss / max(
            batches,
            1,
        )

        elapsed = time.time() - epoch_start

        current_lr = optimizer.param_groups[0]["lr"]

        row = pd.DataFrame([{
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": mean_loss,
            "elapsed_seconds": elapsed,
        }])

        row.to_csv(
            LOG_PATH,
            mode="a",
            header=False,
            index=False,
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": mean_loss,
            "learning_rate": current_lr,
            "seed": SEED,
            "model_name": (
                "Background_Aware_Mask_R-CNN_ResNet50_FPN"
            ),
            "training_positive_images": 1993,
            "training_negative_images": 300,
            "validation_positive_images": 249,
            "validation_negative_images": 75,
            "baseline_checkpoint": CHECKPOINT_IN,
            "protected_test": True,
        }

        epoch_path = os.path.join(
            OUTPUT_DIR,
            f"background_aware_epoch_{epoch}.pth",
        )

        torch.save(
            checkpoint,
            epoch_path,
        )

        torch.save(
            checkpoint,
            resume_path,
        )

        print("\n" + "-" * 80)
        print(
            f"Epoch {epoch} complete | "
            f"Mean loss: {mean_loss:.6f} | "
            f"LR: {current_lr:.7f} | "
            f"Time: {elapsed:.1f}s"
        )
        print(
            f"Saved: {epoch_path}"
        )
        print("-" * 80)

    print("\nTRAINING COMPLETE")
    print(
        f"Checkpoints: {OUTPUT_DIR}"
    )
    print(
        f"Training log: {LOG_PATH}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate positive and empty no-solar targets without saving weights.",
    )

    args = parser.parse_args()

    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(ZIP_PATH)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(CSV_PATH)

    if not os.path.exists(SPLIT_DIR):
        raise FileNotFoundError(SPLIT_DIR)

    if not os.path.exists(CHECKPOINT_IN):
        raise FileNotFoundError(
            "Baseline checkpoint not found:\n"
            + CHECKPOINT_IN
        )

    seed_everything(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"PyTorch: {torch.__version__}")
    print(f"TorchVision: {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if args.dry_run:
        dry_run(device)
    else:
        train(device)


if __name__ == "__main__":
    main()
