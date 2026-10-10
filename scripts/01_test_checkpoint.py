import os
import zipfile
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn


# ============================================================
# PATHS
# ============================================================

ZIP_PATH = r"C:\SolarMap-India\dataset\Solar Images.zip"

CHECKPOINT = (
    r"C:\SolarMap-India\checkpoints"
    r"\maskrcnn_baseline_epoch_10.pth"
)


# ============================================================
# FILE CHECK
# ============================================================

print("=" * 60)
print("SOLARMAP CHECKPOINT TEST")
print("=" * 60)

if not os.path.exists(ZIP_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{ZIP_PATH}"
    )

if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT}"
    )

print("Dataset:    FOUND")
print("Checkpoint: FOUND")


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nPyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Device:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0)
            .total_memory / 1024**3,
            2
        ),
        "GB"
    )


# ============================================================
# CREATE MODEL
# ============================================================

print("\nCreating Mask R-CNN...")

model = maskrcnn_resnet50_fpn(
    weights=None,
    weights_backbone=None,
    num_classes=2
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("Loading checkpoint...")

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False
)

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    elif (
        "model" in checkpoint
        and isinstance(
            checkpoint["model"],
            dict
        )
    ):

        state_dict = checkpoint["model"]

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


# Remove possible DataParallel prefix
clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):
        key = key[7:]

    clean_state_dict[key] = value


model.load_state_dict(
    clean_state_dict,
    strict=True
)

model.to(device)
model.eval()

print("Checkpoint loaded successfully.")


# ============================================================
# FIND ONE IMAGE
# ============================================================

print("\nOpening ZIP...")

with zipfile.ZipFile(
    ZIP_PATH,
    "r"
) as zf:

    png_files = [
        name
        for name in zf.namelist()
        if name.lower().endswith(".png")
    ]

    if not png_files:
        raise RuntimeError(
            "No PNG images found inside ZIP."
        )

    selected = png_files[0]

    print(
        "Testing image:",
        selected
    )

    with zf.open(selected) as f:

        image = Image.open(f).convert(
            "RGB"
        )


# ============================================================
# PREPARE IMAGE
# ============================================================

transform = transforms.ToTensor()

image_tensor = transform(
    image
).to(device)


# ============================================================
# GPU INFERENCE
# ============================================================

print("\nRunning inference...")

if torch.cuda.is_available():

    torch.cuda.synchronize()

with torch.inference_mode():

    output = model(
        [image_tensor]
    )[0]

if torch.cuda.is_available():

    torch.cuda.synchronize()


# ============================================================
# RESULTS
# ============================================================

scores = (
    output["scores"]
    .detach()
    .cpu()
)

boxes = (
    output["boxes"]
    .detach()
    .cpu()
)

masks = (
    output["masks"]
    .detach()
    .cpu()
)

print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)

print(
    "Image size:",
    image.size
)

print(
    "Raw predictions:",
    len(scores)
)

if len(scores) > 0:

    keep = scores >= 0.50

    print(
        "Predictions >= 0.50:",
        int(keep.sum())
    )

    print(
        "Maximum confidence:",
        round(
            float(scores.max()),
            4
        )
    )

    if keep.sum() > 0:

        print(
            "Mean confidence:",
            round(
                float(
                    scores[keep].mean()
                ),
                4
            )
        )

else:

    print(
        "No predictions returned."
    )


# ============================================================
# GPU MEMORY
# ============================================================

if torch.cuda.is_available():

    print("\nGPU memory:")

    print(
        "Allocated:",
        round(
            torch.cuda.memory_allocated()
            / 1024**2,
            2
        ),
        "MB"
    )

    print(
        "Reserved:",
        round(
            torch.cuda.memory_reserved()
            / 1024**2,
            2
        ),
        "MB"
    )

print("\n✅ STEP 3 COMPLETE")