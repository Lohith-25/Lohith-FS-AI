import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn


# ============================================================
# PATHS
# ============================================================

BASE = r"D:\SolarMap-India"

IMAGE_PATH = os.path.join(
    BASE,
    "dataset",
    "Solar Images",
    "images",
    "default",
    "100.0_1.0.png"
)

CHECKPOINT_PATH = os.path.join(
    BASE,
    "checkpoints",
    "maskrcnn_baseline_epoch_10.pth"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "results",
    "baseline_visualization"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "100.0_baseline_input_output.png"
)

# Validation-selected baseline threshold
CONF_THRESHOLD = 0.77


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("SOLARMAP-INDIA BASELINE INFERENCE")
print("=" * 70)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"Input image not found:\n{IMAGE_PATH}"
    )

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT_PATH}"
    )


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(IMAGE_PATH).convert("RGB")

print("\nInput image:")
print("Path:", IMAGE_PATH)
print("Size:", image.size)
print("Mode:", image.mode)


# ============================================================
# BUILD MODEL
# ============================================================

print("\nBuilding Mask R-CNN...")

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
    CHECKPOINT_PATH,
    map_location=device
)

if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

# Remove DataParallel prefix if present
clean_state_dict = {}

for key, value in state_dict.items():
    if key.startswith("module."):
        key = key[len("module."):]
    clean_state_dict[key] = value

model.load_state_dict(clean_state_dict, strict=True)

model.to(device)
model.eval()

print("Checkpoint loaded successfully.")


# ============================================================
# PREPARE INPUT
# ============================================================

tensor = transforms.ToTensor()(image).to(device)


# ============================================================
# INFERENCE
# ============================================================

print("\nRunning inference...")

with torch.inference_mode():
    prediction = model([tensor])[0]


boxes = prediction["boxes"].detach().cpu().numpy()
scores = prediction["scores"].detach().cpu().numpy()
masks = prediction["masks"].detach().cpu().numpy()


# ============================================================
# FILTER PREDICTIONS
# ============================================================

keep = scores >= CONF_THRESHOLD

boxes = boxes[keep]
scores = scores[keep]
masks = masks[keep]

print("\nPrediction summary:")
print("Raw detections:", len(prediction["scores"]))
print(f"Accepted detections (>={CONF_THRESHOLD:.2f}):", len(scores))

if len(scores) > 0:
    print("Highest confidence:", f"{scores.max():.4f}")
    print("Lowest accepted confidence:", f"{scores.min():.4f}")


# ============================================================
# VISUALIZATION
# ============================================================

image_np = np.array(image)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 8)
)

# ------------------------------------------------------------
# INPUT
# ------------------------------------------------------------

axes[0].imshow(image_np)
axes[0].set_title(
    "INPUT IMAGE",
    fontsize=16,
    fontweight="bold"
)
axes[0].axis("off")


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

axes[1].imshow(image_np)

for i, (box, score, mask) in enumerate(
    zip(boxes, scores, masks)
):

    x1, y1, x2, y2 = box.astype(int)

    # Mask R-CNN outputs [1,H,W]
    mask_binary = mask[0] > 0.5

    # Transparent mask overlay
    overlay = np.zeros_like(image_np)
    overlay[:, :, 1] = 255

    alpha = 0.30

    masked = mask_binary

    axes[1].imshow(
        np.where(
            masked[:, :, None],
            overlay,
            image_np
        ),
        alpha=alpha
    )

    # Bounding box
    rect = plt.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=False,
        linewidth=1.5
    )

    axes[1].add_patch(rect)

    # Detection number + confidence
    axes[1].text(
        x1,
        max(y1 - 3, 5),
        f"{i + 1}: {score:.2f}",
        fontsize=8,
        backgroundcolor="white"
    )


axes[1].set_title(
    f"BASELINE PREDICTIONS ({len(scores)} INSTANCES)",
    fontsize=16,
    fontweight="bold"
)

axes[1].axis("off")


# ============================================================
# SAVE
# ============================================================

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)

print("\nVisualization saved:")
print(OUTPUT_PATH)

plt.show()

print("\nDONE.")