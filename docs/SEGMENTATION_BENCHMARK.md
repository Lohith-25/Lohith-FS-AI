# SolarMap-India: Production Segmentation Benchmark Reference

> **Document Status:** Authoritative Benchmark Reference  
> **Release Date:** October 23, 2026  
> **Target Checkpoint:** Production Half U-Net ([`outputs/HalfUNet/models/halfunet_best.pth`](file:///d:/SolarMap-India/outputs/HalfUNet/models/halfunet_best.pth))  
> **Evaluation Split:** Held-Out Unseen Partition ([`outputs/HalfUNet/metrics/split_image_ids.json`](file:///d:/SolarMap-India/outputs/HalfUNet/metrics/split_image_ids.json))

---

## 1. Production Model & Checkpoint Provenance

| Parameter | Specification | Verification Method |
| :--- | :--- | :--- |
| **Model Architecture** | Half U-Net (2-class: Background vs Solar Panel) | [`ml/inference/model.py`](file:///d:/SolarMap-India/ml/inference/model.py) |
| **Input Tensor Size** | $3 \times 256 \times 256$ RGB, normalized to $[0.0, 1.0]$ | `transforms.Resize((256, 256))` + `ToTensor()` |
| **Parameter Count** | 1,928,450 parameters (FP32) | `sum(p.numel() for p in model.parameters())` |
| **Checkpoint Path** | `outputs/HalfUNet/models/halfunet_best.pth` | Local artifact storage |
| **Verified SHA-256 Checksum** | `7a297ade86f7ac8a8c7d6a55c9c0409671f9b5a6f3f0c5c29429610e7e74ac7e` | Python `hashlib.sha256` |
| **Training Epoch** | Epoch 18 (Best Validation Loss: 0.2331) | Checkpoint state dictionary |
| **Inference Threshold** | $\tau = 0.50$ (Constant, non-tuned) | `torch.softmax(logits, dim=1)[:, 1] >= 0.50` |

---

## 2. Dataset & Evaluation Split Provenance

* **Dataset:** SolarMap-India Satellite Imagery Dataset (Surat and regional Gujarat, India).
* **Annotations:** COCO Format ([`dataset/Solar Images/Solar Images/annotations/merged_instances_default.json`](file:///d:/SolarMap-India/dataset/Solar%20Images/Solar%20Images/annotations/merged_instances_default.json)):
  * Total annotated images: 2,500.
  * Total polygon instances: 32,825 (`category_id=1`, `hassolar`).
  * 8 images in COCO contain 0 polygon annotations (verified zero-solar controls).
* **Split Definition:** [`outputs/HalfUNet/metrics/split_image_ids.json`](file:///d:/SolarMap-India/outputs/HalfUNet/metrics/split_image_ids.json) (deterministic partition generated via `random.seed(42)`):
  * **Training Split:** 1,750 images ($70\%$).
  * **Validation Split:** 375 images ($15\%$).
  * **Held-Out Test Split:** **375 images** ($15\%$).
* **Split Isolation Verification:**
  * $\text{Train} \cap \text{Test} = \emptyset$ (0 overlap).
  * $\text{Val} \cap \text{Test} = \emptyset$ (0 overlap).
  * All 375 test images exist on disk in `dataset/Solar Images/Solar Images/images/default/` at native $640 \times 640$ resolution.
  * In the 375 test images, **373 images contain positive solar panel annotations** and **2 images are verified zero-solar negative controls** (sample IDs `526` and `667`).

> [!IMPORTANT]
> **Data Leakage Clarification:**  
> A secondary split file [`splits/positive_test_LOCKED.csv`](file:///d:/SolarMap-India/splits/positive_test_LOCKED.csv) (344 samples) was constructed for Phase 8 / Mask R-CNN experiments. That file contains **165 training images** from Half U-Net's training set.  
> The production Half U-Net must **strictly not** be evaluated on `positive_test_LOCKED.csv` as a held-out benchmark. The true held-out test split is strictly the 375-image partition defined in `split_image_ids.json`.

---

## 3. Quantitative Benchmark Metrics

Evaluated across all 375 unseen test images using the production Half U-Net checkpoint ($\tau = 0.50$):

### 3.1 Micro-Averaged Pixel Metrics (Pooled Confusion Matrix)

Micro metrics pool all pixel classifications across all 375 test images before computing ratios. This provides the direct measure of total physical solar area segmentation fidelity:

$$\text{Micro Dice} = \frac{2 \sum \text{TP}}{2 \sum \text{TP} + \sum \text{FP} + \sum \text{FN}}$$

| Metric | Measured Value | Formula / Derivation |
| :--- | :---: | :--- |
| **Foreground Solar Dice (F1)** | **0.7575** | $2 \times 5,774,340 / (2 \times 5,774,340 + 1,514,688 + 2,182,020)$ |
| **Foreground Solar IoU** | **0.6097** | $5,774,340 / (5,774,340 + 1,514,688 + 2,182,020)$ |
| **Foreground Precision** | **0.7922** | $5,774,340 / (5,774,340 + 1,514,688)$ |
| **Foreground Recall** | **0.7258** | $5,774,340 / (5,774,340 + 2,182,020)$ |
| **Overall Pixel Accuracy** | **0.9759** | $(5,774,340 + 144,128,952) / 153,600,000$ |
| **Pooled Ground-Truth Solar Coverage** | **5.18%** | $7,956,360 / 153,600,000$ total test pixels |
| **Pooled Predicted Solar Coverage** | **4.75%** | $7,289,028 / 153,600,000$ total test pixels |

### 3.2 Confusion Matrix Pixel Counts

| Confusion Element | Pixel Count | Proportion of Test Canvas | Meaning |
| :--- | :---: | :---: | :--- |
| **True Positive (TP)** | **5,774,340** | $3.76\%$ | Correctly segmented solar panel pixels |
| **False Positive (FP)** | **1,514,688** | $0.99\%$ | Background pixels incorrectly flagged as solar |
| **False Negative (FN)** | **2,182,020** | $1.42\%$ | Ground-truth solar pixels missed by model |
| **True Negative (TN)** | **144,128,952** | $93.83\%$ | Correctly identified background (roof, ground, vegetation) |
| **Total Evaluated Pixels** | **153,600,000** | $100.00\%$ | $375 \text{ images} \times 640 \times 640 \text{ pixels}$ |

### 3.3 Macro-Averaged Sample Metrics

Macro metrics compute each metric independently per image and average across images:

| Macro Metric Scope | Mean | Median | Standard Deviation |
| :--- | :---: | :---: | :---: |
| **Foreground Dice (Positive Samples, $N=373$)** | **0.7417** | **0.7716** | 0.1742 |
| **Foreground IoU (Positive Samples, $N=373$)** | **0.6035** | **0.6282** | 0.1798 |
| **Precision (Positive Samples, $N=373$)** | **0.7785** | **0.8354** | 0.2014 |
| **Recall (Positive Samples, $N=373$)** | **0.7380** | **0.8038** | 0.2091 |
| **Foreground Dice (All Samples, $N=375$)** | **0.7377** | **0.7716** | 0.1791 |
| **Foreground IoU (All Samples, $N=375$)** | **0.6003** | **0.6281** | 0.1834 |
| **Notebook Two-Class Macro Dice** | **0.8623** | — | — |

> [!NOTE]
> **Why Two-Class Macro Dice Differs from Foreground Dice:**  
> In `Half U-Net.ipynb`, `SegmentationMetrics` computed two-class macro Dice:
> $$\text{Two-Class Macro Dice} = \frac{\text{Dice}_{\text{background}} + \text{Dice}_{\text{solar}}}{2}$$
> Because background comprises ~95% of pixels, background Dice is consistently ~0.98+. Averaging $0.98$ with $0.74$ produces the previously cited notebook baseline of **0.8742**.  
> In rigorous semantic segmentation reporting, **Foreground Solar Dice (0.7575)** is the standard benchmark metric because it isolates detection of the target object class without background dilution.

---

## 4. Ground-Truth Coverage Distribution & Investigation of Low Coverage

In demonstration testing, an uploaded image exhibited ~2.21% solar coverage (6,513 solar pixels on a 637x463 image).

### Empirical Evidence from Ground-Truth Data
Analysis of the ground truth across the 375 held-out test images demonstrates that low solar coverage is an expected physical property of aerial rooftop scenes:
* **Ground-Truth 25th Percentile Coverage:** **2.26%**
* **Ground-Truth Median Coverage:** **3.92%**
* **Ground-Truth Mean Coverage:** **5.18%**
* **Test Images with GT Coverage $\le 3.0\%$:** **134 / 375 (35.73%)**
* **Test Images with Predicted Coverage $\le 3.0\%$:** **138 / 375 (36.80%)**

**Finding:** Observed solar coverage of ~2.2% is fully aligned with the 25th percentile of the ground-truth distribution. Rooftop panels occupy small localized footprints within large scenes containing roads, barren land, unpaneled roofs, and trees. Low coverage is an accurate reflection of rooftop geometry, not a model defect.

### Status of the Latest Visualization Image
* The image displayed in the latest UI visualization ([`backend/uploads/bbb3a42a-29c3-429c-b061-8f21bb6efba0/input.png`](file:///d:/SolarMap-India/backend/uploads/bbb3a42a-29c3-429c-b061-8f21bb6efba0/input.png)) is an external demonstration upload ($637 \times 463$ RGBA).
* Hash and feature matching confirm it does not exist in the training, validation, or test partitions of the SolarMap-India dataset.
* **Because no verified ground-truth annotation exists for this image, its segmentation accuracy cannot be mathematically established from the prediction alone.**

---

## 5. Known Failure Modes & Scientific Limitations

1. **False Positives on Reflective Clutter:**
   * High-reflectance corrugated tin roofs, skylights, and bright concrete edges can trigger false activations (accounting for 1.51M FP pixels across the test set).
2. **False Negatives on Weathered Panels:**
   * Heavily soiled, dust-covered, or low-contrast solar panels may be partially or completely missed (e.g., sample ID `970`, where 2 low-contrast panels were missed).
3. **Resolution Sensitivity:**
   * The production Half U-Net was trained on $640 \times 640$ satellite images resized to $256 \times 256$ (effective receptive field ~0.5m/px GSD).
   * Tiling and letterboxing allow the pipeline to ingest arbitrary aspect ratios and larger dimensions without distortion, but imagery with GSD outside 0.2m–0.8m represents domain shift.
4. **Physical Area Derivation Safeguard:**
   * Pixel counts are verified. However, physical area ($m^2$) and annual PV energy potential (kWh) strictly require calibrated Ground Sampling Distance (GSD) or GeoTIFF affine tags. Uncalibrated rasters (PNG/JPEG) strictly output `insufficient_data` to prevent fictitious energy projections.

---

## 6. Reproduction Protocol

To reproduce the exact benchmark metrics on the locked 375-image held-out test split:

```powershell
# Activate Python environment and navigate to project root
cd D:\SolarMap-India

# Run the verified standalone evaluation script
python scratch/evaluate_test_split.py
```

### Generated Artifacts
* Per-Image Breakdown: [`scratch/test_split_per_image_metrics.csv`](file:///d:/SolarMap-India/scratch/test_split_per_image_metrics.csv) (375 rows with TP, TN, FP, FN, Dice, IoU, Precision, Recall, and coverage).
* Aggregated JSON Summary: [`scratch/test_split_evaluation_summary.json`](file:///d:/SolarMap-India/scratch/test_split_evaluation_summary.json).
