# SolarMap-India — Project Progress Report

## 1. Project Overview

### Project Theme
**Solar-panel detection and panel-level instance segmentation using satellite imagery**

### Dataset
**SolarMap-India**

### Main Objective
The project started from an existing solar-mapping dataset and baseline detection pipeline. The work focused on understanding the limitations of solar-panel detection in satellite imagery, especially:

- Small solar-panel objects
- Visually similar backgrounds
- False solar-panel detections
- Missed panels
- Difficulty separating nearby instances
- Generalization to unfamiliar imagery

The final goal was not to claim a new architecture without evidence, but to establish a strong baseline, perform systematic error analysis, investigate background-aware training strategies, and evaluate them using a locked test protocol.

---

# 2. Dataset Audit

## 2.1 Dataset Structure

The main dataset files are:

```text
SolarMap-India/
├── dataset/
│   ├── Solar Images.zip
│   └── EI_train_data(Sheet1).csv
├── checkpoints/
├── results/
├── scripts/
├── splits/
└── test_images/
```

## 2.2 CSV Attributes

File:

```text
EI_train_data(Sheet1).csv
```

Columns:

| Attribute | Description |
|---|---|
| `sampleid` | Unique image/sample identifier |
| `latitude` | Geographic latitude |
| `longitude` | Geographic longitude |
| `has_solar` | Solar-presence label: 0 or 1 |

Dataset size:

- Total samples: **3000**
- `has_solar = 1`: **2531**
- `has_solar = 0`: **469**

Class distribution:

- Solar present: **84.37%**
- Solar absent: **15.63%**

## 2.3 Image Dataset

Audit findings:

- Total PNG image files: **3005**
- Unique image/sample IDs represented: **3000**
- COCO images: **2500**
- COCO annotations: **32825**
- Number of categories: **1**
- Category name: `hassolar`
- Image size: **640 × 640**

COCO annotations contain:

- Image ID
- Category ID
- Bounding box
- Segmentation polygon
- Area
- Crowd/instance information
- Annotation ID

## 2.4 Data Quality Findings

The audit identified:

- 5 duplicate sample-ID filename cases
- 1 known zero-byte duplicate file
- 8 COCO images with zero annotations
- 31 positive CSV samples without COCO annotations
- 7 geographic outliers outside approximate India bounds

These cases were not silently deleted. They were explicitly tracked and handled during split construction.

## 2.5 Small-Object Severity

The annotation analysis showed that solar objects are predominantly small.

Approximate annotation area distribution:

- **84.02%** of annotations are below **0.5%** of image area
- **96.84%** are below **1%**
- **99.02%** are below **2%**
- **99.81%** are below **5%**

This supports the importance of small-object and fine-grained segmentation research.

---

# 3. Locked Experimental Split

A fixed train/validation/test split was created before the final experiments.

## Positive images

```text
Train:      1993
Validation:  249
Test:        250
```

## Negative images

```text
Train:       300
Validation:   75
Test:         94
```

## Excluded

```text
Zero-annotation COCO images:       8
Positive-but-unannotated samples: 31
```

All train/validation/test overlap checks passed.

The final test set is:

```text
250 positive + 94 no-solar = 344 images
```

This test set was kept locked throughout the model-development experiments.

---

# 4. Baseline Model

## Architecture

**Mask R-CNN + ResNet-50-FPN**

Mask R-CNN was selected as the baseline because the task requires **instance segmentation**, meaning the system should identify individual detected regions and provide:

- Class
- Confidence
- Bounding box
- Pixel-level segmentation mask

## Baseline Training

The original baseline was trained for **10 epochs**.

Checkpoint:

```text
checkpoints/maskrcnn_baseline_epoch_10.pth
```

Hardware used:

```text
GPU: NVIDIA GeForce RTX 2050 4 GB
PyTorch: 2.14.1+cu130
TorchVision: 0.29.1+cu130
CUDA: True
```

---

# 5. Baseline Evaluation

The baseline was evaluated on the locked test protocol.

At confidence threshold **0.75**:

| Metric | Baseline |
|---|---:|
| Precision | 75.91% |
| Recall | 80.54% |
| F1 | 78.16% |
| Count MAE | 3.404 |
| Mean matched mask IoU | 0.7822 |
| Area MAPE | 27.47% |
| No-solar FP rate | 12.77% |
| No-solar false predictions | 28 |

The baseline therefore provided a useful and measurable reference point.

---

# 6. Baseline Failure Analysis

The baseline produced false solar-panel detections on no-solar images.

Earlier analysis using a broader negative set found:

```text
469 no-solar images
90 images with false detections
19.19% image-level FP rate
```

Common error patterns investigated included:

- Background confusion
- Duplicate detections
- Possible over-segmentation
- Missed panels
- Predictions on visually similar structures

The false predictions were often high-confidence, showing that confidence alone does not guarantee correctness.

---

# 7. Failed Verification Experiments

Several possible post-processing and verification approaches were experimentally investigated.

## 7.1 Geometry-Based Candidate Verification

Features examined included:

- Area
- Aspect ratio
- Compactness
- Solidity
- Rectangularity
- Edge distance
- Neighbor relationships

The best simple geometry signal was not strong enough for reliable separation.

Conclusion:

**Rejected as the main solution.**

## 7.2 Handcrafted Logistic Regression / Random Forest Verifier

A candidate verifier was trained on human-checked examples.

Results were poor:

- Logistic Regression balanced accuracy: approximately **0.336**
- Random Forest balanced accuracy: approximately **0.161**

Conclusion:

**Rejected.**

## 7.3 Generic Visual Embedding Verifier

A ResNet-18 visual embedding approach was tested using grouped cross-validation.

Results were weak:

- Accuracy: approximately **0.211**
- Balanced accuracy: approximately **0.166**
- Macro F1: approximately **0.165**

Conclusion:

**Rejected.**

## 7.4 Global + Tiled Detection

Tiling was tested to improve small-object detection.

Results did not improve the baseline:

```text
Baseline F1:          0.7716
Global + tiles F1:    0.7145
```

Count MAE also became worse.

Conclusion:

**Rejected.**

These failed experiments were important because they prevented arbitrary modifications and helped narrow the actual research problem.

---

# 8. Human False-Positive Verification

A sample of 63 prediction candidates was manually reviewed.

Observed categories:

```text
TRUE_BACKGROUND                 22
POSSIBLE_UNANNOTATED_PANEL      16
DUPLICATE                       10
OVER_SEGMENTATION                9
AMBIGUOUS                        6
```

This showed that the error space was not simply "all false positives are background." Some apparent false detections may relate to annotation incompleteness or instance separation.

This reinforced the need for careful error analysis.

---

# 9. Experiment 1 — Background-Aware Fine-Tuning

## Motivation

Instead of changing the architecture, the first improvement tested whether explicit real no-solar images could reduce false positives.

### Training

Started from the original baseline checkpoint.

Training data:

```text
1993 positive images
+
300 real no-solar images
```

The model was fine-tuned for 10 epochs.

Checkpoint:

```text
checkpoints/background_aware/background_aware_epoch_10.pth
```

## Result

At threshold 0.75 on the final test comparison:

| Metric | Baseline | Background-aware |
|---|---:|---:|
| Precision | 75.91% | 75.53% |
| Recall | 80.54% | 78.88% |
| F1 | **78.16%** | 77.17% |
| Count MAE | 3.404 | **3.400** |
| Mean Mask IoU | **0.7822** | 0.7754 |
| Area MAPE | **27.47%** | 28.33% |
| No-solar FP rate | 12.77% | **9.57%** |
| No-solar false predictions | 28 | **22** |

Interpretation:

- False-positive behavior improved.
- Count MAE improved slightly.
- Overall F1, recall, IoU and area accuracy declined.

Conclusion:

**The approach demonstrated a precision/recall trade-off rather than an overall improvement.**

---

# 10. Experiment 2 — Hard-Negative Replay

## Motivation

Instead of treating every negative image equally, the baseline was run on the 300 negative training images to identify difficult backgrounds.

A hard negative was defined as:

> A known no-solar training image on which the baseline produced at least one prediction with confidence ≥ 0.75.

## Hard-Negative Mining Result

From the 300 negative training images:

```text
Hard negatives: 54
Hard-negative rate: 18.00%
High-confidence false predictions: 136
```

The test set was not used for mining.

## Hard-Negative Replay Training

Started again from the original baseline checkpoint.

Training occurrences per epoch:

```text
1993 positive
+
300 normal negative
+
54 hard-negative replay
=
2347 occurrences
```

Checkpoint:

```text
checkpoints/hard_negative_replay/hard_negative_replay_epoch_10.pth
```

## Result at threshold 0.75

| Metric | Baseline | Hard-negative Replay |
|---|---:|---:|
| Precision | 75.91% | **76.11%** |
| Recall | **80.54%** | 77.71% |
| F1 | **78.16%** | 76.91% |
| Count MAE | 3.404 | **3.344** |
| Mean Mask IoU | **0.7822** | 0.7752 |
| Area MAPE | **27.47%** | 28.39% |
| No-solar FP rate | 12.77% | **10.64%** |
| No-solar false predictions | 28 | **23** |

Interpretation:

- Precision improved slightly.
- Count MAE improved.
- Number of false predictions decreased.
- Recall and F1 declined.

Conclusion:

**Hard-negative replay reduces false positives but does not outperform the baseline overall.**

---

# 11. Validation-Based Threshold Selection

To avoid tuning the locked test set, confidence thresholds were selected using validation data.

Validation split:

```text
249 positive
75 negative
```

Selected thresholds:

| Model | Selected threshold |
|---|---:|
| Baseline | **0.77** |
| Background-aware | **0.66** |
| Hard-negative replay | **0.72** |

Selection rule:

1. Maximize validation F1
2. Tie-break with lower negative-image FP rate
3. Tie-break with higher recall

The test set was not used for threshold selection.

---

# 12. Final Locked Test Evaluation

After validation-based threshold selection, the models were evaluated on the untouched locked test set.

## Final results

| Model | Threshold | Precision | Recall | F1 | Count MAE | Mean Mask IoU | Area MAPE | Negative FP Rate | False Predictions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Baseline** | **0.77** | **76.34%** | 79.93% | **78.10%** | **3.388** | **0.7826** | **27.44%** | 12.77% | 28 |
| Background-aware | 0.66 | 73.46% | **80.89%** | 77.00% | 3.612 | 0.7740 | 31.52% | **10.64%** | 26 |
| Hard-negative replay | 0.72 | 75.38% | 78.85% | 77.07% | 3.400 | 0.7748 | 29.09% | **11.70%** | **24** |

## Final conclusion

The **baseline Mask R-CNN + ResNet-50-FPN remains the best overall model** based on F1, precision, mask IoU, count MAE and area error.

The experimental approaches provide useful research findings:

- Background-aware training reduced false-positive rate and slightly improved recall, but reduced overall F1.
- Hard-negative replay reduced the number of false predictions and improved precision/count MAE, but reduced recall and F1.
- Therefore, the investigated strategies introduce a clear **precision-recall / false-positive trade-off**.

The project therefore does not claim that the proposed training modifications outperform the baseline.

---

# 13. Individual Image Analysis

A representative image was examined:

```text
100.0_1.0.png
```

Ground-truth instances:

```text
18
```

Predictions:

```text
20
```

IoU-based matching:

```text
TP = 18
FP = 2
FN = 0
```

Image-level metrics:

```text
Precision = 90.00%
Recall    = 100.00%
F1        = 94.74%
```

This demonstrates that the model can perform very well on individual images even though its overall dataset performance is lower.

The ground-truth vs prediction visualization was also generated to distinguish:

```text
Green = Ground Truth
Blue  = True Positive
Red   = False Positive
Yellow = Missed Ground Truth
```

---

# 14. External Real-World Testing

Two unseen external images were tested using the final baseline model.

This was a **qualitative external test**, not a numerical benchmark, because the external images did not have ground-truth annotations.

For one external image:

```text
Raw detections: 20
Accepted detections: 4
```

The model generated confidence scores and segmentation masks for the accepted detections.

This test showed:

### Positive observations

- The model can run on unseen imagery.
- It produces panel-level masks and confidence scores.
- Confidence filtering removes many weak candidates.

### Limitation observed

- Some accepted detections can appear visually similar to rooftop/background structures.
- This suggests a domain-generalization challenge.

This supports future research into cross-domain robustness.

---

# 15. Current Final Model

The model currently selected for the project is:

```text
Architecture:
Mask R-CNN + ResNet-50-FPN

Checkpoint:
checkpoints/maskrcnn_baseline_epoch_10.pth

Confidence threshold:
0.77
```

Final locked-test performance:

```text
F1          = 78.10%
Precision   = 76.34%
Recall      = 79.93%
Area MAPE   = 27.44%
Mask IoU    = 78.26%
```

---

# 16. Current Research Finding

The major finding from the completed experiments is:

> **Solar-panel instance segmentation on satellite imagery is strongly affected by small object size and visually confusing backgrounds. Explicit training with real no-solar images and baseline-derived hard negatives can reduce false-positive detections, but these strategies may also reduce overall detection performance.**

This demonstrates that simply adding negative samples is not sufficient to solve the problem.

---

# 17. Research Limitations Identified

The experiments suggest several areas that remain open:

1. **Small-object segmentation**
   - Most annotated solar regions occupy a very small portion of the 640×640 image.

2. **Background confusion**
   - Some rooftop/background structures produce solar-like detections.

3. **Geographic/domain variation**
   - External imagery can have different visual characteristics from the training data.

4. **Instance separation**
   - Nearby solar structures may be difficult to separate correctly.

5. **Annotation uncertainty**
   - Some predicted regions may correspond to visually plausible but unannotated structures, so not every apparent false positive is necessarily a true real-world false positive.

---

# 18. Next Research Direction

Before adding another model modification, the existing solar-panel segmentation literature should be systematically reviewed.

The literature review should compare:

```text
Paper
Dataset
Architecture
Semantic vs Instance segmentation
Problem addressed
Small-object strategy
Background handling
Boundary refinement
Geographic robustness
Reported limitation
Possible research gap
```

The next model change should be based on a clearly identified research gap rather than repeated blind tuning.

---

# 19. Reproducibility

The project contains scripts for:

```text
01_test_checkpoint.py
02_pilot_candidates.py
03_visualize_pilot.py
04_diagnose_fp.py
05_visualize_fp_categories.py
06_human_verify_fp.py
07_analyze_human_labels.py
08_train_pilot_verifier.py
09_visual_verifier.py
10_lock_splits.py
10_tile_pilot.py
11_train_background_aware.py
12_evaluate_final_models.py
13_mine_hard_negatives.py
14_train_hard_negative_replay.py
15_evaluate_three_models.py
16_select_validation_thresholds.py
17_final_locked_test_selected_thresholds.py
18_predict_single_image.py
19_ground_truth_vs_prediction.py
20_analyze_single_image_matching.py
21_select_best_checkpoint_validation.py
22_external_real_world_test.py
```

Important artifacts include:

```text
splits/
checkpoints/
results/
test_images/
```

The project has now been copied from the C: drive to:

```text
D:\SolarMap-India
```

The original working environment remains:

```text
C:\Users\lohit\solarmap-env
```

---

# 20. Current Status

```text
Dataset audit                         ✅
Data-quality analysis                 ✅
Locked split creation                 ✅
Baseline training                    ✅
Baseline evaluation                  ✅
False-positive analysis              ✅
Human verification                   ✅
Candidate verifier experiments       ✅
Tiling experiment                    ✅
Background-aware training            ✅
Hard-negative mining                 ✅
Hard-negative replay training        ✅
Validation threshold selection       ✅
Locked final evaluation              ✅
Individual prediction analysis       ✅
External qualitative testing         ✅
Project moved to D: drive             ✅
```

## Current status

**Baseline Mask R-CNN + ResNet-50-FPN is the current final-performing model.**

The research work now has a complete experimental pipeline and an evidence-based understanding of where the model succeeds and fails.
