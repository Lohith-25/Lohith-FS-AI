SolarMap-India

AI-Based Solar Panel Instance Segmentation and Geospatial Mapping

SolarMap-India is a computer vision project for detecting and segmenting solar-panel regions in satellite imagery. The project studies the practical limitations of solar-panel detection in remote-sensing images and evaluates whether an instance-segmentation approach can provide reliable panel-level predictions.

The project is built around the SolarMap-India dataset and focuses on:

Solar-panel instance detection and segmentation

Panel counting

Mask quality evaluation

False-positive analysis on non-solar imagery

Geographic and small-object challenges

Experimental improvements through background-aware training and hard-negative replay

Reproducible evaluation using a locked train/validation/test split

Research status: The current baseline is the strongest model among the experiments completed so far. The additional training strategies were retained as ablation experiments because they produced useful trade-offs but did not outperform the baseline on the final locked test set.

Problem Statement

Existing solar-panel detection methods can struggle when panels are small, visually similar to surrounding structures, or encountered in geographically different imagery.

This project investigates these challenges using satellite imagery and builds an instance-segmentation pipeline to identify individual solar-panel regions while measuring detection quality, counting accuracy, and false positives on non-solar scenes.

The work is intended to move beyond a simple "solar / no-solar" classifier by evaluating the quality and reliability of panel-level predictions.

Dataset

SolarMap-India

Dataset source:

GitHub: https://github.com/SyedAejazAhmed/SolarMap-India-Dataset

Zenodo: https://zenodo.org/records/21449212

Dataset audit

The audited dataset contains:

3,000 CSV samples

2,531 samples with has_solar = 1

469 samples with has_solar = 0

3,005 PNG image files

2,500 images represented in the COCO instance annotations

32,825 COCO annotations

Image size: 640 × 640

One COCO object category: hassolar

The COCO annotations cover 2,492 images with annotations, while 8 annotated-dataset images contain zero instances. Additional positive samples without COCO annotations were excluded from the segmentation split rather than assigning assumed labels.

Important dataset observations

A dataset audit identified duplicate filename variants for five sample IDs. One of the duplicate files was an empty file and was not treated as a valid image.

The audited annotations are polygon-based instance annotations. The project therefore evaluates them as instance targets, while avoiding an unsupported assumption that every polygon necessarily corresponds to one physical solar module rather than a panel region/array.

Research Focus

The project investigates several real-world challenges:

1. Small-object detection

Solar objects are very small relative to the 640 × 640 image.

Approximately:

84.02% of annotations occupy less than 0.5% of image area

96.84% occupy less than 1%

99.02% occupy less than 2%

This makes solar-panel detection a challenging small-object segmentation problem.

2. False positives on non-solar images

The negative portion of the dataset was used to test whether the detector incorrectly predicts solar panels in visually similar background structures.

3. Geographic variation

The dataset contains geographic variation, so a model that performs well on one region may not necessarily generalize to other scenes.

4. Counting accuracy

A useful mapping system should not only detect solar panels but should also provide a reasonable estimate of the number of detected instances.

Model

Baseline

The main model is:

Mask R-CNN + ResNet-50-FPN

Configuration used for the baseline experiment:

Model: Mask R-CNN

Backbone: ResNet-50-FPN

Classes: 2 (background, hassolar)

Epochs: 10

Learning rate: 0.0005

Optimizer: SGD

Momentum: 0.9

Weight decay: 0.0005

Scheduler: StepLR

Step size: 7

Gamma: 0.1

Mixed precision (AMP): used where supported

The baseline checkpoint is saved locally as:

checkpoints/maskrcnn_baseline_epoch_10.pth

Reproducible Data Split

A locked split was created before final evaluation.

Split

Positive

Negative

Train

1,993

300

Validation

249

75

Test

250

94

Additional samples excluded from the segmentation split:

8 zero-annotation images

31 positive CSV samples without corresponding COCO annotations

The final test set remained locked during the comparison of the evaluated models.

Evaluation Metrics

The project uses both object-detection and application-level metrics.

Detection / segmentation

Precision

Recall

F1-score

Mask IoU

COCO-style AP / AR metrics

Application-level

Count MAE

Area MAPE

Negative image false-positive rate

Number of false predictions on negative images

A prediction was considered a match for the custom evaluation when the predicted and ground-truth masks satisfied the defined IoU matching criterion.

Final Locked Test Results

Thresholds were selected using validation data before evaluating the locked test set.

Model

Precision

Recall

F1

Count MAE

Mask IoU

Area MAPE

Negative FP Rate

Baseline

76.34%

79.93%

78.10%

3.388

0.7826

27.44%

12.77%

Background-aware

73.46%

80.89%

77.00%

3.612

0.7740

31.52%

10.64%

Hard-negative replay

75.38%

78.85%

77.07%

3.400

0.7748

29.09%

11.70%

Interpretation

The baseline currently provides the best overall balance on the locked test set.

The two additional experiments were still valuable:

Background-aware fine-tuning reduced false positives on negative images and increased recall, but reduced overall F1 and worsened area error.

Hard-negative replay reduced the number of false predictions and slightly improved count MAE, but did not beat the baseline on overall F1 or recall.

These experiments are therefore treated as ablation / research findings, not as claimed improvements over the baseline.

Baseline Failure Analysis

The initial baseline evaluation showed a meaningful false-positive problem.

On the negative-image set:

Most images produced no predictions.

A smaller subset produced high-confidence false detections.

Some negative images contained multiple false detections.

False detections could have high confidence, showing that confidence thresholding alone is not sufficient.

Threshold sweeps also showed the expected precision-recall trade-off: increasing the confidence threshold reduced false positives but eventually caused substantial loss of recall.

Experiments

Experiment 1 — Background-Aware Fine-Tuning

Real negative images were included during fine-tuning so that the detector could learn background patterns that should not be classified as solar panels.

Result:

Negative false-positive rate decreased

Recall improved at the selected threshold

Overall F1 did not improve over baseline

Area MAPE became worse

Checkpoint:

checkpoints/background_aware/background_aware_epoch_10.pth

Experiment 2 — Hard-Negative Replay

The original baseline was run on the negative training set. High-confidence false predictions were identified and the corresponding images were replayed during training.

Training statistics:

300 negative training images evaluated

54 hard-negative images selected

Hard-negative rate: 18%

136 high-confidence false predictions observed during mining

Result:

Fewer false predictions on the negative test set

Slight improvement in count MAE relative to the baseline

Overall F1 and recall remained below the baseline

Checkpoint:

checkpoints/hard_negative_replay/hard_negative_replay_epoch_10.pth

Individual Image Test

For the example image:

100.0_1.0.png

the baseline produced:

Raw detections: 38

Accepted detections: 20

Ground-truth instances: 18

True positives: 18

False positives: 2

False negatives: 0

This corresponds to:

Precision: 90.00%

Recall: 100.00%

F1-score: 94.74%

A visualization was generated with:

Green: ground truth

Blue: true positives

Red: false positives

Yellow: false negatives

External Testing

Additional external images were tested qualitatively using the trained baseline model.

Because ground-truth annotations were not available for these images, the external evaluation is treated as qualitative only and is not included in the quantitative benchmark.

Project Structure

SolarMap-India/
│
├── checkpoints/
│   ├── maskrcnn_baseline_epoch_10.pth
│   ├── background_aware/
│   └── hard_negative_replay/
│
├── dataset/
│
├── results/
│   ├── baseline/
│   ├── hard_negative_mining/
│   └── ...
│
├── scripts/
│
├── splits/
│   ├── positive_train.csv
│   ├── positive_val.csv
│   ├── positive_test_LOCKED.csv
│   ├── negative_train.csv
│   ├── negative_val.csv
│   ├── negative_test_LOCKED.csv
│   ├── excluded_zero_annotation.csv
│   ├── excluded_positive_unannotated.csv
│   ├── dataset_manifest.csv
│   └── split_config.json
│
├── src/
├── test_images/
├── requirements.txt
└── SolarMap_India_Project_Progress_Report.md

Installation

Create a Python virtual environment and install the project dependencies:

python -m venv .venv

Windows:

.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

Verify PyTorch can access the GPU:

import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

Running the Project

The exact scripts used during the experiments are located in the scripts/ directory.

For single-image prediction, the baseline prediction workflow uses:

scripts/18_predict_single_image.py

The project also contains scripts for:

dataset inspection

split creation

baseline training

evaluation

threshold analysis

failure analysis

background-aware training

hard-negative mining and replay

visualization

Refer to the script files and project progress report for the exact experiment configurations.

Limitations

The current implementation has several limitations:

The COCO annotations are treated as instance targets, but the project does not claim that every polygon is necessarily one physical solar module.

The dataset is relatively small for a production-grade remote-sensing system.

Most annotated objects are very small, which limits segmentation accuracy.

External test images currently have no ground-truth labels, so their results are qualitative.

The baseline still produces false positives on a subset of non-solar images.

Geographic generalization has not been established as a production-level guarantee.

Area-based estimates depend on image scale assumptions and should not be interpreted as engineering-grade solar capacity measurements without additional calibration data.

Current Research Direction

The current findings suggest that the key research challenge is not simply increasing confidence thresholds.

A stronger future direction is to develop a more robust solar-panel segmentation system that addresses:

small-object detection

difficult background structures

geographic domain shift

confidence calibration

annotation quality

multi-scale feature learning

robust validation on geographically distinct scenes

The baseline and ablation results provide the experimental foundation for evaluating such improvements.

Reproducibility

The project maintains:

a dataset audit

a locked test split

saved model checkpoints

per-image evaluation results

threshold-selection results

failure-analysis outputs

experiment-specific result files

a detailed project progress report

The complete experimental record is documented in:

SolarMap_India_Project_Progress_Report.md

References and Dataset

SolarMap-India

GitHub:
https://github.com/SyedAejazAhmed/SolarMap-India-Dataset

Zenodo:
https://zenodo.org/records/21449212

The project is based on the SolarMap-India dataset and builds on research in satellite-image solar/PV segmentation, small-object detection, instance segmentation, and geographic robustness.

Author

Lohith G

Project: SolarMap-India

Focus: Computer Vision · Remote Sensing · Instance Segmentation · Solar Energy Mapping

