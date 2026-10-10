# SolarMap-India

**AI-Based Solar Panel Instance Segmentation and Geospatial Intelligence**

SolarMap-India is a computer vision and geospatial analysis pipeline designed to detect solar photovoltaic (PV) panel arrays from high-resolution aerial and satellite imagery and extract standardized pixel-level spatial analytics. The project studies the practical limitations of solar-panel detection in remote-sensing images and provides a validated segmentation pipeline with robust pixel-area measurements.

---

## 1. Executive Overview

The project is built around the SolarMap-India aerial dataset and focuses on:

- **Solar-panel segmentation and region detection:** High-accuracy boundary identification.
- **Pixel-based area measurements:** Direct raster-based measurement of solar coverage and connected regions without unverified spatial scale assumptions.
- **Reproducible evaluation:** Benchmark evaluation using a locked train/validation/test split.
- **Ablation research:** Systematic analysis of background-aware training and hard-negative mining to minimize false positives on non-solar imagery.

---

## 2. Model Architecture: Half U-Net

The production segmentation backbone is a lightweight **Half U-Net** deep convolutional network optimized for computational efficiency and spatial boundary fidelity:

- **Model Architecture:** Half U-Net (reduced parameter U-Net variant with depth-scaled skip connections)
- **Input Channels:** 3 (RGB imagery)
- **Output Classes:** 2 (Background class `0`, Solar panel class `1`)
- **Base Feature Channels:** 32
- **Parameter Count:** 926,018 (~0.93M parameters)
- **Standard Input Resolution:** 256 × 256 px (interpolated to native image dimensions during postprocessing)
- **Foreground Probability Threshold:** 0.5 (configurable)

---

## 3. Pixel-Based Solar Area Measurements

SolarMap-India strictly separates 2D image pixel measurements from real-world physical area:

1. **Solar Pixel Counting:** Integer count of all pixels classified as solar panel (`mask == 1`).
2. **Total Pixels:** Calculated directly from native raster dimensions ($W \times H$).
3. **Solar Coverage Percentage:** Exact ratio:
   $$\text{Coverage (\%)} = \frac{\text{Solar Pixels}}{\text{Total Pixels}} \times 100$$
4. **Connected Solar Region Analysis:** 8-connectivity raster component analysis isolating distinct solar clusters with configurable noise filtering ($\text{min\_component\_area} \ge 20\text{ px}$).
5. **Physical Area Safety:** When satellite imagery lacks calibrated Ground Sampling Distance (GSD) or geospatial transform metadata, physical area estimates are not fabricated. The pipeline outputs:
   ```json
   {
       "physical_area_m2": null,
       "physical_area_hectares": null,
       "physical_area_status": "insufficient_data"
   }
   ```

### Standardized Output Schema

```json
{
    "image_width": 640,
    "image_height": 640,
    "total_image_pixels": 409600,
    "solar_area_pixels": 68451,
    "solar_coverage_percent": 16.7117,
    "detected_region_count": 43,
    "largest_region_area_pixels": 4750,
    "smallest_region_area_pixels": 616,
    "mean_region_area_pixels": 1591.8837,
    "physical_area_m2": null,
    "physical_area_hectares": null,
    "physical_area_status": "insufficient_data"
}
```

---

## 4. Dataset Audit & Locked Split

The audited SolarMap-India dataset contains:

- **3,000 CSV samples:** 2,531 positive samples (`has_solar = 1`), 469 negative samples (`has_solar = 0`)
- **3,005 PNG image files:** Dimensions: 640 × 640 px
- **2,500 images** represented in the COCO instance annotations (32,825 annotations)

### Locked Split Distribution

| Split | Positive Images | Negative Images |
| :--- | :---: | :---: |
| **Train** | 1,993 | 300 |
| **Validation** | 249 | 75 |
| **Test (Locked)** | 250 | 94 |

*Additional samples excluded:* 8 zero-annotation images, 31 positive CSV samples without corresponding COCO annotations.

---

## 5. Experimental Baseline & Ablation Results

Thresholds were selected using validation data before evaluating the locked test set:

| Model | Precision | Recall | F1 Score | Count MAE | Mask IoU | Area MAPE | Negative FP Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | **76.34%** | 79.93% | **78.10%** | **3.388** | **0.7826** | **27.44%** | 12.77% |
| **Background-Aware** | 73.46% | **80.89%** | 77.00% | 3.612 | 0.7740 | 31.52% | **10.64%** |
| **Hard-Negative Replay** | 75.38% | 78.85% | 77.07% | 3.400 | 0.7748 | 29.09% | 11.70% |

### Research Findings

- The baseline provides the best overall F1 balance on the locked test set.
- **Background-aware fine-tuning** reduced the false-positive rate on negative images from 12.77% to 10.64% and increased recall, but slightly reduced overall F1.
- **Hard-negative replay** reduced false predictions on negative images and improved count MAE, but did not exceed the baseline F1.
- These experiments serve as valuable ablation baselines rather than claimed improvements.

---

## 6. Project Structure

```
SolarMap-India/
├── config/                  # Environment templates (.env.example)
├── docs/                    # Architecture foundation & roadmap documentation
├── ml/                      # Machine learning modules
│   ├── analytics/           # Connected component analysis & mask metrics
│   ├── area/                # Spatial area & GSD validation pipelines
│   ├── confidence/          # Uncertainty & probability analysis
│   ├── energy/              # Photovoltaic energy model specifications
│   ├── external_data/       # Provenance & imagery resolution verification
│   ├── geospatial/          # Coordinate mapping & scale auditing
│   ├── inference/           # Half U-Net model definition & predict pipeline
│   └── validation/          # Validation runners
├── results/                 # Evaluation metric tables, confusion matrices, logs
├── scripts/                 # Training, evaluation, and experiment scripts
├── splits/                  # Locked train/val/test CSV manifests
├── tests/                   # Automated unit & integration test suites
├── requirements.txt         # Project Python dependencies
└── README.md                # Project documentation
```

---

## 7. Dataset & Model Limitations

- **Spatial Scale (GSD):** The current satellite dataset lacks embedded EXIF geospatial scale or GeoTIFF world file transforms. Physical area conversions ($m^2$, hectares) and capacity estimations require verified external GSD or georeferencing.
- **Panel Clustering vs. Individual Modules:** Contiguous panel arrays in dense solar farms or rooftop arrays appear as connected clusters. Individual panel counts cannot be determined without instance-level boundary models.
- **Out-of-Distribution Data:** Prediction accuracy depends on aerial perspective, resolution, and illumination similarity to the training distribution.

---

## 8. Installation

Ensure Python 3.10+ is installed:

```bash
# Clone the repository
git clone https://github.com/Lohith-25/Lohith-FS-AI.git
cd Lohith-FS-AI

# Install dependencies
pip install -r requirements.txt
```

---

## 9. Usage

### Python Inference API

```python
from ml.inference.predict import predict_image

# Run end-to-end segmentation
result = predict_image(
    image_path="path/to/aerial_image.png",
    checkpoint_path="outputs/HalfUNet/models/halfunet_best.pth",  # local checkpoint
    threshold=0.5,
    min_component_area=20,
)

# Access standardized pixel-area metrics
metrics = result.to_pixel_area_dict()
print(f"Solar Coverage: {metrics['solar_coverage_percent']}%")
print(f"Detected Solar Regions: {metrics['detected_region_count']}")
```

### Command-Line Interface

```bash
python ml/inference/predict.py --image "path/to/aerial_image.png"
```

### Backend REST API (FastAPI)

SolarMap-India includes a production-ready asynchronous REST API for remote inference and web integration:

#### 1. Start the API Server

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. Interactive API Documentation

Once started, open in your browser:
- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

#### 3. Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model": "HalfUNet",
  "checkpoint_available": true,
  "checkpoint_path": "D:\\SolarMap-India\\outputs\\HalfUNet\\models\\halfunet_best.pth",
  "device": "cuda",
  "error": null
}
```

#### 4. Submit an Aerial Image for Solar Prediction

Using `curl`:
```bash
curl -X POST "http://localhost:8000/predict?threshold=0.5&min_component_area=20" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@dataset/Solar Images/Solar Images/images/default/768.0_1.0.png"
```

Using Python (`requests`):
```python
import requests

url = "http://localhost:8000/predict"
with open("dataset/Solar Images/Solar Images/images/default/768.0_1.0.png", "rb") as f:
    response = requests.post(url, files={"file": ("image.png", f, "image/png")})

print(response.json())
```

Example Response:
```json
{
  "success": true,
  "model": "HalfUNet",
  "prediction_id": "c1f7b88e-7117-4952-b8d4-53c4826b01ef",
  "image_width": 640,
  "image_height": 640,
  "total_image_pixels": 409600,
  "solar_area_pixels": 68451,
  "solar_coverage_percent": 16.7117,
  "detected_region_count": 43,
  "largest_region_area_pixels": 4750,
  "smallest_region_area_pixels": 616,
  "mean_region_area_pixels": 1591.8837,
  "physical_area_m2": null,
  "physical_area_hectares": null,
  "physical_area_status": "insufficient_data",
  "mask_url": "/outputs/c1f7b88e-7117-4952-b8d4-53c4826b01ef/mask",
  "overlay_url": "/outputs/c1f7b88e-7117-4952-b8d4-53c4826b01ef/overlay"
}
```

#### 5. Retrieve Segmentation Artifacts

- **Binary Mask:** `http://localhost:8000/outputs/{prediction_id}/mask`
- **Visual Overlay:** `http://localhost:8000/outputs/{prediction_id}/overlay`

---

## 10. Automated Test Suite

Run the full automated test suite (unit tests, integration tests, dimension tests, numerical safety checks, API tests):

```bash
# Run all 108 tests
python -m unittest discover tests

# Or run specific test suites
python -m unittest tests/test_api.py
python -m unittest tests/test_inference.py tests/test_analytics.py
```

---

## 11. References & Credits

- **Author:** Lohith G
- **Focus:** Computer Vision · Remote Sensing · Instance Segmentation · Solar Energy Mapping
- **SolarMap-India Dataset:** [GitHub](https://github.com/SyedAejazAhmed/SolarMap-India-Dataset) · [Zenodo](https://zenodo.org/records/21449212)
- **Detailed Progress Report:** See [`SolarMap_India_Project_Progress_Report.md`](SolarMap_India_Project_Progress_Report.md) for the complete experimental log.
