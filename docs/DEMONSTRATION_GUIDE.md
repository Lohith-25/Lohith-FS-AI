# SolarMap-India: Reviewer Demonstration & Walkthrough Guide
**Complete System Architecture, Demonstration Workflows, and Scientific Verification Manual**

> **Document Version:** 1.1 (Phase 14.5 End-to-End Browser Verification)  
> **Date:** October 2026  
> **Repository:** `d:\SolarMap-India`  
> **Target Audience:** Academic Reviewers, Systems Architects, and Technical Evaluators  
> **Production Model Checkpoint:** `outputs/HalfUNet/models/halfunet_best.pth` (1,928,450 parameters, Frozen)

---

## 1. Project Overview & Architecture

### 1.1 Purpose and Scope
**SolarMap-India** is an end-to-end machine learning and geospatial system engineered to detect rooftop solar photovoltaic (PV) arrays from high-resolution aerial and satellite imagery across India, calculate physical surface areas, and estimate clean energy generation and carbon offset potential using validated physical models.

### 1.2 System Architecture & Component Interactions

```
+---------------------------------------------------------------------------------+
|                                USER INTERFACE                                   |
|   React 19 + Vite (Vanilla CSS Tokens, Leaflet Geospatial Map, Interactive UI)  |
+---------------------------------------+-----------------------------------------+
                                        | HTTP REST (Port 8000)
                                        v
+---------------------------------------------------------------------------------+
|                               FASTAPI BACKEND                                   |
|   - Lifespan Model Manager (Preloads Half U-Net checkpoint into memory)         |
|   - Content Negotiation (Serves SPA HTML to browsers; JSON to API clients)      |
|   - Upload Validation (Allowed MIME types, 50 MB payload ceiling)                |
+-------------------+-------------------+--------------------+--------------------+
                    |                   |                    |
                    v                   v                    v
+-----------------------+ +-----------------------+ +-----------------------+
|  VARIABLE-RESOLUTION  | |  GEOSPATIAL METADATA  | |   PHYSICAL & ENERGY   |
|   INFERENCE PIPELINE  | |   EXTRACTION MODULE   | |    MODELING SUITE     |
| - Aspect letterbox    | | - GeoTIFF tags (PIL)  | | - Determinant Area    |
| - Tiled Hann blending | | - pyproj CRS resolver | | - First-order Energy  |
| - Direct evaluation   | | - Geodesic WGS84 dist | | - Grid CO2 Avoidance  |
+-----------+-----------+ +-----------+-----------+ +-----------+-----------+
            |                         |                         |
            +-------------------------+-------------------------+
                                      |
                                      v
+---------------------------------------------------------------------------------+
|                         PERSISTENCE & STORAGE (OPTIONAL)                        |
|   - MongoDB (Motor Async Driver) for Prediction History                         |
|   - Non-blocking: System operates with 100% feature fidelity if DB is offline   |
+---------------------------------------------------------------------------------+
```

---

## 2. Environment Setup & Prerequisites

### 2.1 System Requirements
* **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS
* **Python Runtime:** Python 3.10, 3.11, 3.12, or 3.13
* **Node.js Runtime:** Node.js v18+ and npm v9+ (only required if recompiling the frontend bundle)
* **GPU / Hardware:** CUDA-capable NVIDIA GPU recommended for sub-second tiled inference; CPU fallback is fully supported out of the box.

### 2.2 Installing Python Dependencies
Activate your virtual environment and install dependencies:
```bash
# In repository root: D:\SolarMap-India
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2.3 Model Checkpoint Verification
Verify that the production checkpoint is in place and undamaged before running the application:
```bash
python scripts/verify_model_artifact.py
```
Expected output:
* Target Checkpoint: `outputs/HalfUNet/models/halfunet_best.pth`
* Total Parameter Count: `1,928,450`
* Checkpoint Status: `PASSED`

---

## 3. Starting the Application

### 3.1 Pre-Flight Check
Run the quickstart launcher in check-only mode to verify dependencies, ports, and model weights without starting servers:
```bash
python scripts/start_demo.py --check-only
```

### 3.2 Standard Production Demonstration Mode (Unified Server)
The unified launcher starts the FastAPI backend, which serves both the REST API endpoints and the compiled React 19 single-page application from `frontend/dist` on port 8000:
```bash
python scripts/start_demo.py
```
* **Unified Web Application:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
* **Interactive API Documentation (Swagger UI):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **Health & Checkpoint Diagnostic Probe:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### 3.3 Development Mode (Vite Hot-Reloading)
To run the Vite development server with hot-reloading alongside the backend:
```bash
python scripts/start_demo.py --dev
```
* **Vite Development UI:** [http://localhost:5173/](http://localhost:5173/)
* **Backend API:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

### 3.4 Clean Process Termination
Press `Ctrl+C` in the terminal. The launcher captures the interrupt signal and terminates all child processes cleanly without leaving orphaned ports.

---

## 4. Step-by-Step Reviewer Demonstration Workflow

### Step 1: Uploading Satellite / Aerial Imagery
1. Navigate to [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in any modern web browser.
2. Under the **Upload Imagery** section:
   * Drag and drop an image file or click **Browse Files**.
   * Supported formats: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.webp`, `.bmp` (up to 50 MB).
   * Sample imagery is available in the repository at `dataset/Solar Images/Solar Images/images/` or `frontend/public/sample_aerial.png`.
3. Configure inference options:
   * **Inference Strategy:** `Auto (Recommended)`, `Tiled (Overlapping Hann)`, `Letterbox (Aspect-preserving)`, or `Direct`.
   * **Confidence Threshold:** Default is `0.50` (range 0.05 to 0.95).
   * **Minimum Component Area:** Default is `10 px` to filter isolated pixel noise.
4. Click **Run Solar Detection**.

### Step 2: Visual Inspection of Segmentation Output
Once inference completes, three synchronized image cards are presented:
1. **Source Image:** Original raster in native pixel dimensions and aspect ratio.
2. **Discrete Binary Mask:** High-contrast binary mask where detected solar arrays appear in emerald green (`#10b981`) over a dark background.
3. **Translucent Alpha Overlay:** Color-blended overlay showing exact spatial alignment of detected panels atop building rooftops.

### Step 3: Interpreting Pixel-Space Metrics
The **Metrics Dashboard** displays exact geometric measurements in image coordinate space:
* **Total Image Pixels:** Dimensions of the input raster ($W \times H$).
* **Detected Solar Pixels:** Exact count of positive classifications after thresholding and morphological filtering.
* **Coverage Percentage:** $\frac{\text{Solar Pixels}}{\text{Total Pixels}} \times 100\%$.
* **Connected Component Count:** Number of distinct, separated solar panel clusters.
* **Mean Cluster Size:** Average pixel footprint per detected solar installation.

### Step 4: Two-Path Physical Area & Ground Sampling Distance (GSD) Calibration
The **Physical Area Notice** clearly differentiates raw pixel counts from physical ground metric areas via two scientifically defensible pathways:

#### Path A: Automatic GeoTIFF Calibration
* When imagery contains embedded georeferencing tags (ModelPixelScaleTag 33550, ModelTiepointTag 33922, ModelTransformationTag 34264, GeoKeyDirectoryTag 34735):
  * **Projected CRS (e.g. UTM Zone 43N - EPSG:32643):** Displays `GSD: VALIDATED` | `Calibration: EXTRACTED_METADATA`. Calculated via affine Jacobian determinant $|a \cdot e - b \cdot d|$, handling rotated and sheared transformations.
  * **Geographic CRS (e.g. WGS84 - EPSG:4326):** Displays `GSD: VALIDATED (Geodesic Ellipsoidal)`. Metric ground distances are derived along latitude and longitude using WGS84 reference ellipsoid geodesics at the scene centroid.
  * **Strict Precedence:** If both embedded GeoTIFF metadata and user-entered custom GSD are provided, authentic GeoTIFF metadata takes strict precedence.

#### Path B: Manual User-Supplied Calibration (Standard PNG/JPEG Rasters)
* For standard rasters lacking embedded geospatial headers, the frontend and API provide an optional spatial calibration interface:
  * **User Inputs:**
    * Ground Sampling Distance (GSD): Strictly positive, finite number in meters per pixel.
    * Calibration Source: Documentation of flight log, drone model, camera sensor, or orthophoto report.
  * **Scientific Validation Safeguards:**
    * Enforces physical plausibility bounds for terrestrial Earth observation ($0.005\text{ m/px} \le \text{GSD} \le 100.0\text{ m/px}$).
    * Malformed, negative, zero, NaN, or infinite inputs return HTTP 400 with descriptive error messages; inputs are **never silently clamped**.
    * Prohibited unbacked generic assumptions (e.g. $1.0$, $0.5$, $0.3\text{ m/px}$) without documented citation are rejected.
  * **Provenance & Labeling Integrity:**
    * Flagged strictly as `USER_SUPPLIED` (`is_user_supplied=True`, `is_validated_metadata=False`).
    * **Never automatically labeled `VERIFIED`** without independent on-site ground-truth validation.
  * **Physical Area Calculation:**
    $$\text{Area per pixel} = \text{GSD} \times \text{GSD}$$
    $$\text{Detected Physical Area } (m^2) = \text{Solar Pixel Count} \times \text{Area per pixel}$$

#### Uncalibrated Fallback (Zero Calibration Provided)
* Displays: `GSD: UNAVAILABLE` | `Physical Area: null (insufficient_data)`.
* Safeguard: SolarMap-India strictly refuses to fabricate arbitrary pixel-to-meter multipliers.

### Step 5: Photovoltaic Energy & CO₂ Avoidance Modeling
Under **Solar Potential & Environmental Impact**:
1. When physical area ($m^2$) is available (via either Path A or Path B):
   $$E = A \times \text{GHI} \times \eta \times \text{PR}$$
   * $A$: Calibrated solar rooftop area ($m^2$).
   * $\text{GHI}$: Annual Global Horizontal Irradiation ($1,800\text{ kWh/m}^2/\text{year}$ default / NASA POWER / CEA scenario).
   * $\eta$: PV module efficiency ($20.0\%$).
   * $\text{PR}$: System Performance Ratio ($75.0\%$).
2. Avoided Carbon Emissions:
   $$\Delta \text{CO}_2 = E \times \text{EF}$$
   * $\text{EF}$: Central Electricity Authority (CEA) Indian national grid baseline emission factor ($0.716\text{ kg CO}_2/\text{kWh}$).
3. **Cascading Scientific Safeguards:**
   * If physical area is unavailable, energy potential strictly returns `energy_status = 'insufficient_data'` and `estimated_energy_kwh_year = null` with explicit insufficient-data reasons.
   * If annual energy is unavailable, avoided CO₂ emissions strictly returns `co2_status = 'insufficient_data'` and `co2_avoided_kg_year = null`.
   * GSD, solar irradiance, module efficiency, performance ratios, and emission factors are **never fabricated**.

### Step 6: Geospatial Leaflet Map Inspection
* When georeferencing coordinates are present (via GeoTIFF metadata or user-supplied latitude/longitude):
  * The **Geospatial Viewer** activates an interactive Leaflet map.
  * Overlays the geographic bounding box of the satellite scene over OpenStreetMap/satellite base layers.
  * Enables layer toggling (satellite base layer, bounding footprint, detection overlay).
* When imagery is unreferenced, the map displays a clear "Unreferenced Scene" empty state without crashing or rendering inaccurate coordinate fallbacks.

### Step 7: Prediction History (When MongoDB is Configured)
* If `MONGODB_URI` is set in the environment:
  * Click **History** in the top navigation bar to slide open the Prediction History Drawer.
  * Displays past prediction timestamps, filenames, solar pixel counts, and verified physical areas.
  * Clicking any past record reloads that prediction into the dashboard.
* If MongoDB is unconfigured or offline:
  * The history drawer displays an informative status: `Prediction history is disabled (database unconfigured)`.
  * All inference, visualization, and export capabilities continue to operate normally.

---

## 5. Verification Commands for Reviewers

Reviewers can execute the automated test suites using the commands below:

### 5.1 Backend Test Suite (235 Tests)
```bash
python -m unittest discover tests
```
* **Coverage:** Model loading, variable-resolution tiling, Hann blending, letterboxing, GeoTIFF IFD extraction, affine determinant area math, geographic geodesic distances, two-path image calibration (automatic GeoTIFF vs user-supplied GSD with scientific bounds), energy/CO2 models, database persistence, SPA static serving, launcher safety, and variable-dimension regression suite.
* **Expected Result:** `Ran 235 tests in ~35s ... OK` (0 failures, 0 errors across 20 test suites).

### 5.2 Frontend Code Quality & Static Analysis
```bash
cd frontend
npm run lint
```
* **Expected Result:** `Found 0 warnings and 0 errors.`

### 5.3 Frontend Production Build
```bash
cd frontend
npm run build
```
* **Expected Result:** `built in ~1.2s` generating `dist/index.html` and assets.

### 5.4 End-to-End Browser Automation Suite (Google Chrome via CDP)
```bash
python scratch/e2e_complete_workflow_test.py
```
* **Coverage:** Native Google Chrome 155 DevTools Protocol execution exercising all 20 end-to-end browser workflows:
  1. Zero fatal JavaScript errors (monitored via CDP `Runtime.exceptionThrown`).
  2. Upload interface visibility and parameter controls.
  3. Interactive benchmark sample loading (640 × 640).
  4. Real-time inference execution and loading state transitions.
  5. Dynamic results container rendering.
  6. Source aerial imagery presentation.
  7. High-contrast binary segmentation mask display.
  8. Spatial alignment of red-orange solar detection overlay.
  9. Strict aspect-ratio preservation without distortion across aspect ratios.
  10. Exact numeric agreement between UI reported solar pixels/coverage and raw binary mask artifacts.
  11. Functional artifact download buttons returning valid PNGs.
  12. Side-by-Side (3-Up) synchronized grid view.
  13. Interactive Compare Slider with draggable divider (tested at 25% and 75% splits).
  14. Large Focus View with pill toggling between Original, Overlay, and Mask.
  15. Mode switching state hygiene without residual DOM elements or stale state.
  16. Scientific safeguards: explicit `insufficient_data` for unreferenced rasters.
  17. Prediction history integration.
  18. Graceful offline resilience when MongoDB is unconfigured or unreachable.
  19. Comprehensive error handling for empty files, invalid file extensions, and corrupted rasters.
  20. Variable-dimension layout verification (256×256, 640×640, 637×463, 1920×1080, 4000×3000) with automatic strategy selection (direct, letterbox, tiled).
* **Expected Result:** `ALL 20 E2E BROWSER WORKFLOW VERIFICATIONS PASSED!` with 10 full-resolution screenshots saved to `scratch/browser_screenshots/`.

---

## 6. Troubleshooting Common Issues

| Issue / Symptom | Root Cause | Solution |
| :--- | :--- | :--- |
| **Port 8000 already in use** | An existing Uvicorn or other web server is running. | `python scripts/start_demo.py --backend-port 8080` |
| **Checkpoint not found** | Weights not placed in `outputs/HalfUNet/models/`. | Ensure `halfunet_best.pth` is at `outputs/HalfUNet/models/halfunet_best.pth`. |
| **Frontend UI not updating** | Browser cached old production assets. | Perform a hard refresh (`Ctrl + Shift + R`) or build frontend (`cd frontend && npm run build`). |
| **MongoDB connection timeout** | `MONGODB_URI` points to an unreachable host. | SolarMap-India gracefully falls back to stateless mode. To disable timeout warnings, unset `MONGODB_URI`. |
| **Payload Too Large (HTTP 413)** | Uploaded raster exceeds 50 MB limit. | Downsample the raster or crop the area of interest to under 50 MB. |

---

## 7. Scientific Assumptions, Boundaries & Known Limitations

To maintain uncompromising scientific honesty, reviewers should take note of the following architectural and empirical boundaries:

1. **Pixel Area vs. Physical Area:**
   * Pixel count is an image-space property. Converting pixels to square meters requires calibrated Ground Sampling Distance (GSD).
   * Unreferenced images (PNG/JPEG) **never produce physical area, annual energy, or carbon offsets**.
2. **GeoTIFF Verification Methodology:**
   * GeoTIFF metadata parsing was verified against synthetically constructed TIFF IFD tags matching the OpenGIS GeoTIFF specification across projected (UTM 43N) and geographic (WGS84) CRS.
   * Field validation across diverse, multi-sensor raw satellite GeoTIFF files (e.g. Sentinel-2 L2A, WorldView-3) remains an intentional future validation phase.
3. **Variable-Resolution Alignment vs. Segmentation Accuracy:**
   * Tiled inference with Hann blending preserves 1:1 coordinate alignment and small-object spatial features on large images without memory overflows.
   * **However**, output alignment does not establish that the convolutional network's accuracy generalizes across all spatial resolutions. If imagery is captured at 2 cm/pixel (drone) while the model was trained at 30 cm/pixel (satellite), visual features appear at a completely different scale.
4. **GSD Normalization Status:**
   * GSD normalization (artificially resampling images to a fixed GSD) has been **deliberately deferred**. Rescaling without multi-scale ground truth introduces severe interpolation artifacts into the Half U-Net.
5. **Testing Category Taxonomy & Verified Capabilities:**
   * **Automated Unit Tests:** 225 test cases in `tests/` covering isolated math, models, coordinate conversions, and database abstractions.
   * **API Integration Tests:** Exercised via FastAPI TestClient and live HTTP requests (`tests/test_production_serving.py`, `tests/test_demo_launcher.py`, `scratch/verify_api_and_workflows.py`).
   * **Actual Browser Tests:** Executed against native Google Chrome 155 via Chrome DevTools Protocol (`scratch/e2e_complete_workflow_test.py`) with 10 real screenshot captures.
   * **Scientific Evaluation on Annotated Images:** 375-image held-out test split benchmark with confusion matrix and per-image metrics (`docs/SEGMENTATION_BENCHMARK.md`).
   * **Manual / Demonstration Checks:** Pre-flight sanity checks via `scripts/start_demo.py --check-only`.
   * **Unverified Capabilities / Safe Deferrals:** Live MongoDB Atlas connectivity (stateless fallback verified), multi-sensor real-world satellite GeoTIFF ingest, and multi-scale GSD normalization.
