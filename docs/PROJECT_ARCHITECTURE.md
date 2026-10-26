# Project Architecture: SolarMap-India

**Project Title:** SolarMap-India: AI-Based Geospatial Solar Intelligence for Solar Panel Mapping and Deployment Assessment  
**Document Status:** Architecture Foundation (Phase 0)  
**Last Updated:** Phase 0 Baseline  

---

## 1. Executive Overview

SolarMap-India is an AI-powered geospatial platform designed for detecting existing rooftop solar photovoltaic (PV) installations and evaluating potential solar deployment capacity across India using high-resolution satellite/aerial imagery.

The core computer vision engine is powered by a trained **Half U-Net** segmentation model that achieved state-of-the-art results during experimental evaluation (Test Dice/F1: **0.8742**, IoU: **0.7946**).

---

## 2. Integrated Full-Stack Architecture Pipeline (Phase 13)

The diagram below reflects the active, verified end-to-end data and execution flow across the integrated frontend, backend, analytics engines, geospatial map, and MongoDB persistence layer:

```mermaid
flowchart TD
    User["End User / Analyst"]
    ReactFrontend["React 19 Frontend Dashboard & History Drawer [VERIFIED]"]
    FastAPIBackend["FastAPI Backend Service [VERIFIED]"]
    MLInference["ML Inference Engine (PyTorch CUDA/CPU) [VERIFIED]"]
    HalfUNet["Half U-Net Model (Trained Checkpoint: halfunet_best.pth) [VERIFIED]"]
    SolarAnalytics["Solar Segmentation Analytics (Connected Components) [VERIFIED]"]
    GeoIntel["Geospatial Catalog (Level B Georeferencing) [VERIFIED]"]
    LeafletMap["Interactive Leaflet Map (CartoDB Dark / OSM) [VERIFIED]"]
    AreaEngine["Physical Area Calculator (GSD Safeguards) [VERIFIED]"]
    EnergyEngine["Solar Energy Model (E = A * GHI * eta * PR) [VERIFIED]"]
    CO2Engine["CO2 Avoidance Model (CEA Grid Factor) [VERIFIED]"]
    MongoDB[("MongoDB Database (Async Motor Persistence) [VERIFIED]")]

    User -->|Image Upload / Sample Load| ReactFrontend
    ReactFrontend -->|POST /predict multipart/form-data| FastAPIBackend
    FastAPIBackend -->|Preprocessed Tensor| MLInference
    MLInference --> HalfUNet
    HalfUNet --> SolarAnalytics
    SolarAnalytics --> GeoIntel
    GeoIntel --> LeafletMap
    SolarAnalytics --> AreaEngine
    AreaEngine --> EnergyEngine
    EnergyEngine --> CO2Engine
    CO2Engine -->|Structured Analytics & Provenance| MongoDB
    FastAPIBackend -->|Standardized PredictResponse| ReactFrontend
    ReactFrontend -->|GET /api/predictions/history| MongoDB
```

---

## 3. Component Status & Description

| Component | Status | Description |
| :--- | :--- | :--- |
| **User Interface** | `COMPLETE & VERIFIED` | Interactive web dashboard with 3-up inspection, comparison slider, Leaflet geospatial map, parameter controls, and analytics. |
| **React Frontend** | `COMPLETE & VERIFIED` | React 19 + Vite 8 SPA with Leaflet v1.9.4 interactive mapping, Obsidian dark theme, and dual basemap support. |
| **FastAPI Backend** | `COMPLETE & VERIFIED` | Asynchronous REST API providing upload handling, model inference orchestration, geospatial queries, history, and reporting. |
| **ML Inference Engine** | `COMPLETE & VERIFIED` | Standalone, lightweight inference module executing PyTorch tensor preprocessing, model forward pass, and mask postprocessing. |
| **Half U-Net Model** | `VERIFIED` | 2-class semantic segmentation architecture trained on 2500 aerial images (640x640 / resized to 256x256). Checkpoint verified at `outputs/HalfUNet/models/halfunet_best.pth`. |
| **Solar Segmentation Analytics** | `COMPLETE & VERIFIED` | Converts raw probability masks into strictly derived pixel metrics, connected components, bounding boxes, centroids, filtering, and machine-readable JSON/CSV reports. |
| **Confidence & Uncertainty Analysis** | `COMPLETE & VERIFIED` | Computes per-pixel softmax probability maps, prediction confidence, probability-derived uncertainty, 10-bin histograms, and region-level confidence categorization. |
| **Geospatial Intelligence & Mapping** | `LEVEL B (COMPLETE & VERIFIED)` | Maps images to scene centroid coordinates via sample catalog (`GeospatialCatalog`); visualized interactively via `GeospatialMap.jsx` with Level B constraints; physical scale uncalibrated. |
| **Energy Estimation** | `COMPLETE & VERIFIED` | Estimates annual kWh generation potential based on panel area, panel efficiency, standard solar irradiance (GHI/DNI), and system performance ratio (module verified; physical scale uncalibrated). |
| **CO2 Estimation** | `COMPLETE & VERIFIED` | Estimates metric tonnes of greenhouse gas ($CO_2$) emissions avoided per year using regional grid emission factors (e.g., CEA baseline factors for India). |
| **MongoDB** | `COMPLETE & VERIFIED` | Persistent asynchronous document storage for prediction records, segmentation statistics, physical area / energy / CO2 statuses, artifact references, and pagination query endpoints. |
| **Deployment & Containerization** | `COMPLETE & VERIFIED (Task 14.1)` | Multi-stage Docker container (Node 22 + Python 3.11 slim), Docker Compose orchestration, unified static asset serving from FastAPI, and model verification script (`scripts/verify_model_artifact.py`). |
| **Deployment Assessment** | `PLANNED` | Rooftop/ground suitability assessment, roof orientation, shading impact assessment, and deployment recommendation scoring. |

---

## 4. Scientific Requirements and Principles

To ensure scientific rigor and prevent misleading outputs, all future analytical modules must adhere to the following 10 foundational principles:

1. **Solar segmentation is the primary ML task:** The machine learning model is exclusively trained to segment solar panels from imagery; downstream energy and environmental metrics are analytical derivations, not direct model predictions.
2. **Pixel area is not automatically physical area:** Pixel counts cannot be directly treated as square meters without calibrating spatial scale.
3. **Physical area requires spatial resolution:** Calculating physical area ($m^2$) strictly requires the Ground Sampling Distance (GSD in meters/pixel) or geographic bounding coordinates.
4. **Energy generation is an estimate:** Photovoltaic electricity generation calculations ($E = A \cdot \eta \cdot G \cdot PR$) are model-based estimates and must clearly disclose assumptions regarding solar irradiance ($G$), panel efficiency ($\eta$), and performance ratio ($PR$).
5. **CO2 avoidance is an estimate:** Carbon offset metrics depend on regional grid emission factors (e.g., CEA India Grid Emission Factor ~0.71–0.82 kg $CO_2$/kWh) and are estimates subject to grid composition changes.
6. **Model confidence is not model accuracy:** Softmax/sigmoid probability scores indicate network activation confidence, not factual correctness. High confidence can still occur on out-of-distribution false positives.
7. **Geographic coordinates must come from actual data or explicit user input:** Coordinates must be extracted from genuine image metadata (GeoTIFF/EXIF), official datasets, or explicit user geocoding—never fabricated or assumed.
8. **Deployment suitability requires additional data:** Assessment of installation feasibility requires structural, tilt, azimuth, local shading, and electrical grid access parameters beyond 2D satellite segmentation alone.
9. **No unsupported scientific claims should be made:** Every report and UI metric must cite the formula, data source, and assumptions used.
10. **No fake values should be used in production:** If spatial resolution, irradiance, or location metadata is unavailable, the system must report "Unknown / Insufficient Data" rather than generating mock metrics.

---

## 5. Solar Segmentation Analytics Layer (Phase 2)

The Phase 2 analytics layer directly transforms raw 2D binary segmentation masks into discrete, mathematically validated geometric and statistical measurements.

### 5.1 Metrics Definition & Scope

- **Total Pixels ($W \times H$):** Exact spatial pixel count of the analyzed raster image.
- **Solar Pixels:** Count of pixels where $\text{mask} == 1$ (derived directly by integer counting).
- **Background Pixels:** Count of pixels where $\text{mask} == 0$. Verified: $\text{solar\_pixels} + \text{background\_pixels} \equiv \text{total\_pixels}$.
- **Solar Pixel Coverage (%):** $(\text{solar\_pixels} / \text{total\_pixels}) \times 100$.
  - *What it means:* Proportion of image pixel raster classified as photovoltaic panel surface.
  - *What it does NOT mean:* Rooftop area, physical panel area ($m^2$), or generation capacity.
- **Connected Components (8-connectivity):** Individual contiguous clusters of solar pixels isolated via raster component analysis.
  - *Bounding Box:* Minimum bounding rectangle $[x, y, w, h]$ in 2D image coordinates.
  - *Centroid:* Spatial center of mass $[\bar{x}, \bar{y}]$ in pixel coordinates (not geographic coordinates).
- **Component Filtering ($\text{min\_component\_area}$):** Configurable threshold (default: 20 pixels) to filter isolated pixel noise artifacts while preserving genuine panel clusters. Reports raw count, filtered count, and percentage of retained solar pixels.
- **Region Density:** Number of filtered solar regions divided by total image pixels ($\text{regions} / \text{pixel}$).

### 5.2 Scientific Limitations & Physical Constraints
- **Absence of Ground Sampling Distance (GSD):** Because dataset images currently lack calibrated spatial resolution (meters/pixel), no pixel-to-meter conversion is performed.
- **No Physical Claims:** No estimates of square meters ($m^2$), kilowatt-peak ($kW_p$), annual megawatt-hours ($MWh$), or tons of avoided $CO_2$ are generated at this layer.

---

## 6. Confidence & Uncertainty Analysis Layer (Phase 3)

The Phase 3 analytical layer evaluates model output probabilities and prediction ambiguity directly from the neural network's softmax distribution.

### 6.1 Mathematical Formulations
- **Softmax Probabilities:** For each pixel, $P(\text{background}) + P(\text{solar}) \equiv 1.0$, verified with numerical tolerance $\epsilon < 10^{-4}$.
- **Prediction Confidence:** 
  $$\text{Confidence}(x, y) = \max\left(P(\text{background}), P(\text{solar})\right) \in [0.5, 1.0]$$
- **Prediction Uncertainty (Ambiguity):**
  $$\text{Uncertainty}(x, y) = 1.0 - \text{Confidence}(x, y) \in [0.0, 0.5]$$
  Pixels near $0.5$ confidence exhibit maximum prediction ambiguity, whereas pixels near $1.0$ indicate strong model activation.
- **Confidence Distribution:** Evaluated across a 10-bin deterministic histogram $[0.0, 0.1), \dots, [0.9, 1.0]$.
- **Ambiguity Regions:** Configurable uncertainty threshold (default: $\text{Uncertainty} \ge 0.40$, corresponding to $\text{Confidence} \le 0.60$).

### 6.2 Region-Level Confidence Integration
- Integrates directly with Phase 2 connected solar components without altering region geometries.
- For each retained component, extracts localized mean, median, min, max, and standard deviation of solar probability and confidence.
- **Operational Categories:**
  - **HIGH:** $\text{Mean } P(\text{solar}) \ge 0.90$
  - **MEDIUM:** $0.70 \le \text{Mean } P(\text{solar}) < 0.90$
  - **LOW:** $\text{Mean } P(\text{solar}) < 0.70$

### 6.3 Scientific Limitations
- **Not Calibrated Accuracy:** Softmax confidence scores indicate network activation strength, not empirically calibrated empirical accuracy or Bayesian posterior probability.
- **Not Epistemic/Aleatoric Uncertainty:** The uncertainty metric measures decision boundary ambiguity, not formal Monte Carlo dropout or ensemble epistemic variance.

---

## 7. Geospatial & Physical Area Foundation (Phase 4)

Phase 4 evaluated the scientific legitimacy of converting 2D image-space detections into geographic coordinates and physical areas ($m^2$).

### 7.1 Dataset Metadata Audit & Georeferencing Classification
- **Georeferencing Classification:** **LEVEL B** (*Per-image geographic coordinates available; physical pixel scale unavailable*).
- **Image-to-Coordinate Mapping:** Every image in the default dataset ($2,505$ files) was deterministically matched ($100\%$ match rate, $0$ unmatched) to ground coordinates in `EI_train_data(Sheet1).csv` and `splits/dataset_manifest.csv` via the validated naming pattern `<sampleid>[.0]_1[.0].png`.
- **Coordinate Boundaries:** All positive solar images ($2,531$ records) are concentrated in Gujarat, Western India (lat: $21.05^\circ \text{N}$ to $23.45^\circ \text{N}$, lon: $70.06^\circ \text{E}$ to $73.19^\circ \text{E}$). Exactly 6 negative background images were identified outside the mainland India bounding box.
- **Component Coordinates:** NOT supported; individual solar panel centroids cannot be legitimately geocoded without an affine georeferencing transform matrix (absent in PNG format).

### 7.2 Physical Area & Energy Readiness
- **Absence of Ground Sampling Distance (GSD):** Raster metadata contains zero EXIF GPS tags, world files (`.tfw`/`.pgw`), or spatial resolution parameters.
- **Physical Area Status:** Marked strictly as `insufficient_data` (`area_m2 = null`), while preserving exact `area_pixels`. No arbitrary pixel-to-meter multipliers ($1\text{ px} = 1\text{ m}$, $0.5\text{ m}$, etc.) are permitted.
- **Energy Estimation Readiness:** **`BLOCKED`**. Downstream photovoltaic energy yield calculations ($E = A \cdot \eta \cdot G \cdot PR$) require physical square meters ($A$) and are halted until calibrated GSD metadata or imagery is integrated.

---

## 8. External Spatial Resolution & Solar Resource Foundation (Phase 5)

Phase 5 investigated whether legitimate external or embedded sources can provide the two required inputs for physical solar energy estimation:
1. Physical spatial resolution / Ground Sampling Distance (GSD)
2. Authoritative solar resource (irradiance/irradiation) data

### 8.1 Empirical Imagery Source & GSD Audit
- **Source Platform & Provider:** UNKNOWN / UNDOCUMENTED in dataset manifests and headers.
- **COCO Metadata:** Header `info` fields in `merged_instances_default.json` are blank strings (`contributor`, `description`, `url`, `version`, `year` all empty).
- **Embedded Rasters:** Standard 8-bit PNG images contain 0 embedded EXIF tags, 0 pHYs resolution chunks, 0 georeferencing tags, and 0 ESRI world files (`.tfw`/`.pgw`).
- **Spatial Resolution Status:** **`UNAVAILABLE`** (`gsd_meters = null`).
- **Strict No-Fabrication Policy:** Arbitrary resolution assumptions ($1\text{ px} = 1\text{ m}$, $0.5\text{ m/px}$, or generic Sentinel-2 $10\text{ m/px}$) are strictly rejected. Physical area ($m^2$) calculation remains blocked.

### 8.2 Solar Resource Acquisition (NASA POWER)
- **Source Selected:** NASA Langley Research Center — POWER Climatology Project (`SYN1DEG` / CERES).
- **API Endpoint:** `https://power.larc.nasa.gov/api/temporal/climatology/point`
- **Variable Retrieved:** `ALLSKY_SFC_SW_DWN` (All Sky Surface Shortwave Downward Irradiance).
- **Native Units:** $\text{kW-hr/m}^2/\text{day}$ ($\text{kWh/m}^2/\text{day}$).
- **Derived Annual Irradiation:** $\text{kWh/m}^2/\text{year} = \text{daily GHI} \times 365.25$.
- **Temporal Basis:** 20-Year Multi-Annual Climatology (January 2001 – December 2020).
- **Spatial Resolution:** $1.0^\circ \times 1.0^\circ$ native global grid ($\sim 110\text{ km}$).
- **Caching & Provenance:** Deterministic disk caching (`outputs/external_data/cache/`) keyed by coordinates and parameters; full scientific provenance recorded for every retrieval.
- **Primary Test Image (`768.0_1.0.png`):** Coordinates $(21.197148^\circ\text{N}, 72.780643^\circ\text{E})$ yielded $5.4041\text{ kWh/m}^2/\text{day}$ ($1,973.85\text{ kWh/m}^2/\text{year}$).

### 8.3 Energy Estimation Readiness
- **Status:** **`PARTIALLY_READY`** (Solar resource data is validated and available; physical spatial resolution remains UNAVAILABLE, blocking physical area derivation and final energy calculations).

---

## 9. Physical Area & Estimated Solar Energy Potential Foundation (Phase 6)

Phase 6 built the production-grade physical area and estimated solar energy potential modeling layer, enforcing strict scientific anti-fabrication standards.

### 9.1 Physical Area Methodology & GSD Reassessment
- **Mathematical Scaling:** For validated spatial scale, $\text{pixel\_area\_m2} = \text{GSD}^2$ (or $\text{GSD}_x \times \text{GSD}_y$). Physical panel area is $A = \text{solar\_pixels} \times \text{pixel\_area\_m2}$.
- **Empirical Status:** Because no sensor certificate, world file, or tile zoom level exists in the dataset, $\text{gsd\_status} = \text{"UNAVAILABLE"}$.
- **Enforced Result:** $\text{physical\_area\_status} = \text{"insufficient\_data"}$ and $\text{solar\_area\_m2} = \text{null}$. No heuristic multipliers (such as $1\text{ px} = 1\text{ m}$) are tolerated.

### 9.2 First-Order Photovoltaic Energy Model
- **Mathematical Formulation:**
  $$E = A \times \text{GHI} \times \eta \times \text{PR}$$
  where:
  - $E$: Estimated Solar Energy Potential ($\text{kWh/year}$)
  - $A$: Detected physical solar panel area ($m^2$)
  - $\text{GHI}$: Annual Global Horizontal Irradiation ($\text{kWh/m}^2/\text{year}$) derived from NASA POWER multi-annual climatology ($\text{daily GHI} \times 365.25$)
  - $\eta$: Photovoltaic module efficiency (fraction; configurable, benchmark commercial mono-Si $\sim 0.20$)
  - $\text{PR}$: System performance ratio (fraction; configurable, benchmark tropical rooftop $\sim 0.75$)
- **Scientific Labeling:** Outputs are strictly designated as **"Estimated Solar Energy Potential"** and never labeled "Actual Energy Generated".

### 9.3 Explicit Assumptions & Missing Input Behavior
- Parameters $\eta$ and $\text{PR}$ default strictly to `null`. Assumptions are never silently applied.
- If any input ($A$, $\text{GHI}$, $\eta$, $\text{PR}$) is `null`, the model returns $\text{energy\_status} = \text{"insufficient\_data"}$ and $\text{estimated\_energy\_kwh\_year} = \text{null}$.
- **Primary Test Sample (`768.0_1.0.png`):**
  - Solar pixels: $68,451 / 409,600$ ($16.71\%$ coverage)
  - Coordinates: $(21.197148^\circ\text{N}, 72.780643^\circ\text{E})$
  - Solar Resource: $1,973.85\text{ kWh/m}^2/\text{year}$ ($5.4041\text{ kWh/m}^2/\text{day}$)
  - Physical Area: `insufficient_data` (`null`)
  - Estimated Solar Energy Potential: `insufficient_data` (`null`)
  - Overall Phase 6 Readiness: **`BLOCKED`** (scientifically defensible due to uncalibrated raster GSD).

### 9.4 Comprehensive Qualitative Limitations
1. **Spatial Scale Uncertainty:** Ground Sampling Distance cannot be derived from image dimensions ($640 \times 640$ px does not equal $640\text{ m} \times 640\text{ m}$).
2. **Segmentation Edge Ambiguity:** 2D pixel classifications are subject to edge resolution effects and confidence thresholds.
3. **Coarse Solar Climatology:** NASA CERES native grid is $1.0^\circ \times 1.0^\circ$ ($\sim 110\text{ km}$), omitting local microclimates.
4. **Orientation and Tilt Unknowns:** Satellite nadir view lacks 3D rooftop tilt and compass azimuth angles.
5. **Loss Factors:** Local shading obstacles, soiling/dust accumulation, temperature derating coefficients, inverter clipping, and annual panel degradation ($\sim 0.5\%/\text{year}$) are not modeled.

---

## 10. CO₂ & Environmental Avoidance Analysis Layer (Phase 7)

Phase 7 implemented the greenhouse gas ($CO_2$) emission avoidance modeling layer, connecting photovoltaic energy estimates to authoritative electricity grid emission factors published by India's Central Electricity Authority (CEA).

### 10.1 Mathematical Formulations
- **Annual Avoided Carbon ($kg\text{ }CO_2/\text{year}$):**
  $$\text{CO}_2\text{ avoided (kg/year)} = E \times \text{EF}$$
  where:
  - $E$: Annual solar electricity generation ($kWh/\text{year}$) from Phase 6.
  - $\text{EF}$: Grid emission factor ($kg\text{ }CO_2/kWh$), defaulting to the CEA India National Grid Weighted Average ($\sim 0.716\text{ }kg\text{ }CO_2/kWh$).
- **Metric Tonnes Avoidance ($\text{tonnes }CO_2/\text{year}$):**
  $$\text{CO}_2\text{ avoided (tonnes/year)} = \frac{\text{CO}_2\text{ avoided (kg/year)}}{1,000}$$

### 10.2 Strict Scientific Anti-Fabrication Guarantees
- If physical area or annual energy generation $E$ is unavailable, uncalibrated, or $null$ (as in the uncalibrated benchmark imagery), the environmental model strictly reports:
  ```json
  {
    "co2_status": "insufficient_data",
    "co2_avoided_kg_year": null,
    "co2_avoided_tonnes_year": null
  }
  ```
- No zeros or synthetic default multipliers are substituted for unverified inputs.

### 10.3 Authoritative Grid Emission Factors (CEA India)
- **Primary Source:** Central Electricity Authority (CEA), Ministry of Power, Government of India.
- **Reference Document:** *CO₂ Baseline Database for the Indian Power Sector (User Guide Version 19.0 / 20.0)*.
- **Scope:** Indian Unified National Synchronous Grid (NEWNE + Southern Region).
- **Default Baseline Factor:** $0.716\text{ }kg\text{ }CO_2/kWh$ ($tCO_2/MWh$) weighted average across all grid generation including renewables.
- **Thermal Generation Baseline:** $0.820\text{ }kg\text{ }CO_2/kWh$ weighted average excluding renewables.

---

## 11. MongoDB Integration & Prediction History Layer (Phase 8)

Phase 8 implemented asynchronous document persistence and query endpoints for prediction history, capturing full multi-phase analytics and provenance while respecting strict scientific anti-fabrication rules.

### 11.1 Architecture & Connection Lifecycle
- **Driver:** Asynchronous Motor client (`motor.motor_asyncio.AsyncIOMotorClient`) running on the native asyncio event loop.
- **Connection Management:** Initialized during FastAPI startup within the application lifespan handler (`@asynccontextmanager lifespan`); disconnected cleanly during shutdown.
- **Offline & Import Resilience:** The application and automated test suites import and run without requiring a live MongoDB instance. If the database is unconfigured or offline, inference execution (`POST /predict`) continues normally while gracefully setting `saved_to_history: false`.
- **Credential Safety:** Connection strings and passwords are never logged; diagnostic endpoints and logs mask credentials (e.g. `sanitize_mongodb_uri`).

### 11.2 Prediction History Schema
Each persisted prediction record preserves:
- `prediction_id`: Unique UUID4 identifier.
- `created_at_utc`: ISO 8601 UTC timestamp.
- `filename`: Original aerial/satellite image filename.
- `status`: Execution status (`success`).
- `model_name`: Detection model (`HalfUNet`).
- `segmentation`: Dimensions, solar pixels, coverage percent, and connected component metrics.
- `spatial_resolution`: Ground Sampling Distance (`gsd_status`, `gsd_meters`, `physical_area_status`, `solar_area_m2`).
- `energy_model`: Estimated solar energy potential status (`insufficient_data` or `estimated`) and annual kWh.
- `environmental_model`: Avoided CO₂ status (`insufficient_data` or `estimated`), kg/year, and metric tonnes/year.
- `artifacts`: Safe relative URLs for binary masks and visual detection overlays (`/outputs/{id}/mask`, `/outputs/{id}/overlay`).
- `applied_threshold` & `applied_min_component_area`: Parameter configuration used during inference.

### 11.3 Indexing Strategy
- **Unique Lookup Index:** `prediction_id` (ascending, unique) for constant-time $O(1)$ single record retrieval.
- **Chronological Sorting Index:** `created_at_utc` (descending) for high-performance paginated queries (`sort("created_at_utc", -1)`).

### 11.4 REST API Endpoints
- `GET /api/predictions/history` (and `/predictions/history`): Retrieves paginated prediction records with configurable `skip` and `limit` (max 100).
- `GET /api/predictions/history/{prediction_id}` (and `/predictions/history/{prediction_id}`): Retrieves a single prediction record by UUID4 identifier.
- `GET /health/db`: Lightweight database readiness and connectivity check.

---

## 12. Full Integration and Testing Layer (Phase 13)

Phase 13 establishes the cohesive end-to-end integration between the machine learning segmentation pipeline, analytical derivation engines, geospatial map viewer, and prediction history database.

### 12.1 Cross-Phase Engine Integration
1. **Physical Area & Spatial Resolution (Phase 5 & 6):**
   - Integrates `ml.energy.area.PhysicalAreaCalculator`.
   - Without verified GSD calibration (isotropic or anisotropic meters/pixel), `physical_area_status` strictly evaluates to `insufficient_data` and physical dimensions remain `null`.
2. **Photovoltaic Energy Potential (Phase 6):**
   - Integrates `ml.energy.energy_model.SolarEnergyModel`.
   - Evaluates annual photovoltaic generation via $E = A \cdot GHI \cdot \eta \cdot PR$.
   - Adheres to zero-fabrication rules: when physical area or annual irradiance ($GHI$) is uncalibrated, `energy_status` evaluates to `insufficient_data` with detailed diagnostic reasons logged in provenance metadata.
3. **Avoided CO₂ Emissions (Phase 7):**
   - Integrates `ml.environmental.co2_model.CO2AvoidanceModel`.
   - Derives avoided grid emissions using Central Electricity Authority (CEA) Indian national grid baseline factors ($0.716\text{ }kg\text{ }CO_2/kWh$).
   - Safely preserves `insufficient_data` when upstream generation inputs are unverified.

### 12.2 Frontend UI Integration & State Management
1. **Prediction History Drawer (`PredictionHistoryDrawer.jsx`):**
   - Slide-out drawer displaying persisted inference runs sorted newest first.
   - Built-in pagination controls (`skip`, `limit`, page navigation).
   - Real-time MongoDB connection health monitoring (`checkDatabaseHealth`).
   - Graceful offline fallback: displays clear notification when MongoDB is offline without interfering with the live image inference workflow.
2. **Active History Inspection Banner:**
   - When a historical prediction is selected, the application dynamically displays a prominent banner indicating that a historical run is being inspected, with an instant button to return to live upload.
   - Updates visual inspection overlays and synchronizes the Leaflet map coordinates.
3. **Geospatial Map Synchronization (`GeospatialMap.jsx`):**
   - Synchronizes marker location and popup metadata directly with the active prediction.
   - Accurately distinguishes cataloged Level B scene centroids from unreferenced images.
   - Completely prevents fallback coordinate fabrication for unreferenced scenes.

### 12.3 Automated Regression & Integration Testing
- **Backend Test Suite:** 171 automated tests passing across 10 test modules (`python -m unittest discover tests`).
- **Dedicated Phase 13 Suite:** `tests/test_phase13_integration.py` verifies 10 critical integration contracts (energy/CO₂ defaults, geospatial catalogs, MongoDB persistence details, offline resilience, pagination, UUID validation, and mathematical correctness under explicit scenarios).
- **Frontend Code Quality:** 0 lint errors/warnings (`oxlint`) across 13 source files; production bundle builds cleanly in <300ms (`vite build`).







