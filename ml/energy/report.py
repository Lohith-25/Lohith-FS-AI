"""
SolarMap-India — Phase 6 Physical Area & Estimated Solar Energy Potential CLI & Reporter.

Executes end-to-end integration:
    Binary Segmentation (Phase 1)
          ↓
    Image Coordinates & Catalog (Phase 4)
          ↓
    GSD Reassessment & Physical Area (Phase 6)
          ↓
    NASA POWER Climatology (Phase 5)
          ↓
    Solar Energy Potential Modeling (Phase 6)
          ↓
    Standard Output Display & Structured JSON Reports
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.geospatial.coordinate_mapping import GeospatialCatalog
from ml.external_data.imagery_source import inspect_imagery_source
from ml.external_data.solar_resource import SolarResourceService
from ml.inference.predict import predict_image
from ml.energy.assumptions import EnergyModelConfig, ScientificAssumptions
from ml.energy.area import PhysicalAreaCalculator, AreaCalculationResult
from ml.energy.energy_model import SolarEnergyModel, EnergyCalculationResult

OUTPUT_ENERGY_DIR = PROJECT_ROOT / "outputs" / "energy"
REPORTS_DIR = OUTPUT_ENERGY_DIR / "reports"
CALCULATIONS_DIR = OUTPUT_ENERGY_DIR / "calculations"


def run_energy_analysis_pipeline(
    image_path: Union[str, Path],
    config: Optional[EnergyModelConfig] = None,
    candidate_gsd: Optional[float] = None,
    is_candidate_gsd_validated: bool = False,
    catalog: Optional[GeospatialCatalog] = None,
    solar_service: Optional[SolarResourceService] = None,
) -> Dict[str, Any]:
    """
    Executes the complete Phase 6 energy analysis pipeline for an input image.
    """
    img_file = Path(image_path)
    if not img_file.is_file():
        # Check standard default image folder if relative filename provided
        default_candidate = (
            PROJECT_ROOT / "dataset" / "Solar Images" / "Solar Images" / "images" / "default" / img_file.name
        )
        if default_candidate.is_file():
            img_file = default_candidate
        else:
            raise FileNotFoundError(f"Image file not found: {image_path}")

    if catalog is None:
        catalog = GeospatialCatalog()
    if solar_service is None:
        solar_service = SolarResourceService()
    if config is None:
        config = EnergyModelConfig.create_default()

    # 1. Run Phase 1 Inference (Model forward pass -> Binary mask -> Solar pixels)
    inf_res = predict_image(image_path=img_file)
    solar_pixels = inf_res.solar_pixels
    total_pixels = inf_res.total_pixels

    # 2. Resolve genuine geographic coordinates from Phase 4 Catalog
    coord_rec = catalog.lookup(img_file.name)
    if coord_rec is None:
        raise ValueError(f"Could not resolve geographic coordinates for image: {img_file.name}")
    latitude = coord_rec.latitude
    longitude = coord_rec.longitude

    # 3. Reassess GSD from Phase 5 empirical findings or explicit candidate
    source_audit = inspect_imagery_source()
    if candidate_gsd is not None and is_candidate_gsd_validated:
        eff_gsd = candidate_gsd
        is_gsd_val = True
    else:
        eff_gsd = source_audit.gsd_meters  # None in actual dataset
        is_gsd_val = (source_audit.spatial_resolution_status == "VALIDATED")

    # 4. Calculate Physical Area (Strictly 'insufficient_data' if GSD is unverified)
    area_res = PhysicalAreaCalculator.calculate(
        solar_pixels=solar_pixels,
        total_pixels=total_pixels,
        gsd_meters=eff_gsd,
        is_gsd_validated=is_gsd_val,
    )

    # 5. Retrieve NASA POWER Climatology Solar Resource for coordinates
    solar_res = solar_service.get_solar_resource(latitude=latitude, longitude=longitude)

    # 6. Estimate Solar Energy Potential via transparent model
    energy_res = SolarEnergyModel.estimate_energy(
        area_result=area_res,
        annual_ghi_kwh_m2_year=solar_res.annual_value_kwh_m2_year,
        config=config,
        source_daily_ghi=solar_res.daily_value_kwh_m2_day,
        source_unit=solar_res.unit,
        solar_resource_provenance=solar_res.provenance,
    )

    pipeline_output = {
        "image_name": img_file.name,
        "sampleid": coord_rec.sampleid,
        "coordinates": {
            "latitude": latitude,
            "longitude": longitude,
        },
        "segmentation": {
            "solar_pixels": solar_pixels,
            "total_pixels": total_pixels,
            "pixel_coverage_percent": inf_res.solar_coverage_percent,
        },
        "spatial_resolution": {
            "gsd_status": area_res.gsd_status,
            "gsd_meters": area_res.gsd_meters,
            "physical_area_status": area_res.physical_area_status,
            "solar_area_m2": area_res.solar_area_m2,
            "status_reason": area_res.status_reason,
        },
        "solar_resource": {
            "source": solar_res.source,
            "status": solar_res.status,
            "daily_ghi_kwh_m2_day": solar_res.daily_value_kwh_m2_day,
            "annual_ghi_kwh_m2_year": solar_res.annual_value_kwh_m2_year,
            "unit": solar_res.unit,
            "temporal_basis": solar_res.temporal_basis,
            "provenance": solar_res.provenance,
        },
        "energy_model": {
            "energy_status": energy_res.energy_status,
            "energy_label": energy_res.energy_label,
            "formula": energy_res.formula,
            "module_efficiency": energy_res.module_efficiency,
            "performance_ratio": energy_res.performance_ratio,
            "estimated_energy_kwh_year": energy_res.estimated_energy_kwh_year,
            "unavailability_reasons": energy_res.unavailability_reasons,
        },
        "scientific_assumptions": ScientificAssumptions().to_dict(),
    }

    return pipeline_output


def print_manual_energy_report(result: Dict[str, Any]) -> None:
    """
    Displays the standardized terminal report strictly adhering to Section 18 format.
    Outputs 'insufficient_data' whenever a metric cannot be legitimately calculated.
    """
    img_name = result["image_name"]
    solar_pixels = result["segmentation"]["solar_pixels"]
    total_pixels = result["segmentation"]["total_pixels"]
    cov_pct = result["segmentation"]["pixel_coverage_percent"]

    gsd_status = result["spatial_resolution"]["gsd_status"]
    gsd_val = result["spatial_resolution"]["gsd_meters"]
    gsd_str = f"{gsd_val:.4f} m/px" if gsd_val is not None else "insufficient_data"

    phys_status = result["spatial_resolution"]["physical_area_status"]
    area_val = result["spatial_resolution"]["solar_area_m2"]
    area_str = f"{area_val:.2f} m^2" if area_val is not None else "insufficient_data"

    sol_src = result["solar_resource"]["source"].replace("—", "-")
    sol_status = result["solar_resource"]["status"]
    ann_ghi = result["solar_resource"]["annual_ghi_kwh_m2_year"]
    ghi_str = (
        f"{ann_ghi:.2f} kWh/m^2/year" if ann_ghi is not None and sol_status == "available"
        else "insufficient_data"
    )

    eff = result["energy_model"]["module_efficiency"]
    eff_str = f"{eff * 100:.1f}% ({eff})" if eff is not None else "insufficient_data"

    pr = result["energy_model"]["performance_ratio"]
    pr_str = f"{pr * 100:.1f}% ({pr})" if pr is not None else "insufficient_data"

    eng_status = result["energy_model"]["energy_status"]
    eng_val = result["energy_model"]["estimated_energy_kwh_year"]
    eng_str = f"{eng_val:,.2f} kWh/year" if eng_val is not None else "insufficient_data"

    print("-" * 50)
    print("SolarMap-India Energy Analysis")
    print("-" * 50)
    print(f"Image:                            {img_name}")
    print(f"Solar pixels:                     {solar_pixels:,} / {total_pixels:,}")
    print(f"Pixel coverage:                   {cov_pct:.2f}%")
    print()
    print(f"GSD status:                       {gsd_status}")
    print(f"GSD:                              {gsd_str}")
    print(f"Physical area status:             {phys_status}")
    print(f"Solar area:                       {area_str}")
    print()
    print(f"Solar resource:                   {sol_src} ({sol_status})")
    print(f"Annual GHI:                       {ghi_str}")
    print()
    print(f"Module efficiency:                {eff_str}")
    print(f"Performance ratio:                {pr_str}")
    print()
    print(f"Energy status:                    {eng_status}")
    print(f"Estimated Solar Energy Potential: {eng_str}")
    print("-" * 50)


def build_and_save_phase6_reports(
    target_image: str = "768.0_1.0.png",
    module_efficiency: Optional[float] = None,
    performance_ratio: Optional[float] = None,
    candidate_gsd: Optional[float] = None,
    is_candidate_gsd_validated: bool = False,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Executes Phase 6 energy analysis and saves phase6_energy_report.json and calculations.
    """
    rep_dir = (output_dir or REPORTS_DIR)
    calc_dir = (output_dir.parent / "calculations" if output_dir else CALCULATIONS_DIR)

    rep_dir.mkdir(parents=True, exist_ok=True)
    calc_dir.mkdir(parents=True, exist_ok=True)

    if module_efficiency is not None or performance_ratio is not None:
        cfg = EnergyModelConfig(
            module_efficiency=module_efficiency,
            performance_ratio=performance_ratio,
            efficiency_source_note="User-specified scenario parameter" if module_efficiency else None,
            pr_source_note="User-specified scenario parameter" if performance_ratio else None,
        )
    else:
        cfg = EnergyModelConfig.create_default()

    result = run_energy_analysis_pipeline(
        image_path=target_image,
        config=cfg,
        candidate_gsd=candidate_gsd,
        is_candidate_gsd_validated=is_candidate_gsd_validated,
    )

    report_payload = {
        "phase": "PHASE 6 — PHYSICAL AREA & ESTIMATED SOLAR ENERGY POTENTIAL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_analysis": result,
        "energy_estimation_readiness": (
            "READY" if result["energy_model"]["energy_status"] == "estimated"
            else "BLOCKED" if result["spatial_resolution"]["physical_area_status"] == "insufficient_data"
            else "PARTIALLY_READY"
        ),
        "scientific_integrity_guarantees": {
            "gsd_fabricated": False,
            "physical_area_fabricated": False,
            "energy_fabricated": False,
            "actual_energy_claimed": False,
        },
    }

    report_path = rep_dir / "phase6_energy_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    calc_path = calc_dir / f"energy_calc_{Path(target_image).stem}.json"
    with open(calc_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return report_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SolarMap-India — Phase 6 Physical Area & Estimated Solar Energy Potential"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="768.0_1.0.png",
        help="Path or filename of target image (default: 768.0_1.0.png)",
    )
    parser.add_argument(
        "--module-efficiency",
        type=float,
        default=None,
        help="Photovoltaic module efficiency as fraction (e.g., 0.20 for 20%%; default: None)",
    )
    parser.add_argument(
        "--performance-ratio",
        type=float,
        default=None,
        help="System performance ratio as fraction (e.g., 0.75 for 75%%; default: None)",
    )
    parser.add_argument(
        "--gsd",
        type=float,
        default=None,
        help="Validated GSD in meters/pixel (optional; strictly requires authenticated source)",
    )
    parser.add_argument(
        "--validated-gsd-source",
        type=str,
        default=None,
        help="Official citation or certificate authenticating the candidate GSD",
    )

    args = parser.parse_args()

    is_gsd_val = False
    if args.gsd is not None and args.validated_gsd_source:
        is_gsd_val = True

    report_payload = build_and_save_phase6_reports(
        target_image=args.image,
        module_efficiency=args.module_efficiency,
        performance_ratio=args.performance_ratio,
        candidate_gsd=args.gsd,
        is_candidate_gsd_validated=is_gsd_val,
    )

    print_manual_energy_report(report_payload["primary_analysis"])
    print(f"\nPhase 6 report saved to: {REPORTS_DIR / 'phase6_energy_report.json'}")


if __name__ == "__main__":
    main()
