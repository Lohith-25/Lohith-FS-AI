"""
SolarMap-India — Phase 7 CO2 & Environmental Analysis CLI & Reporter.

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
    CO2 Emission Avoidance Modeling (Phase 7)
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

from ml.energy.assumptions import EnergyModelConfig
from ml.energy.report import run_energy_analysis_pipeline
from ml.environmental.assumptions import (
    EnvironmentalAssumptions,
    STANDARD_CEA_GRID_EMISSION_FACTOR,
)
from ml.environmental.emission_factors import GridEmissionConfig
from ml.environmental.co2_model import CO2AvoidanceModel, CO2CalculationResult

OUTPUT_ENV_DIR = PROJECT_ROOT / "outputs" / "environmental"
REPORTS_DIR = OUTPUT_ENV_DIR / "reports"
CALCULATIONS_DIR = OUTPUT_ENV_DIR / "calculations"


def run_environmental_analysis_pipeline(
    image_path: Union[str, Path],
    energy_config: Optional[EnergyModelConfig] = None,
    emission_config: Optional[GridEmissionConfig] = None,
    candidate_gsd: Optional[float] = None,
    is_candidate_gsd_validated: bool = False,
    emission_factor_kg_kwh: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Executes the complete Phase 7 environmental analysis pipeline for an input image.
    Reuses existing Phase 6 energy analysis to ensure scientific continuity.
    """
    # 1. Run upstream Phase 6 energy analysis pipeline
    energy_pipeline_res = run_energy_analysis_pipeline(
        image_path=image_path,
        config=energy_config,
        candidate_gsd=candidate_gsd,
        is_candidate_gsd_validated=is_candidate_gsd_validated,
    )

    # 2. Configure grid emission parameters
    if emission_config is None:
        if emission_factor_kg_kwh is not None:
            emission_config = GridEmissionConfig.create_custom_scenario(
                emission_factor=emission_factor_kg_kwh,
                source_note="Direct parameter override provided.",
            )
        else:
            emission_config = GridEmissionConfig.create_default()

    # 3. Calculate avoided CO2 emissions
    co2_result = CO2AvoidanceModel.estimate_avoided_emissions(
        energy_input=energy_pipeline_res["energy_model"],
        config=emission_config,
        emission_factor_kg_kwh=emission_factor_kg_kwh,
    )

    # 4. Synthesize complete structured environmental report payload
    env_assumptions = EnvironmentalAssumptions()

    pipeline_output = {
        "image_name": energy_pipeline_res["image_name"],
        "sampleid": energy_pipeline_res["sampleid"],
        "coordinates": energy_pipeline_res["coordinates"],
        "segmentation": energy_pipeline_res["segmentation"],
        "spatial_resolution": energy_pipeline_res["spatial_resolution"],
        "solar_resource": energy_pipeline_res["solar_resource"],
        "energy_model": energy_pipeline_res["energy_model"],
        "environmental_model": {
            "co2_status": co2_result.co2_status,
            "co2_label": co2_result.co2_label,
            "formula": co2_result.formula,
            "units": co2_result.units,
            "annual_energy_kwh_year": co2_result.annual_energy_kwh_year,
            "emission_factor_kg_kwh": co2_result.emission_factor_kg_kwh,
            "co2_avoided_kg_year": co2_result.co2_avoided_kg_year,
            "co2_avoided_tonnes_year": co2_result.co2_avoided_tonnes_year,
            "unavailability_reasons": co2_result.unavailability_reasons,
            "emission_factor_provenance": emission_config.to_dict(),
        },
        "scientific_assumptions": {
            "energy_assumptions": energy_pipeline_res["scientific_assumptions"],
            "environmental_assumptions": env_assumptions.to_dict(),
        },
    }

    return pipeline_output


def print_manual_environmental_report(result: Dict[str, Any]) -> None:
    """
    Displays the standardized terminal report for Phase 7 environmental analysis.
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

    eng_status = result["energy_model"]["energy_status"]
    eng_val = result["energy_model"]["estimated_energy_kwh_year"]
    eng_str = f"{eng_val:,.2f} kWh/year" if eng_val is not None else "insufficient_data"

    env = result["environmental_model"]
    co2_status = env["co2_status"]
    ef_val = env["emission_factor_kg_kwh"]
    ef_str = f"{ef_val:.4f} kg CO2/kWh" if ef_val is not None else "insufficient_data"

    kg_val = env["co2_avoided_kg_year"]
    kg_str = f"{kg_val:,.2f} kg CO2/year" if kg_val is not None else "insufficient_data"

    tonne_val = env["co2_avoided_tonnes_year"]
    tonne_str = f"{tonne_val:,.4f} metric tonnes CO2/year" if tonne_val is not None else "insufficient_data"

    print("-" * 60)
    print("SolarMap-India Phase 7: CO2 & Environmental Analysis")
    print("-" * 60)
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
    print(f"Energy status:                    {eng_status}")
    print(f"Estimated Solar Energy Potential: {eng_str}")
    print()
    print(f"Environmental Analysis Status:    {co2_status}")
    print(f"Grid Emission Factor:             {ef_str}")
    print(f"Estimated CO2 Avoided (kg/year):  {kg_str}")
    print(f"Estimated CO2 Avoided (tonnes):   {tonne_str}")
    print("-" * 60)


def build_and_save_phase7_reports(
    target_image: str = "768.0_1.0.png",
    module_efficiency: Optional[float] = None,
    performance_ratio: Optional[float] = None,
    candidate_gsd: Optional[float] = None,
    is_candidate_gsd_validated: bool = False,
    emission_factor_kg_kwh: Optional[float] = None,
    use_cea_benchmark: bool = False,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Executes Phase 7 environmental analysis and saves phase7_environmental_report.json and calculations.
    """
    rep_dir = (output_dir or REPORTS_DIR)
    calc_dir = (output_dir.parent / "calculations" if output_dir else CALCULATIONS_DIR)

    rep_dir.mkdir(parents=True, exist_ok=True)
    calc_dir.mkdir(parents=True, exist_ok=True)

    # 1. Configure energy parameters
    if module_efficiency is not None or performance_ratio is not None:
        energy_cfg = EnergyModelConfig(
            module_efficiency=module_efficiency,
            performance_ratio=performance_ratio,
            efficiency_source_note="User-specified scenario parameter" if module_efficiency else None,
            pr_source_note="User-specified scenario parameter" if performance_ratio else None,
        )
    else:
        energy_cfg = EnergyModelConfig.create_default()

    # 2. Configure grid emission parameters
    if emission_factor_kg_kwh is not None:
        emission_cfg = GridEmissionConfig.create_custom_scenario(
            emission_factor=emission_factor_kg_kwh,
            source_note="User-specified custom scenario parameter.",
        )
    elif use_cea_benchmark:
        emission_cfg = GridEmissionConfig.create_cea_benchmark_scenario()
    else:
        emission_cfg = GridEmissionConfig.create_default()

    # 3. Run pipeline
    result = run_environmental_analysis_pipeline(
        image_path=target_image,
        energy_config=energy_cfg,
        emission_config=emission_cfg,
        candidate_gsd=candidate_gsd,
        is_candidate_gsd_validated=is_candidate_gsd_validated,
        emission_factor_kg_kwh=emission_factor_kg_kwh,
    )

    # Determine readiness
    if result["environmental_model"]["co2_status"] == "estimated":
        readiness = "READY"
    elif result["energy_model"]["energy_status"] == "estimated":
        readiness = "PARTIALLY_READY"
    else:
        readiness = "BLOCKED"

    report_payload = {
        "phase": "PHASE 7 — CO2 & ENVIRONMENTAL ANALYSIS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_analysis": result,
        "environmental_readiness": readiness,
        "scientific_integrity_guarantees": {
            "gsd_fabricated": False,
            "physical_area_fabricated": False,
            "energy_fabricated": False,
            "co2_avoided_fabricated": False,
            "actual_reduction_claimed": False,
        },
    }

    # Save reports
    report_file = rep_dir / "phase7_environmental_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    safe_stem = Path(target_image).stem
    calc_file = calc_dir / f"environmental_calc_{safe_stem}.json"
    with open(calc_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return report_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SolarMap-India Phase 7: CO2 & Environmental Analysis")
    parser.add_argument("--image", type=str, default="768.0_1.0.png", help="Target aerial image")
    parser.add_argument("--efficiency", type=float, default=None, help="PV module efficiency fraction (e.g. 0.20)")
    parser.add_argument("--pr", type=float, default=None, help="Performance ratio fraction (e.g. 0.75)")
    parser.add_argument("--gsd", type=float, default=None, help="Candidate GSD in meters/pixel")
    parser.add_argument("--validate-gsd", action="store_true", help="Flag confirming candidate GSD is validated")
    parser.add_argument("--emission-factor", type=float, default=None, help="Grid emission factor in kg CO2/kWh")
    parser.add_argument("--cea-benchmark", action="store_true", help="Use authoritative CEA India baseline factor (0.716 kg CO2/kWh)")

    args = parser.parse_args()

    report_dict = build_and_save_phase7_reports(
        target_image=args.image,
        module_efficiency=args.efficiency,
        performance_ratio=args.pr,
        candidate_gsd=args.gsd,
        is_candidate_gsd_validated=args.validate_gsd,
        emission_factor_kg_kwh=args.emission_factor,
        use_cea_benchmark=args.cea_benchmark,
    )

    print_manual_environmental_report(report_dict["primary_analysis"])
    print(f"\n[Artifact] Phase 7 report saved to: {REPORTS_DIR / 'phase7_environmental_report.json'}")
