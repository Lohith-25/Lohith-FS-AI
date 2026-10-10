"""
SolarMap-India — Phase 5 External Data Readiness & Reporting.

Compiles imagery source findings, GSD validation status, and solar resource
acquisition into structured research-grade JSON reports.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.geospatial.coordinate_mapping import GeospatialCatalog
from ml.external_data.imagery_source import inspect_imagery_source
from ml.external_data.spatial_resolution import SpatialResolutionValidator
from ml.external_data.solar_resource import SolarResourceService

OUTPUT_REPORTS_DIR = PROJECT_ROOT / "outputs" / "external_data" / "reports"


def generate_image_readiness_assessment(
    image_filename: str,
    catalog: Optional[GeospatialCatalog] = None,
    solar_service: Optional[SolarResourceService] = None,
) -> Dict[str, Any]:
    """
    Generates structured per-image assessment including coordinate lookup,
    imagery source audit, GSD validation, and real solar resource acquisition.
    """
    if catalog is None:
        catalog = GeospatialCatalog()
    if solar_service is None:
        solar_service = SolarResourceService()

    # 1. Lookup verified coordinates from Phase 4 catalog
    rec = catalog.lookup(image_filename)
    if not rec:
        raise ValueError(f"Image '{image_filename}' could not be matched to any catalog coordinates.")

    latitude = rec.latitude
    longitude = rec.longitude

    # 2. Inspect imagery source & validate GSD
    source_audit = inspect_imagery_source()
    gsd_val = SpatialResolutionValidator.evaluate(
        candidate_gsd=source_audit.gsd_meters,
        source_evidence=None,
    )

    # 3. Retrieve solar resource data for verified coordinates
    solar_res = solar_service.get_solar_resource(latitude=latitude, longitude=longitude)

    # 4. Determine energy estimation readiness
    # If solar resource is available but GSD is unavailable, readiness is PARTIALLY_READY
    # (Physical area is blocked).
    if gsd_val.can_derive_physical_area and solar_res.status == "available":
        readiness = "READY"
        readiness_rationale = (
            "Valid physical area can be derived AND an appropriate solar-resource "
            "variable is available."
        )
    elif solar_res.status == "available" and not gsd_val.can_derive_physical_area:
        readiness = "PARTIALLY_READY"
        readiness_rationale = (
            "Solar resource data (GHI daily and annual climatology) is successfully retrieved "
            "and validated for verified coordinates, but physical spatial resolution (GSD) "
            "is UNAVAILABLE from the original imagery source. Physical area and energy calculation "
            "remain BLOCKED."
        )
    else:
        readiness = "BLOCKED"
        readiness_rationale = (
            "Required physical spatial information and/or solar resource data remain unavailable."
        )

    # Standard per-image schema matching Phase 5 specification
    per_image_record = {
        "image": image_filename,
        "sampleid": rec.sampleid,
        "latitude": latitude,
        "longitude": longitude,
        "imagery_source": {
            "provider": source_audit.provider,
            "product": source_audit.product_name,
            "source_evidence": "; ".join(source_audit.evidence_notes[:2]),
            "spatial_resolution_status": gsd_val.spatial_resolution_status,
            "gsd_meters": gsd_val.gsd_meters,
            "can_derive_physical_area": gsd_val.can_derive_physical_area,
        },
        "solar_resource": {
            "source": solar_res.source,
            "variable": solar_res.variable,
            "value": solar_res.daily_value_kwh_m2_day,
            "unit": solar_res.unit,
            "annual_equivalent_kwh_m2_year": solar_res.annual_value_kwh_m2_year,
            "monthly_values": solar_res.monthly_values,
            "temporal_basis": solar_res.temporal_basis,
            "retrieval_date": solar_res.retrieval_date,
            "status": solar_res.status,
            "error_message": solar_res.error_message,
            "from_cache": solar_res.from_cache,
            "provenance": solar_res.provenance,
        },
        "energy_estimation_readiness": readiness,
        "readiness_rationale": readiness_rationale,
    }

    return per_image_record


def build_and_save_phase5_reports(
    target_image: str = "768.0_1.0.png",
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Executes full Phase 5 audit, assesses target image, and generates
    all three required JSON reports in outputs/external_data/reports/.
    """
    out_dir = output_dir or OUTPUT_REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog = GeospatialCatalog()
    solar_service = SolarResourceService()

    # 1. Image assessment
    image_record = generate_image_readiness_assessment(
        image_filename=target_image,
        catalog=catalog,
        solar_service=solar_service,
    )

    source_audit = inspect_imagery_source()

    # 2. Master readiness report (phase5_data_readiness.json)
    master_report = {
        "phase": "PHASE 5 — EXTERNAL SPATIAL RESOLUTION & SOLAR RESOURCE FOUNDATION",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_test_image": image_record,
        "imagery_source_summary": {
            "provider": source_audit.provider,
            "product_name": source_audit.product_name,
            "spatial_resolution_status": source_audit.spatial_resolution_status,
            "gsd_meters": source_audit.gsd_meters,
            "has_world_files": source_audit.has_world_files,
            "has_georeference_tags": source_audit.has_georeference_tags,
            "has_exif_metadata": source_audit.has_exif_metadata,
        },
        "solar_resource_summary": {
            "source_selected": "NASA Langley Research Center — POWER Project",
            "api_endpoint": "https://power.larc.nasa.gov/api/temporal/climatology/point",
            "variable_name": "ALLSKY_SFC_SW_DWN",
            "native_units": "kW-hr/m^2/day",
            "temporal_basis": "20-year multi-annual climatology (2001-2020)",
            "spatial_resolution": "1.0 x 1.0 degree global grid (~110 km)",
            "test_image_daily_ghi": image_record["solar_resource"]["value"],
            "test_image_annual_ghi": image_record["solar_resource"]["annual_equivalent_kwh_m2_year"],
            "status": image_record["solar_resource"]["status"],
        },
        "scientific_conclusions": {
            "can_derive_physical_scale": False,
            "can_derive_physical_area": False,
            "is_solar_resource_available": image_record["solar_resource"]["status"] == "available",
            "energy_estimation_readiness": image_record["energy_estimation_readiness"],
            "energy_readiness_rationale": image_record["readiness_rationale"],
        },
    }

    # Save master report
    master_path = out_dir / "phase5_data_readiness.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master_report, f, indent=2)

    # Save imagery source report
    imagery_report_path = out_dir / "imagery_source_report.json"
    with open(imagery_report_path, "w", encoding="utf-8") as f:
        json.dump(source_audit.to_dict(), f, indent=2)

    # Save solar resource report
    solar_report_path = out_dir / "solar_resource_report.json"
    with open(solar_report_path, "w", encoding="utf-8") as f:
        json.dump(image_record["solar_resource"], f, indent=2)

    return master_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SolarMap-India Phase 5 Readiness Reporter")
    parser.add_argument(
        "--image",
        type=str,
        default="768.0_1.0.png",
        help="Target image filename for external readiness evaluation (default: 768.0_1.0.png)",
    )
    args = parser.parse_args()

    print(f"Executing Phase 5 Readiness Evaluation for '{args.image}'...")
    report = build_and_save_phase5_reports(target_image=args.image)
    print(f"Master report saved to: {OUTPUT_REPORTS_DIR / 'phase5_data_readiness.json'}")
    print(f"Energy Estimation Readiness: {report['scientific_conclusions']['energy_estimation_readiness']}")
    print(f"Rationale: {report['scientific_conclusions']['energy_readiness_rationale']}")
