"""
Unit and Integration Tests for Phase 5 External Spatial Resolution & Solar Resource Foundation.

Covers:
1. Coordinate validation.
2. Source metadata validation.
3. GSD validation.
4. Missing GSD handling.
5. Solar-resource response validation.
6. Missing solar-resource value handling.
7. Unit validation.
8. Provenance record creation.
9. Cache-key generation.
10. External-data failure handling.
11. Real image coordinate lookup.
12. Real external solar-resource retrieval (with cache or live network).
13. No-fabrication safeguards.
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from ml.geospatial.coordinate_mapping import GeospatialCatalog
from ml.external_data.imagery_source import inspect_imagery_source
from ml.external_data.spatial_resolution import SpatialResolutionValidator
from ml.external_data.provenance import DataProvenance
from ml.external_data.validation import ExternalDataValidator
from ml.external_data.solar_resource import SolarResourceService, SolarResourceResult
from ml.external_data.report import generate_image_readiness_assessment, build_and_save_phase5_reports


class TestExternalDataFoundation(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.temp_dir.name) / "cache"
        self.solar_service = SolarResourceService(cache_dir=self.cache_dir)
        self.catalog = GeospatialCatalog()

    def tearDown(self):
        self.temp_dir.cleanup()

    # 1. Coordinate validation
    def test_coordinate_validation_global_and_india(self):
        # Valid Gujarat coordinates
        valid, errs = ExternalDataValidator.validate_coordinates(21.197148, 72.780643)
        self.assertTrue(valid)
        self.assertEqual(len(errs), 0)

        # Invalid latitude out of global range
        valid, errs = ExternalDataValidator.validate_coordinates(105.0, 72.0)
        self.assertFalse(valid)
        self.assertTrue(any("global bounds" in e for e in errs))

        # Coordinates outside India
        valid, errs = ExternalDataValidator.validate_coordinates(45.0, 5.0)
        self.assertFalse(valid)
        self.assertTrue(any("India bounding box" in e for e in errs))

    # 2. Source metadata validation
    def test_imagery_source_inspection(self):
        audit = inspect_imagery_source()
        self.assertEqual(audit.provider, "UNKNOWN_UNDOCUMENTED")
        self.assertEqual(audit.spatial_resolution_status, "UNAVAILABLE")
        self.assertIsNone(audit.gsd_meters)
        self.assertFalse(audit.has_world_files)
        self.assertFalse(audit.has_georeference_tags)
        self.assertGreater(len(audit.evidence_notes), 3)

    # 3. GSD validation (with valid source) & 4. Missing GSD handling
    def test_gsd_validation_rules(self):
        # Missing candidate
        res_missing = SpatialResolutionValidator.evaluate(candidate_gsd=None)
        self.assertEqual(res_missing.spatial_resolution_status, "UNAVAILABLE")
        self.assertIsNone(res_missing.gsd_meters)
        self.assertFalse(res_missing.can_derive_physical_area)

        # Candidate with missing evidence
        res_no_doc = SpatialResolutionValidator.evaluate(candidate_gsd=0.25, source_evidence="")
        self.assertEqual(res_no_doc.spatial_resolution_status, "UNAVAILABLE")

        # Legitimate candidate with authoritative evidence
        res_valid = SpatialResolutionValidator.evaluate(
            candidate_gsd=0.25,
            source_evidence="Official calibration document from survey drone mission 2024",
        )
        self.assertEqual(res_valid.spatial_resolution_status, "VALIDATED")
        self.assertEqual(res_valid.gsd_meters, 0.25)
        self.assertTrue(res_valid.can_derive_physical_area)

    # 13. No-fabrication safeguards for prohibited generic constants
    def test_no_fabrication_safeguards(self):
        # Generic 1 pixel = 1 meter
        res_1m = SpatialResolutionValidator.evaluate(candidate_gsd=1.0)
        self.assertEqual(res_1m.spatial_resolution_status, "UNAVAILABLE")
        self.assertIn("generic assumption", res_1m.source_description)

        # Generic Sentinel-2 10m
        res_10m = SpatialResolutionValidator.evaluate(candidate_gsd=10.0)
        self.assertEqual(res_10m.spatial_resolution_status, "UNAVAILABLE")
        self.assertIn("Sentinel-2", res_10m.source_description)

        # Arbitrary 0.5m without proof
        res_half = SpatialResolutionValidator.evaluate(candidate_gsd=0.5)
        self.assertEqual(res_half.spatial_resolution_status, "UNAVAILABLE")

    # 5. Solar-resource response validation
    def test_nasa_power_payload_validation(self):
        valid_payload = {
            "properties": {
                "parameter": {
                    "ALLSKY_SFC_SW_DWN": {
                        "ANN": 5.40,
                        "JAN": 4.69,
                    }
                }
            }
        }
        ok, err = ExternalDataValidator.validate_nasa_power_payload(valid_payload)
        self.assertTrue(ok)
        self.assertIsNone(err)

    # 6. Missing solar-resource value handling (-999 fill value)
    def test_missing_solar_resource_fill_value(self):
        missing_payload = {
            "properties": {
                "parameter": {
                    "ALLSKY_SFC_SW_DWN": {
                        "ANN": -999.0,
                    }
                }
            }
        }
        ok, err = ExternalDataValidator.validate_nasa_power_payload(missing_payload)
        self.assertFalse(ok)
        self.assertIn("fill-value", str(err))

    # 7. Unit validation
    def test_unit_validation(self):
        self.assertTrue(
            ExternalDataValidator.validate_units("kW-hr/m^2/day", ["kW-hr/m^2/day", "kWh/m^2/day"])
        )
        self.assertTrue(
            ExternalDataValidator.validate_units("kwh/m2/day", ["kW-hr/m^2/day", "kwh/m2/day"])
        )
        self.assertFalse(
            ExternalDataValidator.validate_units("W/m^2", ["kW-hr/m^2/day"])
        )

    # 8. Provenance record creation
    def test_provenance_creation(self):
        prov = DataProvenance.create_nasa_power_provenance(
            latitude=21.197148,
            longitude=72.780643,
            returned_value=5.4041,
        )
        d = prov.to_dict()
        self.assertIn("NASA", d["source_name"])
        self.assertEqual(d["variable_name"], "ALLSKY_SFC_SW_DWN")
        self.assertEqual(d["native_units"], "kW-hr/m^2/day")
        self.assertEqual(d["returned_value"], 5.4041)
        self.assertIn("SYN1DEG", d["dataset_product_name"])
        self.assertIn("Open Data Policy", d["licensing"])

    # 9. Cache-key generation
    def test_cache_key_generation(self):
        key1 = self.solar_service.generate_cache_key(21.197148, 72.780643)
        key2 = self.solar_service.generate_cache_key(21.197148, 72.780643)
        key3 = self.solar_service.generate_cache_key(28.6139, 77.2090)
        self.assertEqual(key1, key2)
        self.assertNotEqual(key1, key3)
        self.assertTrue(key1.startswith("solar_res_21.1971_72.7806_"))

    # 10. External-data failure handling (mock network timeout & 500 error)
    @patch("urllib.request.urlopen")
    def test_external_data_failure_handling(self, mock_urlopen):
        # Simulate network timeout
        mock_urlopen.side_effect = TimeoutError("Connection timed out after 12s")
        res = self.solar_service.get_solar_resource(
            latitude=21.197148, longitude=72.780643, use_cache=False
        )
        self.assertEqual(res.status, "unavailable")
        self.assertIsNone(res.daily_value_kwh_m2_day)
        self.assertIsNone(res.annual_value_kwh_m2_year)
        self.assertIn("Connection timed out", str(res.error_message))

    # 11. Real image coordinate lookup for 768.0_1.0.png
    def test_real_image_coordinate_lookup(self):
        rec = self.catalog.lookup("768.0_1.0.png")
        self.assertIsNotNone(rec)
        self.assertAlmostEqual(rec.latitude, 21.197148, places=4)
        self.assertAlmostEqual(rec.longitude, 72.780643, places=4)
        self.assertEqual(rec.sampleid, 768)

    # 12. Real external solar-resource retrieval integration
    def test_real_external_solar_resource_retrieval(self):
        # Live / cached query for primary test coordinates (21.197148, 72.780643)
        res = self.solar_service.get_solar_resource(
            latitude=21.197148, longitude=72.780643, use_cache=True
        )
        self.assertEqual(res.status, "available")
        self.assertIsNotNone(res.daily_value_kwh_m2_day)
        self.assertGreater(res.daily_value_kwh_m2_day, 3.5)
        self.assertLess(res.daily_value_kwh_m2_day, 8.0)
        self.assertIsNotNone(res.annual_value_kwh_m2_year)
        self.assertGreater(len(res.monthly_values), 10)
        self.assertIsNotNone(res.provenance)

    # Full report generation integration
    def test_report_generation(self):
        report = build_and_save_phase5_reports(
            target_image="768.0_1.0.png",
            output_dir=Path(self.temp_dir.name) / "reports",
        )
        self.assertEqual(report["phase"], "PHASE 5 — EXTERNAL SPATIAL RESOLUTION & SOLAR RESOURCE FOUNDATION")
        self.assertEqual(report["scientific_conclusions"]["energy_estimation_readiness"], "PARTIALLY_READY")
        self.assertFalse(report["scientific_conclusions"]["can_derive_physical_area"])
        self.assertTrue(report["scientific_conclusions"]["is_solar_resource_available"])


if __name__ == "__main__":
    unittest.main()
