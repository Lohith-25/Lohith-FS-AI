"""
SolarMap-India — Test Suite for Defensible Image Calibration Workflow.

Comprehensive tests covering all 10 requirements:
1. Valid GeoTIFF automatic calibration.
2. PNG/JPEG standard raster without calibration (safeguards active).
3. Valid user-supplied GSD with scientific validation.
4. Zero, negative, NaN, infinite, malformed, out-of-bounds, and prohibited GSD inputs.
5. Pixel-area and total-area arithmetic consistency.
6. Calibration provenance tracking and labeling integrity (USER_SUPPLIED, never automatically VERIFIED).
7. Precedence enforcement: preventing automatic and manual calibration from being applied simultaneously or twice.
8. Energy calculation with missing required inputs.
9. CO2 calculation with missing energy input.
10. Full FastAPI regression tests for live inference with and without calibration parameters.
"""

from pathlib import Path
import io
import math
import sys
import unittest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.geospatial.metadata import GeoMetadata
from ml.area.calibration import (
    ImageCalibration,
    validate_user_gsd,
    resolve_image_calibration,
    calculate_calibrated_area,
    MIN_PLAUSIBLE_GSD_M,
    MAX_PLAUSIBLE_GSD_M,
)
from ml.energy.assumptions import EnergyModelConfig
from ml.energy.energy_model import SolarEnergyModel
from ml.environmental.emission_factors import GridEmissionConfig
from ml.environmental.co2_model import CO2AvoidanceModel
from backend.main import app
from backend.model_manager import model_manager
from fastapi.testclient import TestClient


class TestImageCalibrationWorkflow(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ensure model is loaded for API integration tests
        model_manager.load_model()
        cls.client = TestClient(app)

    # -------------------------------------------------------------------------
    # TEST 1: Valid GeoTIFF automatic calibration
    # -------------------------------------------------------------------------
    def test_valid_geotiff_automatic_calibration(self):
        """Valid GeoTIFF with projected CRS and pixel scale calibrates automatically."""
        geo_meta = GeoMetadata(
            is_georeferenced=True,
            format="GeoTIFF",
            width=640,
            height=640,
            crs_epsg=32643,
            crs_name="WGS 84 / UTM zone 43N",
            is_projected=True,
            is_geographic=False,
            map_units="metre",
            transform_matrix=(0.25, 0.0, 500000.0, 0.0, -0.25, 2300000.0),
            is_rotated_or_sheared=False,
            pixel_width=0.25,
            pixel_height=0.25,
            gsd_x_meters=0.25,
            gsd_y_meters=0.25,
            pixel_area_m2=0.0625,  # 0.25 * 0.25 m^2
            gsd_status="VALIDATED",
            metadata_provenance="geotiff_embedded_tags",
            status_reason="Projected CRS with verified metric units.",
        )

        cal = resolve_image_calibration(geo_metadata=geo_meta, custom_gsd=None)

        self.assertTrue(cal.is_calibrated)
        self.assertEqual(cal.calibration_type, "AUTOMATIC")
        self.assertEqual(cal.calibration_status, "EXTRACTED_METADATA")
        self.assertFalse(cal.is_user_supplied)
        self.assertTrue(cal.is_validated_metadata)
        self.assertAlmostEqual(cal.pixel_area_m2, 0.0625, places=6)
        self.assertAlmostEqual(cal.gsd_meters, 0.25, places=4)
        self.assertIn("WGS 84 / UTM zone 43N", cal.calibration_source)

        # Physical area calculation
        area_res = calculate_calibrated_area(solar_pixels=400, total_pixels=10000, calibration=cal)
        self.assertEqual(area_res.physical_area_status, "calculated")
        self.assertAlmostEqual(area_res.solar_area_m2, 400 * 0.0625, places=4)
        self.assertAlmostEqual(area_res.pixel_area_m2, 0.0625, places=6)

    # -------------------------------------------------------------------------
    # TEST 2: PNG/JPEG without calibration
    # -------------------------------------------------------------------------
    def test_uncalibrated_raster_safeguards(self):
        """Standard unreferenced raster with no calibration must remain uncalculated."""
        unreferenced_meta = GeoMetadata(
            is_georeferenced=False,
            format="PNG",
            width=640,
            height=640,
            crs_epsg=None,
            crs_name=None,
            is_projected=False,
            is_geographic=False,
            map_units="pixels",
            transform_matrix=None,
            is_rotated_or_sheared=False,
            pixel_width=None,
            pixel_height=None,
            gsd_x_meters=None,
            gsd_y_meters=None,
            pixel_area_m2=None,
            gsd_status="UNAVAILABLE",
            metadata_provenance="unreferenced_raster",
            status_reason="No embedded geospatial tags.",
        )

        cal = resolve_image_calibration(geo_metadata=unreferenced_meta, custom_gsd=None)

        self.assertFalse(cal.is_calibrated)
        self.assertEqual(cal.calibration_type, "NONE")
        self.assertEqual(cal.calibration_status, "UNAVAILABLE")
        self.assertIsNone(cal.gsd_meters)
        self.assertIsNone(cal.pixel_area_m2)

        area_res = calculate_calibrated_area(solar_pixels=500, total_pixels=409600, calibration=cal)
        self.assertEqual(area_res.physical_area_status, "insufficient_data")
        self.assertIsNone(area_res.solar_area_m2)
        self.assertIsNotNone(area_res.insufficient_reason)
        self.assertIn("unavailable or unverified", area_res.insufficient_reason.lower())

    # -------------------------------------------------------------------------
    # TEST 3: Valid user-supplied GSD
    # -------------------------------------------------------------------------
    def test_valid_user_supplied_gsd(self):
        """User providing a valid positive GSD with source is accepted as USER_SUPPLIED."""
        citation = "Flight log DJI Mavic 3 Enterprise, 120m AGL, GSD 0.05m"
        is_valid, gsd_val, eff_source, reason = validate_user_gsd(0.05, calibration_source=citation)

        self.assertTrue(is_valid)
        self.assertAlmostEqual(gsd_val, 0.05, places=4)
        self.assertEqual(eff_source, citation)
        self.assertIsNone(reason)

        cal = resolve_image_calibration(
            geo_metadata=None,
            custom_gsd=0.05,
            calibration_source=citation,
        )

        self.assertTrue(cal.is_calibrated)
        self.assertEqual(cal.calibration_type, "MANUAL")
        self.assertEqual(cal.calibration_status, "USER_SUPPLIED")
        self.assertTrue(cal.is_user_supplied)
        self.assertFalse(cal.is_validated_metadata)  # Must NEVER be labeled VERIFIED automatically
        self.assertAlmostEqual(cal.gsd_meters, 0.05, places=4)
        self.assertAlmostEqual(cal.pixel_area_m2, 0.0025, places=6)
        self.assertEqual(cal.calibration_source, citation)

    # -------------------------------------------------------------------------
    # TEST 4: Zero, negative, NaN, infinite, malformed, and out-of-bounds GSD
    # -------------------------------------------------------------------------
    def test_invalid_gsd_inputs_rejected(self):
        """Strictly rejects zero, negative, NaN, infinite, malformed, and out-of-bounds GSD."""
        source = "Survey specification document #42"

        # Zero GSD
        ok, val, src, err = validate_user_gsd(0.0, source)
        self.assertFalse(ok)
        self.assertIn("strictly positive", err.lower())

        # Negative GSD
        ok, val, src, err = validate_user_gsd(-0.05, source)
        self.assertFalse(ok)
        self.assertIn("strictly positive", err.lower())

        # NaN
        ok, val, src, err = validate_user_gsd(float("nan"), source)
        self.assertFalse(ok)
        self.assertIn("finite", err.lower())

        # Infinite (+inf and -inf)
        ok, val, src, err = validate_user_gsd(float("inf"), source)
        self.assertFalse(ok)
        self.assertIn("finite", err.lower())

        ok, val, src, err = validate_user_gsd(float("-inf"), source)
        self.assertFalse(ok)
        self.assertIn("finite", err.lower())

        # Malformed strings
        ok, val, src, err = validate_user_gsd("invalid_number", source)
        self.assertFalse(ok)
        self.assertIn("numeric", err.lower())

        # None / empty
        ok, val, src, err = validate_user_gsd(None, source)
        self.assertFalse(ok)

        ok, val, src, err = validate_user_gsd("", source)
        self.assertFalse(ok)

        # Below lower physical bound (< 0.005 m/px = 5 mm/px)
        ok, val, src, err = validate_user_gsd(0.001, source)
        self.assertFalse(ok)
        self.assertIn(f"{MIN_PLAUSIBLE_GSD_M}", err)

        # Above upper physical bound (> 100.0 m/px)
        ok, val, src, err = validate_user_gsd(150.0, source)
        self.assertFalse(ok)
        self.assertIn(f"{MAX_PLAUSIBLE_GSD_M}", err)

        # Prohibited generic assumption without source description (0.5 m/px is in PROHIBITED_GENERIC_GSDS)
        ok, val, src, err = validate_user_gsd(0.5, calibration_source="")
        self.assertFalse(ok)
        self.assertIn("rejected unverified generic assumption", err.lower())

    # -------------------------------------------------------------------------
    # TEST 5: Pixel-area and total-area arithmetic
    # -------------------------------------------------------------------------
    def test_area_arithmetic_precision(self):
        """Verifies exact arithmetic: area_per_pixel = GSD^2, solar_area = count * area_per_pixel."""
        gsd = 0.08  # 8 cm/pixel
        cal = resolve_image_calibration(
            geo_metadata=None,
            custom_gsd=gsd,
            calibration_source="Drone photogrammetry report",
        )

        solar_pixels = 2500
        total_pixels = 640 * 640  # 409,600

        res = calculate_calibrated_area(solar_pixels, total_pixels, cal)

        expected_pixel_area = 0.08 * 0.08  # 0.0064 m^2
        expected_solar_area = 2500 * expected_pixel_area  # 16.0 m^2

        self.assertAlmostEqual(res.pixel_area_m2, expected_pixel_area, places=6)
        self.assertAlmostEqual(res.solar_area_m2, expected_solar_area, places=4)
        self.assertEqual(res.physical_area_status, "calculated")

    # -------------------------------------------------------------------------
    # TEST 6: Calibration provenance tracking
    # -------------------------------------------------------------------------
    def test_calibration_provenance_labels(self):
        """Verifies provenance tags distinguish metadata from user inputs."""
        # 1. User supplied
        cal_user = resolve_image_calibration(
            geo_metadata=None,
            custom_gsd=0.15,
            calibration_source="Survey citation #99",
        )
        self.assertEqual(cal_user.calibration_status, "USER_SUPPLIED")
        self.assertTrue(cal_user.is_user_supplied)
        self.assertFalse(cal_user.is_validated_metadata)

        # 2. Metadata derived
        geo_meta = GeoMetadata(
            is_georeferenced=True,
            format="GeoTIFF",
            width=640,
            height=640,
            crs_epsg=32643,
            crs_name="WGS 84 / UTM zone 43N",
            is_projected=True,
            is_geographic=False,
            map_units="metre",
            transform_matrix=(0.1, 0.0, 500000.0, 0.0, -0.1, 2300000.0),
            is_rotated_or_sheared=False,
            pixel_width=0.1,
            pixel_height=0.1,
            gsd_x_meters=0.1,
            gsd_y_meters=0.1,
            pixel_area_m2=0.01,
            gsd_status="VALIDATED",
            metadata_provenance="geotiff_embedded_tags",
            status_reason="Projected CRS verified.",
        )
        cal_meta = resolve_image_calibration(geo_metadata=geo_meta)
        self.assertEqual(cal_meta.calibration_status, "EXTRACTED_METADATA")
        self.assertFalse(cal_meta.is_user_supplied)
        self.assertTrue(cal_meta.is_validated_metadata)

    # -------------------------------------------------------------------------
    # TEST 7: Precedence: Preventing automatic & manual from being applied twice
    # -------------------------------------------------------------------------
    def test_precedence_automatic_over_manual(self):
        """When BOTH GeoTIFF metadata and custom GSD exist, GeoTIFF takes strict precedence."""
        geo_meta = GeoMetadata(
            is_georeferenced=True,
            format="GeoTIFF",
            width=640,
            height=640,
            crs_epsg=32643,
            crs_name="WGS 84 / UTM zone 43N",
            is_projected=True,
            is_geographic=False,
            map_units="metre",
            transform_matrix=(0.20, 0.0, 500000.0, 0.0, -0.20, 2300000.0),
            is_rotated_or_sheared=False,
            pixel_width=0.20,
            pixel_height=0.20,
            gsd_x_meters=0.20,
            gsd_y_meters=0.20,
            pixel_area_m2=0.04,  # From GeoTIFF: 0.04 m^2/px
            gsd_status="VALIDATED",
            metadata_provenance="geotiff_embedded_tags",
            status_reason="Projected CRS verified.",
        )

        # User attempts to supply custom GSD = 0.50 (which would give 0.25 m^2/px)
        cal = resolve_image_calibration(
            geo_metadata=geo_meta,
            custom_gsd=0.50,
            calibration_source="Conflicting user GSD parameter",
        )

        # GeoTIFF must win
        self.assertEqual(cal.calibration_type, "AUTOMATIC")
        self.assertEqual(cal.calibration_status, "EXTRACTED_METADATA")
        self.assertFalse(cal.is_user_supplied)
        self.assertAlmostEqual(cal.gsd_meters, 0.20, places=4)
        self.assertIn("GeoTIFF metadata takes strict precedence", cal.provenance.get("user_override_notice", ""))

    # -------------------------------------------------------------------------
    # TEST 8: Energy calculation with missing required inputs
    # -------------------------------------------------------------------------
    def test_energy_safeguard_missing_area_or_ghi(self):
        """Energy potential must remain insufficient_data when area or GHI is missing."""
        # Case A: Uncalibrated area (area is None)
        uncal = ImageCalibration.create_uncalibrated()
        area_res_null = calculate_calibrated_area(500, 10000, uncal)

        energy_res_no_area = SolarEnergyModel.estimate_energy(
            area_result=area_res_null,
            annual_ghi_kwh_m2_year=1850.0,
        )
        self.assertEqual(energy_res_no_area.energy_status, "insufficient_data")
        self.assertIsNone(energy_res_no_area.estimated_energy_kwh_year)
        self.assertTrue(len(energy_res_no_area.unavailability_reasons) > 0)

        # Case B: Calibrated area but missing GHI
        cal = resolve_image_calibration(geo_metadata=None, custom_gsd=0.05, calibration_source="Drone survey")
        area_res_valid = calculate_calibrated_area(500, 10000, cal)

        energy_res_no_ghi = SolarEnergyModel.estimate_energy(
            area_result=area_res_valid,
            annual_ghi_kwh_m2_year=None,
        )
        self.assertEqual(energy_res_no_ghi.energy_status, "insufficient_data")
        self.assertIsNone(energy_res_no_ghi.estimated_energy_kwh_year)
        self.assertTrue(len(energy_res_no_ghi.unavailability_reasons) > 0)

    # -------------------------------------------------------------------------
    # TEST 9: CO2 calculation with missing energy
    # -------------------------------------------------------------------------
    def test_co2_safeguard_missing_energy(self):
        """CO2 avoidance must remain insufficient_data when energy is missing."""
        uncal = ImageCalibration.create_uncalibrated()
        area_res_null = calculate_calibrated_area(500, 10000, uncal)

        energy_res_no_area = SolarEnergyModel.estimate_energy(
            area_result=area_res_null,
            annual_ghi_kwh_m2_year=1850.0,
        )

        co2_res = CO2AvoidanceModel.estimate_avoided_emissions(
            energy_input=energy_res_no_area,
        )
        self.assertEqual(co2_res.co2_status, "insufficient_data")
        self.assertIsNone(co2_res.co2_avoided_kg_year)
        self.assertIsNone(co2_res.co2_avoided_tonnes_year)
        self.assertTrue(len(co2_res.unavailability_reasons) > 0)

    # -------------------------------------------------------------------------
    # TEST 10: Regression tests for API prediction with calibration
    # -------------------------------------------------------------------------
    def test_api_predict_with_valid_and_invalid_custom_gsd(self):
        """Tests POST /predict API with valid custom_gsd and rejects invalid custom_gsd with 400."""
        # Create a small valid test PNG
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), color=(100, 150, 200)).save(buf, format="PNG")
        buf.seek(0)
        img_bytes = buf.getvalue()

        # 1. Uncalibrated request (no GSD)
        res_uncal = self.client.post(
            "/predict",
            files={"file": ("test.png", io.BytesIO(img_bytes), "image/png")},
        )
        self.assertEqual(res_uncal.status_code, 200)
        data_uncal = res_uncal.json()
        self.assertIsNone(data_uncal["physical_area_m2"])
        self.assertEqual(data_uncal["physical_area_status"], "insufficient_data")
        self.assertEqual(data_uncal["calibration_type"], "NONE")
        self.assertEqual(data_uncal["calibration_status"], "UNAVAILABLE")

        # 2. Calibrated request with valid GSD and CEA benchmark
        res_cal = self.client.post(
            "/predict?custom_gsd=0.05&calibration_source=Aerial+inspection+survey&use_cea_benchmark=true",
            files={"file": ("test.png", io.BytesIO(img_bytes), "image/png")},
        )
        self.assertEqual(res_cal.status_code, 200)
        data_cal = res_cal.json()
        self.assertEqual(data_cal["calibration_type"], "MANUAL")
        self.assertEqual(data_cal["calibration_status"], "USER_SUPPLIED")
        self.assertEqual(data_cal["physical_area_status"], "calculated")
        self.assertIsNotNone(data_cal["physical_area_m2"])
        # Pixel area = 0.05^2 = 0.0025 m^2
        expected_area = data_cal["solar_area_pixels"] * 0.0025
        self.assertAlmostEqual(data_cal["physical_area_m2"], expected_area, places=3)

        # 3. Request with invalid negative GSD -> HTTP 400
        res_invalid = self.client.post(
            "/predict?custom_gsd=-0.5",
            files={"file": ("test.png", io.BytesIO(img_bytes), "image/png")},
        )
        self.assertEqual(res_invalid.status_code, 400)
        err_msg = res_invalid.json().get("error", "")
        self.assertIn("strictly positive", err_msg.lower())

        # 4. Request with out-of-bounds GSD -> HTTP 400
        res_oob = self.client.post(
            "/predict?custom_gsd=500.0",
            files={"file": ("test.png", io.BytesIO(img_bytes), "image/png")},
        )
        self.assertEqual(res_oob.status_code, 400)
        oob_msg = res_oob.json().get("error", "")
        self.assertIn("exceeds plausible rooftop solar observation scale", oob_msg.lower())


if __name__ == "__main__":
    unittest.main()
