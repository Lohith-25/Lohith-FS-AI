"""
SolarMap-India — Phase 14.2: Geospatial Metadata & GSD Test Suite.

Automated tests for:
1. GeoTIFF with valid projected CRS (UTM 43N) and metric pixel resolution.
2. GeoTIFF with geographic CRS (WGS84 degrees) and ellipsoidal GSD computation.
3. TIFF without georeferencing metadata tags.
4. Invalid or missing CRS in GeoTIFF.
5. Invalid affine transform (missing coordinates, corrupt tags).
6. Rotated or sheared affine transform matrix via determinant area calculation.
7. Unsupported or unknown linear units.
8. Standard PNG/JPEG without spatial metadata.
9. Physical-area calculations with valid metadata (m² and hectares).
10. Physical-area calculations when metadata is insufficient (strict null & insufficient_data).
11. Missing metadata must not produce fabricated energy or CO2 values.
12. Existing PNG/JPEG prediction behavior remains fully compatible.
13. Prediction history preserves supported geospatial metadata.
14. API response compatibility (fields exposed, backward-compatible schemas).
"""

import io
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image
from PIL.TiffImagePlugin import ImageFileDirectory_v2
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app
from backend.model_manager import model_manager
from ml.energy.area import PhysicalAreaCalculator
from ml.energy.energy_model import EnergyModelConfig, SolarEnergyModel
from ml.environmental.co2_model import CO2AvoidanceModel
from ml.geospatial.metadata import (
    GeoMetadata,
    extract_geospatial_metadata,
)

SAMPLE_REAL_IMAGE = (
    PROJECT_ROOT
    / "dataset"
    / "Solar Images"
    / "Solar Images"
    / "images"
    / "default"
    / "768.0_1.0.png"
)


class TestPhase142GeospatialMetadata(unittest.TestCase):
    """Test suite covering GeoTIFF spatial metadata extraction, affine geometry, and area calculation."""

    @classmethod
    def setUpClass(cls):
        model_manager.load_model()
        cls.client = TestClient(app)

    def _create_projected_geotiff_bytes(
        self, width: int = 100, height: int = 100, scale_x: float = 0.5, scale_y: float = 0.5, epsg: int = 32643
    ) -> bytes:
        """Helper creating valid GeoTIFF bytes in a Projected CRS (e.g. UTM 43N)."""
        buf = io.BytesIO()
        im = Image.new("RGB", (width, height), color=(100, 150, 200))
        ifd = ImageFileDirectory_v2()
        ifd[33550] = (float(scale_x), float(scale_y), 0.0)
        ifd[33922] = (0.0, 0.0, 0.0, 269000.0, 2345000.0, 0.0)
        ifd[34735] = (1, 1, 0, 2, 1024, 0, 1, 1, 3072, 0, 1, int(epsg))
        im.save(buf, format="TIFF", tiffinfo=ifd)
        return buf.getvalue()

    def _create_geographic_geotiff_bytes(
        self, width: int = 100, height: int = 100, deg_scale: float = 0.00001, lon: float = 72.83, lat: float = 21.17
    ) -> bytes:
        """Helper creating valid GeoTIFF bytes in a Geographic CRS (WGS84 EPSG:4326)."""
        buf = io.BytesIO()
        im = Image.new("RGB", (width, height), color=(100, 150, 200))
        ifd = ImageFileDirectory_v2()
        ifd[33550] = (float(deg_scale), float(deg_scale), 0.0)
        ifd[33922] = (0.0, 0.0, 0.0, float(lon), float(lat), 0.0)
        ifd[34735] = (1, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 4326)
        im.save(buf, format="TIFF", tiffinfo=ifd)
        return buf.getvalue()

    def _create_matrix_geotiff_bytes(
        self, width: int = 100, height: int = 100, matrix_16: tuple = None, epsg: int = 32643
    ) -> bytes:
        """Helper creating GeoTIFF bytes with ModelTransformationTag 34264."""
        buf = io.BytesIO()
        im = Image.new("RGB", (width, height), color=(100, 150, 200))
        ifd = ImageFileDirectory_v2()
        if matrix_16 is None:
            # 30 degree rotation with 0.5m scale
            theta = np.pi / 6.0
            c_val, s_val = 0.5 * np.cos(theta), 0.5 * np.sin(theta)
            matrix_16 = (
                c_val, -s_val, 0.0, 269000.0,
                s_val,  c_val, 0.0, 2345000.0,
                0.0,    0.0,   0.0, 0.0,
                0.0,    0.0,   0.0, 1.0,
            )
        ifd[34264] = matrix_16
        ifd[34735] = (1, 1, 0, 2, 1024, 0, 1, 1, 3072, 0, 1, int(epsg))
        im.save(buf, format="TIFF", tiffinfo=ifd)
        return buf.getvalue()

    def test_01_projected_geotiff_extraction(self):
        """Test 1: Valid projected CRS (UTM 43N) extracts metric resolution and pixel area."""
        raw_bytes = self._create_projected_geotiff_bytes(scale_x=0.5, scale_y=0.5, epsg=32643)
        meta = extract_geospatial_metadata(raw_bytes)

        self.assertTrue(meta.is_georeferenced)
        self.assertTrue(meta.is_projected)
        self.assertFalse(meta.is_geographic)
        self.assertEqual(meta.crs_epsg, 32643)
        self.assertIn("UTM", meta.crs_name)
        self.assertEqual(meta.map_units, "metre")
        self.assertEqual(meta.gsd_x_meters, 0.5)
        self.assertEqual(meta.gsd_y_meters, 0.5)
        self.assertEqual(meta.pixel_area_m2, 0.25)
        self.assertEqual(meta.gsd_status, "VALIDATED")
        self.assertFalse(meta.is_rotated_or_sheared)

    def test_02_geographic_geotiff_ellipsoidal_gsd(self):
        """Test 2: Geographic CRS (degrees) converts to metric GSD via WGS84 geodesic math."""
        raw_bytes = self._create_geographic_geotiff_bytes(deg_scale=0.00001, lon=72.83, lat=21.17)
        meta = extract_geospatial_metadata(raw_bytes)

        self.assertTrue(meta.is_georeferenced)
        self.assertFalse(meta.is_projected)
        self.assertTrue(meta.is_geographic)
        self.assertEqual(meta.crs_epsg, 4326)
        self.assertEqual(meta.map_units, "degree")
        self.assertIsNotNone(meta.gsd_x_meters)
        self.assertIsNotNone(meta.gsd_y_meters)
        self.assertIsNotNone(meta.pixel_area_m2)
        self.assertEqual(meta.gsd_status, "VALIDATED")

        # In Surat (~21.17 deg N): 0.00001 deg lon is ~1.038 m; 0.00001 deg lat is ~1.107 m
        self.assertAlmostEqual(meta.gsd_x_meters, 1.0385, delta=0.05)
        self.assertAlmostEqual(meta.gsd_y_meters, 1.1072, delta=0.05)
        self.assertGreater(meta.pixel_area_m2, 1.0)

    def test_03_tiff_without_georeferencing_tags(self):
        """Test 3: Plain TIFF without spatial tags returns UNAVAILABLE."""
        buf = io.BytesIO()
        im = Image.new("RGB", (64, 64), (100, 100, 100))
        im.save(buf, format="TIFF")
        meta = extract_geospatial_metadata(buf.getvalue())

        self.assertFalse(meta.is_georeferenced)
        self.assertEqual(meta.gsd_status, "UNAVAILABLE")
        self.assertIsNone(meta.pixel_area_m2)
        self.assertIsNone(meta.gsd_x_meters)

    def test_04_geotiff_missing_or_invalid_crs(self):
        """Test 4: GeoTIFF with pixel scale but missing CRS returns INSUFFICIENT_DATA."""
        buf = io.BytesIO()
        im = Image.new("RGB", (64, 64), (100, 100, 100))
        ifd = ImageFileDirectory_v2()
        ifd[33550] = (0.5, 0.5, 0.0)
        ifd[33922] = (0.0, 0.0, 0.0, 1000.0, 2000.0, 0.0)
        # No GeoKeyDirectoryTag 34735
        im.save(buf, format="TIFF", tiffinfo=ifd)
        meta = extract_geospatial_metadata(buf.getvalue())

        self.assertFalse(meta.is_georeferenced)
        self.assertEqual(meta.gsd_status, "INSUFFICIENT_DATA")
        self.assertIn("missing", meta.status_reason.lower())

    def test_05_invalid_affine_transform(self):
        """Test 5: GeoTIFF with degenerate (singular) matrix returns INSUFFICIENT_DATA."""
        singular_matrix = (
            0.0, 0.0, 0.0, 1000.0,
            0.0, 0.0, 0.0, 2000.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
        raw_bytes = self._create_matrix_geotiff_bytes(matrix_16=singular_matrix)
        meta = extract_geospatial_metadata(raw_bytes)

        self.assertFalse(meta.is_georeferenced)
        self.assertEqual(meta.gsd_status, "INSUFFICIENT_DATA")
        self.assertIn("degenerate", meta.status_reason.lower())

    def test_06_rotated_and_sheared_affine_transform(self):
        """Test 6: Rotated GeoTIFF calculates true pixel area using determinant without error."""
        raw_bytes = self._create_matrix_geotiff_bytes()
        meta = extract_geospatial_metadata(raw_bytes)

        self.assertTrue(meta.is_georeferenced)
        self.assertTrue(meta.is_rotated_or_sheared)
        self.assertEqual(meta.gsd_status, "VALIDATED")
        # Scale was 0.5m rotated 30 deg -> determinant area = 0.5^2 = 0.25 m^2
        self.assertAlmostEqual(meta.pixel_area_m2, 0.25, places=4)

    def test_07_unsupported_or_unknown_units(self):
        """Test 7: Projected CRS with Foot units converts correctly to meters."""
        buf = io.BytesIO()
        im = Image.new("RGB", (50, 50), (100, 100, 100))
        ifd = ImageFileDirectory_v2()
        ifd[33550] = (3.0, 3.0, 0.0)  # 3 feet per pixel
        ifd[33922] = (0.0, 0.0, 0.0, 1000000.0, 2000000.0, 0.0)
        # Linear units code 9002 = Linear_Foot
        ifd[34735] = (1, 1, 0, 3, 1024, 0, 1, 1, 3072, 0, 1, 32643, 3076, 0, 1, 9002)
        im.save(buf, format="TIFF", tiffinfo=ifd)
        meta = extract_geospatial_metadata(buf.getvalue())

        self.assertTrue(meta.is_georeferenced)
        # 3 feet * 0.3048 = 0.9144 meters per pixel
        self.assertAlmostEqual(meta.gsd_x_meters, 0.9144, places=4)
        self.assertAlmostEqual(meta.pixel_area_m2, 0.9144 ** 2, places=4)

    def test_08_standard_png_and_jpeg_without_spatial_metadata(self):
        """Test 8: PNG and JPEG rasters gracefully return UNAVAILABLE without error."""
        buf_png = io.BytesIO()
        Image.new("RGB", (64, 64), (120, 140, 160)).save(buf_png, format="PNG")
        meta_png = extract_geospatial_metadata(buf_png.getvalue())
        self.assertFalse(meta_png.is_georeferenced)
        self.assertEqual(meta_png.gsd_status, "UNAVAILABLE")

        buf_jpg = io.BytesIO()
        Image.new("RGB", (64, 64), (120, 140, 160)).save(buf_jpg, format="JPEG")
        meta_jpg = extract_geospatial_metadata(buf_jpg.getvalue())
        self.assertFalse(meta_jpg.is_georeferenced)
        self.assertEqual(meta_jpg.gsd_status, "UNAVAILABLE")

    def test_09_physical_area_calculation_with_valid_metadata(self):
        """Test 9: PhysicalAreaCalculator computes verified solar area (m²) when metadata is VALIDATED."""
        area_calc = PhysicalAreaCalculator.calculate(
            solar_pixels=400,
            total_pixels=10000,
            pixel_area_m2=0.25,
            is_gsd_validated=True,
        )
        self.assertEqual(area_calc.physical_area_status, "calculated")
        self.assertEqual(area_calc.gsd_status, "VALIDATED")
        # 400 pixels * 0.25 m²/pixel = 100.0 m²
        self.assertEqual(area_calc.solar_area_m2, 100.0)
        self.assertEqual(area_calc.pixel_area_m2, 0.25)

        # Energy and CO2 models consume valid area and produce estimated yields
        benchmark_cfg = EnergyModelConfig.create_benchmark_scenario(efficiency=0.20, performance_ratio=0.75)
        energy = SolarEnergyModel.estimate_energy(
            area_result=area_calc,
            annual_ghi_kwh_m2_year=1800.0,
            config=benchmark_cfg,
        )
        self.assertEqual(energy.energy_status, "estimated")
        self.assertGreater(energy.estimated_energy_kwh_year, 0.0)

        co2 = CO2AvoidanceModel.estimate_avoided_emissions(
            energy_input=energy,
            emission_factor_kg_kwh=0.716,
        )
        self.assertEqual(co2.co2_status, "estimated")
        self.assertGreater(co2.co2_avoided_kg_year, 0.0)

    def test_10_physical_area_calculation_when_metadata_is_insufficient(self):
        """Test 10: Physical area remains strictly null and insufficient_data when GSD is missing."""
        area_calc = PhysicalAreaCalculator.calculate(
            solar_pixels=400,
            total_pixels=10000,
            pixel_area_m2=None,
            is_gsd_validated=False,
        )
        self.assertEqual(area_calc.physical_area_status, "insufficient_data")
        self.assertEqual(area_calc.gsd_status, "UNAVAILABLE")
        self.assertIsNone(area_calc.solar_area_m2)
        self.assertIsNone(area_calc.pixel_area_m2)

    def test_11_missing_metadata_prevents_fabricated_energy_or_co2(self):
        """Test 11: Missing GSD strictly propagates insufficient_data to energy and CO2 models."""
        area_calc = PhysicalAreaCalculator.calculate(
            solar_pixels=500,
            total_pixels=2500,
            is_gsd_validated=False,
        )
        energy = SolarEnergyModel.estimate_energy(area_calc)
        co2 = CO2AvoidanceModel.estimate_avoided_emissions(energy)

        self.assertEqual(energy.energy_status, "insufficient_data")
        self.assertIsNone(energy.estimated_energy_kwh_year)
        self.assertEqual(co2.co2_status, "insufficient_data")
        self.assertIsNone(co2.co2_avoided_kg_year)
        self.assertIsNone(co2.co2_avoided_tonnes_year)

    def test_12_existing_png_jpeg_prediction_compatibility(self):
        """Test 12: Standard PNG benchmark prediction preserves exact Phase 1 & 13 metrics."""
        self.assertTrue(SAMPLE_REAL_IMAGE.is_file())
        with open(SAMPLE_REAL_IMAGE, "rb") as fp:
            file_bytes = fp.read()

        res = self.client.post("/predict", files={"file": ("768.0_1.0.png", file_bytes, "image/png")})
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["image_width"], 640)
        self.assertEqual(data["image_height"], 640)
        self.assertEqual(data["solar_area_pixels"], 68451)
        self.assertEqual(data["detected_region_count"], 43)
        self.assertEqual(data["gsd_status"], "UNAVAILABLE")
        self.assertIsNone(data["physical_area_m2"])
        self.assertEqual(data["physical_area_status"], "insufficient_data")

    def test_13_prediction_history_preserves_geospatial_metadata(self):
        """Test 13: Prediction endpoint extracts GeoTIFF and includes metadata in response payload."""
        geotiff_bytes = self._create_projected_geotiff_bytes(scale_x=0.5, scale_y=0.5, epsg=32643)
        res = self.client.post(
            "/predict",
            files={"file": ("solar_farm_utm43n.tif", geotiff_bytes, "image/tiff")}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["gsd_status"], "VALIDATED")
        self.assertIn("0.500m x 0.500m/px", data["source_resolution"])
        self.assertIn("UTM zone 43N", data["source_crs"])
        self.assertEqual(data["metadata_provenance"], "geotiff_projected_crs")
        self.assertEqual(data["georeferencing_level"], "LEVEL_A")

    def test_14_api_response_schema_compatibility(self):
        """Test 14: PredictResponse contains all required fields without breaking changes."""
        raw_bytes = self._create_geographic_geotiff_bytes()
        res = self.client.post(
            "/predict",
            files={"file": ("geographic_scene.tiff", raw_bytes, "image/tiff")}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()

        expected_fields = [
            "success", "model", "prediction_id", "image_width", "image_height",
            "total_image_pixels", "solar_area_pixels", "solar_coverage_percent",
            "detected_region_count", "physical_area_m2", "physical_area_status",
            "energy_status", "co2_status", "georeferencing_level", "mask_url", "overlay_url",
            "processing_strategy", "gsd_status", "source_resolution", "source_crs", "metadata_provenance",
        ]
        for field in expected_fields:
            self.assertIn(field, data, f"Field '{field}' missing from API response")


if __name__ == "__main__":
    unittest.main()
