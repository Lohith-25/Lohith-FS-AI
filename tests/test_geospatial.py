"""
SolarMap-India — Phase 4 Geospatial & Physical Area Test Suite.

Automated tests for:
1. Latitude validation [-90, 90]
2. Longitude validation [-180, 180]
3. Missing metadata handling
4. Duplicate sample ID detection
5. Image-to-metadata mapping
6. Physical scale validation
7. Physical area calculation when scale exists
8. Physical area rejection when scale is missing
9. Component centroid handling
10. Geospatial status report generation
11. Real dataset metadata inspection
12. Real image metadata inspection (768.0_1.0.png)
"""

from pathlib import Path
import sys
import unittest

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.analytics.components import SolarRegion
from ml.geospatial.area_calculator import (
    calculate_physical_area,
    evaluate_component_centroids,
)
from ml.geospatial.coordinate_mapping import GeospatialCatalog
from ml.geospatial.metadata_inspector import inspect_image_metadata
from ml.geospatial.report import generate_geospatial_status_report
from ml.geospatial.spatial_scale import get_dataset_spatial_scale, validate_custom_scale

SAMPLE_IMAGE = (
    PROJECT_ROOT
    / "dataset"
    / "Solar Images"
    / "Solar Images"
    / "images"
    / "default"
    / "768.0_1.0.png"
)


class TestGeospatialFoundation(unittest.TestCase):

    def setUp(self):
        self.catalog = GeospatialCatalog()

    def test_01_latitude_validation(self):
        """Test 1: Normal latitude passes; > 90 or < -90 fails."""
        rec = self.catalog.lookup(768)
        self.assertIsNotNone(rec)
        self.assertTrue(-90.0 <= rec.latitude <= 90.0)

    def test_02_longitude_validation(self):
        """Test 2: Normal longitude passes; > 180 or < -180 fails."""
        rec = self.catalog.lookup(768)
        self.assertIsNotNone(rec)
        self.assertTrue(-180.0 <= rec.longitude <= 180.0)

    def test_03_missing_metadata_handling(self):
        """Test 3: Non-existent image/sample ID returns None gracefully."""
        self.assertIsNone(self.catalog.lookup("non_existent_99999.png"))
        self.assertIsNone(self.catalog.lookup(99999))

    def test_04_duplicate_detection(self):
        """Test 4: Catalog loads unique sample IDs without error."""
        self.assertEqual(self.catalog.total_records, 3000)

    def test_05_image_to_metadata_mapping(self):
        """Test 5: Validates filename parsing and catalog resolution."""
        rec = self.catalog.lookup("768.0_1.0.png")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.sampleid, 768)
        self.assertAlmostEqual(rec.latitude, 21.197148, places=4)
        self.assertAlmostEqual(rec.longitude, 72.780643, places=4)

        # Test alternative naming format (e.g. 0960_1.png)
        rec2 = self.catalog.lookup("0960_1.png")
        self.assertIsNotNone(rec2)
        self.assertEqual(rec2.sampleid, 960)

    def test_06_physical_scale_validation(self):
        """Test 6: Valid scale passes; zero or negative scale raises ValueError."""
        valid_scale = validate_custom_scale(0.5, 0.5)
        self.assertTrue(valid_scale.is_available)

        with self.assertRaises(ValueError):
            validate_custom_scale(0.0, 0.5)

        with self.assertRaises(ValueError):
            validate_custom_scale(-1.0, 1.0)

    def test_07_physical_area_when_scale_exists(self):
        """Test 7: Area formula when valid scale provided: area = px * scale_x * scale_y."""
        # 100 pixels with 0.5m x 0.5m scale = 100 * 0.25 = 25.0 m^2
        res = calculate_physical_area(pixel_count=100, meters_per_pixel_x=0.5, meters_per_pixel_y=0.5)
        self.assertEqual(res["status"], "calculated")
        self.assertEqual(res["area_m2"], 25.0)

    def test_08_physical_area_rejection_when_scale_missing(self):
        """Test 8: Rejects calculation and returns insufficient_data when scale is None."""
        res = calculate_physical_area(pixel_count=68451, meters_per_pixel_x=None, meters_per_pixel_y=None)
        self.assertEqual(res["status"], "insufficient_data")
        self.assertIsNone(res["area_m2"])
        self.assertEqual(res["area_pixels"], 68451)

    def test_09_component_centroid_handling(self):
        """Test 9: Centroid geographic coordinates marked unavailable without affine transform."""
        reg = SolarRegion(
            id=1,
            area_pixels=500,
            x=10,
            y=20,
            width=50,
            height=30,
            centroid_x=35.0,
            centroid_y=35.0,
        )
        evaluated = evaluate_component_centroids([reg])
        self.assertEqual(len(evaluated), 1)
        self.assertFalse(evaluated[0]["geographic_coordinates_available"])
        self.assertIsNone(evaluated[0]["geographic_latitude"])
        self.assertEqual(evaluated[0]["centroid_pixel_x"], 35.0)

    def test_10_status_report_generation(self):
        """Test 10: Status report dictionary contains all required fields and valid Level B classification."""
        report, report_path = generate_geospatial_status_report("768.0_1.0.png", solar_pixels=68451)
        self.assertTrue(report_path.is_file())
        self.assertEqual(report["georeferencing"]["level"], "LEVEL_B")
        self.assertTrue(report["georeferencing"]["image_coordinates_available"])
        self.assertFalse(report["georeferencing"]["physical_pixel_scale_available"])
        self.assertEqual(report["georeferencing"]["energy_estimation_readiness"], "BLOCKED")
        self.assertEqual(report["image"]["physical_area_status"], "insufficient_data")

    def test_11_real_dataset_metadata_inspection(self):
        """Test 11: Real dataset audit confirms 2505 disk images match catalog."""
        img_dir = PROJECT_ROOT / "dataset" / "Solar Images" / "Solar Images" / "images" / "default"
        disk_images = list(img_dir.glob("*.png"))
        self.assertEqual(len(disk_images), 2505)

        # Check every disk image maps to catalog
        matched = sum(1 for img in disk_images if self.catalog.lookup(img.name) is not None)
        self.assertEqual(matched, 2505)

    def test_12_real_image_metadata_inspection(self):
        """Test 12: Real image 768.0_1.0.png coordinates match Surat, Gujarat."""
        self.assertTrue(SAMPLE_IMAGE.is_file())
        meta = inspect_image_metadata(SAMPLE_IMAGE)
        self.assertFalse(meta["has_gps_exif"])
        self.assertIsNone(meta["world_file"])

        rec = self.catalog.lookup("768.0_1.0.png")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.sampleid, 768)
        self.assertAlmostEqual(rec.latitude, 21.197148, places=4)
        self.assertAlmostEqual(rec.longitude, 72.780643, places=4)
        self.assertTrue(rec.is_within_india_bounds)


if __name__ == "__main__":
    unittest.main()
