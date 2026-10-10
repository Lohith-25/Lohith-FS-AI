"""
Unit Tests for Phase 7A GSD-Based Physical Area Calculation.

Validates:
1. Valid GSD calculation.
2. Missing GSD handling.
3. Invalid GSD handling (negative, zero, prohibited assumed constants).
4. Physical area formula correctness (pixel_area = gsd^2, solar_area = solar_pixels * pixel_area).
5. Zero solar pixels behavior.
6. Provenance structure and validation.
7. Real image integration test on 768.0_1.0.png (verifying GSD=unavailable, Solar area=insufficient_data).
"""

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.area.gsd_provider import GSDProvider, GSDResult, PROHIBITED_GENERIC_GSDS
from ml.area.calculator import calculate_physical_area, PhysicalAreaResult
from ml.area.pipeline import compute_image_solar_area


class TestPhase7APhysicalArea(unittest.TestCase):

    # 1. Valid GSD
    def test_valid_gsd_provider(self):
        citation = "Official Calibration Certificate for Aerial Survey Mission #104"
        res = GSDProvider.get_gsd(custom_gsd=0.25, provenance_citation=citation)
        self.assertEqual(res.status, "VALIDATED")
        self.assertEqual(res.gsd_m_per_pixel, 0.25)
        self.assertEqual(res.source, citation)
        self.assertIsNone(res.rejection_reason)
        self.assertTrue(res.provenance["validated"])
        self.assertEqual(res.provenance["citation"], citation)

    # 2. Missing GSD
    def test_missing_gsd_provider(self):
        # Default query without external authenticated input
        res = GSDProvider.get_gsd()
        self.assertEqual(res.status, "UNAVAILABLE")
        self.assertIsNone(res.gsd_m_per_pixel)
        self.assertIn("Phase 5", res.source)
        self.assertFalse(res.provenance["validated"])
        self.assertIsNotNone(res.rejection_reason)

    # 3. Invalid GSD (negative, zero, prohibited assumed values)
    def test_invalid_and_prohibited_gsd(self):
        # Negative GSD
        res_neg = GSDProvider.get_gsd(custom_gsd=-0.5, provenance_citation="Test citation")
        self.assertEqual(res_neg.status, "UNAVAILABLE")
        self.assertIsNone(res_neg.gsd_m_per_pixel)

        # Zero GSD
        res_zero = GSDProvider.get_gsd(custom_gsd=0.0, provenance_citation="Test citation")
        self.assertEqual(res_zero.status, "UNAVAILABLE")
        self.assertIsNone(res_zero.gsd_m_per_pixel)

        # Prohibited generic 1 px = 1 m
        res_1m = GSDProvider.get_gsd(custom_gsd=1.0, provenance_citation="Some claim")
        self.assertEqual(res_1m.status, "UNAVAILABLE")
        self.assertIn("1 pixel = 1 meter", res_1m.rejection_reason)

        # Prohibited generic 0.5 m/pixel
        res_half = GSDProvider.get_gsd(custom_gsd=0.5, provenance_citation="Some claim")
        self.assertEqual(res_half.status, "UNAVAILABLE")
        self.assertIn("0.5 m/pixel", res_half.rejection_reason)

        # Prohibited Sentinel-2 10m
        res_10m = GSDProvider.get_gsd(custom_gsd=10.0, provenance_citation="Some claim")
        self.assertEqual(res_10m.status, "UNAVAILABLE")
        self.assertIn("Sentinel-2", res_10m.rejection_reason)

        # Missing provenance citation
        res_no_cite = GSDProvider.get_gsd(custom_gsd=0.25, provenance_citation="")
        self.assertEqual(res_no_cite.status, "UNAVAILABLE")
        self.assertIn("lacks authoritative provenance citation", res_no_cite.rejection_reason)

    # 4. Area calculation with validated GSD
    def test_area_calculation_validated(self):
        # 10,000 pixels at 0.2 m/pixel
        # pixel_area = 0.2 * 0.2 = 0.04 m2
        # solar_area = 10,000 * 0.04 = 400.0 m2
        gsd_res = GSDResult(
            gsd_m_per_pixel=0.2,
            status="VALIDATED",
            source="Calibrated Sensor",
            provenance={"validated": True},
        )
        area_res = calculate_physical_area(solar_pixels=10000, gsd_result=gsd_res, total_pixels=409600)
        self.assertEqual(area_res.area_status, "calculated")
        self.assertEqual(area_res.gsd_status, "VALIDATED")
        self.assertAlmostEqual(area_res.pixel_area_m2, 0.04)
        self.assertAlmostEqual(area_res.solar_area_m2, 400.0)
        self.assertEqual(area_res.solar_pixels, 10000)
        self.assertEqual(area_res.total_pixels, 409600)

    # Area calculation with UNAVAILABLE GSD
    def test_area_calculation_unavailable_gsd(self):
        gsd_res = GSDResult(
            gsd_m_per_pixel=None,
            status="UNAVAILABLE",
            source="Unknown Source",
            provenance={"validated": False},
            rejection_reason="No GSD metadata",
        )
        area_res = calculate_physical_area(solar_pixels=68451, gsd_result=gsd_res, total_pixels=409600)
        self.assertEqual(area_res.area_status, "insufficient_data")
        self.assertEqual(area_res.gsd_status, "UNAVAILABLE")
        self.assertIsNone(area_res.solar_area_m2)
        self.assertIsNone(area_res.pixel_area_m2)
        self.assertEqual(area_res.solar_pixels, 68451)

    # 5. Zero solar pixels behavior
    def test_zero_solar_pixels(self):
        # Valid GSD + 0 solar pixels -> solar_area = 0.0
        gsd_res = GSDResult(
            gsd_m_per_pixel=0.25,
            status="VALIDATED",
            source="Test Sensor",
            provenance={"validated": True},
        )
        area_res = calculate_physical_area(solar_pixels=0, gsd_result=gsd_res, total_pixels=409600)
        self.assertEqual(area_res.area_status, "calculated")
        self.assertAlmostEqual(area_res.solar_area_m2, 0.0)

        # Unavailable GSD + 0 solar pixels -> solar_area = None, insufficient_data
        gsd_unavail = GSDResult(
            gsd_m_per_pixel=None,
            status="UNAVAILABLE",
            source="Unknown",
            provenance={"validated": False},
        )
        area_res_unavail = calculate_physical_area(solar_pixels=0, gsd_result=gsd_unavail, total_pixels=409600)
        self.assertEqual(area_res_unavail.area_status, "insufficient_data")
        self.assertIsNone(area_res_unavail.solar_area_m2)

    # 6. Provenance validation
    def test_provenance_validation(self):
        citation = "Airborne Orthophoto Survey 2024 - Gujarat Energy Development Agency"
        res = GSDProvider.get_gsd(custom_gsd=0.15, provenance_citation=citation)
        self.assertIn("retrieval_date_utc", res.provenance)
        self.assertEqual(res.provenance["citation"], citation)
        self.assertEqual(res.provenance["gsd_m_per_pixel"], 0.15)
        self.assertTrue(res.provenance["validated"])

    # 7. Real image integration test with 768.0_1.0.png
    def test_real_image_768_returns_unavailable_and_insufficient_data(self):
        """
        Requirements 9 & 10:
        Test with 768.0_1.0.png. Because current dataset has no validated GSD,
        the real image must currently return:
            GSD = unavailable
            Solar area = insufficient_data
        """
        area_res = compute_image_solar_area("768.0_1.0.png")
        self.assertEqual(area_res.solar_pixels, 68451)
        self.assertEqual(area_res.total_pixels, 409600)
        self.assertEqual(area_res.gsd_status, "UNAVAILABLE")
        self.assertIsNone(area_res.gsd_m_per_pixel)
        self.assertEqual(area_res.area_status, "insufficient_data")
        self.assertIsNone(area_res.solar_area_m2)
        self.assertIsNone(area_res.pixel_area_m2)


if __name__ == "__main__":
    unittest.main()
