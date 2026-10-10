"""
SolarMap-India — Phase 6 Energy & Physical Area Test Suite.

Automated verification covering:
1. Pixel-area calculation.
2. Solar-area calculation.
3. Invalid GSD.
4. Missing GSD.
5. Zero solar pixels.
6. Missing solar resource.
7. Invalid efficiency.
8. Invalid performance ratio.
9. Annual GHI conversion.
10. Energy formula correctness.
11. Unit consistency.
12. Insufficient-data behavior.
13. No-fabrication safeguards.
14. End-to-end integration with mock and real data.
"""

from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.energy.assumptions import (
    EnergyModelConfig,
    ScientificAssumptions,
    STANDARD_EFFICIENCY_BENCHMARK,
    STANDARD_PERFORMANCE_RATIO_BENCHMARK,
)
from ml.energy.validation import EnergyValidator
from ml.energy.area import PhysicalAreaCalculator, AreaCalculationResult
from ml.energy.energy_model import SolarEnergyModel, EnergyCalculationResult
from ml.energy.report import run_energy_analysis_pipeline, build_and_save_phase6_reports


class TestPhysicalAreaAndSolarEnergy(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name) / "reports"

    def tearDown(self):
        self.temp_dir.cleanup()

    # 1. Pixel-area calculation & 2. Solar-area calculation
    def test_pixel_and_solar_area_calculation_validated(self):
        # 1000 solar pixels, GSD = 0.5 m/px -> pixel_area = 0.25 m2 -> solar_area = 250 m2
        res = PhysicalAreaCalculator.calculate(
            solar_pixels=1000,
            total_pixels=409600,
            gsd_meters=0.5,
            is_gsd_validated=True,
        )
        self.assertEqual(res.physical_area_status, "calculated")
        self.assertEqual(res.gsd_status, "VALIDATED")
        self.assertAlmostEqual(res.pixel_area_m2, 0.25)
        self.assertAlmostEqual(res.solar_area_m2, 250.0)
        self.assertEqual(res.solar_pixels, 1000)
        self.assertEqual(res.total_pixels, 409600)

    # Anisotropic resolution (dx != dy)
    def test_anisotropic_gsd_calculation(self):
        res = PhysicalAreaCalculator.calculate(
            solar_pixels=500,
            total_pixels=100000,
            gsd_x_meters=0.4,
            gsd_y_meters=0.5,
            is_gsd_validated=True,
        )
        self.assertEqual(res.physical_area_status, "calculated")
        self.assertAlmostEqual(res.pixel_area_m2, 0.20)
        self.assertAlmostEqual(res.solar_area_m2, 100.0)

    # 3. Invalid GSD
    def test_invalid_gsd_rejection(self):
        # Negative GSD
        res_neg = PhysicalAreaCalculator.calculate(
            solar_pixels=100, total_pixels=10000, gsd_meters=-0.5, is_gsd_validated=True
        )
        self.assertEqual(res_neg.physical_area_status, "insufficient_data")
        self.assertIsNone(res_neg.solar_area_m2)

        # Zero GSD
        res_zero = PhysicalAreaCalculator.calculate(
            solar_pixels=100, total_pixels=10000, gsd_meters=0.0, is_gsd_validated=True
        )
        self.assertEqual(res_zero.physical_area_status, "insufficient_data")
        self.assertIsNone(res_zero.solar_area_m2)

    # 4. Missing GSD handling
    def test_missing_gsd_handling(self):
        res_none = PhysicalAreaCalculator.calculate(
            solar_pixels=100, total_pixels=10000, gsd_meters=None, is_gsd_validated=False
        )
        self.assertEqual(res_none.physical_area_status, "insufficient_data")
        self.assertEqual(res_none.gsd_status, "UNAVAILABLE")
        self.assertIsNone(res_none.solar_area_m2)
        self.assertIsNone(res_none.pixel_area_m2)

    # 5. Zero solar pixels handling
    def test_zero_solar_pixels(self):
        res_zero = PhysicalAreaCalculator.calculate(
            solar_pixels=0,
            total_pixels=409600,
            gsd_meters=0.5,
            is_gsd_validated=True,
        )
        self.assertEqual(res_zero.physical_area_status, "calculated")
        self.assertEqual(res_zero.solar_area_m2, 0.0)
        self.assertEqual(res_zero.pixel_coverage_percent, 0.0)

    # 6. Missing solar resource handling
    def test_missing_solar_resource_handling(self):
        area_res = PhysicalAreaCalculator.calculate(
            solar_pixels=1000, total_pixels=409600, gsd_meters=0.5, is_gsd_validated=True
        )
        cfg = EnergyModelConfig.create_benchmark_scenario()
        energy_res = SolarEnergyModel.estimate_energy(
            area_result=area_res,
            annual_ghi_kwh_m2_year=None,
            source_daily_ghi=None,
            config=cfg,
        )
        self.assertEqual(energy_res.energy_status, "insufficient_data")
        self.assertIsNone(energy_res.estimated_energy_kwh_year)
        self.assertTrue(any("Solar resource" in r for r in energy_res.unavailability_reasons))

    # 7. Invalid efficiency handling
    def test_invalid_efficiency_handling(self):
        ok, err = EnergyValidator.validate_efficiency(0.01)  # 1% is below plausible commercial PV
        self.assertFalse(ok)
        ok, err = EnergyValidator.validate_efficiency(0.55)  # 55% exceeds terrestrial single-junction
        self.assertFalse(ok)

    # 8. Invalid performance ratio handling
    def test_invalid_pr_handling(self):
        ok, err = EnergyValidator.validate_performance_ratio(0.20)  # 20% is below plausible range
        self.assertFalse(ok)
        ok, err = EnergyValidator.validate_performance_ratio(1.20)  # > 100% physically impossible
        self.assertFalse(ok)

    # 9. Annual GHI conversion
    def test_annual_ghi_conversion(self):
        daily_ghi = 5.4041
        area_res = PhysicalAreaCalculator.calculate(
            solar_pixels=1000, total_pixels=409600, gsd_meters=0.5, is_gsd_validated=True
        )
        cfg = EnergyModelConfig.create_benchmark_scenario()
        energy_res = SolarEnergyModel.estimate_energy(
            area_result=area_res,
            annual_ghi_kwh_m2_year=None,
            source_daily_ghi=daily_ghi,
            source_unit="kW-hr/m^2/day",
            config=cfg,
        )
        expected_annual = round(daily_ghi * 365.25, 2)
        self.assertEqual(energy_res.annual_ghi_kwh_m2_year, expected_annual)

    # 10. Energy formula: E = A * GHI * eta * PR
    def test_energy_formula_mathematics(self):
        # A = 250 m2, GHI = 2000 kWh/m2/year, eta = 0.20 (20%), PR = 0.75 (75%)
        # E = 250 * 2000 * 0.20 * 0.75 = 75,000 kWh/year
        area_res = AreaCalculationResult(
            solar_pixels=1000,
            total_pixels=409600,
            pixel_coverage_fraction=0.00244,
            pixel_coverage_percent=0.244,
            gsd_status="VALIDATED",
            gsd_meters=0.5,
            gsd_x_meters=0.5,
            gsd_y_meters=0.5,
            pixel_area_m2=0.25,
            solar_area_m2=250.0,
            physical_area_status="calculated",
            status_reason="Validated test scale",
        )
        cfg = EnergyModelConfig(
            module_efficiency=0.20,
            performance_ratio=0.75,
        )
        res = SolarEnergyModel.estimate_energy(
            area_result=area_res,
            annual_ghi_kwh_m2_year=2000.0,
            config=cfg,
        )
        self.assertEqual(res.energy_status, "estimated")
        self.assertAlmostEqual(res.estimated_energy_kwh_year, 75000.0)
        self.assertIn("NOT Actual Energy Generated", res.energy_label)

    # 11. Unit consistency
    def test_unit_consistency(self):
        area_res = PhysicalAreaCalculator.calculate(
            solar_pixels=1000, total_pixels=409600, gsd_meters=0.5, is_gsd_validated=True
        )
        cfg = EnergyModelConfig.create_benchmark_scenario()
        res = SolarEnergyModel.estimate_energy(
            area_result=area_res,
            source_daily_ghi=5.0,
            source_unit="kW-hr/m^2/day",
            config=cfg,
        )
        self.assertEqual(res.solar_resource_details["derived_unit"], "kWh/m2/year")
        self.assertEqual(res.solar_resource_details["conversion"], "daily × 365.25")

    # 12. Insufficient-data behavior when GSD is unverified
    def test_insufficient_data_behavior(self):
        # Default unverified pipeline run
        area_res = PhysicalAreaCalculator.calculate(
            solar_pixels=68451, total_pixels=409600, gsd_meters=None, is_gsd_validated=False
        )
        res = SolarEnergyModel.estimate_energy(
            area_result=area_res,
            annual_ghi_kwh_m2_year=1973.85,
            config=EnergyModelConfig.create_default(),  # null efficiency and PR
        )
        self.assertEqual(area_res.physical_area_status, "insufficient_data")
        self.assertIsNone(area_res.solar_area_m2)
        self.assertEqual(res.energy_status, "insufficient_data")
        self.assertIsNone(res.estimated_energy_kwh_year)
        self.assertGreaterEqual(len(res.unavailability_reasons), 2)

    # 13. No-fabrication safeguards
    def test_no_fabrication_safeguards(self):
        # Passing an unvalidated candidate GSD without flag must remain insufficient_data
        res = PhysicalAreaCalculator.calculate(
            solar_pixels=1000,
            total_pixels=409600,
            gsd_meters=1.0,  # 1 px = 1 m
            is_gsd_validated=False,
        )
        self.assertEqual(res.physical_area_status, "insufficient_data")
        self.assertEqual(res.gsd_status, "UNAVAILABLE")
        self.assertIsNone(res.solar_area_m2)

    # 14. Real image pipeline test on 768.0_1.0.png
    def test_real_image_pipeline(self):
        report = build_and_save_phase6_reports(
            target_image="768.0_1.0.png",
            output_dir=self.output_dir,
        )
        self.assertEqual(report["phase"], "PHASE 6 — PHYSICAL AREA & ESTIMATED SOLAR ENERGY POTENTIAL")
        prim = report["primary_analysis"]
        self.assertEqual(prim["image_name"], "768.0_1.0.png")
        self.assertEqual(prim["segmentation"]["solar_pixels"], 68451)
        self.assertEqual(prim["spatial_resolution"]["gsd_status"], "UNAVAILABLE")
        self.assertEqual(prim["spatial_resolution"]["physical_area_status"], "insufficient_data")
        self.assertIsNone(prim["spatial_resolution"]["solar_area_m2"])
        self.assertEqual(prim["solar_resource"]["status"], "available")
        self.assertEqual(prim["energy_model"]["energy_status"], "insufficient_data")
        self.assertIsNone(prim["energy_model"]["estimated_energy_kwh_year"])
        self.assertEqual(report["energy_estimation_readiness"], "BLOCKED")


if __name__ == "__main__":
    unittest.main()
