"""
SolarMap-India — Phase 7 CO2 & Environmental Analysis Test Suite.

Automated verification covering:
1. Correct CO2 avoidance calculation with valid energy generation.
2. Conversion from kilograms to metric tonnes (1000 kg = 1 metric tonne).
3. Missing energy generation (returns insufficient_data and nulls).
4. Missing or unverified physical-area/energy inputs (unverified GSD strictly yields insufficient_data).
5. Negative energy values rejection.
6. Negative, zero, and non-finite emission factor rejection.
7. Invalid input types rejection (boolean, strings, non-numeric).
8. Reproducibility, deterministic calculations, and output schema completeness.
9. Existing energy-model compatibility (direct integration with EnergyCalculationResult).
10. Unconfigured default baseline behavior (no silent assumptions).
11. Authoritative CEA benchmark scenario configuration and provenance.
12. End-to-end real image pipeline evaluation on 768.0_1.0.png.
13. Validated candidate GSD scenario end-to-end simulation.
"""

import math
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.energy.area import PhysicalAreaCalculator
from ml.energy.assumptions import EnergyModelConfig
from ml.energy.energy_model import SolarEnergyModel, EnergyCalculationResult
from ml.environmental.assumptions import (
    EnvironmentalAssumptions,
    STANDARD_CEA_GRID_EMISSION_FACTOR,
    CEA_THERMAL_GRID_EMISSION_FACTOR,
    MIN_PLAUSIBLE_EMISSION_FACTOR,
    MAX_PLAUSIBLE_EMISSION_FACTOR,
)
from ml.environmental.emission_factors import (
    GridEmissionConfig,
    EmissionFactorProvenance,
)
from ml.environmental.validation import EnvironmentalValidator
from ml.environmental.co2_model import (
    CO2AvoidanceModel,
    CO2CalculationResult,
)
from ml.environmental.report import (
    run_environmental_analysis_pipeline,
    build_and_save_phase7_reports,
)


class TestEnvironmentalAnalysis(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name) / "reports"

    def tearDown(self):
        self.temp_dir.cleanup()

    # 1. Correct calculation with valid energy generation
    def test_valid_co2_calculation(self):
        # 100,000 kWh/year * 0.716 kg CO2/kWh = 71,600 kg CO2/year = 71.6 metric tonnes/year
        cfg = GridEmissionConfig.create_cea_benchmark_scenario(emission_factor=0.716)
        res = CO2AvoidanceModel.estimate_avoided_emissions(
            energy_input=100000.0,
            config=cfg,
        )
        self.assertEqual(res.co2_status, "estimated")
        self.assertAlmostEqual(res.annual_energy_kwh_year, 100000.0)
        self.assertAlmostEqual(res.emission_factor_kg_kwh, 0.716)
        self.assertAlmostEqual(res.co2_avoided_kg_year, 71600.0, places=2)
        self.assertAlmostEqual(res.co2_avoided_tonnes_year, 71.6, places=4)
        self.assertEqual(len(res.unavailability_reasons), 0)

    # 2. Conversion from kilograms to metric tonnes
    def test_kg_to_metric_tonnes_conversion(self):
        # 12345.67 kWh * 0.80 kg/kWh = 9876.54 kg = 9.8765 tonnes
        res = CO2AvoidanceModel.estimate_avoided_emissions(
            energy_input=12345.67,
            emission_factor_kg_kwh=0.80,
        )
        self.assertEqual(res.co2_status, "estimated")
        expected_kg = round(12345.67 * 0.80, 2)
        expected_tonnes = round(expected_kg / 1000.0, 4)
        self.assertAlmostEqual(res.co2_avoided_kg_year, expected_kg)
        self.assertAlmostEqual(res.co2_avoided_tonnes_year, expected_tonnes)
        self.assertAlmostEqual(res.co2_avoided_kg_year, res.co2_avoided_tonnes_year * 1000.0, delta=0.1)

    # 3. Missing energy generation
    def test_missing_energy_generation(self):
        cfg = GridEmissionConfig.create_cea_benchmark_scenario()
        res = CO2AvoidanceModel.estimate_avoided_emissions(
            energy_input=None,
            config=cfg,
        )
        self.assertEqual(res.co2_status, "insufficient_data")
        self.assertIsNone(res.co2_avoided_kg_year)
        self.assertIsNone(res.co2_avoided_tonnes_year)
        self.assertTrue(any("missing" in r.lower() for r in res.unavailability_reasons))

    # 4. Missing or unverified physical-area/energy inputs
    def test_unverified_gsd_energy_result_integration(self):
        # Simulate an EnergyCalculationResult where area is unverified
        area_res = PhysicalAreaCalculator.calculate(
            solar_pixels=68451, total_pixels=409600, gsd_meters=None, is_gsd_validated=False
        )
        energy_res = SolarEnergyModel.estimate_energy(
            area_result=area_res,
            annual_ghi_kwh_m2_year=1973.85,
            config=EnergyModelConfig.create_default(),
        )
        self.assertEqual(energy_res.energy_status, "insufficient_data")
        self.assertIsNone(energy_res.estimated_energy_kwh_year)

        # Feed the unverified energy result to CO2 model
        cfg = GridEmissionConfig.create_cea_benchmark_scenario()
        co2_res = CO2AvoidanceModel.estimate_avoided_emissions(
            energy_input=energy_res,
            config=cfg,
        )
        self.assertEqual(co2_res.co2_status, "insufficient_data")
        self.assertIsNone(co2_res.co2_avoided_kg_year)
        self.assertIsNone(co2_res.co2_avoided_tonnes_year)
        self.assertTrue(any("unavailable" in r.lower() for r in co2_res.unavailability_reasons))

    # 5. Negative energy values rejection
    def test_negative_energy_rejection(self):
        ok, err = EnvironmentalValidator.validate_energy_generation(-500.0)
        self.assertFalse(ok)
        self.assertIn("cannot be negative", err)

        cfg = GridEmissionConfig.create_cea_benchmark_scenario()
        with self.assertRaises(ValueError):
            CO2AvoidanceModel.estimate_avoided_emissions(
                energy_input=-1000.0,
                config=cfg,
            )

    # 6. Negative, zero, or non-finite emission factors rejection
    def test_invalid_emission_factors_rejection(self):
        # Zero
        ok_zero, err_zero = EnvironmentalValidator.validate_emission_factor(0.0)
        self.assertFalse(ok_zero)
        with self.assertRaises(ValueError):
            CO2AvoidanceModel.estimate_avoided_emissions(energy_input=1000.0, emission_factor_kg_kwh=0.0)

        # Negative
        ok_neg, err_neg = EnvironmentalValidator.validate_emission_factor(-0.5)
        self.assertFalse(ok_neg)
        with self.assertRaises(ValueError):
            CO2AvoidanceModel.estimate_avoided_emissions(energy_input=1000.0, emission_factor_kg_kwh=-0.5)

        # NaN
        ok_nan, err_nan = EnvironmentalValidator.validate_emission_factor(float("nan"))
        self.assertFalse(ok_nan)
        with self.assertRaises(ValueError):
            CO2AvoidanceModel.estimate_avoided_emissions(energy_input=1000.0, emission_factor_kg_kwh=float("nan"))

        # Inf
        ok_inf, err_inf = EnvironmentalValidator.validate_emission_factor(float("inf"))
        self.assertFalse(ok_inf)
        with self.assertRaises(ValueError):
            CO2AvoidanceModel.estimate_avoided_emissions(energy_input=1000.0, emission_factor_kg_kwh=float("inf"))

        # Out of physical bounds (e.g. 5.0 kg CO2/kWh)
        ok_high, err_high = EnvironmentalValidator.validate_emission_factor(5.0)
        self.assertFalse(ok_high)
        with self.assertRaises(ValueError):
            CO2AvoidanceModel.estimate_avoided_emissions(energy_input=1000.0, emission_factor_kg_kwh=5.0)

    # 7. Invalid input types
    def test_invalid_input_types_rejection(self):
        # Booleans (which are instances of int in Python)
        ok_b_eng, _ = EnvironmentalValidator.validate_energy_generation(True)
        self.assertFalse(ok_b_eng)
        ok_b_ef, _ = EnvironmentalValidator.validate_emission_factor(False)
        self.assertFalse(ok_b_ef)

        # Strings
        ok_s_eng, _ = EnvironmentalValidator.validate_energy_generation("5000")
        self.assertFalse(ok_s_eng)
        ok_s_ef, _ = EnvironmentalValidator.validate_emission_factor("0.716")
        self.assertFalse(ok_s_ef)

        with self.assertRaises(ValueError):
            CO2AvoidanceModel.estimate_avoided_emissions(energy_input="invalid_string")

    # 8. Reproducibility and output schema
    def test_reproducibility_and_schema(self):
        cfg = GridEmissionConfig.create_cea_benchmark_scenario()
        res = CO2AvoidanceModel.estimate_avoided_emissions(
            energy_input=50000.0,
            config=cfg,
        )
        d = res.to_dict()
        required_keys = [
            "co2_label",
            "co2_status",
            "annual_energy_kwh_year",
            "emission_factor_kg_kwh",
            "co2_avoided_kg_year",
            "co2_avoided_tonnes_year",
            "formula",
            "units",
            "unavailability_reasons",
            "energy_details",
            "emission_factor_details",
            "assumptions_and_limitations",
        ]
        for key in required_keys:
            self.assertIn(key, d)

        # Determinism check
        res2 = CO2AvoidanceModel.estimate_avoided_emissions(energy_input=50000.0, config=cfg)
        self.assertEqual(res.co2_avoided_kg_year, res2.co2_avoided_kg_year)
        self.assertEqual(res.co2_avoided_tonnes_year, res2.co2_avoided_tonnes_year)

    # 9. Existing energy-model compatibility
    def test_energy_model_compatibility_valid(self):
        # Create a valid calculated EnergyCalculationResult
        area_res = PhysicalAreaCalculator.calculate(
            solar_pixels=1000, total_pixels=409600, gsd_meters=0.5, is_gsd_validated=True
        )
        energy_cfg = EnergyModelConfig(module_efficiency=0.20, performance_ratio=0.75)
        energy_res = SolarEnergyModel.estimate_energy(
            area_result=area_res,
            annual_ghi_kwh_m2_year=2000.0,
            config=energy_cfg,
        )
        self.assertEqual(energy_res.energy_status, "estimated")
        self.assertAlmostEqual(energy_res.estimated_energy_kwh_year, 75000.0)

        # Compute avoided emissions directly from the energy result
        co2_res = CO2AvoidanceModel.estimate_avoided_emissions(
            energy_input=energy_res,
            config=GridEmissionConfig.create_cea_benchmark_scenario(0.716),
        )
        self.assertEqual(co2_res.co2_status, "estimated")
        self.assertAlmostEqual(co2_res.annual_energy_kwh_year, 75000.0)
        # 75,000 * 0.716 = 53,700 kg = 53.7 metric tonnes
        self.assertAlmostEqual(co2_res.co2_avoided_kg_year, 53700.0)
        self.assertAlmostEqual(co2_res.co2_avoided_tonnes_year, 53.7)

    # 10. Unconfigured default baseline behavior (no silent assumptions)
    def test_unconfigured_default_baseline(self):
        # When no emission factor is configured, model strictly refuses to fabricate
        co2_res = CO2AvoidanceModel.estimate_avoided_emissions(
            energy_input=50000.0,
            config=GridEmissionConfig.create_default(),
        )
        self.assertEqual(co2_res.co2_status, "insufficient_data")
        self.assertIsNone(co2_res.co2_avoided_kg_year)
        self.assertIsNone(co2_res.co2_avoided_tonnes_year)
        self.assertTrue(any("unconfigured" in r.lower() for r in co2_res.unavailability_reasons))

    # 11. Authoritative CEA benchmark scenario configuration and provenance
    def test_cea_provenance_structure(self):
        cfg = GridEmissionConfig.create_cea_benchmark_scenario()
        prov = cfg.provenance
        self.assertIsNotNone(prov)
        self.assertIn("Central Electricity Authority", prov.source_agency)
        self.assertIn("CO2 Baseline Database", prov.database_product)
        self.assertEqual(prov.factor_value, STANDARD_CEA_GRID_EMISSION_FACTOR)
        self.assertEqual(prov.factor_unit, "kg CO2/kWh")
        self.assertTrue(prov.is_official_reference)

    # 12. End-to-end real image pipeline evaluation on 768.0_1.0.png
    def test_real_image_pipeline_insufficient_data(self):
        report = build_and_save_phase7_reports(
            target_image="768.0_1.0.png",
            output_dir=self.output_dir,
        )
        self.assertEqual(report["phase"], "PHASE 7 — CO2 & ENVIRONMENTAL ANALYSIS")
        prim = report["primary_analysis"]
        self.assertEqual(prim["image_name"], "768.0_1.0.png")
        self.assertEqual(prim["spatial_resolution"]["gsd_status"], "UNAVAILABLE")
        self.assertEqual(prim["spatial_resolution"]["physical_area_status"], "insufficient_data")
        self.assertIsNone(prim["spatial_resolution"]["solar_area_m2"])
        self.assertEqual(prim["energy_model"]["energy_status"], "insufficient_data")
        self.assertIsNone(prim["energy_model"]["estimated_energy_kwh_year"])
        self.assertEqual(prim["environmental_model"]["co2_status"], "insufficient_data")
        self.assertIsNone(prim["environmental_model"]["co2_avoided_kg_year"])
        self.assertIsNone(prim["environmental_model"]["co2_avoided_tonnes_year"])
        self.assertEqual(report["environmental_readiness"], "BLOCKED")
        self.assertFalse(report["scientific_integrity_guarantees"]["co2_avoided_fabricated"])

    # 13. Validated candidate GSD scenario end-to-end simulation
    def test_validated_candidate_gsd_pipeline(self):
        # If imagery has an authenticated GSD (e.g. 0.30 m/px) and benchmark scenario parameters
        report = build_and_save_phase7_reports(
            target_image="768.0_1.0.png",
            module_efficiency=0.20,
            performance_ratio=0.75,
            candidate_gsd=0.30,
            is_candidate_gsd_validated=True,
            use_cea_benchmark=True,
            output_dir=self.output_dir,
        )
        prim = report["primary_analysis"]
        self.assertEqual(prim["spatial_resolution"]["gsd_status"], "VALIDATED")
        self.assertEqual(prim["spatial_resolution"]["physical_area_status"], "calculated")
        self.assertIsNotNone(prim["spatial_resolution"]["solar_area_m2"])
        self.assertEqual(prim["energy_model"]["energy_status"], "estimated")
        self.assertIsNotNone(prim["energy_model"]["estimated_energy_kwh_year"])
        self.assertEqual(prim["environmental_model"]["co2_status"], "estimated")
        self.assertIsNotNone(prim["environmental_model"]["co2_avoided_kg_year"])
        self.assertIsNotNone(prim["environmental_model"]["co2_avoided_tonnes_year"])
        self.assertEqual(report["environmental_readiness"], "READY")


if __name__ == "__main__":
    unittest.main()
