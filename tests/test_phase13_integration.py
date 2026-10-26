"""
SolarMap-India — Phase 13: Full Integration & Testing Suite.

Automated verification covering:
1. End-to-end solar energy and CO2 environmental analytics integration in POST /predict.
2. Missing scientific input safeguards (null-value & insufficient_data preservation).
3. Image-level geospatial coordinate lookup (Level B vs UNREFERENCED, no fallback fabrication).
4. MongoDB history persistence with full provenance in details dictionary.
5. Database offline resilience (inference completes with saved_to_history=False).
6. Prediction history retrieval with pagination, limit/offset enforcement.
7. Prediction history single-record retrieval by UUID4 identifier with 400/404 handling.
8. Database health endpoint (/health/db) status reporting without credential leaks.
9. Scientific calculation module validation when valid physical inputs exist vs when absent.
10. Backward compatibility of all schemas (PredictResponse, PredictionHistoryItem).
"""

from datetime import datetime, timezone
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch
import uuid

from fastapi.testclient import TestClient
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.database import DatabaseManager, db_manager
from backend.main import app
from backend.model_manager import model_manager
from backend.schemas import PredictResponse, PredictionHistoryItem, PredictionHistoryListResponse
from ml.energy.area import PhysicalAreaCalculator
from ml.energy.assumptions import EnergyModelConfig
from ml.energy.energy_model import SolarEnergyModel
from ml.environmental.co2_model import CO2AvoidanceModel


class TestPhase13FullIntegration(unittest.TestCase):
    """Phase 13 Full Integration Test Suite."""

    @classmethod
    def setUpClass(cls):
        model_manager.load_model()
        cls.client = TestClient(app)

    def _create_synthetic_image_bytes(
        self, width: int = 128, height: int = 128, color: tuple = (120, 160, 200)
    ) -> bytes:
        img = Image.new("RGB", (width, height), color=color)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_01_predict_endpoint_energy_and_co2_integration(self):
        """Verify POST /predict returns energy and CO2 fields with insufficient_data defaults."""
        img_bytes = self._create_synthetic_image_bytes(width=150, height=150)
        res = self.client.post(
            "/predict",
            files={"file": ("test_aerial.png", img_bytes, "image/png")},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()

        # Segmentation pixel metrics
        self.assertTrue(data["success"])
        self.assertEqual(data["image_width"], 150)
        self.assertEqual(data["image_height"], 150)
        self.assertIn("solar_area_pixels", data)
        self.assertIn("solar_coverage_percent", data)
        self.assertIn("detected_region_count", data)

        # Scientific safeguards: Physical scale & area
        self.assertIsNone(data["physical_area_m2"])
        self.assertIsNone(data["physical_area_hectares"])
        self.assertEqual(data["physical_area_status"], "insufficient_data")

        # Phase 6 & 7 integration fields
        self.assertIn("energy_status", data)
        self.assertEqual(data["energy_status"], "insufficient_data")
        self.assertIsNone(data["estimated_energy_kwh_year"])

        self.assertIn("co2_status", data)
        self.assertEqual(data["co2_status"], "insufficient_data")
        self.assertIsNone(data["co2_avoided_kg_year"])
        self.assertIsNone(data["co2_avoided_tonnes_year"])

    def test_02_geospatial_metadata_handling_catalog_vs_unreferenced(self):
        """Verify geospatial lookup correctly distinguishes cataloged Level B from UNREFERENCED without fabrication."""
        img_bytes = self._create_synthetic_image_bytes(width=100, height=100)

        # 1. Cataloged image: 768.0_1.0.png
        res_cat = self.client.post(
            "/predict",
            files={"file": ("768.0_1.0.png", img_bytes, "image/png")},
        )
        self.assertEqual(res_cat.status_code, 200)
        data_cat = res_cat.json()
        self.assertEqual(data_cat["georeferencing_level"], "LEVEL_B")
        self.assertIsNotNone(data_cat["latitude"])
        self.assertIsNotNone(data_cat["longitude"])
        self.assertIn("Gujarat", data_cat["location_name"])
        self.assertAlmostEqual(data_cat["latitude"], 21.197148, places=4)
        self.assertAlmostEqual(data_cat["longitude"], 72.780643, places=4)

        # 2. Unknown unreferenced image: custom_upload.png
        res_unref = self.client.post(
            "/predict",
            files={"file": ("custom_unknown_aerial.png", img_bytes, "image/png")},
        )
        self.assertEqual(res_unref.status_code, 200)
        data_unref = res_unref.json()
        self.assertEqual(data_unref["georeferencing_level"], "UNREFERENCED")
        self.assertIsNone(data_unref["latitude"])
        self.assertIsNone(data_unref["longitude"])
        self.assertIsNone(data_unref["location_name"])

    def test_03_prediction_history_persistence_with_analytics_details(self):
        """Verify that when database is connected, history record contains analytics details and provenance."""
        img_bytes = self._create_synthetic_image_bytes(width=100, height=100)
        saved_record_holder = []

        async def mock_save(record):
            saved_record_holder.append(record)
            return True

        with patch.object(DatabaseManager, "is_connected", new_callable=PropertyMock, return_value=True):
            with patch.object(db_manager, "save_prediction", side_effect=mock_save):
                res = self.client.post(
                    "/predict",
                    files={"file": ("history_test.png", img_bytes, "image/png")},
                )
                self.assertEqual(res.status_code, 200)
                data = res.json()
                self.assertTrue(data["saved_to_history"])

        self.assertEqual(len(saved_record_holder), 1)
        record = saved_record_holder[0]
        self.assertEqual(record["energy_status"], "insufficient_data")
        self.assertIsNone(record["estimated_energy_kwh_year"])
        self.assertEqual(record["co2_status"], "insufficient_data")
        self.assertIsNone(record["co2_avoided_kg_year"])
        self.assertIn("details", record)
        self.assertIn("area", record["details"])
        self.assertIn("energy", record["details"])
        self.assertIn("co2", record["details"])

    def test_04_database_offline_resilience(self):
        """Verify prediction succeeds even when database is disconnected or raises write error."""
        img_bytes = self._create_synthetic_image_bytes(width=100, height=100)

        # Case A: DB is disconnected
        with patch.object(DatabaseManager, "is_connected", new_callable=PropertyMock, return_value=False):
            res = self.client.post(
                "/predict",
                files={"file": ("offline_db.png", img_bytes, "image/png")},
            )
            self.assertEqual(res.status_code, 200)
            self.assertFalse(res.json()["saved_to_history"])

        # Case B: DB connected but throws write exception
        with patch.object(DatabaseManager, "is_connected", new_callable=PropertyMock, return_value=True):
            with patch.object(db_manager, "save_prediction", side_effect=RuntimeError("DB write timeout")):
                res = self.client.post(
                    "/predict",
                    files={"file": ("timeout_db.png", img_bytes, "image/png")},
                )
                self.assertEqual(res.status_code, 200)
                self.assertFalse(res.json()["saved_to_history"])

    def test_05_history_api_pagination_and_analytics_schema(self):
        """Verify GET /api/predictions/history enforces pagination and serializes analytics fields."""
        fake_items = [
            {
                "prediction_id": str(uuid.uuid4()),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "filename": f"sample_{i}.png",
                "status": "success",
                "model_name": "HalfUNet",
                "image_width": 256,
                "image_height": 256,
                "total_image_pixels": 65536,
                "solar_area_pixels": 1000 * (i + 1),
                "solar_coverage_percent": 1.5 * (i + 1),
                "detected_region_count": i + 1,
                "physical_area_m2": None,
                "physical_area_hectares": None,
                "physical_area_status": "insufficient_data",
                "energy_status": "insufficient_data",
                "estimated_energy_kwh_year": None,
                "co2_status": "insufficient_data",
                "co2_avoided_kg_year": None,
                "co2_avoided_tonnes_year": None,
                "latitude": 21.197148 if i == 0 else None,
                "longitude": 72.780643 if i == 0 else None,
                "georeferencing_level": "LEVEL_B" if i == 0 else "UNREFERENCED",
                "location_name": "Surat, Gujarat, India" if i == 0 else None,
                "mask_url": f"/outputs/{i}/mask",
                "overlay_url": f"/outputs/{i}/overlay",
            }
            for i in range(3)
        ]

        async def mock_list(skip=0, limit=20):
            return fake_items[skip : skip + limit], len(fake_items)

        with patch.object(DatabaseManager, "is_connected", new_callable=PropertyMock, return_value=True):
            with patch.object(db_manager, "list_predictions", side_effect=mock_list):
                res = self.client.get("/api/predictions/history?skip=0&limit=2")
                self.assertEqual(res.status_code, 200)
                data = res.json()
                self.assertTrue(data["success"])
                self.assertEqual(data["total"], 3)
                self.assertEqual(data["skip"], 0)
                self.assertEqual(data["limit"], 2)
                self.assertEqual(len(data["items"]), 2)

                # Check first item analytics
                first = data["items"][0]
                self.assertEqual(first["energy_status"], "insufficient_data")
                self.assertEqual(first["co2_status"], "insufficient_data")
                self.assertEqual(first["georeferencing_level"], "LEVEL_B")

    def test_06_history_api_offline_returns_503(self):
        """Verify GET /api/predictions/history returns 503 when MongoDB is offline."""
        with patch.object(DatabaseManager, "is_connected", new_callable=PropertyMock, return_value=False):
            res = self.client.get("/api/predictions/history")
            self.assertEqual(res.status_code, 503)
            self.assertIn("unavailable", res.json()["error"].lower())

    def test_07_history_single_item_lookup_and_error_handling(self):
        """Verify GET /api/predictions/history/{id} handles valid ID, invalid ID (400), and not found (404)."""
        valid_uuid = str(uuid.uuid4())
        fake_record = {
            "prediction_id": valid_uuid,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "filename": "single_lookup.png",
            "status": "success",
            "model_name": "HalfUNet",
            "image_width": 256,
            "image_height": 256,
            "total_image_pixels": 65536,
            "solar_area_pixels": 500,
            "solar_coverage_percent": 0.76,
            "detected_region_count": 2,
            "physical_area_m2": None,
            "physical_area_status": "insufficient_data",
            "energy_status": "insufficient_data",
            "co2_status": "insufficient_data",
            "mask_url": f"/outputs/{valid_uuid}/mask",
            "overlay_url": f"/outputs/{valid_uuid}/overlay",
        }

        # 1. Invalid UUID format -> 400
        res_bad = self.client.get("/api/predictions/history/not-a-valid-uuid")
        self.assertEqual(res_bad.status_code, 400)

        # 2. Database offline -> 503
        with patch.object(DatabaseManager, "is_connected", new_callable=PropertyMock, return_value=False):
            res_off = self.client.get(f"/api/predictions/history/{valid_uuid}")
            self.assertEqual(res_off.status_code, 503)

        # 3. Not found -> 404
        with patch.object(DatabaseManager, "is_connected", new_callable=PropertyMock, return_value=True):
            with patch.object(db_manager, "get_prediction", return_value=None):
                res_404 = self.client.get(f"/api/predictions/history/{valid_uuid}")
                self.assertEqual(res_404.status_code, 404)

        # 4. Successful retrieval -> 200
        with patch.object(DatabaseManager, "is_connected", new_callable=PropertyMock, return_value=True):
            with patch.object(db_manager, "get_prediction", return_value=fake_record):
                res_ok = self.client.get(f"/api/predictions/history/{valid_uuid}")
                self.assertEqual(res_ok.status_code, 200)
                self.assertEqual(res_ok.json()["prediction_id"], valid_uuid)

    def test_08_database_health_endpoint_no_secrets(self):
        """Verify GET /health/db reports connection status without leaking credentials."""
        res = self.client.get("/health/db")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("status", data)
        self.assertIn("configured", data)
        self.assertIn("connected", data)
        # Verify no URI or password appears in keys
        for key in data.keys():
            self.assertNotIn("password", key.lower())
            self.assertNotIn("uri", key.lower())

    def test_09_scientific_calculation_pipeline_when_valid_vs_missing(self):
        """Verify calculation logic of PhysicalAreaCalculator, SolarEnergyModel, and CO2AvoidanceModel."""
        # Case A: Uncalibrated inputs (Phase 13 standard behavior)
        area_calc = PhysicalAreaCalculator.calculate(
            solar_pixels=5000,
            total_pixels=100000,
            is_gsd_validated=False,
        )
        self.assertEqual(area_calc.physical_area_status, "insufficient_data")
        self.assertIsNone(area_calc.solar_area_m2)

        energy_calc = SolarEnergyModel.estimate_energy(area_result=area_calc)
        self.assertEqual(energy_calc.energy_status, "insufficient_data")
        self.assertIsNone(energy_calc.estimated_energy_kwh_year)

        co2_calc = CO2AvoidanceModel.estimate_avoided_emissions(energy_input=energy_calc)
        self.assertEqual(co2_calc.co2_status, "insufficient_data")
        self.assertIsNone(co2_calc.co2_avoided_kg_year)
        self.assertIsNone(co2_calc.co2_avoided_tonnes_year)

        # Case B: Valid calibrated inputs with explicit benchmark scenario
        calibrated_area = PhysicalAreaCalculator.calculate(
            solar_pixels=5000,
            total_pixels=100000,
            gsd_meters=0.1,  # 0.1 m/pixel -> 0.01 m2/pixel -> 50 m2
            is_gsd_validated=True,
        )
        self.assertEqual(calibrated_area.physical_area_status, "calculated")
        self.assertAlmostEqual(calibrated_area.solar_area_m2, 50.0, places=2)

        # E = A * GHI * eta * PR = 50 * 1800 * 0.20 * 0.75 = 13,500 kWh/yr
        benchmark_cfg = EnergyModelConfig.create_benchmark_scenario(efficiency=0.20, performance_ratio=0.75)
        calibrated_energy = SolarEnergyModel.estimate_energy(
            area_result=calibrated_area,
            annual_ghi_kwh_m2_year=1800.0,
            config=benchmark_cfg,
        )
        self.assertEqual(calibrated_energy.energy_status, "estimated")
        self.assertAlmostEqual(calibrated_energy.estimated_energy_kwh_year, 13500.0, places=1)

        # CO2 = E * EF = 13500 * 0.716 = 9666 kg = 9.666 tonnes
        calibrated_co2 = CO2AvoidanceModel.estimate_avoided_emissions(
            energy_input=calibrated_energy,
            emission_factor_kg_kwh=0.716,
        )
        self.assertEqual(calibrated_co2.co2_status, "estimated")
        self.assertAlmostEqual(calibrated_co2.co2_avoided_kg_year, 9666.0, places=1)
        self.assertAlmostEqual(calibrated_co2.co2_avoided_tonnes_year, 9.666, places=3)

    def test_10_backward_compatibility_schemas(self):
        """Verify PredictResponse and PredictionHistoryItem remain fully backward-compatible."""
        resp = PredictResponse(
            success=True,
            model="HalfUNet",
            prediction_id=str(uuid.uuid4()),
            image_width=256,
            image_height=256,
            total_image_pixels=65536,
            solar_area_pixels=1200,
            solar_coverage_percent=1.83,
            detected_region_count=2,
            mask_url="/outputs/xyz/mask",
            overlay_url="/outputs/xyz/overlay",
        )
        self.assertEqual(resp.physical_area_status, "insufficient_data")
        self.assertEqual(resp.energy_status, "insufficient_data")
        self.assertEqual(resp.co2_status, "insufficient_data")
        self.assertIsNone(resp.estimated_energy_kwh_year)
        self.assertIsNone(resp.co2_avoided_kg_year)


if __name__ == "__main__":
    unittest.main()
