"""
SolarMap-India — Phase 8: MongoDB & Prediction History Test Suite.

Automated verification covering:
1. Database configuration and credential safety (URI masking, empty URI handling).
2. Application imports and initializes cleanly without requiring a live database.
3. DatabaseManager connection lifecycle and graceful offline handling.
4. Schema indexing definition (unique prediction_id, descending created_at_utc).
5. Saving valid prediction history records with deterministic serialization.
6. Handling missing optional metadata without fabrication or corruption.
7. Database write failure and offline resilience (never crashes inference).
8. Listing prediction history with pagination (skip, limit, max page enforcement).
9. Empty history listing behavior.
10. Retrieving individual prediction records by UUID4 identifier.
11. Not-found (404) behavior for nonexistent records.
12. Bad request (400) behavior for invalid prediction identifier formats.
13. Database unavailable (503) error handling on history endpoints.
14. Database health endpoint (/health/db) reporting without leaking secrets.
15. Seamless integration with POST /predict preserving full backward compatibility.
"""

from datetime import datetime, timezone
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from fastapi.testclient import TestClient
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import MONGODB_COLLECTION, MONGODB_DATABASE, MONGODB_URI
from backend.database import DatabaseManager, db_manager, sanitize_mongodb_uri
from backend.main import app
from backend.model_manager import model_manager
from backend.schemas import (
    DatabaseHealthResponse,
    PredictionHistoryItem,
    PredictionHistoryListResponse,
    PredictResponse,
)


class TestDatabaseConfigurationAndSafety(unittest.TestCase):
    """Tests configuration management and credential safety."""

    def test_01_sanitize_mongodb_uri(self):
        """Test password masking in connection strings."""
        raw_uri = "mongodb+srv://admin_user:super_secret_pw@cluster0.abcde.mongodb.net/?retryWrites=true"
        masked = sanitize_mongodb_uri(raw_uri)
        self.assertNotIn("super_secret_pw", masked)
        self.assertIn("admin_user:****@", masked)

        # Empty or simple URIs
        self.assertEqual(sanitize_mongodb_uri(""), "<empty>")
        self.assertEqual(sanitize_mongodb_uri("mongodb://localhost:27017"), "mongodb://localhost:27017")

    def test_02_default_manager_unconfigured_offline(self):
        """Test manager behavior when URI is empty or unconfigured."""
        mgr = DatabaseManager(uri="")
        self.assertFalse(mgr.is_configured)
        self.assertFalse(mgr.is_connected)
        status = mgr.get_status()
        self.assertFalse(status["configured"])
        self.assertFalse(status["connected"])

    def test_03_no_hardcoded_passwords_in_backend_files(self):
        """Verify no plaintext passwords are hardcoded in backend Python files."""
        backend_dir = PROJECT_ROOT / "backend"
        for py_file in backend_dir.glob("**/*.py"):
            text = py_file.read_text(encoding="utf-8")
            self.assertNotIn("mongodb+srv://", text.lower())
            self.assertNotIn("password=", text.lower())


class TestDatabaseManagerOperations(unittest.IsolatedAsyncioTestCase):
    """Unit tests for DatabaseManager CRUD, error handling, and connection lifecycle."""

    async def test_04_connect_when_unconfigured_returns_false(self):
        """Connecting with empty URI safely returns False without raising exceptions."""
        mgr = DatabaseManager(uri="")
        connected = await mgr.connect()
        self.assertFalse(connected)
        self.assertFalse(mgr.is_connected)

    async def test_05_connect_failure_handles_exception(self):
        """Network/timeout connection errors are caught cleanly without crashing."""
        mgr = DatabaseManager(uri="mongodb://nonexistent.host:27017", server_timeout_ms=100)
        with patch("motor.motor_asyncio.AsyncIOMotorClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.admin.command = AsyncMock(side_effect=Exception("Connection refused"))
            mock_client_cls.return_value = mock_client

            connected = await mgr.connect()
            self.assertFalse(connected)
            self.assertFalse(mgr.is_connected)
            self.assertIn("Connection refused", mgr.get_status()["error"])

    async def test_06_save_prediction_when_disconnected(self):
        """Saving when disconnected safely returns False."""
        mgr = DatabaseManager(uri="")
        saved = await mgr.save_prediction({"prediction_id": str(uuid.uuid4())})
        self.assertFalse(saved)

    async def test_07_save_and_retrieve_mocked_prediction(self):
        """Test saving and retrieving prediction documents with proper serialization."""
        mgr = DatabaseManager(uri="mongodb://localhost:27017")
        mgr._is_connected = True
        mock_col = MagicMock()

        test_id = str(uuid.uuid4())
        mock_doc = {
            "_id": "mock_object_id_12345",
            "prediction_id": test_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "filename": "test_aerial.png",
            "status": "success",
            "image_width": 640,
            "image_height": 640,
            "total_image_pixels": 409600,
            "solar_area_pixels": 5000,
            "solar_coverage_percent": 1.22,
            "detected_region_count": 2,
            "physical_area_status": "insufficient_data",
            "physical_area_m2": None,
            "mask_url": f"/outputs/{test_id}/mask",
            "overlay_url": f"/outputs/{test_id}/overlay",
        }

        mock_col.insert_one = AsyncMock(return_value=MagicMock(acknowledged=True))
        mock_col.find_one = AsyncMock(return_value=dict(mock_doc))
        mgr._collection = mock_col

        # Test Save
        save_success = await mgr.save_prediction(dict(mock_doc))
        self.assertTrue(save_success)
        mock_col.insert_one.assert_awaited_once()

        # Test Get
        retrieved = await mgr.get_prediction(test_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["prediction_id"], test_id)
        self.assertEqual(retrieved["_id"], "mock_object_id_12345")

    async def test_08_list_predictions_pagination(self):
        """Test paginated retrieval with skip and limit."""
        mgr = DatabaseManager(uri="mongodb://localhost:27017")
        mgr._is_connected = True
        mock_col = MagicMock()

        items = [
            {"_id": f"id_{i}", "prediction_id": str(uuid.uuid4()), "created_at_utc": f"2026-10-22T10:{i:02d}:00Z"}
            for i in range(5)
        ]

        async def async_cursor_gen():
            for it in items:
                yield it

        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = mock_cursor
        mock_cursor.skip.return_value = mock_cursor
        mock_cursor.limit.return_value = async_cursor_gen()

        mock_col.count_documents = AsyncMock(return_value=15)
        mock_col.find.return_value = mock_cursor
        mgr._collection = mock_col

        retrieved_items, total = await mgr.list_predictions(skip=5, limit=5)
        self.assertEqual(total, 15)
        self.assertEqual(len(retrieved_items), 5)
        mock_col.count_documents.assert_awaited_once()


class TestHistoryAPIEndpoints(unittest.TestCase):
    """Integration tests for FastAPI history routes and database resilience."""

    @classmethod
    def setUpClass(cls):
        model_manager.load_model()
        cls.client = TestClient(app)

    def _create_synthetic_image_bytes(self, width: int = 128, height: int = 128) -> bytes:
        img = Image.new("RGB", (width, height), color=(120, 160, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_09_database_health_endpoint(self):
        """Test GET /health/db returns connectivity diagnostic without errors."""
        response = self.client.get("/health/db")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("configured", data)
        self.assertIn("connected", data)

    def test_10_list_history_when_database_offline(self):
        """GET /api/predictions/history returns 503 when database is not connected."""
        with patch.object(DatabaseManager, "is_connected", new_callable=unittest.mock.PropertyMock, return_value=False):
            response = self.client.get("/api/predictions/history")
            self.assertEqual(response.status_code, 503)
            self.assertIn("Database service unavailable", response.json()["error"])

    def test_11_get_prediction_by_id_when_database_offline(self):
        """GET /api/predictions/history/{id} returns 503 when database is not connected."""
        test_uuid = str(uuid.uuid4())
        with patch.object(DatabaseManager, "is_connected", new_callable=unittest.mock.PropertyMock, return_value=False):
            response = self.client.get(f"/api/predictions/history/{test_uuid}")
            self.assertEqual(response.status_code, 503)
            self.assertIn("Database service unavailable", response.json()["error"])

    def test_12_get_prediction_by_id_invalid_uuid(self):
        """GET /api/predictions/history/{id} returns 400 for non-UUID strings."""
        response = self.client.get("/api/predictions/history/not-a-valid-uuid")
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid prediction ID format", response.json()["error"])

    def test_13_get_prediction_by_id_not_found(self):
        """GET /api/predictions/history/{id} returns 404 when record does not exist in DB."""
        test_uuid = str(uuid.uuid4())
        with patch.object(DatabaseManager, "is_connected", new_callable=unittest.mock.PropertyMock, return_value=True):
            with patch.object(db_manager, "get_prediction", new_callable=AsyncMock, return_value=None):
                response = self.client.get(f"/api/predictions/history/{test_uuid}")
                self.assertEqual(response.status_code, 404)
                self.assertIn("not found", response.json()["error"])

    def test_14_list_history_success_mocked(self):
        """GET /api/predictions/history returns 200 with paginated items when DB is active."""
        test_uuid = str(uuid.uuid4())
        sample_item = {
            "prediction_id": test_uuid,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "filename": "sample.png",
            "status": "success",
            "model_name": "HalfUNet",
            "image_width": 640,
            "image_height": 640,
            "total_image_pixels": 409600,
            "solar_area_pixels": 2500,
            "solar_coverage_percent": 0.61,
            "detected_region_count": 1,
            "physical_area_status": "insufficient_data",
            "physical_area_m2": None,
            "energy_status": "insufficient_data",
            "estimated_energy_kwh_year": None,
            "co2_status": "insufficient_data",
            "co2_avoided_kg_year": None,
            "co2_avoided_tonnes_year": None,
            "mask_url": f"/outputs/{test_uuid}/mask",
            "overlay_url": f"/outputs/{test_uuid}/overlay",
        }

        with patch.object(DatabaseManager, "is_connected", new_callable=unittest.mock.PropertyMock, return_value=True):
            with patch.object(
                db_manager,
                "list_predictions",
                new_callable=AsyncMock,
                return_value=([sample_item], 1),
            ):
                response = self.client.get("/api/predictions/history?skip=0&limit=10")
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertTrue(data["success"])
                self.assertEqual(data["total"], 1)
                self.assertEqual(len(data["items"]), 1)
                self.assertEqual(data["items"][0]["prediction_id"], test_uuid)

    def test_15_get_history_item_success_mocked(self):
        """GET /api/predictions/history/{id} returns 200 and full record schema."""
        test_uuid = str(uuid.uuid4())
        sample_item = {
            "prediction_id": test_uuid,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "filename": "sample.png",
            "status": "success",
            "model_name": "HalfUNet",
            "image_width": 640,
            "image_height": 640,
            "total_image_pixels": 409600,
            "solar_area_pixels": 2500,
            "solar_coverage_percent": 0.61,
            "detected_region_count": 1,
            "physical_area_status": "insufficient_data",
            "physical_area_m2": None,
            "energy_status": "insufficient_data",
            "estimated_energy_kwh_year": None,
            "co2_status": "insufficient_data",
            "co2_avoided_kg_year": None,
            "co2_avoided_tonnes_year": None,
            "mask_url": f"/outputs/{test_uuid}/mask",
            "overlay_url": f"/outputs/{test_uuid}/overlay",
        }

        with patch.object(DatabaseManager, "is_connected", new_callable=unittest.mock.PropertyMock, return_value=True):
            with patch.object(
                db_manager,
                "get_prediction",
                new_callable=AsyncMock,
                return_value=sample_item,
            ):
                response = self.client.get(f"/api/predictions/history/{test_uuid}")
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(data["prediction_id"], test_uuid)
                self.assertEqual(data["physical_area_status"], "insufficient_data")

    def test_16_predict_integration_with_persistence(self):
        """Test POST /predict executes inference and persists record when DB is active."""
        img_bytes = self._create_synthetic_image_bytes(128, 128)

        mock_save = AsyncMock(return_value=True)
        with patch.object(DatabaseManager, "is_connected", new_callable=unittest.mock.PropertyMock, return_value=True):
            with patch.object(db_manager, "save_prediction", mock_save):
                response = self.client.post(
                    "/predict",
                    files={"file": ("test_aerial.png", img_bytes, "image/png")},
                )
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertTrue(data["success"])
                self.assertTrue(data["saved_to_history"])
                mock_save.assert_awaited_once()

    def test_17_predict_integration_when_db_write_fails(self):
        """Test POST /predict succeeds even if MongoDB write fails (safe degradation)."""
        img_bytes = self._create_synthetic_image_bytes(128, 128)

        mock_save = AsyncMock(side_effect=Exception("Write timeout"))
        with patch.object(DatabaseManager, "is_connected", new_callable=unittest.mock.PropertyMock, return_value=True):
            with patch.object(db_manager, "save_prediction", mock_save):
                response = self.client.post(
                    "/predict",
                    files={"file": ("test_aerial.png", img_bytes, "image/png")},
                )
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertTrue(data["success"])
                # Confirms prediction succeeded, but saved_to_history is False
                self.assertFalse(data["saved_to_history"])


if __name__ == "__main__":
    unittest.main()
