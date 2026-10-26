"""
SolarMap-India — Production Serving & Deployment Integration Tests.

Validates:
1. Production frontend static asset serving and index.html delivery.
2. Content negotiation at GET / (HTML for web browsers vs. JSON for API clients).
3. Explicit GET /api service metadata endpoint.
4. Static file delivery for /assets and root assets (favicon.svg, sample_aerial.png).
5. SPA client-side route fallback for non-API routes.
6. Guardrails ensuring unknown API routes (/api/*, /predict/*) return 404 JSON, never HTML.
7. Model artifact verification script behavior (both valid and missing cases).
8. Graceful operation when MongoDB is unconfigured or offline.
"""

from pathlib import Path
import subprocess
import sys
import unittest

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import FRONTEND_DIST_DIR
from backend.main import app
from backend.model_manager import model_manager
from scripts.verify_model_artifact import verify_checkpoint


class TestProductionServingAndDeployment(unittest.TestCase):
    """Integration test suite for production serving and container readiness."""

    @classmethod
    def setUpClass(cls):
        model_manager.load_model()
        cls.client = TestClient(app)

    def test_01_root_returns_html_when_browser_accepts_html(self):
        """Test GET / with Accept: text/html serves index.html."""
        if not (FRONTEND_DIST_DIR / "index.html").is_file():
            self.skipTest("frontend/dist/index.html not built; skipping HTML delivery test.")

        response = self.client.get("/", headers={"Accept": "text/html,application/xhtml+xml"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))
        self.assertIn('<div id="root">', response.text)
        self.assertIn("<title>SolarMap-India", response.text)

    def test_02_root_returns_json_when_client_requests_json(self):
        """Test GET / with Accept: application/json returns service metadata."""
        response = self.client.get("/", headers={"Accept": "application/json"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("application/json", response.headers.get("content-type", ""))
        data = response.json()
        self.assertIn("docs", data)
        self.assertIn("health", data)
        self.assertIn("predict", data)
        self.assertIn("history", data)

    def test_03_api_root_endpoint_returns_json(self):
        """Test GET /api explicitly returns JSON service metadata regardless of Accept header."""
        response = self.client.get("/api", headers={"Accept": "text/html,application/json"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("application/json", response.headers.get("content-type", ""))
        data = response.json()
        self.assertEqual(data["docs"], "/docs")
        self.assertEqual(data["health"], "/health")

    def test_04_static_assets_mounted_and_served(self):
        """Test static assets in /assets/ are served with correct content types."""
        assets_dir = FRONTEND_DIST_DIR / "assets"
        if not assets_dir.is_dir():
            self.skipTest("frontend/dist/assets not found; skipping asset mount test.")

        # Find any .js or .css file in assets
        js_files = list(assets_dir.glob("*.js"))
        if js_files:
            rel_name = js_files[0].name
            response = self.client.get(f"/assets/{rel_name}")
            self.assertEqual(response.status_code, 200)
            self.assertTrue(
                "javascript" in response.headers.get("content-type", "")
                or "text/" in response.headers.get("content-type", "")
            )

    def test_05_root_static_files_served(self):
        """Test root static files such as favicon.svg or sample_aerial.png."""
        if (FRONTEND_DIST_DIR / "favicon.svg").is_file():
            response = self.client.get("/favicon.svg")
            self.assertEqual(response.status_code, 200)
            self.assertIn("image/svg+xml", response.headers.get("content-type", ""))

        if (FRONTEND_DIST_DIR / "sample_aerial.png").is_file():
            response = self.client.get("/sample_aerial.png")
            self.assertEqual(response.status_code, 200)
            self.assertIn("image/png", response.headers.get("content-type", ""))

    def test_06_unknown_api_routes_return_404_json_not_html(self):
        """Guardrail: Unknown API routes must NEVER return frontend HTML."""
        # 1. Unknown path under /api/
        response = self.client.get("/api/unknown_nonexistent_route")
        self.assertEqual(response.status_code, 404)
        self.assertNotIn("text/html", response.headers.get("content-type", ""))
        data = response.json()
        self.assertIn("error", data)

        # 2. Unknown path under /predict/
        response = self.client.get("/predict/invalid_subroute")
        self.assertEqual(response.status_code, 404)
        self.assertNotIn("text/html", response.headers.get("content-type", ""))

        # 3. Unknown path under /outputs/
        response = self.client.get("/outputs/invalid_subroute")
        self.assertEqual(response.status_code, 404)
        self.assertNotIn("text/html", response.headers.get("content-type", ""))

    def test_07_spa_fallback_serves_html_for_client_routes(self):
        """Test that browser navigation to non-API SPA routes falls back to index.html."""
        if not (FRONTEND_DIST_DIR / "index.html").is_file():
            self.skipTest("frontend/dist/index.html not built; skipping SPA fallback test.")

        # Simulate browser requesting a client route like /dashboard or /map
        response = self.client.get("/geospatial_view", headers={"Accept": "text/html,application/xhtml+xml"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))
        self.assertIn('<div id="root">', response.text)

    def test_08_path_traversal_protection_on_static_serving(self):
        """Guardrail: Directory traversal attempts on static assets must be rejected."""
        response = self.client.get("/..%2F..%2Fbackend%2Fconfig.py")
        self.assertEqual(response.status_code, 404)

    def test_09_model_artifact_verification_helper(self):
        """Test verify_checkpoint logic for present and absent checkpoints."""
        # Valid baseline checkpoint
        baseline_ckpt = PROJECT_ROOT / "outputs" / "HalfUNet" / "models" / "halfunet_best.pth"
        if baseline_ckpt.is_file():
            self.assertTrue(verify_checkpoint(baseline_ckpt, verbose=False))

        # Nonexistent checkpoint
        fake_ckpt = PROJECT_ROOT / "outputs" / "HalfUNet" / "models" / "nonexistent_model.pth"
        self.assertFalse(verify_checkpoint(fake_ckpt, verbose=False))

    def test_10_health_endpoints_without_mongodb(self):
        """Test /health and /health/db behavior when database persistence is unconfigured."""
        # /health must remain 200 healthy when model is loaded
        health_resp = self.client.get("/health")
        self.assertEqual(health_resp.status_code, 200)
        self.assertEqual(health_resp.json()["status"], "healthy")

        # /health/db reports status without crashing
        db_resp = self.client.get("/health/db")
        self.assertEqual(db_resp.status_code, 200)
        db_data = db_resp.json()
        self.assertIn(db_data["status"], ["connected", "disconnected", "unconfigured"])
        self.assertIn("database", db_data)
        # Verify secrets are not exposed in health response
        self.assertNotIn("password", str(db_data).lower())


if __name__ == "__main__":
    unittest.main()
