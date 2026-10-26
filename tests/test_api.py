"""
SolarMap-India — Backend FastAPI Integration & Unit Test Suite.

Tests:
1. Health endpoint (GET /health) when healthy and unhealthy
2. Root metadata endpoint (GET /)
3. Valid synthetic image upload (POST /predict)
4. Response structure & pixel coverage percentage calculation
5. Binary mask retrieval (GET /outputs/{id}/mask)
6. Visual overlay retrieval (GET /outputs/{id}/overlay)
7. Invalid upload validation (empty file, non-image, unsupported format)
8. Artifact security & path traversal protection
9. Real dataset image end-to-end inference (768.0_1.0.png)
10. Inference error handling (HTTP 500 & 503)
"""

import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import uuid

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app
from backend.model_manager import ModelManager, model_manager

REAL_SAMPLE_IMAGE = (
    PROJECT_ROOT
    / "dataset"
    / "Solar Images"
    / "Solar Images"
    / "images"
    / "default"
    / "768.0_1.0.png"
)


class TestSolarMapAPI(unittest.TestCase):
    """Automated integration and unit test suite for FastAPI backend endpoints."""

    @classmethod
    def setUpClass(cls):
        # Preload the model once for test session
        model_manager.load_model()
        cls.client = TestClient(app)

    def _create_synthetic_image_bytes(
        self, width: int = 128, height: int = 128, color: tuple = (100, 150, 200)
    ) -> bytes:
        """Helper to create valid in-memory RGB PNG bytes."""
        img = Image.new("RGB", (width, height), color=color)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_01_root_endpoint(self):
        """Test GET / returns service information and documentation links."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("docs", data)
        self.assertIn("health", data)
        self.assertIn("predict", data)

    def test_02_health_endpoint_healthy(self):
        """Test GET /health returns 200 and healthy status when model is loaded."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["model"], "HalfUNet")
        self.assertTrue(data["checkpoint_available"])
        self.assertIsNone(data["error"])
        self.assertIn("checkpoint_path", data)
        self.assertIn("device", data)

    def test_03_health_endpoint_unhealthy(self):
        """Test GET /health returns 503 when model checkpoint is not ready."""
        with patch.object(ModelManager, "is_ready", new_callable=unittest.mock.PropertyMock, return_value=False):
            with patch.object(
                model_manager,
                "get_status",
                return_value={
                    "model_name": "HalfUNet",
                    "is_ready": False,
                    "checkpoint_available": False,
                    "checkpoint_path": "fake/path.pth",
                    "device": "cpu",
                    "error": "Checkpoint missing for test",
                },
            ):
                response = self.client.get("/health")
                self.assertEqual(response.status_code, 503)
                data = response.json()
                self.assertEqual(data["status"], "unhealthy")
                self.assertFalse(data["checkpoint_available"])
                self.assertIsNotNone(data["error"])

    def test_04_predict_valid_synthetic_image(self):
        """Test POST /predict with valid synthetic image returns standardized response schema."""
        img_bytes = self._create_synthetic_image_bytes(width=200, height=200)
        response = self.client.post(
            "/predict",
            files={"file": ("synthetic_test.png", img_bytes, "image/png")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Schema checks
        self.assertTrue(data["success"])
        self.assertEqual(data["model"], "HalfUNet")
        self.assertEqual(data["image_width"], 200)
        self.assertEqual(data["image_height"], 200)
        self.assertEqual(data["total_image_pixels"], 40000)

        # Numerical consistency checks
        solar_px = data["solar_area_pixels"]
        total_px = data["total_image_pixels"]
        expected_coverage = round((solar_px / total_px) * 100.0, 4)
        self.assertAlmostEqual(data["solar_coverage_percent"], expected_coverage, places=4)

        # Region metrics
        self.assertIsInstance(data["detected_region_count"], int)
        self.assertGreaterEqual(data["detected_region_count"], 0)

        # Physical area safety
        self.assertIsNone(data["physical_area_m2"])
        self.assertIsNone(data["physical_area_hectares"])
        self.assertEqual(data["physical_area_status"], "insufficient_data")

        # Artifact URLs
        pred_id = data["prediction_id"]
        self.assertEqual(data["mask_url"], f"/outputs/{pred_id}/mask")
        self.assertEqual(data["overlay_url"], f"/outputs/{pred_id}/overlay")

    def test_05_mask_and_overlay_retrieval(self):
        """Test GET /outputs/{id}/mask and /overlay securely stream PNG artifacts."""
        img_bytes = self._create_synthetic_image_bytes(width=100, height=100)
        predict_res = self.client.post(
            "/predict",
            files={"file": ("artifact_test.png", img_bytes, "image/png")},
        )
        self.assertEqual(predict_res.status_code, 200)
        pred_data = predict_res.json()
        mask_url = pred_data["mask_url"]
        overlay_url = pred_data["overlay_url"]

        # Fetch mask
        mask_res = self.client.get(mask_url)
        self.assertEqual(mask_res.status_code, 200)
        self.assertEqual(mask_res.headers["content-type"], "image/png")
        # Validate returned bytes decode as valid image
        with Image.open(io.BytesIO(mask_res.content)) as mask_img:
            self.assertEqual(mask_img.size, (100, 100))

        # Fetch overlay
        overlay_res = self.client.get(overlay_url)
        self.assertEqual(overlay_res.status_code, 200)
        self.assertEqual(overlay_res.headers["content-type"], "image/png")
        with Image.open(io.BytesIO(overlay_res.content)) as overlay_img:
            self.assertEqual(overlay_img.size, (100, 100))

    def test_06_predict_invalid_uploads(self):
        """Test POST /predict input validation: empty file, corrupt image, unsupported format."""
        # 1. Empty file (0 bytes)
        res_empty = self.client.post(
            "/predict",
            files={"file": ("empty.png", b"", "image/png")},
        )
        self.assertEqual(res_empty.status_code, 400)
        self.assertIn("empty", res_empty.json()["error"].lower())

        # 2. Corrupt / non-image data
        res_corrupt = self.client.post(
            "/predict",
            files={"file": ("corrupt.png", b"not-a-valid-image-content", "image/png")},
        )
        self.assertEqual(res_corrupt.status_code, 400)
        self.assertIn("corrupt", res_corrupt.json()["error"].lower())

        # 3. Unsupported extension
        res_unsupported = self.client.post(
            "/predict",
            files={"file": ("document.pdf", b"%PDF-1.4...", "application/pdf")},
        )
        self.assertEqual(res_unsupported.status_code, 400)
        self.assertIn("unsupported", res_unsupported.json()["error"].lower())

    def test_07_artifact_security_and_traversal(self):
        """Test artifact endpoints reject invalid UUIDs, missing files, and path traversals."""
        # Non-existent UUID4
        fake_uuid = str(uuid.uuid4())
        res_404 = self.client.get(f"/outputs/{fake_uuid}/mask")
        self.assertEqual(res_404.status_code, 404)

        # Invalid UUID format
        res_bad_id = self.client.get("/outputs/invalid-not-a-uuid/mask")
        self.assertEqual(res_bad_id.status_code, 400)

        # Path traversal attempt
        res_traversal = self.client.get("/outputs/..%2F..%2Fconfig/mask")
        self.assertIn(res_traversal.status_code, [400, 404])

    def test_08_real_image_end_to_end_prediction(self):
        """Test POST /predict with real benchmark image (768.0_1.0.png) reproduces Phase 1 ground metrics."""
        self.assertTrue(REAL_SAMPLE_IMAGE.is_file(), f"Sample image missing: {REAL_SAMPLE_IMAGE}")
        with open(REAL_SAMPLE_IMAGE, "rb") as fp:
            file_bytes = fp.read()

        response = self.client.post(
            "/predict",
            files={"file": ("768.0_1.0.png", file_bytes, "image/png")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Check exact Phase 1 benchmark metrics
        self.assertEqual(data["image_width"], 640)
        self.assertEqual(data["image_height"], 640)
        self.assertEqual(data["total_image_pixels"], 409600)
        self.assertEqual(data["solar_area_pixels"], 68451)
        self.assertAlmostEqual(data["solar_coverage_percent"], 16.7117, places=3)
        self.assertEqual(data["detected_region_count"], 43)
        self.assertEqual(data["largest_region_area_pixels"], 4750)
        self.assertEqual(data["smallest_region_area_pixels"], 616)
        self.assertAlmostEqual(data["mean_region_area_pixels"], 1591.8837, places=2)
        self.assertIsNone(data["physical_area_m2"])
        self.assertIsNone(data["physical_area_hectares"])
        self.assertEqual(data["physical_area_status"], "insufficient_data")

    def test_09_inference_error_handling(self):
        """Test POST /predict returns 500 if inference pipeline encounters an unhandled error."""
        img_bytes = self._create_synthetic_image_bytes(width=64, height=64)
        with patch("backend.routes.predict.predict_image", side_effect=RuntimeError("Simulated pipeline failure")):
            response = self.client.post(
                "/predict",
                files={"file": ("fail_test.png", img_bytes, "image/png")},
            )
            self.assertEqual(response.status_code, 500)
            data = response.json()
            self.assertFalse(data["success"])
            self.assertIn("Simulated pipeline failure", data["error"])

    def test_10_predict_when_model_unavailable(self):
        """Test POST /predict returns 503 when model is not ready."""
        img_bytes = self._create_synthetic_image_bytes(width=64, height=64)
        with patch.object(ModelManager, "is_ready", new_callable=unittest.mock.PropertyMock, return_value=False):
            response = self.client.post(
                "/predict",
                files={"file": ("unavail_test.png", img_bytes, "image/png")},
            )
            self.assertEqual(response.status_code, 503)
            data = response.json()
            self.assertFalse(data["success"])
            self.assertIn("unavailable", data["error"].lower())

    def test_11_predict_multiple_supported_formats(self):
        """Test POST /predict accepts and processes JPEG, TIFF, BMP, and WebP images."""
        formats = [
            ("JPEG", "jpg", "image/jpeg"),
            ("TIFF", "tif", "image/tiff"),
            ("TIFF", "tiff", "image/tiff"),
            ("BMP", "bmp", "image/bmp"),
            ("WEBP", "webp", "image/webp"),
        ]
        for pil_fmt, ext, mime in formats:
            img = Image.new("RGB", (100, 100), color=(150, 180, 210))
            buf = io.BytesIO()
            img.save(buf, format=pil_fmt)
            raw_bytes = buf.getvalue()

            res = self.client.post(
                "/predict",
                files={"file": (f"format_test.{ext}", raw_bytes, mime)},
            )
            self.assertEqual(
                res.status_code,
                200,
                f"Failed for format {pil_fmt} (.{ext}): {res.text}",
            )
            data = res.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["image_width"], 100)
            self.assertEqual(data["image_height"], 100)

    def test_12_predict_parameter_validation(self):
        """Test POST /predict query parameter boundary constraints."""
        img_bytes = self._create_synthetic_image_bytes(width=64, height=64)

        # Threshold > 1.0
        res_high = self.client.post(
            "/predict?threshold=1.2",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        self.assertEqual(res_high.status_code, 422)

        # Threshold < 0.0
        res_low = self.client.post(
            "/predict?threshold=-0.1",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        self.assertEqual(res_low.status_code, 422)

        # Min component area < 1
        res_zero_area = self.client.post(
            "/predict?min_component_area=0",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        self.assertEqual(res_zero_area.status_code, 422)

    def test_13_predict_file_size_limit_rejection(self):
        """Test POST /predict returns 413 when file exceeds configured limit."""
        img_bytes = self._create_synthetic_image_bytes(width=64, height=64)
        with patch("backend.routes.predict.MAX_FILE_SIZE_BYTES", 50):
            res_too_large = self.client.post(
                "/predict",
                files={"file": ("oversized.png", img_bytes, "image/png")},
            )
            self.assertEqual(res_too_large.status_code, 413)
            self.assertIn("exceeds limit", res_too_large.json()["error"].lower())

    def test_14_predict_threshold_effect(self):
        """Test that changing probability threshold modifies segmentation pixel count."""
        self.assertTrue(REAL_SAMPLE_IMAGE.is_file(), f"Sample image missing: {REAL_SAMPLE_IMAGE}")
        with open(REAL_SAMPLE_IMAGE, "rb") as fp:
            file_bytes = fp.read()

        # Lower threshold -> more permissive
        res_low = self.client.post(
            "/predict?threshold=0.2",
            files={"file": ("sample.png", file_bytes, "image/png")},
        )
        self.assertEqual(res_low.status_code, 200)
        low_solar_px = res_low.json()["solar_area_pixels"]

        # Higher threshold -> more conservative
        res_high = self.client.post(
            "/predict?threshold=0.8",
            files={"file": ("sample.png", file_bytes, "image/png")},
        )
        self.assertEqual(res_high.status_code, 200)
        high_solar_px = res_high.json()["solar_area_pixels"]

        # Permissive threshold must yield greater or equal solar pixels than conservative threshold
        self.assertGreaterEqual(
            low_solar_px,
            high_solar_px,
            f"Expected threshold 0.2 ({low_solar_px} px) >= threshold 0.8 ({high_solar_px} px)",
        )

    def test_15_predict_geospatial_metadata(self):
        """Test POST /predict includes Level B coordinates for benchmark image and unreferenced for synthetic."""
        # 1. Synthetic image should be UNREFERENCED with None coordinates
        synthetic_bytes = self._create_synthetic_image_bytes(width=64, height=64)
        res_syn = self.client.post(
            "/predict",
            files={"file": ("unknown_rooftop.png", synthetic_bytes, "image/png")},
        )
        self.assertEqual(res_syn.status_code, 200)
        syn_data = res_syn.json()
        self.assertEqual(syn_data["georeferencing_level"], "UNREFERENCED")
        self.assertIsNone(syn_data["latitude"])
        self.assertIsNone(syn_data["longitude"])

        # 2. Benchmark catalog image (768.0_1.0.png) should return Level B coordinates
        self.assertTrue(REAL_SAMPLE_IMAGE.is_file())
        with open(REAL_SAMPLE_IMAGE, "rb") as fp:
            file_bytes = fp.read()

        res_cat = self.client.post(
            "/predict",
            files={"file": ("768.0_1.0.png", file_bytes, "image/png")},
        )
        self.assertEqual(res_cat.status_code, 200)
        cat_data = res_cat.json()
        self.assertEqual(cat_data["georeferencing_level"], "LEVEL_B")
        self.assertIsNotNone(cat_data["latitude"])
        self.assertIsNotNone(cat_data["longitude"])
        self.assertAlmostEqual(cat_data["latitude"], 21.197148, places=4)
        self.assertAlmostEqual(cat_data["longitude"], 72.780643, places=4)
        self.assertEqual(cat_data["location_name"], "Surat, Gujarat, India")

    def test_16_original_image_artifact_retrieval(self):
        """Test GET /outputs/{id}/original retrieves the exact uploaded original raster."""
        img_bytes = self._create_synthetic_image_bytes(width=160, height=90)
        predict_res = self.client.post(
            "/predict",
            files={"file": ("orig_test.png", img_bytes, "image/png")},
        )
        self.assertEqual(predict_res.status_code, 200)
        data = predict_res.json()
        orig_url = data.get("original_image_url")
        self.assertIsNotNone(orig_url)
        self.assertEqual(orig_url, f"/outputs/{data['prediction_id']}/original")

        res = self.client.get(orig_url)
        self.assertEqual(res.status_code, 200)
        with Image.open(io.BytesIO(res.content)) as im:
            self.assertEqual(im.size, (160, 90))

    def test_17_representative_aspect_ratios_alignment(self):
        """Test representative image aspect ratios (1:1, 4:3, 16:9, and 4:1 panorama) maintain strict dimensional and pixel alignment."""
        test_dimensions = [
            (200, 200),   # 1:1 Square
            (400, 300),   # 4:3 Standard
            (480, 270),   # 16:9 Widescreen
            (600, 150),   # 4:1 Panoramic
        ]

        for w, h in test_dimensions:
            with self.subTest(width=w, height=h):
                img_bytes = self._create_synthetic_image_bytes(width=w, height=h)
                res = self.client.post(
                    "/predict",
                    files={"file": (f"test_{w}x{h}.png", img_bytes, "image/png")},
                )
                self.assertEqual(res.status_code, 200)
                data = res.json()

                self.assertEqual(data["image_width"], w)
                self.assertEqual(data["image_height"], h)

                # Fetch mask and overlay
                mask_res = self.client.get(data["mask_url"])
                overlay_res = self.client.get(data["overlay_url"])
                orig_res = self.client.get(data["original_image_url"])

                self.assertEqual(mask_res.status_code, 200)
                self.assertEqual(overlay_res.status_code, 200)
                self.assertEqual(orig_res.status_code, 200)

                with Image.open(io.BytesIO(mask_res.content)) as mask_img:
                    self.assertEqual(mask_img.size, (w, h), f"Mask size mismatch for {w}x{h}")
                    mask_arr = np.array(mask_img)

                with Image.open(io.BytesIO(overlay_res.content)) as overlay_img:
                    self.assertEqual(overlay_img.size, (w, h), f"Overlay size mismatch for {w}x{h}")
                    overlay_arr = np.array(overlay_img.convert("RGB"))

                with Image.open(io.BytesIO(orig_res.content)) as orig_img:
                    self.assertEqual(orig_img.size, (w, h), f"Original size mismatch for {w}x{h}")
                    orig_arr = np.array(orig_img.convert("RGB"))

                # Mask pixel alignment with overlay highlight
                diff = np.abs(overlay_arr.astype(int) - orig_arr.astype(int))
                diff_pixels = int(np.count_nonzero(np.any(diff > 0, axis=-1)))
                solar_pixels = int(np.count_nonzero(mask_arr == 255))
                self.assertEqual(
                    diff_pixels,
                    solar_pixels,
                    f"Overlay highlight pixels ({diff_pixels}) must match mask positive pixels ({solar_pixels}) for {w}x{h}",
                )


if __name__ == "__main__":
    unittest.main()
