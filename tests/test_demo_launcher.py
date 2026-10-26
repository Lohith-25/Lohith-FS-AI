"""
SolarMap-India — Phase 14.3: Demonstration Launcher Automated Test Suite.

Tests for:
1. Secret masking for database connection strings and URIs.
2. Python dependency check reporting.
3. Model checkpoint presence and validation.
4. Missing checkpoint error reporting.
5. Frontend build directory validation.
6. Missing frontend build instructions reporting.
7. Port checking logic and socket probe.
8. Port conflict detection and failure reporting.
9. Environment auditing without secret leakage.
10. Launcher CLI --check-only execution and exit codes.
"""

from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.start_demo import (
    DEFAULT_CHECKPOINT,
    DEFAULT_FRONTEND_DIST,
    audit_environment,
    check_frontend_build,
    check_model_checkpoint,
    check_python_dependencies,
    is_port_in_use,
    mask_secret,
    run_checks,
)


class TestDemoLauncher(unittest.TestCase):
    """Automated tests for demonstration quickstart launcher logic and safety guardrails."""

    def test_01_secret_masking_standard_mongodb_uri(self):
        """Test 1: Standard MongoDB URI masks username:password credentials."""
        raw_uri = "mongodb://solarmap_admin:SuperSecretPass123@localhost:27017/solarmap_india"
        masked = mask_secret(raw_uri)
        self.assertNotIn("SuperSecretPass123", masked)
        self.assertIn("solarmap_admin:***@", masked)
        self.assertTrue(masked.startswith("mongodb://solarmap_admin:***@localhost:27017/"))

    def test_02_secret_masking_atlas_srv_uri(self):
        """Test 2: MongoDB Atlas SRV URI masks sensitive cluster passwords."""
        raw_uri = "mongodb+srv://admin_user:P%40ssw0rd!@cluster0.mongodb.net/?retryWrites=true"
        masked = mask_secret(raw_uri)
        self.assertNotIn("P%40ssw0rd!", masked)
        self.assertIn("admin_user:***@", masked)
        self.assertTrue(masked.startswith("mongodb+srv://admin_user:***@cluster0.mongodb.net/"))

    def test_03_secret_masking_empty_or_none(self):
        """Test 3: Empty or none inputs handle gracefully without exceptions."""
        self.assertEqual(mask_secret(""), "")
        self.assertEqual(mask_secret(None), "")
        self.assertEqual(mask_secret("mongodb://localhost:27017"), "mongodb://localhost:27017")

    def test_04_python_dependencies_check(self):
        """Test 4: Core Python packages are identified and verified."""
        all_ok, missing_req, missing_opt = check_python_dependencies()
        self.assertTrue(all_ok)
        self.assertEqual(len(missing_req), 0)

    def test_05_production_model_checkpoint_verification(self):
        """Test 5: Production checkpoint at default path is verified."""
        cp_ok, msg = check_model_checkpoint(DEFAULT_CHECKPOINT)
        self.assertTrue(cp_ok)
        self.assertIn("Valid checkpoint found", msg)

    def test_06_missing_model_checkpoint_produces_clear_error(self):
        """Test 6: Missing checkpoint path produces a descriptive failure message."""
        fake_path = PROJECT_ROOT / "outputs" / "nonexistent_model.pth"
        cp_ok, msg = check_model_checkpoint(fake_path)
        self.assertFalse(cp_ok)
        self.assertIn("Checkpoint file not found", msg)
        self.assertIn("Expected location", msg)

    def test_07_frontend_build_verification(self):
        """Test 7: Frontend build dist verification."""
        fe_ok, msg = check_frontend_build(DEFAULT_FRONTEND_DIST)
        self.assertTrue(fe_ok)
        self.assertIn("Compiled frontend assets verified", msg)

    def test_08_missing_frontend_build_guidance(self):
        """Test 8: Missing frontend dist provides actionable build instructions."""
        fake_dist = PROJECT_ROOT / "frontend" / "nonexistent_dist"
        fe_ok, msg = check_frontend_build(fake_dist)
        self.assertFalse(fe_ok)
        self.assertIn("npm run build", msg)
        self.assertIn("Compiled frontend not found", msg)

    def test_09_port_availability_check(self):
        """Test 9: High random port is verified as available (not in use)."""
        # Test an unlikely high port
        in_use = is_port_in_use(59921)
        self.assertFalse(in_use)

    def test_10_port_conflict_detection(self):
        """Test 10: Occupied port triggers failure in pre-flight checks."""
        with patch("scripts.start_demo.is_port_in_use", return_value=True):
            passed = run_checks(
                checkpoint_path=DEFAULT_CHECKPOINT,
                dist_dir=DEFAULT_FRONTEND_DIST,
                backend_port=8000,
            )
            self.assertFalse(passed)

    def test_11_environment_audit_masks_secrets(self):
        """Test 11: Environment auditing masks configured MONGODB_URI."""
        with patch.dict("os.environ", {"MONGODB_URI": "mongodb://user:my_secret_token@host:27017/db"}):
            env_info = audit_environment()
            self.assertTrue(env_info["MONGODB_CONFIGURED"])
            self.assertNotIn("my_secret_token", env_info["MONGODB_URI_MASKED"])
            self.assertIn("user:***@", env_info["MONGODB_URI_MASKED"])

    def test_12_cli_check_only_flag_execution(self):
        """Test 12: CLI execution with --check-only exits with code 0 on valid environment."""
        res = subprocess.run(
            [sys.executable, "scripts/start_demo.py", "--check-only", "--backend-port", "59981"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("All required Python packages are installed", res.stdout)
        self.assertIn("Valid checkpoint found", res.stdout)


if __name__ == "__main__":
    unittest.main()
