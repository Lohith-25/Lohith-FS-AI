#!/usr/bin/env python3
"""
SolarMap-India — Demonstration Quickstart Launcher (Phase 14.3).

Validates prerequisites, environment settings, and port availability before
launching the SolarMap-India application processes in a clean, predictable manner.

Features:
1. Validates required Python dependencies without auto-installing.
2. Verifies Half U-Net production model checkpoint existence and size.
3. Checks for compiled frontend assets (frontend/dist) or provides build instructions.
4. Audits environment variables with strict secrets masking for database URIs.
5. Checks port availability before attempting process binding.
6. Gracefully handles SIGINT/SIGTERM with process cleanup.
7. Explains MongoDB status (offline/unconfigured vs connected) and non-blocking operation.
"""

import argparse
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Locate project root directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Default paths
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "HalfUNet" / "models" / "halfunet_best.pth"
DEFAULT_FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
DEFAULT_FRONTEND_DIR = PROJECT_ROOT / "frontend"

REQUIRED_PYTHON_MODULES = [
    ("torch", "PyTorch deep learning framework"),
    ("fastapi", "FastAPI asynchronous web framework"),
    ("uvicorn", "Uvicorn ASGI server"),
    ("pydantic", "Pydantic data validation"),
    ("PIL", "Pillow image processing library"),
    ("numpy", "NumPy numerical computing"),
    ("shapely", "Shapely geometric operations"),
    ("pyproj", "PyProj cartographic projections & geodesic calculations"),
]

OPTIONAL_PYTHON_MODULES = [
    ("motor", "Async MongoDB driver (required only for prediction history persistence)"),
    ("httpx", "HTTP client (used for integration testing and API probes)"),
]


def mask_secret(uri: str) -> str:
    """
    Masks credentials in database URIs or connection strings to prevent leaking secrets.

    Example:
        mongodb+srv://admin:mypassword@cluster0.net -> mongodb+srv://admin:***@cluster0.net
        mongodb://user:pass@localhost:27017/db -> mongodb://user:***@localhost:27017/db
    """
    if not uri or not isinstance(uri, str):
        return ""
    # Mask password in URI scheme:://username:password@host
    masked = re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", uri)
    return masked


def check_python_dependencies() -> Tuple[bool, List[str], List[str]]:
    """
    Validates presence of required and optional Python packages.

    Returns:
        (all_required_present, missing_required_list, missing_optional_list)
    """
    missing_required = []
    missing_optional = []

    for mod_name, desc in REQUIRED_PYTHON_MODULES:
        try:
            __import__(mod_name)
        except ImportError:
            missing_required.append(f"{mod_name} ({desc})")

    for mod_name, desc in OPTIONAL_PYTHON_MODULES:
        try:
            __import__(mod_name)
        except ImportError:
            missing_optional.append(f"{mod_name} ({desc})")

    return (len(missing_required) == 0, missing_required, missing_optional)


def check_model_checkpoint(checkpoint_path: Path) -> Tuple[bool, str]:
    """
    Verifies that the Half U-Net production checkpoint exists and is non-empty.

    Returns:
        (exists_and_valid, message)
    """
    cp = Path(checkpoint_path).resolve()
    if not cp.is_file():
        return (
            False,
            f"Checkpoint file not found: {cp}\n"
            f"  Expected location: {DEFAULT_CHECKPOINT}\n"
            f"  Note: Model weights must be placed at outputs/HalfUNet/models/halfunet_best.pth",
        )

    file_size_mb = cp.stat().st_size / (1024 * 1024)
    if file_size_mb < 1.0:
        return (
            False,
            f"Checkpoint file appears incomplete or truncated ({file_size_mb:.2f} MB): {cp}",
        )

    return (True, f"Valid checkpoint found ({file_size_mb:.2f} MB): {cp}")


def check_frontend_build(dist_dir: Path) -> Tuple[bool, str]:
    """
    Checks whether compiled React frontend assets exist for direct serving.

    Returns:
        (is_built, message)
    """
    dist = Path(dist_dir).resolve()
    index_html = dist / "index.html"
    if index_html.is_file():
        return (True, f"Compiled frontend assets verified at: {dist}")

    return (
        False,
        f"Compiled frontend not found at: {dist}\n"
        f"  To compile production frontend:\n"
        f"    cd frontend && npm install && npm run build\n"
        f"  Alternatively, run with --dev to launch the Vite dev server alongside the backend.",
    )


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """
    Checks whether a network port is already in use by attempting a socket connection.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        # If connect_ex returns 0, the port is accepting connections (in use)
        return s.connect_ex((host, port)) == 0


def audit_environment() -> Dict[str, Any]:
    """
    Audits relevant environment variables with secrets masking.
    """
    raw_mongo = os.getenv("MONGODB_URI", "")
    return {
        "MODEL_PATH": os.getenv("MODEL_PATH", str(DEFAULT_CHECKPOINT)),
        "MONGODB_CONFIGURED": bool(raw_mongo.strip()),
        "MONGODB_URI_MASKED": mask_secret(raw_mongo) if raw_mongo.strip() else "(unconfigured - history disabled)",
        "MONGODB_DATABASE": os.getenv("MONGODB_DATABASE", os.getenv("DATABASE_NAME", "solarmap_india")),
        "FRONTEND_DIST_DIR": os.getenv("FRONTEND_DIST_DIR", str(DEFAULT_FRONTEND_DIST)),
        "CORS_ORIGINS": os.getenv("CORS_ORIGINS", "*"),
    }


def print_banner() -> None:
    print("=" * 76)
    print("   SOLARMAP-INDIA -- DEMONSTRATION & REVIEWER QUICKSTART LAUNCHER")
    print("=" * 76)
    print("Satellite-based rooftop solar potential & variable-resolution inference")
    print("-" * 76)


def run_checks(
    checkpoint_path: Path,
    dist_dir: Path,
    backend_port: int,
    vite_port: Optional[int] = None,
) -> bool:
    """
    Executes all pre-flight checks and prints a structured readiness report.

    Returns:
        True if all required checks pass, False otherwise.
    """
    print_banner()
    all_ok = True

    # 1. Python Dependencies
    print("[1/5] Checking Python Dependencies...")
    deps_ok, missing_req, missing_opt = check_python_dependencies()
    if deps_ok:
        print("  [OK] All required Python packages are installed.")
    else:
        all_ok = False
        print("  [FAIL] Missing required Python packages:")
        for m in missing_req:
            print(f"    - {m}")
        print("  Resolution: pip install -r requirements.txt")

    if missing_opt:
        print("  [INFO] Missing optional packages:")
        for m in missing_opt:
            print(f"    - {m}")

    # 2. Model Checkpoint
    print("\n[2/5] Checking Production Model Checkpoint...")
    cp_ok, cp_msg = check_model_checkpoint(checkpoint_path)
    if cp_ok:
        print(f"  [OK] {cp_msg}")
    else:
        all_ok = False
        print(f"  [FAIL] {cp_msg}")

    # 3. Frontend Build Output
    print("\n[3/5] Checking Frontend Assets...")
    fe_ok, fe_msg = check_frontend_build(dist_dir)
    if fe_ok:
        print(f"  [OK] {fe_msg}")
    else:
        print(f"  [WARN] {fe_msg}")

    # 4. Environment & Database Configuration
    print("\n[4/5] Checking Environment Configuration (Secrets Masked)...")
    env = audit_environment()
    print(f"  Model Path:          {env['MODEL_PATH']}")
    print(f"  MongoDB URI:         {env['MONGODB_URI_MASKED']}")
    print(f"  MongoDB Database:    {env['MONGODB_DATABASE']}")
    print(f"  Frontend Dist:       {env['FRONTEND_DIST_DIR']}")
    print(f"  CORS Allowed:        {env['CORS_ORIGINS']}")
    if not env["MONGODB_CONFIGURED"]:
        print("  [NOTE] MongoDB is unconfigured or offline. SolarMap-India will run in")
        print("         stateless mode: predictions, masks, and calculations work normally,")
        print("         while prediction history persistence (/api/predictions/history) is disabled.")

    # 5. Port Availability
    print("\n[5/5] Checking Network Port Availability...")
    if is_port_in_use(backend_port):
        all_ok = False
        print(f"  [FAIL] Backend port {backend_port} is already in use by another process.")
        print(f"  Resolution: Terminate existing server or specify another port via --backend-port.")
    else:
        print(f"  [OK] Backend port {backend_port} is available.")

    if vite_port is not None:
        if is_port_in_use(vite_port):
            all_ok = False
            print(f"  [FAIL] Vite dev port {vite_port} is already in use.")
            print(f"  Resolution: Terminate existing Vite instance or specify another port via --vite-port.")
        else:
            print(f"  [OK] Vite dev port {vite_port} is available.")

    print("-" * 76)
    return all_ok


def launch_demo(
    backend_host: str = "127.0.0.1",
    backend_port: int = 8000,
    run_vite_dev: bool = False,
    vite_port: int = 5173,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    dist_dir: Path = DEFAULT_FRONTEND_DIST,
) -> int:
    """
    Validates environment and launches application processes with clean lifecycle management.
    """
    check_vite_port = vite_port if run_vite_dev else None
    passed = run_checks(
        checkpoint_path=checkpoint_path,
        dist_dir=dist_dir,
        backend_port=backend_port,
        vite_port=check_vite_port,
    )

    if not passed:
        print("\n[ABORT] Demonstration pre-flight checks failed. Please resolve above issues.")
        return 1

    processes: List[subprocess.Popen] = []

    def shutdown_handler(signum, frame):
        print("\n\n[SHUTDOWN] Interruption signal received. Terminating running demo processes...")
        for p in processes:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    p.kill()
        print("[SHUTDOWN] All demo processes stopped cleanly.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    print("\n[STARTUP] Starting application processes...")

    # Start FastAPI backend
    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.main:app",
        "--host",
        backend_host,
        "--port",
        str(backend_port),
    ]
    env = os.environ.copy()
    env["MODEL_PATH"] = str(checkpoint_path.resolve())
    env["FRONTEND_DIST_DIR"] = str(dist_dir.resolve())

    try:
        backend_proc = subprocess.Popen(
            backend_cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        processes.append(backend_proc)
    except Exception as exc:
        print(f"[FAIL] Failed to spawn FastAPI backend process: {exc}")
        return 1

    vite_proc = None
    if run_vite_dev:
        # User explicitly requested Vite dev server alongside backend
        npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
        vite_cmd = [npm_cmd, "run", "dev", "--", "--port", str(vite_port)]
        try:
            vite_proc = subprocess.Popen(
                vite_cmd,
                cwd=str(DEFAULT_FRONTEND_DIR),
            )
            processes.append(vite_proc)
        except Exception as exc:
            print(f"[WARN] Failed to spawn Vite dev server ({exc}).")
            print("       Backend will continue running.")

    time.sleep(1.5)

    print("\n" + "=" * 76)
    print("   SOLARMAP-INDIA APPLICATION IS RUNNING")
    print("=" * 76)
    if (dist_dir / "index.html").is_file():
        print(f"  * Web Application (Unified UI):  http://{backend_host}:{backend_port}/")
    if run_vite_dev:
        print(f"  * Vite Development UI:          http://localhost:{vite_port}/")
    print(f"  * Interactive API Docs (Swagger): http://{backend_host}:{backend_port}/docs")
    print(f"  * Health & Checkpoint Status:    http://{backend_host}:{backend_port}/health")
    print(f"  * Prediction Endpoint:           http://{backend_host}:{backend_port}/predict")
    print("-" * 76)
    print("Press Ctrl+C to shut down all demo processes cleanly.\n")

    # Monitor running processes until interrupted
    try:
        while True:
            for p in processes:
                exit_code = p.poll()
                if exit_code is not None:
                    print(f"\n[ALERT] A demo process exited unexpectedly with code {exit_code}.")
                    shutdown_handler(None, None)
            time.sleep(1.0)
    except KeyboardInterrupt:
        shutdown_handler(None, None)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SolarMap-India Demonstration Quickstart Launcher (Phase 14.3)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run pre-flight validation checks and exit without starting servers.",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Launch the Vite development server (port 5173) alongside FastAPI backend.",
    )
    parser.add_argument(
        "--backend-host",
        default="127.0.0.1",
        help="Host interface for the FastAPI backend (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--backend-port",
        type=int,
        default=8000,
        help="Network port for the FastAPI backend (default: 8000).",
    )
    parser.add_argument(
        "--vite-port",
        type=int,
        default=5173,
        help="Network port for the Vite dev server (default: 5173).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to Half U-Net checkpoint (default: {DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=DEFAULT_FRONTEND_DIST,
        help=f"Path to compiled frontend distribution (default: {DEFAULT_FRONTEND_DIST}).",
    )

    args = parser.parse_args()

    if args.check_only:
        check_vite_port = args.vite_port if args.dev else None
        ok = run_checks(
            checkpoint_path=args.checkpoint,
            dist_dir=args.dist_dir,
            backend_port=args.backend_port,
            vite_port=check_vite_port,
        )
        return 0 if ok else 1

    return launch_demo(
        backend_host=args.backend_host,
        backend_port=args.backend_port,
        run_vite_dev=args.dev,
        vite_port=args.vite_port,
        checkpoint_path=args.checkpoint,
        dist_dir=args.dist_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
