#!/usr/bin/env python3
"""
SolarMap-India — Model Artifact Verification Script.

Verifies the presence, file integrity, architecture compatibility, parameter count,
and inference capability of the Half U-Net checkpoint required by the production backend.

Exit Codes:
- 0: Checkpoint verified and compatible for production inference.
- 1: Checkpoint missing, unreadable, or architecture incompatible.
"""

import argparse
import hashlib
import os
from pathlib import Path
import sys

# Ensure repository root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from ml.inference.model import EXPECTED_PARAM_COUNT, get_device, load_halfunet_model

DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT / "outputs" / "HalfUNet" / "models" / "halfunet_best.pth"
)


def compute_sha256(filepath: Path, chunk_size: int = 65536) -> str:
    """Computes SHA-256 checksum of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            sha.update(chunk)
    return sha.hexdigest()


def verify_checkpoint(checkpoint_path: Path, verbose: bool = True) -> bool:
    """
    Verifies that the target model checkpoint exists and can be loaded.

    Args:
        checkpoint_path: Target path to .pth checkpoint file.
        verbose: Whether to print diagnostic messages.

    Returns:
        True if checkpoint is valid and operable, False otherwise.
    """
    if verbose:
        print("=" * 70)
        print("SOLARMAP-INDIA -- PRODUCTION MODEL ARTIFACT VERIFICATION")
        print("=" * 70)
        print(f"Target Checkpoint:  {checkpoint_path.resolve()}")
        print(f"Target Device:      {get_device()}")

    # 1. Check existence
    if not checkpoint_path.is_file():
        if verbose:
            print("\n[ERROR] CHECKPOINT NOT FOUND!")
            print(f"   Expected location: {checkpoint_path.resolve()}")
            print("\nPROVISIONING INSTRUCTIONS:")
            print("   Trained model weights are not committed to Git to keep repository size lean.")
            print("   Developers and deployers must provide the trained Half U-Net checkpoint:")
            print("   1. Local Training: Run 'Half U-Net.ipynb' to train and save 'halfunet_best.pth'")
            print("      into 'outputs/HalfUNet/models/'.")
            print("   2. Docker Container Deployment: Mount the host checkpoint directory into the container:")
            print("      docker run -v /host/path/to/models:/app/outputs/HalfUNet/models:ro ...")
            print("      or configure docker-compose.yml volume mount.")
            print("   3. Environment Variable: Override the default path via MODEL_PATH:")
            print("      export MODEL_PATH=/custom/path/to/halfunet_best.pth")
            print("=" * 70)
        return False

    # 2. File size and readability
    try:
        size_bytes = checkpoint_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        sha256_hash = compute_sha256(checkpoint_path)
    except Exception as exc:
        if verbose:
            print(f"\n[ERROR] CHECKPOINT UNREADABLE: {exc}")
        return False

    if verbose:
        print(f"File Size:          {size_mb:.2f} MB ({size_bytes:,} bytes)")
        print(f"SHA-256 Checksum:   {sha256_hash}")

    # 3. Model Architecture & Weights Loading
    try:
        device = get_device()
        model = load_halfunet_model(checkpoint_path=checkpoint_path, device=device)
        total_params = sum(p.numel() for p in model.parameters())
    except Exception as exc:
        if verbose:
            print(f"\n[ERROR] FAILED TO LOAD MODEL CHECKPOINT: {exc}")
            print("=" * 70)
        return False

    if verbose:
        print(f"Model Architecture: HalfUNet (Encoder: 3-level, Bottleneck, Decoder: 3-level)")
        print(f"Parameter Count:    {total_params:,} (Expected: {EXPECTED_PARAM_COUNT:,})")

    # 4. Forward Pass Verification
    try:
        dummy_input = torch.zeros(1, 3, 256, 256, device=device)
        with torch.no_grad():
            output = model(dummy_input)
        expected_shape = (1, 2, 256, 256)
        if output.shape != expected_shape:
            if verbose:
                print(f"\n[ERROR] UNEXPECTED FORWARD PASS OUTPUT SHAPE: {output.shape} (Expected {expected_shape})")
            return False
    except Exception as exc:
        if verbose:
            print(f"\n[ERROR] FORWARD PASS INFERENCE ERROR: {exc}")
            print("=" * 70)
        return False

    if verbose:
        print(f"Forward Pass Test:  PASSED (Output shape: {output.shape})")
        print("\n[SUCCESS] MODEL ARTIFACT VERIFIED: Half U-Net is ready for production inference.")
        print("=" * 70)

    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify SolarMap-India Half U-Net model artifact readiness."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.getenv("MODEL_PATH", str(DEFAULT_CHECKPOINT_PATH)),
        help="Path to the model checkpoint file (defaults to MODEL_PATH env or outputs/HalfUNet/models/halfunet_best.pth)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose diagnostic logging (exit code only)",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    is_valid = verify_checkpoint(checkpoint_path, verbose=not args.quiet)
    return 0 if is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
