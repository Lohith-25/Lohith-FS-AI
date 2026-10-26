"""
SolarMap-India — Tiled & Variable Resolution Inference Audit.
Examines tile coordinates, Hann blending, probability accumulation,
and compares direct vs letterbox vs tiled on identical scenes.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path("d:/SolarMap-India")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference.model import load_halfunet_model, get_device
from ml.inference.tiling import predict_tiled, TilingConfig
from ml.inference.preprocessing import compute_tile_coordinates, generate_hann_window
from ml.inference.predict import predict_image

def main():
    print("=" * 70)
    print("SOLARMAP-INDIA: VARIABLE-RESOLUTION & TILED INFERENCE AUDIT")
    print("=" * 70)

    device = get_device()
    ckpt = PROJECT_ROOT / "outputs" / "HalfUNet" / "models" / "halfunet_best.pth"
    model = load_halfunet_model(checkpoint_path=ckpt, device=device)

    # 1. Inspect tile coordinate generation & edge handling
    print("\n1. VERIFYING TILE COORDINATES & COVERAGE:")
    for (dim_name, w, h) in [("Square 640x640", 640, 640),
                             ("Odd 637x463", 637, 463),
                             ("HD 1920x1080", 1920, 1080),
                             ("4K 4000x3000", 4000, 3000),
                             ("Small 200x150", 200, 150)]:
        xs = compute_tile_coordinates(w, tile_dim=256, overlap_ratio=0.25)
        ys = compute_tile_coordinates(h, tile_dim=256, overlap_ratio=0.25)
        total_tiles = len(xs) * len(ys)
        print(f"  {dim_name}:")
        print(f"    xs: {xs} (covers 0 to {xs[-1] + 256} of {w})")
        print(f"    ys: {ys} (covers 0 to {ys[-1] + 256} of {h})")
        print(f"    total tiles: {total_tiles}")
        assert xs[0] == 0 and xs[-1] + 256 == max(w, 256)
        assert ys[0] == 0 and ys[-1] + 256 == max(h, 256)

    # 2. Inspect Hann window properties
    print("\n2. VERIFYING HANN WINDOW BLENDING PROPERTIES:")
    hw = generate_hann_window(tile_size=(256, 256), min_weight=1e-3)
    print(f"  Hann window shape: {hw.shape}, dtype: {hw.dtype}")
    print(f"  Hann min value: {hw.min():.6f}, max value: {hw.max():.6f}, center: {hw[128, 128]:.6f}")
    assert hw.min() >= 1e-3 and hw.max() <= 1.0

    # 3. Inspect Fixtures Content
    print("\n3. INSPECTING BROWSER FIXTURES & SOLAR CONTENT:")
    fixtures = [
        ("sample_256x256.png", Path("scratch/browser_fixtures/sample_256x256.png")),
        ("sample_640x640.png", Path("scratch/browser_fixtures/sample_640x640.png")),
        ("sample_637x463.png", Path("scratch/browser_fixtures/sample_637x463.png")),
        ("sample_1920x1080.jpg", Path("scratch/browser_fixtures/sample_1920x1080.jpg")),
        ("sample_4000x3000.jpg", Path("scratch/browser_fixtures/sample_4000x3000.jpg")),
    ]

    for name, path in fixtures:
        img = Image.open(path)
        w, h = img.size
        print(f"\n  Fixture: {name} ({w}x{h})")

        # Test predict_image with auto, direct, letterbox, tiled (where applicable)
        res_auto = predict_image(path, model=model, device=device, strategy="auto", min_component_area=0)
        print(f"    [AUTO] strat={res_auto.processing_strategy}, solar_px={res_auto.solar_pixels:,}, cov={res_auto.solar_coverage_percent:.4f}%")
        print(f"           prob min={res_auto.solar_probabilities.min():.4f}, max={res_auto.solar_probabilities.max():.4f}, mean={res_auto.solar_probabilities.mean():.4f}")

        # Also run TILED explicitly on each fixture to see what tiled produces!
        res_tiled = predict_image(path, model=model, device=device, strategy="tiled", min_component_area=0)
        print(f"    [TILED] strat={res_tiled.processing_strategy}, solar_px={res_tiled.solar_pixels:,}, cov={res_tiled.solar_coverage_percent:.4f}%")
        print(f"            prob min={res_tiled.solar_probabilities.min():.4f}, max={res_tiled.solar_probabilities.max():.4f}, mean={res_tiled.solar_probabilities.mean():.4f}")

    # 4. Same Scene Multi-Resolution Evaluation:
    # Take a known ground-truth benchmark image containing solar panels
    print("\n4. SAME-SCENE CONTROLLED MULTI-RESOLUTION & MULTI-STRATEGY EVALUATION:")
    benchmark_img_path = Path("frontend/public/sample_aerial.png")
    if not benchmark_img_path.is_file():
        benchmark_img_path = Path("scratch/browser_fixtures/sample_640x640.png")

    bm_img = Image.open(benchmark_img_path)
    print(f"  Benchmark image: {benchmark_img_path.name} ({bm_img.size})")

    for strat in ["direct", "letterbox", "tiled"]:
        res = predict_image(benchmark_img_path, model=model, device=device, strategy=strat, min_component_area=0)
        print(f"    Strategy: {strat:10s} -> solar_px={res.solar_pixels:,} ({res.solar_coverage_percent:.4f}%), max_prob={res.solar_probabilities.max():.4f}, mean_prob={res.solar_probabilities.mean():.4f}")

    # Now downscale / upscale the benchmark image and compare strategies
    print("\n  Evaluating Rescaled Benchmark Image (Scale Effect on CNN receptive field):")
    for scale in [0.5, 1.0, 1.5, 2.0]:
        target_s = (int(bm_img.size[0] * scale), int(bm_img.size[1] * scale))
        scaled_img = bm_img.resize(target_s, Image.BILINEAR)
        scaled_path = PROJECT_ROOT / "scratch" / f"scaled_benchmark_{scale:.1f}.png"
        scaled_img.save(scaled_path)

        for strat in ["direct", "letterbox", "tiled"]:
            res = predict_image(scaled_path, model=model, device=device, strategy=strat, min_component_area=0)
            print(f"    Scale {scale:.1f}x ({target_s[0]}x{target_s[1]}) [{strat:10s}]: solar_px={res.solar_pixels:6,}, cov={res.solar_coverage_percent:7.4f}%, max_p={res.solar_probabilities.max():.4f}")

if __name__ == "__main__":
    main()
