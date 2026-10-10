"""
SolarMap-India — Probability, Confidence & Uncertainty Extraction.

Calculates:
- Softmax probability maps (solar and background)
- Prediction confidence: max(P(bg), P(solar))
- Probability-derived prediction uncertainty: 1 - confidence
- Statistical summaries (solar, background, global)
- 10-bin confidence distribution histogram
- Configurable ambiguity/uncertainty threshold metrics
"""

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference.model import HalfUNet, get_device, load_halfunet_model
from ml.inference.postprocessing import process_logits
from ml.inference.preprocessing import load_and_validate_image, preprocess_image


@dataclass
class ProbabilityStats:
    """Statistical summary of a probability or confidence array."""
    mean: float
    median: float
    min: float
    max: float
    std: float

    def to_dict(self) -> dict:
        return {
            "mean": round(float(self.mean), 4),
            "median": round(float(self.median), 4),
            "min": round(float(self.min), 4),
            "max": round(float(self.max), 4),
            "std": round(float(self.std), 4),
        }


def validate_probabilities(
    bg_prob: np.ndarray,
    solar_prob: np.ndarray,
    tolerance: float = 1e-4,
) -> float:
    """
    Validates probability maps:
    1. No NaN or Inf values.
    2. Values within [0.0, 1.0].
    3. Sum(bg_prob + solar_prob) == 1.0 within numerical tolerance.

    Returns:
        Maximum absolute deviation from 1.0.

    Raises:
        ValueError: If any probability validation check fails.
    """
    if np.isnan(bg_prob).any() or np.isnan(solar_prob).any():
        raise ValueError("Probability map contains NaN values.")
    if np.isinf(bg_prob).any() or np.isinf(solar_prob).any():
        raise ValueError("Probability map contains infinite values.")

    # Tolerant bounding check (within eps)
    if (bg_prob < -tolerance).any() or (bg_prob > 1.0 + tolerance).any():
        raise ValueError(
            f"Background probability out of [0, 1] bounds: range [{bg_prob.min()}, {bg_prob.max()}]"
        )
    if (solar_prob < -tolerance).any() or (solar_prob > 1.0 + tolerance).any():
        raise ValueError(
            f"Solar probability out of [0, 1] bounds: range [{solar_prob.min()}, {solar_prob.max()}]"
        )

    prob_sum = bg_prob + solar_prob
    max_dev = float(np.max(np.abs(prob_sum - 1.0)))
    if max_dev > tolerance:
        raise ValueError(
            f"Probability sum deviation ({max_dev:.6f}) exceeds tolerance ({tolerance:.6f})"
        )

    return max_dev


def compute_confidence_and_uncertainty(
    bg_prob: np.ndarray,
    solar_prob: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes per-pixel prediction confidence and prediction uncertainty.

    Confidence: max(P(bg), P(solar)) ∈ [0.5, 1.0]
    Uncertainty: 1 - confidence ∈ [0.0, 0.5]
    """
    confidence = np.maximum(bg_prob, solar_prob).astype(np.float32)
    uncertainty = (1.0 - confidence).astype(np.float32)
    return confidence, uncertainty


def compute_stats(values: np.ndarray) -> Optional[ProbabilityStats]:
    """Computes summary statistics over non-empty 1D/2D array."""
    if values.size == 0:
        return None
    return ProbabilityStats(
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        min=float(np.min(values)),
        max=float(np.max(values)),
        std=float(np.std(values)),
    )


def compute_confidence_histogram(
    confidence: np.ndarray,
    num_bins: int = 10,
) -> List[dict]:
    """
    Computes 10-bin histogram of confidence distribution across all pixels.
    Bins: [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0].
    Sum of counts equals total pixels.
    """
    total_pixels = int(confidence.size)
    bins_info: List[dict] = []
    accumulated_count = 0

    for i in range(num_bins):
        low = i / num_bins
        high = (i + 1) / num_bins
        if i == num_bins - 1:
            # Include 1.0 in final bin
            mask = (confidence >= low) & (confidence <= high)
            bin_label = f"[{low:.1f}, {high:.1f}]"
        else:
            mask = (confidence >= low) & (confidence < high)
            bin_label = f"[{low:.1f}, {high:.1f})"

        count = int(np.count_nonzero(mask))
        accumulated_count += count
        pct = (count / total_pixels) * 100.0 if total_pixels > 0 else 0.0

        bins_info.append({
            "bin": i,
            "range": bin_label,
            "lower_bound": round(low, 2),
            "upper_bound": round(high, 2),
            "pixel_count": count,
            "percentage": round(pct, 2),
        })

    if accumulated_count != total_pixels:
        raise ValueError(
            f"Histogram sum ({accumulated_count}) does not equal total pixels ({total_pixels})"
        )

    return bins_info


def compute_ambiguity_stats(
    uncertainty: np.ndarray,
    solar_mask: np.ndarray,
    ambiguity_threshold: float = 0.40,
) -> dict:
    """
    Calculates pixel counts and percentages exceeding the operational uncertainty/ambiguity threshold.
    Default threshold: uncertainty >= 0.40 (equivalent to confidence <= 0.60).
    """
    total_pixels = int(uncertainty.size)
    global_ambiguous = (uncertainty >= ambiguity_threshold)
    global_count = int(np.count_nonzero(global_ambiguous))
    global_pct = (global_count / total_pixels) * 100.0 if total_pixels > 0 else 0.0

    solar_pixels = int(np.count_nonzero(solar_mask == 1))
    if solar_pixels > 0:
        solar_ambiguous = global_ambiguous & (solar_mask == 1)
        solar_count = int(np.count_nonzero(solar_ambiguous))
        solar_pct = (solar_count / solar_pixels) * 100.0
    else:
        solar_count = 0
        solar_pct = 0.0

    return {
        "ambiguity_threshold_uncertainty": float(ambiguity_threshold),
        "confidence_cutoff": float(1.0 - ambiguity_threshold),
        "global_ambiguous_pixels": global_count,
        "global_ambiguous_percentage": round(global_pct, 2),
        "solar_ambiguous_pixels": solar_count,
        "solar_ambiguous_percentage": round(solar_pct, 2),
    }


def extract_model_probabilities(
    image_path: Union[str, Path],
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Executes Half U-Net forward pass and extracts full-resolution probability maps and binary mask.

    Returns:
        (bg_prob [H, W], solar_prob [H, W], binary_mask [H, W], (width, height))
    """
    if device is None:
        device = get_device()

    model = load_halfunet_model(checkpoint_path=checkpoint_path, device=device)
    orig_img, (w, h) = load_and_validate_image(image_path)
    tensor = preprocess_image(orig_img, device=device)

    with torch.no_grad():
        logits = model(tensor)

    # Softmax on logits [1, 2, 256, 256]
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.float32)
    bg_prob_256 = probs[0]
    solar_prob_256 = probs[1]

    # Resize probability maps to original dimensions using nearest-neighbor interpolation
    # for strict, pixel-for-pixel consistency with Phase 1 binary mask
    bg_prob_orig = cv2.resize(bg_prob_256, (w, h), interpolation=cv2.INTER_NEAREST)
    solar_prob_orig = cv2.resize(solar_prob_256, (w, h), interpolation=cv2.INTER_NEAREST)

    # Thresholding generates exact Phase 1 binary mask
    binary_mask = (solar_prob_orig >= 0.5).astype(np.uint8)

    # Validate probabilities
    validate_probabilities(bg_prob_orig, solar_prob_orig)

    return bg_prob_orig, solar_prob_orig, binary_mask, (w, h)


def run_confidence_pipeline(
    image_path: Union[str, Path],
    checkpoint_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    ambiguity_threshold: float = 0.40,
    high_threshold: float = 0.90,
    medium_threshold: float = 0.70,
    min_component_area: int = 20,
    device: Optional[torch.device] = None,
) -> Tuple[dict, dict]:
    """
    Executes the complete Phase 3 confidence & uncertainty analysis workflow.

    Returns:
        (report_dict, artifact_paths_dict)
    """
    from ml.analytics.components import analyze_components
    from ml.confidence.confidence_maps import generate_all_confidence_maps
    from ml.confidence.region_confidence import (
        compute_region_confidences,
        summarize_region_confidences,
    )
    from ml.confidence.report import (
        build_confidence_report,
        save_confidence_json,
        save_region_confidence_csv,
    )
    from ml.inference.predict import DEFAULT_CHECKPOINT

    img_path = Path(image_path)
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT
    ckpt_path = Path(checkpoint_path)

    if output_dir is None:
        out_base = PROJECT_ROOT / "outputs" / "confidence"
    else:
        out_base = Path(output_dir)

    reports_dir = out_base / "reports"
    maps_dir = out_base / "maps"
    regions_dir = out_base / "regions"
    for d in [reports_dir, maps_dir, regions_dir]:
        d.mkdir(parents=True, exist_ok=True)

    stem = img_path.stem

    # 1. Extract model probabilities and Phase 1-consistent binary mask
    bg_prob, solar_prob, binary_mask, (w, h) = extract_model_probabilities(
        image_path=img_path,
        checkpoint_path=ckpt_path,
        device=device,
    )

    # 2. Validation
    max_dev = validate_probabilities(bg_prob, solar_prob)

    # 3. Confidence and uncertainty
    confidence, uncertainty = compute_confidence_and_uncertainty(bg_prob, solar_prob)

    # 4. Statistical summaries
    solar_pixels = binary_mask == 1
    bg_pixels = binary_mask == 0

    solar_stats = compute_stats(solar_prob[solar_pixels])
    bg_stats = compute_stats(bg_prob[bg_pixels])
    global_conf_stats = compute_stats(confidence)
    global_unc_stats = {
        "mean": round(float(np.mean(uncertainty)), 4),
        "median": round(float(np.median(uncertainty)), 4),
        "max": round(float(np.max(uncertainty)), 4),
    }

    # 5. Histogram & ambiguity metrics
    histogram = compute_confidence_histogram(confidence, num_bins=10)
    ambiguity_stats = compute_ambiguity_stats(
        uncertainty=uncertainty,
        solar_mask=binary_mask,
        ambiguity_threshold=ambiguity_threshold,
    )

    # 6. Connected component analysis (from Phase 2)
    component_stats, label_map = analyze_components(
        binary_mask=binary_mask,
        min_component_area=min_component_area,
    )

    # 7. Region-level confidence
    region_records = compute_region_confidences(
        regions=component_stats.regions,
        label_map=label_map,
        solar_prob=solar_prob,
        confidence_map=confidence,
        uncertainty_map=uncertainty,
        high_threshold=high_threshold,
        medium_threshold=medium_threshold,
    )
    region_summary = summarize_region_confidences(region_records)
    region_summary["high_threshold"] = high_threshold
    region_summary["medium_threshold"] = medium_threshold

    # 8. Generate maps
    maps_paths = generate_all_confidence_maps(
        solar_prob=solar_prob,
        confidence=confidence,
        uncertainty=uncertainty,
        stem=stem,
        output_dir=maps_dir,
    )

    # 9. Build and save reports
    report_dict = build_confidence_report(
        image_name=img_path.name,
        width=w,
        height=h,
        max_sum_dev=max_dev,
        solar_stats=solar_stats.to_dict() if solar_stats else None,
        bg_stats=bg_stats.to_dict() if bg_stats else None,
        global_conf_stats=global_conf_stats.to_dict() if global_conf_stats else {},
        global_unc_stats=global_unc_stats,
        histogram=histogram,
        ambiguity_stats=ambiguity_stats,
        region_summary=region_summary,
        region_records=region_records,
    )

    json_report_path = reports_dir / f"{stem}_confidence.json"
    save_confidence_json(report_dict, json_report_path)

    csv_region_path = regions_dir / f"{stem}_region_confidence.csv"
    save_region_confidence_csv(region_records, csv_region_path)

    artifacts = {
        "json_report": json_report_path,
        "region_csv": csv_region_path,
        **maps_paths,
    }

    return report_dict, artifacts


def print_summary(report: dict, artifacts: dict) -> None:
    """Formats and prints summary to console."""
    img = report["image"]
    probs = report["probabilities"]
    sol_p = probs["solar_pixels_probability"]
    g_conf = probs["global_prediction_confidence"]
    g_unc = probs["global_prediction_uncertainty"]
    amb = report["ambiguity"]
    reg_sum = report["region_confidence_summary"]

    print("-" * 54)
    print("SolarMap-India Prediction Confidence & Uncertainty")
    print("-" * 54)
    print(f"Image:                     {img['name']}")
    print(f"Dimensions:                {img['width']}x{img['height']}")
    print(f"Total Pixels:              {img['total_pixels']:,}")
    print(f"Max Prob Sum Deviation:    {report['probability_validation']['max_probability_sum_deviation']:.6f}")
    if sol_p:
        print(f"Mean Solar Probability:    {sol_p['mean']:.4f}")
        print(f"Median Solar Probability:  {sol_p['median']:.4f}")
    print(f"Mean Prediction Confidence:{g_conf['mean']:.4f}")
    print(f"Mean Uncertainty:          {g_unc['mean']:.4f}")
    print(f"Ambiguous Pixels (>=0.40): {amb['global_ambiguous_pixels']:,} ({amb['global_ambiguous_percentage']}%)")
    print(f"Total Solar Regions:       {reg_sum['total_regions']}")
    print(f"High Confidence Regions:   {reg_sum['high_confidence_count']} ({reg_sum['high_confidence_percentage']}%)")
    print(f"Medium Confidence Regions: {reg_sum['medium_confidence_count']} ({reg_sum['medium_confidence_percentage']}%)")
    print(f"Low Confidence Regions:    {reg_sum['low_confidence_count']} ({reg_sum['low_confidence_percentage']}%)")
    print(f"JSON Report:               {artifacts['json_report'].resolve()}")
    print(f"Solar Prob Map:            {artifacts['solar_probability_map'].resolve()}")
    print(f"Confidence Map:            {artifacts['confidence_map'].resolve()}")
    print(f"Uncertainty Map:           {artifacts['uncertainty_map'].resolve()}")
    print("-" * 54)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="SolarMap-India — Prediction Confidence & Uncertainty Analysis"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--ambiguity-threshold", type=float, default=0.40, help="Uncertainty ambiguity threshold (default: 0.40)")
    parser.add_argument("--high-threshold", type=float, default=0.90, help="High confidence threshold (default: 0.90)")
    parser.add_argument("--medium-threshold", type=float, default=0.70, help="Medium confidence threshold (default: 0.70)")
    parser.add_argument("--min-component-area", type=int, default=20, help="Min component area filter (default: 20)")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda', 'cpu')")

    args = parser.parse_args()

    try:
        dev = get_device(args.device) if args.device else None
        report, artifacts = run_confidence_pipeline(
            image_path=args.image,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            ambiguity_threshold=args.ambiguity_threshold,
            high_threshold=args.high_threshold,
            medium_threshold=args.medium_threshold,
            min_component_area=args.min_component_area,
            device=dev,
        )
        print_summary(report, artifacts)
    except Exception as exc:
        import sys
        print(f"\n[ERROR] Confidence analysis failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

