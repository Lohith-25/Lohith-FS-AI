"""
SolarMap-India — Full-Resolution Probability, Confidence & Uncertainty Maps.

Generates:
1. Solar probability map: P(solar) ∈ [0.0, 1.0]
2. Prediction confidence map: max(P(bg), P(solar)) ∈ [0.5, 1.0]
3. Prediction uncertainty map: 1 - confidence ∈ [0.0, 0.5]
Preserves original spatial dimensions with calibrated colorbars and labels.
"""

from pathlib import Path
from typing import Optional, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def save_heatmap_visualization(
    data: np.ndarray,
    title: str,
    colorbar_label: str,
    output_path: Union[str, Path],
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "inferno",
) -> Path:
    """
    Renders and saves a scientific heatmap with an explicit colorbar.

    Args:
        data: 2D numpy float array.
        title: Plot title.
        colorbar_label: Label for the colorbar axis.
        output_path: Destination PNG file path.
        vmin: Minimum value for color scaling.
        vmax: Maximum value for color scaling.
        cmap: Matplotlib colormap name.

    Returns:
        Path to the saved visualization image.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    h, w = data.shape
    aspect = w / h
    fig_w = 7.0
    fig_h = max(4.0, fig_w / aspect)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight", dpi=150)
    plt.close(fig)

    return out_file


def generate_all_confidence_maps(
    solar_prob: np.ndarray,
    confidence: np.ndarray,
    uncertainty: np.ndarray,
    stem: str,
    output_dir: Union[str, Path],
) -> dict:
    """
    Generates and saves the three standard probability and confidence visualizations.

    Returns:
        dict containing paths to:
        - solar_probability_map
        - confidence_map
        - uncertainty_map
    """
    maps_dir = Path(output_dir)
    maps_dir.mkdir(parents=True, exist_ok=True)

    prob_path = maps_dir / f"{stem}_solar_probability.png"
    conf_path = maps_dir / f"{stem}_confidence.png"
    unc_path = maps_dir / f"{stem}_uncertainty.png"

    save_heatmap_visualization(
        data=solar_prob,
        title="Model Output Probability — P(Solar)",
        colorbar_label="P(Solar Panel)",
        output_path=prob_path,
        vmin=0.0,
        vmax=1.0,
        cmap="inferno",
    )

    save_heatmap_visualization(
        data=confidence,
        title="Prediction Confidence — max(P(bg), P(solar))",
        colorbar_label="Confidence [0.5, 1.0]",
        output_path=conf_path,
        vmin=0.5,
        vmax=1.0,
        cmap="viridis",
    )

    save_heatmap_visualization(
        data=uncertainty,
        title="Probability-Derived Prediction Uncertainty — (1 - Confidence)",
        colorbar_label="Uncertainty [0.0, 0.5]",
        output_path=unc_path,
        vmin=0.0,
        vmax=0.5,
        cmap="cividis",
    )

    return {
        "solar_probability_map": prob_path,
        "confidence_map": conf_path,
        "uncertainty_map": unc_path,
    }
