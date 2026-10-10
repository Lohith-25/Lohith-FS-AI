"""
SolarMap-India — Scientific Assumptions & Energy Configuration.

Defines the scientific baselines, validation ranges, qualitative limitations,
and user-configurable parameters for photovoltaic potential modeling.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# Scientific benchmarks from peer-reviewed literature and standards (IEC 61724, NREL)
STANDARD_EFFICIENCY_BENCHMARK = 0.20  # Typical modern commercial monocrystalline Si module (~20%)
STANDARD_PERFORMANCE_RATIO_BENCHMARK = 0.75  # Typical tropical rooftop balance-of-system PR (~75%)

MIN_PLAUSIBLE_EFFICIENCY = 0.05
MAX_PLAUSIBLE_EFFICIENCY = 0.35

MIN_PLAUSIBLE_PR = 0.40
MAX_PLAUSIBLE_PR = 1.00


@dataclass
class EnergyModelConfig:
    """
    Configuration parameters for the first-order photovoltaic energy model.
    Values default strictly to None so that assumptions are never silently applied.
    """
    module_efficiency: Optional[float] = None
    performance_ratio: Optional[float] = None
    efficiency_source_note: Optional[str] = None
    pr_source_note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def create_default(cls) -> "EnergyModelConfig":
        """Returns the unconfigured baseline where assumptions are explicit nulls."""
        return cls(
            module_efficiency=None,
            performance_ratio=None,
            efficiency_source_note="No module efficiency provided; explicit assumption required.",
            pr_source_note="No performance ratio provided; explicit assumption required.",
        )

    @classmethod
    def create_benchmark_scenario(
        cls,
        efficiency: float = STANDARD_EFFICIENCY_BENCHMARK,
        performance_ratio: float = STANDARD_PERFORMANCE_RATIO_BENCHMARK,
    ) -> "EnergyModelConfig":
        """
        Creates an explicit benchmark scenario documenting scientific justifications.
        """
        return cls(
            module_efficiency=efficiency,
            performance_ratio=performance_ratio,
            efficiency_source_note=(
                f"Standard commercial crystalline silicon PV module efficiency ({efficiency*100:.1f}%), "
                "based on IEC 61215 / NREL PV baseline specifications."
            ),
            pr_source_note=(
                f"Benchmark rooftop Performance Ratio ({performance_ratio*100:.1f}%), accounting for "
                "tropical thermal derating, inverter clipping, and balance-of-system losses (IEC 61724)."
            ),
        )


@dataclass
class ScientificAssumptions:
    """Documents the comprehensive physical assumptions and limitations of the model."""
    formula: str = "E = A * GHI * eta * PR"
    model_name: str = "First-Order Photovoltaic Energy Potential Model"
    energy_label: str = "Estimated Solar Energy Potential (NOT Actual Energy Generated)"
    limitations: List[str] = field(default_factory=lambda: [
        "Spatial Scale Uncertainty: Ground Sampling Distance (GSD) must be authenticated from original imagery metadata.",
        "Segmentation Boundary Precision: Model outputs are 2D pixel classifications subject to edge uncertainty and false positives.",
        "Solar Resource Grid Resolution: NASA POWER climatology has a native 1.0 x 1.0 degree spatial grid (~110 km), not localized micro-climate irradiance.",
        "Module Efficiency Variance: Actual PV panel efficiency depends on cell technology (mono-Si, poly-Si, thin-film) and nominal STC rating.",
        "Performance Ratio Derating: True system performance depends on site-specific ambient temperature, wiring losses, and inverter efficiency.",
        "Tilt and Azimuth Geometry: 2D satellite views do not measure panel inclination angle or compass azimuth relative to solar path.",
        "Local Shading & Soiling: Trees, parapets, dust accumulation, and adjacent structures create localized yield degradation not modeled in 2D nadir views.",
        "Degradation & Availability: PV modules experience annual degradation (~0.5-0.8%/year) and downtime which are not factored into first-year potential.",
    ])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
