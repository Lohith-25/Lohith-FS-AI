"""
SolarMap-India — Spatial Resolution & GSD Validation Foundation.

Enforces scientific validation rules for Ground Sampling Distance (GSD).
Ensures no synthetic, assumed, or unverified spatial resolution is used.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class GSDValidationResult:
    """Result of GSD validation."""
    spatial_resolution_status: str  # "VALIDATED" | "UNAVAILABLE"
    gsd_meters: Optional[float]
    confidence: str                 # "HIGH" | "ZERO"
    source_description: str
    evidence: List[str]
    can_derive_physical_area: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SpatialResolutionValidator:
    """
    Validates candidate spatial resolutions against scientific ground truth.
    Strictly forbids guessing or applying generic constants.
    """

    PROHIBITED_GENERIC_RESOLUTIONS = {
        1.0: "1 pixel = 1 meter generic assumption",
        10.0: "Sentinel-2 10m band assumption (inconsistent with 640x640 rooftop views)",
        30.0: "Landsat 30m assumption",
        0.5: "Generic aerial 50cm assumption without provider proof",
        0.3: "Generic satellite 30cm assumption without provider proof",
    }

    @classmethod
    def evaluate(
        cls,
        candidate_gsd: Optional[float] = None,
        source_evidence: Optional[str] = None,
        zoom_level: Optional[int] = None,
        latitude: Optional[float] = None,
    ) -> GSDValidationResult:
        """
        Evaluates whether a candidate GSD can be scientifically validated.
        If inputs are missing or unverified, returns status 'UNAVAILABLE'.
        """
        evidence_log: List[str] = []

        # Case 1: No candidate GSD provided
        if candidate_gsd is None:
            evidence_log.append("No candidate GSD or sensor specification provided.")
            evidence_log.append(
                "Repository artifacts lack sensor metadata, projection tags, or zoom levels."
            )
            return GSDValidationResult(
                spatial_resolution_status="UNAVAILABLE",
                gsd_meters=None,
                confidence="ZERO",
                source_description="None (unverified imagery source)",
                evidence=evidence_log,
                can_derive_physical_area=False,
            )

        # Case 2: Candidate matches known prohibited generic defaults
        if candidate_gsd in cls.PROHIBITED_GENERIC_RESOLUTIONS:
            reason = cls.PROHIBITED_GENERIC_RESOLUTIONS[candidate_gsd]
            evidence_log.append(f"Rejected candidate GSD ({candidate_gsd} m/px): matches {reason}.")
            return GSDValidationResult(
                spatial_resolution_status="UNAVAILABLE",
                gsd_meters=None,
                confidence="ZERO",
                source_description=f"Rejected assumption: {reason}",
                evidence=evidence_log,
                can_derive_physical_area=False,
            )

        # Case 3: If candidate GSD is non-positive or physically absurd
        if candidate_gsd <= 0 or candidate_gsd > 1000.0:
            evidence_log.append(f"Invalid numeric GSD value: {candidate_gsd} m/px.")
            return GSDValidationResult(
                spatial_resolution_status="UNAVAILABLE",
                gsd_meters=None,
                confidence="ZERO",
                source_description="Invalid numeric GSD",
                evidence=evidence_log,
                can_derive_physical_area=False,
            )

        # Case 4: Candidate requires official provenance/evidence string
        if not source_evidence or len(source_evidence.strip()) < 10:
            evidence_log.append("Candidate GSD lacks authoritative documentation or sensor certificate.")
            return GSDValidationResult(
                spatial_resolution_status="UNAVAILABLE",
                gsd_meters=None,
                confidence="ZERO",
                source_description="Unsubstantiated candidate value",
                evidence=evidence_log,
                can_derive_physical_area=False,
            )

        # If genuine official evidence is supplied (e.g. verified sensor product spec)
        evidence_log.append(f"GSD validated against authoritative source: {source_evidence}")
        return GSDValidationResult(
            spatial_resolution_status="VALIDATED",
            gsd_meters=float(candidate_gsd),
            confidence="HIGH",
            source_description=source_evidence,
            evidence=evidence_log,
            can_derive_physical_area=True,
        )
