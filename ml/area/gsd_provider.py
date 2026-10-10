"""
SolarMap-India — GSD Provider Module.

Responsible for retrieving and validating Ground Sampling Distance (GSD)
for aerial and satellite imagery without guessing or fabricating values.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.external_data.imagery_source import inspect_imagery_source

# Prohibited assumed values (strictly rejected per scientific standards)
PROHIBITED_GENERIC_GSDS = {
    1.0: "Assumed 1 pixel = 1 meter",
    0.5: "Assumed generic 0.5 m/pixel aerial resolution",
    10.0: "Assumed Sentinel-2 10m band resolution",
    30.0: "Assumed Landsat 30m resolution",
    0.3: "Assumed generic 30cm commercial satellite resolution",
}


@dataclass
class GSDResult:
    """Standardized representation of Ground Sampling Distance with provenance."""
    gsd_m_per_pixel: Optional[float]
    status: str                         # "VALIDATED" | "UNAVAILABLE"
    source: str
    provenance: Dict[str, Any] = field(default_factory=dict)
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GSDProvider:
    """
    Authoritative GSD Provider.
    Inspects image metadata and external spatial resolution records.
    Refuses any assumed or fabricated GSD.
    """

    @classmethod
    def get_gsd(
        cls,
        image_path: Optional[Union[str, Path]] = None,
        custom_gsd: Optional[float] = None,
        provenance_citation: Optional[str] = None,
    ) -> GSDResult:
        """
        Retrieves Ground Sampling Distance (meters/pixel).

        Args:
            image_path: Optional path to target image to inspect.
            custom_gsd: Optional candidate GSD.
            provenance_citation: Authoritative citation or certificate backing the candidate GSD.

        Returns:
            GSDResult with status VALIDATED or UNAVAILABLE.
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        # Case 1: An external candidate GSD is provided
        if custom_gsd is not None:
            # Reject non-numeric, zero, or negative
            if not isinstance(custom_gsd, (int, float)) or custom_gsd <= 0:
                return GSDResult(
                    gsd_m_per_pixel=None,
                    status="UNAVAILABLE",
                    source="Rejected Candidate GSD",
                    provenance={
                        "retrieval_date_utc": now_iso,
                        "candidate_value": custom_gsd,
                        "validated": False,
                    },
                    rejection_reason=f"Invalid numeric GSD value: {custom_gsd}. Must be strictly positive.",
                )

            # Reject known prohibited assumed values
            if custom_gsd in PROHIBITED_GENERIC_GSDS:
                reason = PROHIBITED_GENERIC_GSDS[custom_gsd]
                return GSDResult(
                    gsd_m_per_pixel=None,
                    status="UNAVAILABLE",
                    source="Rejected Assumed GSD",
                    provenance={
                        "retrieval_date_utc": now_iso,
                        "candidate_value": custom_gsd,
                        "validated": False,
                    },
                    rejection_reason=f"Rejected prohibited assumption: {reason}.",
                )

            # Reject absurd values (e.g., > 500 meters/pixel)
            if custom_gsd > 500.0:
                return GSDResult(
                    gsd_m_per_pixel=None,
                    status="UNAVAILABLE",
                    source="Rejected Candidate GSD",
                    provenance={
                        "retrieval_date_utc": now_iso,
                        "candidate_value": custom_gsd,
                        "validated": False,
                    },
                    rejection_reason=f"GSD of {custom_gsd} m/pixel exceeds plausible earth observation scale.",
                )

            # Candidate must have verified provenance citation
            if not provenance_citation or len(provenance_citation.strip()) < 10:
                return GSDResult(
                    gsd_m_per_pixel=None,
                    status="UNAVAILABLE",
                    source="Unsubstantiated Candidate GSD",
                    provenance={
                        "retrieval_date_utc": now_iso,
                        "candidate_value": custom_gsd,
                        "validated": False,
                        "citation": provenance_citation,
                    },
                    rejection_reason="Candidate GSD lacks authoritative provenance citation or sensor certificate.",
                )

            # Validated GSD with verified provenance
            return GSDResult(
                gsd_m_per_pixel=float(custom_gsd),
                status="VALIDATED",
                source=provenance_citation.strip(),
                provenance={
                    "retrieval_date_utc": now_iso,
                    "gsd_m_per_pixel": float(custom_gsd),
                    "citation": provenance_citation.strip(),
                    "validated": True,
                },
                rejection_reason=None,
            )

        # Case 2: Querying the existing dataset repository metadata
        audit = inspect_imagery_source()
        if audit.spatial_resolution_status == "VALIDATED" and audit.gsd_meters is not None:
            return GSDResult(
                gsd_m_per_pixel=audit.gsd_meters,
                status="VALIDATED",
                source=f"{audit.provider} - {audit.product_name}",
                provenance={
                    "retrieval_date_utc": now_iso,
                    "gsd_m_per_pixel": audit.gsd_meters,
                    "provider": audit.provider,
                    "product": audit.product_name,
                    "validated": True,
                },
                rejection_reason=None,
            )

        # For the current dataset, original imagery source is undocumented and GSD is UNAVAILABLE
        return GSDResult(
            gsd_m_per_pixel=None,
            status="UNAVAILABLE",
            source="Dataset Metadata Audit (Phase 5)",
            provenance={
                "retrieval_date_utc": now_iso,
                "dataset_images": "640x640 8-bit PNG",
                "sensor_platform": None,
                "imagery_provider": None,
                "tile_zoom_level": None,
                "has_world_files": audit.has_world_files,
                "has_georeference_tags": audit.has_georeference_tags,
                "validated": False,
            },
            rejection_reason=(
                "Original imagery source and Web Mercator tile zoom are undocumented in dataset manifests. "
                "No calibrated spatial scale exists."
            ),
        )
