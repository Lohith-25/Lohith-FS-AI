"""
SolarMap-India — External Spatial Resolution & Solar Resource Package.

Phase 5 foundation providing:
- Imagery source investigation and evidence recording.
- Spatial resolution / GSD verification (strictly no-fabrication).
- Solar resource data acquisition via NASA POWER Climatology.
- Scientific provenance tracking and deterministic caching.
- External data readiness reporting.
"""

from ml.external_data.imagery_source import ImagerySourceAudit, inspect_imagery_source
from ml.external_data.spatial_resolution import SpatialResolutionValidator, GSDValidationResult
from ml.external_data.provenance import DataProvenance
from ml.external_data.validation import ExternalDataValidator
from ml.external_data.solar_resource import SolarResourceService, SolarResourceResult

__all__ = [
    "ImagerySourceAudit",
    "inspect_imagery_source",
    "SpatialResolutionValidator",
    "GSDValidationResult",
    "DataProvenance",
    "ExternalDataValidator",
    "SolarResourceService",
    "SolarResourceResult",
]
