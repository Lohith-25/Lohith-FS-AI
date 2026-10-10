"""
SolarMap-India — Image-to-Coordinate Mapping & Validation.

Parses, validates, and matches image filenames to authoritative
geographic coordinates from EI_train_data(Sheet1).csv and splits/dataset_manifest.csv.
"""

from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CSV_PATH = PROJECT_ROOT / "dataset" / "EI_train_data(Sheet1).csv"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "splits" / "dataset_manifest.csv"

# Geographic reference bounding box for mainland India & islands
INDIA_LAT_BOUNDS = (8.0, 38.0)
INDIA_LON_BOUNDS = (68.0, 98.0)


@dataclass
class CoordinateRecord:
    """Stores verified geospatial metadata for an image sample."""
    sampleid: int
    filename: str
    latitude: float
    longitude: float
    has_solar: int
    is_valid_geo_range: bool
    is_within_india_bounds: bool


class GeospatialCatalog:
    """
    Catalog indexing the relationship between images, sample IDs, and coordinates.
    """

    def __init__(
        self,
        csv_path: Union[str, Path] = DEFAULT_CSV_PATH,
        manifest_path: Union[str, Path] = DEFAULT_MANIFEST_PATH,
    ):
        self.csv_path = Path(csv_path)
        self.manifest_path = Path(manifest_path)
        self._df_csv: Optional[pd.DataFrame] = None
        self._df_manifest: Optional[pd.DataFrame] = None
        self._sample_lookup: Dict[int, CoordinateRecord] = {}
        self._filename_lookup: Dict[str, CoordinateRecord] = {}
        self._load_and_validate()

    def _load_and_validate(self) -> None:
        """Loads CSVs, validates coordinates, and builds deterministic lookups."""
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Coordinate CSV missing: {self.csv_path.resolve()}")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Dataset manifest missing: {self.manifest_path.resolve()}")

        self._df_csv = pd.read_csv(self.csv_path)
        self._df_manifest = pd.read_csv(self.manifest_path)

        # 1. Row count and duplicate checks
        if self._df_csv["sampleid"].duplicated().any():
            dups = self._df_csv[self._df_csv["sampleid"].duplicated()]["sampleid"].tolist()
            raise ValueError(f"Duplicate sample IDs detected in CSV: {dups}")

        if self._df_manifest["sampleid"].duplicated().any():
            dups = self._df_manifest[self._df_manifest["sampleid"].duplicated()]["sampleid"].tolist()
            raise ValueError(f"Duplicate sample IDs detected in manifest: {dups}")

        # 2. Merge manifest and CSV on sampleid
        merged = pd.merge(
            self._df_csv,
            self._df_manifest[["sampleid", "filename"]],
            on="sampleid",
            how="inner",
        )

        for _, row in merged.iterrows():
            sid = int(row["sampleid"])
            fn = str(row["filename"])
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            solar = int(row["has_solar"])

            # Geographic range checks [-90, 90], [-180, 180]
            valid_geo = (-90.0 <= lat <= 90.0) and (-180.0 <= lon <= 180.0)
            in_india = (
                INDIA_LAT_BOUNDS[0] <= lat <= INDIA_LAT_BOUNDS[1]
                and INDIA_LON_BOUNDS[0] <= lon <= INDIA_LON_BOUNDS[1]
            )

            rec = CoordinateRecord(
                sampleid=sid,
                filename=fn,
                latitude=lat,
                longitude=lon,
                has_solar=solar,
                is_valid_geo_range=valid_geo,
                is_within_india_bounds=in_india,
            )

            self._sample_lookup[sid] = rec
            self._filename_lookup[fn] = rec

    @property
    def total_records(self) -> int:
        return len(self._sample_lookup)

    def extract_sample_id_from_filename(self, filename: str) -> Optional[int]:
        """
        Extracts integer sampleid from standard filenames:
        Examples: '768.0_1.0.png' -> 768, '0960_1.png' -> 960, '100.0_1.0.png' -> 100.
        """
        clean_name = Path(filename).name
        # Match pattern <sampleid>[.0]_1[.0].png
        m = re.match(r"^(\d+)(?:\.0)?_1(?:\.0)?\.(?:png|jpg|jpeg)$", clean_name, re.IGNORECASE)
        if m:
            return int(m.group(1))

        # Direct numeric stem
        m2 = re.match(r"^(\d+)\.(?:png|jpg|jpeg)$", clean_name, re.IGNORECASE)
        if m2:
            return int(m2.group(1))

        return None

    def lookup(self, identifier: Union[str, int, Path]) -> Optional[CoordinateRecord]:
        """
        Looks up coordinate record by filename or integer sample ID.
        """
        if isinstance(identifier, int):
            return self._sample_lookup.get(identifier)

        path_obj = Path(str(identifier))
        filename = path_obj.name

        # Direct filename match
        if filename in self._filename_lookup:
            return self._filename_lookup[filename]

        # Parsed sample ID match
        sid = self.extract_sample_id_from_filename(filename)
        if sid is not None and sid in self._sample_lookup:
            return self._sample_lookup[sid]

        return None

    def audit_bounds(self) -> dict:
        """Audits overall coordinate statistics and boundary anomalies."""
        latitudes = [r.latitude for r in self._sample_lookup.values()]
        longitudes = [r.longitude for r in self._sample_lookup.values()]

        anomalies = [r for r in self._sample_lookup.values() if not r.is_within_india_bounds]

        return {
            "total_samples": len(self._sample_lookup),
            "min_latitude": float(min(latitudes)),
            "max_latitude": float(max(latitudes)),
            "min_longitude": float(min(longitudes)),
            "max_longitude": float(max(longitudes)),
            "anomalous_out_of_bounds_count": len(anomalies),
            "anomalous_samples": [
                {
                    "sampleid": a.sampleid,
                    "filename": a.filename,
                    "latitude": a.latitude,
                    "longitude": a.longitude,
                    "has_solar": a.has_solar,
                }
                for a in anomalies
            ],
        }
