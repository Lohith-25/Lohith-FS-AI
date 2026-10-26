import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import './GeospatialMap.css';

// Fix Leaflet's default marker asset resolution under Vite
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png';
import markerIcon from 'leaflet/dist/images/marker-icon.png';
import markerShadow from 'leaflet/dist/images/marker-shadow.png';

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconUrl: markerIcon,
  iconRetinaUrl: markerIcon2x,
  shadowUrl: markerShadow,
});

// Tile Layer Definitions (Free, keyless, zero watermarks)
const BASEMAP_TILES = {
  dark: {
    name: 'Obsidian Dark',
    url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap</a> contributors',
    subdomains: 'abc',
    maxZoom: 19,
  },
  satellite: {
    name: 'Satellite Imagery',
    url: 'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attribution:
      'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
    subdomains: 'abc',
    maxZoom: 19,
  },
  street: {
    name: 'Street Map',
    url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap</a> contributors',
    subdomains: 'abc',
    maxZoom: 19,
  },
};

export default function GeospatialMap({ predictionResult, selectedFile }) {
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const tileLayerRef = useRef(null);
  const markerRef = useRef(null);

  const [basemapStyle, setBasemapStyle] = useState('dark');
  const [initError, setInitError] = useState(null);

  const hasCoordinates =
    typeof predictionResult?.latitude === 'number' &&
    typeof predictionResult?.longitude === 'number';

  const latitude = hasCoordinates ? predictionResult.latitude : null;
  const longitude = hasCoordinates ? predictionResult.longitude : null;
  const georefLevel = predictionResult?.georeferencing_level || 'UNREFERENCED';
  const locationName = predictionResult?.location_name || 'Unmapped Location';

  // Initialize and update the Leaflet map instance
  useEffect(() => {
    if (!hasCoordinates || !mapContainerRef.current) {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
        tileLayerRef.current = null;
        markerRef.current = null;
      }
      return;
    }

    try {
      if (!mapInstanceRef.current) {
        // Create Leaflet map centered at image-level scene coordinates
        const map = L.map(mapContainerRef.current, {
          center: [latitude, longitude],
          zoom: 14,
          minZoom: 4,
          maxZoom: 18,
          scrollWheelZoom: true,
          zoomControl: true,
        });

        // Add initial tile layer
        const config = BASEMAP_TILES[basemapStyle] || BASEMAP_TILES.dark;
        const tileLayer = L.tileLayer(config.url, {
          attribution: config.attribution,
          subdomains: config.subdomains,
          maxZoom: config.maxZoom,
        }).addTo(map);

        tileLayerRef.current = tileLayer;

        // Build popup HTML adhering strictly to scientific Level B metadata
        const popupContent = `
          <div class="geo-popup-card">
            <div class="geo-popup-header">
              <h4 class="geo-popup-title">${locationName}</h4>
              <span class="map-hud-badge">${georefLevel}</span>
            </div>
            <div class="geo-popup-grid">
              <span class="geo-popup-label">Latitude:</span>
              <span class="geo-popup-val">${latitude.toFixed(6)}°</span>
              <span class="geo-popup-label">Longitude:</span>
              <span class="geo-popup-val">${longitude.toFixed(6)}°</span>
              <span class="geo-popup-label">Solar Area:</span>
              <span class="geo-popup-val">${predictionResult?.solar_area_pixels?.toLocaleString() || 0} px</span>
              <span class="geo-popup-label">Coverage:</span>
              <span class="geo-popup-val">${predictionResult?.solar_coverage_percent || 0}%</span>
              <span class="geo-popup-label">Solar Arrays:</span>
              <span class="geo-popup-val">${predictionResult?.detected_region_count || 0} regions</span>
            </div>
            <div class="geo-popup-note">
              <strong>Level B Scene Centroid:</strong> Represents approximate image scene center. Does not delineate individual panel polygons.
            </div>
          </div>
        `;

        // Add scene marker
        const marker = L.marker([latitude, longitude], {
          title: `Scene Location: ${latitude.toFixed(6)}, ${longitude.toFixed(6)}`,
        })
          .addTo(map)
          .bindPopup(popupContent, { maxWidth: 300 });

        // Auto-open popup on initial load
        marker.openPopup();

        markerRef.current = marker;
        mapInstanceRef.current = map;

        // Invalidate size on next tick to handle CSS transitions & flex containers
        setTimeout(() => {
          map.invalidateSize();
        }, 150);
      } else {
        // Build popup HTML adhering strictly to scientific Level B metadata
        const popupContent = `
          <div class="geo-popup-card">
            <div class="geo-popup-header">
              <h4 class="geo-popup-title">${locationName}</h4>
              <span class="map-hud-badge">${georefLevel}</span>
            </div>
            <div class="geo-popup-grid">
              <span class="geo-popup-label">Latitude:</span>
              <span class="geo-popup-val">${latitude.toFixed(6)}°</span>
              <span class="geo-popup-label">Longitude:</span>
              <span class="geo-popup-val">${longitude.toFixed(6)}°</span>
              <span class="geo-popup-label">Solar Area:</span>
              <span class="geo-popup-val">${predictionResult?.solar_area_pixels?.toLocaleString() || 0} px</span>
              <span class="geo-popup-label">Coverage:</span>
              <span class="geo-popup-val">${predictionResult?.solar_coverage_percent || 0}%</span>
              <span class="geo-popup-label">Solar Arrays:</span>
              <span class="geo-popup-val">${predictionResult?.detected_region_count || 0} regions</span>
            </div>
            <div class="geo-popup-note">
              <strong>Level B Scene Centroid:</strong> Represents approximate image scene center. Does not delineate individual panel polygons.
            </div>
          </div>
        `;

        // Update existing map view when coordinates or results change
        const map = mapInstanceRef.current;
        map.setView([latitude, longitude], map.getZoom(), { animate: true });

        if (markerRef.current) {
          markerRef.current.setLatLng([latitude, longitude]);
          markerRef.current.setPopupContent(popupContent);
        }

        setTimeout(() => {
          map.invalidateSize();
        }, 100);
      }
    } catch (err) {
      setTimeout(() => {
        setInitError(`Leaflet map initialization failed: ${err.message}`);
      }, 0);
    }

    return () => {
      // Map instance is preserved across minor updates, cleaned up on component unmount
    };
  }, [hasCoordinates, latitude, longitude, georefLevel, locationName, predictionResult, basemapStyle]);

  // Handle basemap style changes
  useEffect(() => {
    if (!mapInstanceRef.current || !tileLayerRef.current) return;

    const map = mapInstanceRef.current;
    map.removeLayer(tileLayerRef.current);

    const config = BASEMAP_TILES[basemapStyle] || BASEMAP_TILES.dark;
    const newTileLayer = L.tileLayer(config.url, {
      attribution: config.attribution,
      subdomains: config.subdomains,
      maxZoom: config.maxZoom,
    }).addTo(map);

    tileLayerRef.current = newTileLayer;
  }, [basemapStyle]);

  // Clean up Leaflet on component unmount
  useEffect(() => {
    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
        tileLayerRef.current = null;
        markerRef.current = null;
      }
    };
  }, []);

  const handleCenterOnScene = () => {
    if (mapInstanceRef.current && hasCoordinates) {
      mapInstanceRef.current.setView([latitude, longitude], 14, { animate: true });
      if (markerRef.current) {
        markerRef.current.openPopup();
      }
    }
  };

  const handleZoomIndiaOverview = () => {
    if (mapInstanceRef.current) {
      // Center of India overview
      mapInstanceRef.current.setView([21.5, 78.0], 5, { animate: true });
    }
  };

  return (
    <div className="geospatial-section" id="geospatial-map-section">
      {/* Component Header */}
      <div className="geospatial-header">
        <div className="geospatial-title-group">
          <h2 className="geospatial-title">
            <span className="geospatial-title-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <line x1="2" y1="12" x2="22" y2="12" />
                <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
              </svg>
            </span>
            Geospatial Scene Mapping
          </h2>
          <p className="geospatial-subtitle">
            Interactive geographic visualization of satellite scene center coordinates (Level B Georeferencing).
          </p>
        </div>

        {hasCoordinates && (
          <div className="map-actions-bar" role="toolbar" aria-label="Map Controls">
            {/* Basemap Switcher */}
            <button
              type="button"
              className={`map-control-btn ${basemapStyle === 'dark' ? 'active' : ''}`}
              onClick={() => setBasemapStyle('dark')}
              aria-label="Switch to Obsidian Dark Basemap"
            >
              Dark Basemap
            </button>
            <button
              type="button"
              className={`map-control-btn ${basemapStyle === 'satellite' ? 'active' : ''}`}
              onClick={() => setBasemapStyle('satellite')}
              aria-label="Switch to High-Resolution Satellite Basemap"
            >
              Satellite
            </button>
            <button
              type="button"
              className={`map-control-btn ${basemapStyle === 'street' ? 'active' : ''}`}
              onClick={() => setBasemapStyle('street')}
              aria-label="Switch to OpenStreetMap Standard Basemap"
            >
              Street Basemap
            </button>

            {/* Recenter Buttons */}
            <button
              type="button"
              className="map-control-btn"
              onClick={handleCenterOnScene}
              aria-label="Center map on scene coordinates"
            >
              Center on Scene
            </button>
            <button
              type="button"
              className="map-control-btn"
              onClick={handleZoomIndiaOverview}
              aria-label="Zoom to India Overview"
            >
              India Overview
            </button>
          </div>
        )}
      </div>

      {/* Map Display or Informative Empty State */}
      {hasCoordinates ? (
        <div className="map-frame-wrapper" role="region" aria-label="Interactive Map View">
          <div
            ref={mapContainerRef}
            className={`leaflet-map-element basemap-${basemapStyle}`}
          />

          {/* Floating HUD Panel */}
          <div className="map-hud-overlay" aria-live="polite">
            <div className="map-hud-row">
              <span className="map-hud-label">Georeferencing:</span>
              <span className="map-hud-badge">{georefLevel}</span>
            </div>
            <div className="map-hud-row">
              <span className="map-hud-label">Scene Center:</span>
              <span className="map-hud-value">
                {latitude.toFixed(4)}°N, {longitude.toFixed(4)}°E
              </span>
            </div>
            <div className="map-hud-row">
              <span className="map-hud-label">Location:</span>
              <span className="map-hud-value">{locationName}</span>
            </div>
            <div className="map-hud-row">
              <span className="map-hud-label">Footprint Type:</span>
              <span className="map-hud-value">Image-Level Scene</span>
            </div>
          </div>
        </div>
      ) : (
        <div className="geo-empty-state" role="status">
          <div className="geo-empty-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="36" height="36" fill="none" stroke="currentColor" strokeWidth="1.5">
              <circle cx="12" cy="12" r="10" />
              <path d="m4.93 4.93 14.14 14.14" />
            </svg>
          </div>
          <span className="geo-empty-badge">Unreferenced Scene</span>
          <h3 className="geo-empty-title">Geographic Coordinates Unavailable</h3>
          <p className="geo-empty-desc">
            {selectedFile
              ? `The uploaded image "${selectedFile.name}" does not contain embedded GPS EXIF tags, world files (.tfw/.pgw), or a matching record in the dataset manifest.`
              : 'No active prediction or image selected. Upload an aerial image or load the benchmark sample to inspect geographic positioning.'}
          </p>
          <p className="geo-empty-desc">
            In accordance with SolarMap-India scientific integrity standards, fallback geographic coordinates are{' '}
            <strong>strictly not fabricated</strong>.
          </p>
        </div>
      )}

      {initError && (
        <div className="error-alert" role="alert">
          <span>{initError}</span>
        </div>
      )}

      {/* Metadata Facts Grid */}
      {hasCoordinates && (
        <div className="geo-metadata-grid">
          <div className="geo-meta-card">
            <span className="geo-meta-label">Geographic Centroid</span>
            <span className="geo-meta-value">
              {latitude.toFixed(5)}°, {longitude.toFixed(5)}°
            </span>
            <span className="geo-meta-desc">WGS84 ellipsoidal coordinates</span>
          </div>

          <div className="geo-meta-card">
            <span className="geo-meta-label">Classification</span>
            <span className="geo-meta-value">{georefLevel}</span>
            <span className="geo-meta-desc">Per-image coordinate cataloging</span>
          </div>

          <div className="geo-meta-card">
            <span className="geo-meta-label">Administrative Region</span>
            <span className="geo-meta-value">{locationName}</span>
            <span className="geo-meta-desc">Western India Solar Belt</span>
          </div>

          <div className="geo-meta-card">
            <span className="geo-meta-label">Pixel vs Ground Status</span>
            <span className="geo-meta-value">Uncalibrated GSD</span>
            <span className="geo-meta-desc">Pixel segmentation only</span>
          </div>
        </div>
      )}

      {/* Scientific Limitation Disclaimer */}
      <section className="scientific-disclaimer-card" aria-labelledby="geospatial-disclaimer-heading">
        <div className="disclaimer-icon-box" aria-hidden="true">
          <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
            <line x1="12" y1="9" x2="12" y2="13" />
            <line x1="12" y1="17" x2="12.01" y2="17" />
          </svg>
        </div>

        <div className="disclaimer-body">
          <h3 id="geospatial-disclaimer-heading" className="disclaimer-title">
            Scientific Accuracy & Georeferencing Limitations (Level B)
          </h3>
          <p className="disclaimer-text">
            SolarMap-India adheres to rigorous geospatial data integrity principles. The interactive map above is subject to the following scientific constraints:
          </p>
          <ul className="disclaimer-list">
            <li>
              <strong>Scene-Level Approximation:</strong> Scene markers represent the center of the satellite scene tile, not the exact geographic position of individual solar panels.
            </li>
            <li>
              <strong>No Affine Ground Transformation:</strong> Input PNG/JPEG tiles do not contain affine transformation matrices (such as GeoTIFF ModelTransformation tags or world files). Consequently, detected pixel polygons cannot be projected into absolute geographic coordinates.
            </li>
            <li>
              <strong>No Fabricated Extrapolations:</strong> Precise sub-meter installation boundaries or physical metric areas are never inferred without verified Ground Sampling Distance (GSD) calibration.
            </li>
          </ul>
        </div>
      </section>
    </div>
  );
}
