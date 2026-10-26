import React from 'react';

export default function MetricsDashboard({ result }) {
  if (!result) return null;

  const {
    image_width,
    image_height,
    total_image_pixels,
    solar_area_pixels,
    solar_coverage_percent,
    detected_region_count,
    largest_region_area_pixels,
    smallest_region_area_pixels,
    mean_region_area_pixels,
    model = 'HalfUNet',
    prediction_id,
    applied_threshold,
    applied_min_component_area,
    saved_to_history = false,
    processing_strategy = 'direct',
  } = result;

  const formatNumber = (val) => {
    if (val === null || val === undefined) return 'N/A';
    return Number(val).toLocaleString();
  };

  const formatDecimal = (val, places = 2) => {
    if (val === null || val === undefined) return 'N/A';
    return Number(val).toLocaleString(undefined, {
      minimumFractionDigits: places,
      maximumFractionDigits: places,
    });
  };

  return (
    <section className="card metrics-card" aria-labelledby="metrics-heading">
      <div className="card-header">
        <div>
          <div className="meta-badge-row">
            <span className="badge badge-success">Inference Successful</span>
            <span className="badge badge-mono">Model: {model}</span>
            {processing_strategy && (
              <span className="badge badge-info" title="Inference resolution strategy">
                Strategy: {processing_strategy.toUpperCase()}
              </span>
            )}
            {applied_threshold !== undefined && (
              <span className="badge badge-mono">Threshold: {applied_threshold}</span>
            )}
            {applied_min_component_area !== undefined && (
              <span className="badge badge-mono">Min Area: {applied_min_component_area} px</span>
            )}
            {prediction_id && (
              <span className="badge badge-uuid" title={prediction_id}>
                ID: {prediction_id.slice(0, 8)}...
              </span>
            )}
            {saved_to_history ? (
              <span className="badge badge-success" title="Persisted to MongoDB history">
                ✓ Saved to DB
              </span>
            ) : (
              <span className="badge badge-warning" title="MongoDB offline or disabled">
                Persistence Offline
              </span>
            )}
          </div>
          <h2 id="metrics-heading" className="card-title">Pixel-Level Analysis Dashboard</h2>
          <p className="card-subtitle">
            Quantitative segmentation metrics computed directly from Half U-Net inference.
            Note: Detected connected regions represent contiguous solar pixel clusters and are not necessarily individual physical solar panels.
          </p>
        </div>
      </div>

      {/* Primary Highlight: Solar Coverage Gauge & Progress Bar */}
      <div className="primary-metric-banner">
        <div className="coverage-visual">
          <div className="coverage-percent-display">
            <span className="coverage-num">{formatDecimal(solar_coverage_percent, 4)}</span>
            <span className="coverage-unit">%</span>
          </div>
          <span className="coverage-label">Rooftop Solar Coverage</span>
        </div>

        <div className="coverage-bar-container">
          <div className="coverage-bar-track">
            <div
              className="coverage-bar-fill"
              style={{ width: `${Math.min(Math.max(solar_coverage_percent || 0, 0), 100)}%` }}
            />
          </div>
          <div className="coverage-bar-scale">
            <span>0%</span>
            <span>25%</span>
            <span>50%</span>
            <span>75%</span>
            <span>100%</span>
          </div>
        </div>
      </div>

      {/* Grid of Key Metrics */}
      <div className="metrics-grid">
        {/* Metric 1: Solar Panel Pixels */}
        <div className="metric-box metric-accent-solar">
          <div className="metric-header">
            <span className="metric-title">Predicted Solar Pixels</span>
            <span className="metric-icon">☀️</span>
          </div>
          <div className="metric-value">{formatNumber(solar_area_pixels)}</div>
          <div className="metric-meta">
            Segmented foreground solar pixels
          </div>
        </div>

        {/* Metric 2: Total Image Pixels */}
        <div className="metric-box metric-accent-cyan">
          <div className="metric-header">
            <span className="metric-title">Total Image Pixels</span>
            <span className="metric-icon">🖼️</span>
          </div>
          <div className="metric-value">{formatNumber(total_image_pixels)}</div>
          <div className="metric-meta">
            Image canvas resolution
          </div>
        </div>

        {/* Metric 3: Dimensions */}
        <div className="metric-box">
          <div className="metric-header">
            <span className="metric-title">Image Dimensions</span>
            <span className="metric-icon">📐</span>
          </div>
          <div className="metric-value">
            {formatNumber(image_width)} × {formatNumber(image_height)}
          </div>
          <div className="metric-meta">
            Width × Height (px)
          </div>
        </div>

        {/* Metric 4: Connected Regions */}
        <div className="metric-box metric-accent-emerald">
          <div className="metric-header">
            <span className="metric-title">Connected Regions Detected</span>
            <span className="metric-icon">🔗</span>
          </div>
          <div className="metric-value">{formatNumber(detected_region_count)}</div>
          <div className="metric-meta">
            Contiguous solar clusters (not individual panels)
          </div>
        </div>

        {/* Metric 5: Largest Region */}
        <div className="metric-box">
          <div className="metric-header">
            <span className="metric-title">Largest Region Area</span>
            <span className="metric-icon">🔺</span>
          </div>
          <div className="metric-value">
            {largest_region_area_pixels !== null && largest_region_area_pixels !== undefined ? (
              <>
                {formatNumber(largest_region_area_pixels)} <span className="val-unit">px</span>
              </>
            ) : (
              'N/A'
            )}
          </div>
          <div className="metric-meta">
            Peak contiguous solar array
          </div>
        </div>

        {/* Metric 6: Smallest Region */}
        <div className="metric-box">
          <div className="metric-header">
            <span className="metric-title">Smallest Region Area</span>
            <span className="metric-icon">🔻</span>
          </div>
          <div className="metric-value">
            {smallest_region_area_pixels !== null && smallest_region_area_pixels !== undefined ? (
              <>
                {formatNumber(smallest_region_area_pixels)} <span className="val-unit">px</span>
              </>
            ) : (
              'N/A'
            )}
          </div>
          <div className="metric-meta">
            Smallest detected solar patch
          </div>
        </div>

        {/* Metric 7: Mean Region Area */}
        <div className="metric-box">
          <div className="metric-header">
            <span className="metric-title">Mean Region Area</span>
            <span className="metric-icon">📊</span>
          </div>
          <div className="metric-value">
            {mean_region_area_pixels !== null && mean_region_area_pixels !== undefined ? (
              <>
                {formatDecimal(mean_region_area_pixels, 2)} <span className="val-unit">px</span>
              </>
            ) : (
              'N/A'
            )}
          </div>
          <div className="metric-meta">
            Average solar cluster area
          </div>
        </div>

        {/* Metric 8: Coverage Ratio */}
        <div className="metric-box">
          <div className="metric-header">
            <span className="metric-title">Solar Coverage Ratio</span>
            <span className="metric-icon">⚡</span>
          </div>
          <div className="metric-value">
            {formatDecimal(solar_coverage_percent, 2)}%
          </div>
          <div className="metric-meta">
            Proportion of solar surface
          </div>
        </div>
      </div>
    </section>
  );
}
