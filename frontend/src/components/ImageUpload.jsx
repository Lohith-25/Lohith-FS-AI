import React, { useState, useRef } from 'react';

const ALLOWED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'];
const MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024; // 50MB

export default function ImageUpload({
  selectedFile,
  previewUrl,
  imageDimensions,
  onFileSelect,
  onClearFile,
  onSubmit,
  onLoadSample,
  isLoading,
  threshold,
  setThreshold,
  minComponentArea,
  setMinComponentArea,
  strategy = 'auto',
  setStrategy,
  backendOnline,
  customGsd = '',
  setCustomGsd,
  calibrationSource = '',
  setCalibrationSource,
  useCeaBenchmark = true,
  setUseCeaBenchmark,
}) {
  const [isDragging, setIsDragging] = useState(false);
  const [validationError, setValidationError] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  const [showCalibration, setShowCalibration] = useState(false);
  const fileInputRef = useRef(null);

  const validateAndProcessFile = (file) => {
    setValidationError(null);

    if (!file) return false;

    if (file.size === 0) {
      onClearFile();
      setValidationError('Uploaded file is empty (0 bytes). Please select a valid image.');
      return false;
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
      onClearFile();
      setValidationError('Uploaded image exceeds the 50 MB size limit.');
      return false;
    }

    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      onClearFile();
      setValidationError(
        `Unsupported format "${ext}". Supported formats: ${ALLOWED_EXTENSIONS.join(', ')}`
      );
      return false;
    }

    onFileSelect(file);
    return true;
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (isLoading) return;
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (isLoading) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      validateAndProcessFile(file);
    }
  };

  const handleFileInputChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndProcessFile(e.target.files[0]);
    }
  };

  const formatFileSize = (bytes) => {
    if (!bytes && bytes !== 0) return '';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <section className="card upload-card" aria-labelledby="upload-heading">
      <div className="card-header">
        <div>
          <h2 id="upload-heading" className="card-title">Upload Your Aerial Imagery</h2>
          <p className="card-subtitle">
            Start with a satellite or aerial image. Our AI model will analyze the image and highlight potential rooftop solar installations.
          </p>
        </div>
        <button
          type="button"
          className="btn-sample"
          onClick={onLoadSample}
          disabled={isLoading}
          title="Load the 768.0_1.0 benchmark dataset image"
        >
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="7 10 12 15 17 10" />
            <line x1="12" y1="15" x2="12" y2="3" />
          </svg>
          Load Benchmark Sample
        </button>
      </div>

      {validationError && (
        <div className="alert alert-warning" role="alert">
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          <span>{validationError}</span>
        </div>
      )}

      {/* Dropzone */}
      <div
        className={`dropzone ${isDragging ? 'dropzone-active' : ''} ${selectedFile ? 'dropzone-has-file' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !selectedFile && fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={ALLOWED_EXTENSIONS.join(',')}
          onChange={handleFileInputChange}
          style={{ display: 'none' }}
          disabled={isLoading}
        />

        {!selectedFile ? (
          <div className="dropzone-empty">
            <div className="dropzone-icon-circle">
              <svg viewBox="0 0 24 24" width="36" height="36" fill="none" stroke="currentColor" strokeWidth="1.75">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                <circle cx="8.5" cy="8.5" r="1.5" />
                <polyline points="21 15 16 10 5 21" />
              </svg>
            </div>
            <div className="dropzone-instructions">
              <p className="dropzone-prompt">
                <span className="dropzone-highlight">Drop your image here to begin</span>
              </p>
              <p className="dropzone-subprompt">
                Drag and drop an image, or click to browse your files.
              </p>
              <p className="dropzone-support">
                PNG, JPG, JPEG, TIFF, BMP, WebP — Maximum 50 MB.
              </p>
            </div>
          </div>
        ) : (
          <div className="dropzone-preview-container">
            <div className="preview-image-box">
              <img src={previewUrl} alt="Selected satellite/aerial preview" className="preview-thumb" />
            </div>
            <div className="preview-meta">
              <div className="preview-file-title">
                <span className="file-name" title={selectedFile.name}>{selectedFile.name}</span>
                <span className="badge badge-file-size">{formatFileSize(selectedFile.size)}</span>
              </div>
              <div className="preview-specs">
                {imageDimensions.width ? (
                  <span className="spec-item">
                    <strong>Dimensions:</strong> {imageDimensions.width} × {imageDimensions.height} px
                  </span>
                ) : (
                  <span className="spec-item">Loading dimensions...</span>
                )}
                <span className="spec-item">
                  <strong>Type:</strong> {selectedFile.type || 'image/*'}
                </span>
              </div>
              <div className="preview-actions">
                <button
                  type="button"
                  className="btn-change-file"
                  onClick={(e) => {
                    e.stopPropagation();
                    fileInputRef.current?.click();
                  }}
                  disabled={isLoading}
                >
                  Change File
                </button>
                <button
                  type="button"
                  className="btn-clear-file"
                  onClick={(e) => {
                    e.stopPropagation();
                    onClearFile();
                  }}
                  disabled={isLoading}
                >
                  Remove
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Inference Parameters Toggle */}
      <div className="params-section">
        <button
          type="button"
          className="btn-toggle-params"
          onClick={() => setShowSettings(!showSettings)}
        >
          <span>Model Inference Parameters</span>
          <span className="params-badge">
            Strategy: {strategy.toUpperCase()} • Threshold: {threshold} • Min Area: {minComponentArea}px
          </span>
          <svg
            className={`chevron ${showSettings ? 'chevron-open' : ''}`}
            viewBox="0 0 24 24"
            width="16"
            height="16"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </button>

        {showSettings && (
          <div className="params-controls">
            <div className="param-group">
              <div className="param-label-row">
                <label htmlFor="threshold-slider">Probability Threshold</label>
                <span className="param-val">{threshold}</span>
              </div>
              <input
                id="threshold-slider"
                type="range"
                min="0.05"
                max="0.95"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                disabled={isLoading}
              />
              <span className="param-hint">
                Classifier cutoff for foreground solar pixel segmentation (default: 0.50).
              </span>
            </div>

            <div className="param-group">
              <div className="param-label-row">
                <label htmlFor="min-area-input">Min Component Area</label>
                <span className="param-val">{minComponentArea} px</span>
              </div>
              <input
                id="min-area-input"
                type="range"
                min="5"
                max="100"
                step="5"
                value={minComponentArea}
                onChange={(e) => setMinComponentArea(parseInt(e.target.value, 10))}
                disabled={isLoading}
              />
              <span className="param-hint">
                Filters isolated noise clusters smaller than this pixel area (default: 20 px).
              </span>
            </div>

            <div className="param-group">
              <div className="param-label-row">
                <label htmlFor="strategy-select">Processing Strategy</label>
                <span className="param-val">{strategy.toUpperCase()}</span>
              </div>
              <select
                id="strategy-select"
                className="param-select"
                value={strategy}
                onChange={(e) => setStrategy(e.target.value)}
                disabled={isLoading}
                aria-label="Inference Processing Strategy"
              >
                <option value="auto">Auto (Adaptive by Resolution)</option>
                <option value="letterbox">Letterbox (Scale to 256x256)</option>
                <option value="tiled">Tiled (1:1 Native Resolution Windows)</option>
                <option value="direct">Direct (Standard Resize)</option>
              </select>
              <span className="param-hint">
                Tiled preserves 1:1 scale for small rooftop panels; Letterbox scales down non-square images.
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Spatial Calibration & Analytics Configuration */}
      <div className="params-section calibration-section">
        <button
          type="button"
          className="btn-toggle-params"
          onClick={() => setShowCalibration(!showCalibration)}
        >
          <span>Spatial Calibration (GSD) & Analytics</span>
          <span className="params-badge">
            {customGsd
              ? `GSD: ${customGsd} m/px (Manual)`
              : 'Auto / Uncalibrated'}
          </span>
          <svg
            className={`chevron ${showCalibration ? 'chevron-open' : ''}`}
            viewBox="0 0 24 24"
            width="16"
            height="16"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </button>

        {showCalibration && (
          <div className="calibration-controls">
            <div className="calibration-info-box">
              <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="16" x2="12" y2="12" />
                <line x1="12" y1="8" x2="12.01" y2="8" />
              </svg>
              <div>
                <strong>Calibration Precedence:</strong> Georeferenced GeoTIFFs (with embedded CRS & geotransform)
                calibrate physical scale automatically. For standard PNG/JPEG rasters, supply a verified
                Ground Sampling Distance (GSD) to compute physical surface area (m²) and energy yield.
              </div>
            </div>

            <div className="param-group">
              <div className="param-label-row">
                <label htmlFor="custom-gsd-input">Ground Sampling Distance (GSD)</label>
                <span className="param-val">{customGsd ? `${customGsd} m/px` : 'None (Uncalibrated)'}</span>
              </div>
              <div className="input-with-unit">
                <input
                  id="custom-gsd-input"
                  type="number"
                  step="0.001"
                  min="0.005"
                  max="100.0"
                  placeholder="e.g. 0.05"
                  value={customGsd}
                  onChange={(e) => setCustomGsd && setCustomGsd(e.target.value)}
                  disabled={isLoading}
                  className="param-text-input"
                />
                <span className="input-unit-label">meters / pixel</span>
              </div>
              <span className="param-hint">
                Strict physical bounds: 0.005 to 100.0 m/pixel. Unverified guesses or non-positive values are rejected.
              </span>
            </div>

            <div className="calibration-presets-row">
              <span className="presets-label">Quick Presets:</span>
              <button
                type="button"
                className="btn-preset"
                disabled={isLoading}
                onClick={() => {
                  if (setCustomGsd) setCustomGsd('0.05');
                  if (setCalibrationSource) setCalibrationSource('Drone UAV aerial inspection (0.05 m/px GSD)');
                }}
              >
                0.05 m/px (Drone UAV)
              </button>
              <button
                type="button"
                className="btn-preset"
                disabled={isLoading}
                onClick={() => {
                  if (setCustomGsd) setCustomGsd('0.10');
                  if (setCalibrationSource) setCalibrationSource('Aerial orthophoto survey report (0.10 m/px GSD)');
                }}
              >
                0.10 m/px (Orthophoto)
              </button>
              <button
                type="button"
                className="btn-preset"
                disabled={isLoading}
                onClick={() => {
                  if (setCustomGsd) setCustomGsd('0.30');
                  if (setCalibrationSource) setCalibrationSource('VHR commercial satellite imagery (0.30 m/px GSD)');
                }}
              >
                0.30 m/px (VHR Satellite)
              </button>
              {customGsd && (
                <button
                  type="button"
                  className="btn-preset btn-preset-clear"
                  disabled={isLoading}
                  onClick={() => {
                    if (setCustomGsd) setCustomGsd('');
                    if (setCalibrationSource) setCalibrationSource('');
                  }}
                >
                  Clear Calibration
                </button>
              )}
            </div>

            <div className="param-group">
              <div className="param-label-row">
                <label htmlFor="calibration-source-input">Calibration Source / Survey Reference</label>
              </div>
              <input
                id="calibration-source-input"
                type="text"
                placeholder="e.g. Flight log DJI Mavic 3 Enterprise, AGL 120m, 2024"
                value={calibrationSource}
                onChange={(e) => setCalibrationSource && setCalibrationSource(e.target.value)}
                disabled={isLoading}
                className="param-text-input"
              />
              <span className="param-hint">
                Document camera specifications, flight parameters, or orthomosaic processing report.
              </span>
            </div>

            <div className="param-group param-checkbox-group">
              <label className="checkbox-label" htmlFor="cea-benchmark-toggle">
                <input
                  id="cea-benchmark-toggle"
                  type="checkbox"
                  checked={useCeaBenchmark}
                  onChange={(e) => setUseCeaBenchmark && setUseCeaBenchmark(e.target.checked)}
                  disabled={isLoading}
                />
                <span>
                  <strong>Enable CEA National Benchmark Energy & CO₂ Estimation</strong>
                  <span className="checkbox-subtext">
                    Applies CEA baseline grid factor (~0.716 kg CO₂/kWh) when physical area is calibrated.
                  </span>
                </span>
              </label>
            </div>
          </div>
        )}
      </div>

      {/* Submit Action */}
      <div className="upload-footer">
        <div className="upload-notice">
          {!backendOnline && (
            <span className="text-warning-sm">
              ⚠ Backend service is unreachable. Starting backend on port 8000 is required.
            </span>
          )}
        </div>
        <button
          type="button"
          className="btn-submit"
          onClick={onSubmit}
          disabled={!selectedFile || isLoading}
        >
          {isLoading ? (
            <>
              <span className="button-spinner" />
              <span>Analyzing Image with Half U-Net...</span>
            </>
          ) : (
            <>
              <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
              </svg>
              <span>Analyze Solar Imagery</span>
            </>
          )}
        </button>
      </div>
    </section>
  );
}
