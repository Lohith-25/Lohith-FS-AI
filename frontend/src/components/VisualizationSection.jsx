import React, { useState, useRef } from 'react';
import { getArtifactUrl } from '../services/api';

export default function VisualizationSection({ result, originalPreviewUrl }) {
  const [activeTab, setActiveTab] = useState('grid'); // 'grid' | 'slider' | 'focus'
  const [focusView, setFocusView] = useState('overlay'); // 'original' | 'mask' | 'overlay'
  const [sliderPos, setSliderPos] = useState(50); // 0 to 100
  const [isDraggingSlider, setIsDraggingSlider] = useState(false);
  const [maskEnhanced, setMaskEnhanced] = useState(false);
  const [imageErrors, setImageErrors] = useState({});
  const [imageReloadKey, setImageReloadKey] = useState(0);
  const [modalImage, setModalImage] = useState(null);
  const sliderViewportRef = useRef(null);

  const updateSliderPosition = (clientX) => {
    if (!sliderViewportRef.current) return;
    const rect = sliderViewportRef.current.getBoundingClientRect();
    if (rect.width <= 0) return;
    const offsetX = clientX - rect.left;
    const clampedRatio = Math.max(0, Math.min(1, offsetX / rect.width));
    const newPos = Math.round(clampedRatio * 1000) / 10;
    setSliderPos(newPos);
  };

  const handleSliderPointerDown = (e) => {
    setIsDraggingSlider(true);
    updateSliderPosition(e.clientX);
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      // Ignore if pointer capture fails
    }
  };

  const handleSliderPointerMove = (e) => {
    if (!isDraggingSlider) return;
    updateSliderPosition(e.clientX);
  };

  const handleSliderPointerUp = (e) => {
    if (isDraggingSlider) {
      setIsDraggingSlider(false);
      try {
        e.currentTarget.releasePointerCapture(e.pointerId);
      } catch {
        // Ignore
      }
    }
  };

  if (!result) return null;

  const maskUrl = result.mask_url
    ? `${getArtifactUrl(result.mask_url)}${imageReloadKey ? `?k=${imageReloadKey}` : ''}`
    : null;

  const overlayUrl = result.overlay_url
    ? `${getArtifactUrl(result.overlay_url)}${imageReloadKey ? `?k=${imageReloadKey}` : ''}`
    : null;

  const originalUrl =
    originalPreviewUrl ||
    (result.original_image_url ? getArtifactUrl(result.original_image_url) : null);

  // Dynamic aspect ratio calculation matching source image dimensions
  const hasDimensions = Boolean(result.image_width && result.image_height);
  const imageAspectRatio = hasDimensions
    ? `${result.image_width} / ${result.image_height}`
    : '16 / 10';

  const containerStyle = {
    aspectRatio: imageAspectRatio,
  };

  const handleImageError = (type) => {
    setImageErrors((prev) => ({ ...prev, [type]: true }));
  };

  const handleImageLoad = (type) => {
    setImageErrors((prev) => ({ ...prev, [type]: false }));
  };

  const retryImageLoad = (type) => {
    setImageErrors((prev) => ({ ...prev, [type]: false }));
    setImageReloadKey((k) => k + 1);
  };

  const downloadArtifact = async (url, filename) => {
    try {
      const res = await fetch(url);
      const blob = await res.blob();
      const blobUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = blobUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(blobUrl);
    } catch {
      window.open(url, '_blank');
    }
  };

  return (
    <section className="card visual-card" aria-labelledby="visual-heading">
      <div className="card-header">
        <div>
          <h2 id="visual-heading" className="card-title">Segmentation Visualizations</h2>
          <p className="card-subtitle">
            Visual inspection comparing original aerial capture, binary mask, and solar overlay
          </p>
        </div>

        {/* View Mode Tabs */}
        <div className="view-mode-tabs" role="tablist">
          <button
            type="button"
            className={`tab-btn ${activeTab === 'grid' ? 'tab-btn-active' : ''}`}
            onClick={() => setActiveTab('grid')}
            role="tab"
            aria-selected={activeTab === 'grid'}
          >
            Side-by-Side (3-Up)
          </button>
          <button
            type="button"
            className={`tab-btn ${activeTab === 'slider' ? 'tab-btn-active' : ''}`}
            onClick={() => setActiveTab('slider')}
            role="tab"
            aria-selected={activeTab === 'slider'}
          >
            Compare Slider
          </button>
          <button
            type="button"
            className={`tab-btn ${activeTab === 'focus' ? 'tab-btn-active' : ''}`}
            onClick={() => setActiveTab('focus')}
            role="tab"
            aria-selected={activeTab === 'focus'}
          >
            Focus View
          </button>
        </div>
      </div>

      {/* Mode 1: 3-Up Grid View */}
      {activeTab === 'grid' && (
        <div className="visual-grid-3">
          {/* 1. Original Image */}
          <div className="visual-panel">
            <div className="panel-header">
              <span className="panel-tag tag-original">Original Aerial</span>
              <span className="panel-dims">
                {result.image_width} × {result.image_height}
              </span>
            </div>
            <div
              className="panel-image-container"
              style={containerStyle}
              onClick={() =>
                originalUrl && setModalImage({ src: originalUrl, title: 'Original Aerial Image' })
              }
            >
              {originalUrl ? (
                <img
                  src={originalUrl}
                  alt="Original aerial source"
                  className="visual-img"
                />
              ) : (
                <div className="panel-error-box">No preview available</div>
              )}
              <div className="zoom-hint">Click to enlarge</div>
            </div>
            <div className="panel-footer">
              <span className="panel-desc">Raw input imagery</span>
            </div>
          </div>

          {/* 2. Binary Mask */}
          <div className="visual-panel">
            <div className="panel-header">
              <div className="panel-header-title">
                <span className="panel-tag tag-mask">Binary Mask</span>
                <span className="panel-dims">
                  {result.solar_area_pixels != null
                    ? `${result.solar_area_pixels.toLocaleString()} px (${result.solar_coverage_percent?.toFixed(2)}%)`
                    : '1-bit'}
                </span>
              </div>
              <button
                type="button"
                className={`btn-mask-toggle ${maskEnhanced ? 'active' : ''}`}
                onClick={(e) => {
                  e.stopPropagation();
                  setMaskEnhanced((prev) => !prev);
                }}
                title="Toggle high-contrast solar glow on mask"
              >
                {maskEnhanced ? 'Solar Glow: ON' : 'Solar Glow: OFF'}
              </button>
            </div>
            <div
              className="panel-image-container"
              style={containerStyle}
              onClick={() =>
                !imageErrors.mask &&
                setModalImage({ src: maskUrl, title: 'Binary Segmentation Mask' })
              }
            >
              {imageErrors.mask ? (
                <div className="panel-error-box">
                  <span className="error-icon">⚠</span>
                  <p>Failed to load mask artifact</p>
                  <button
                    type="button"
                    className="btn-retry"
                    onClick={(e) => {
                      e.stopPropagation();
                      retryImageLoad('mask');
                    }}
                  >
                    Retry Artifact Fetch
                  </button>
                </div>
              ) : (
                <img
                  src={maskUrl}
                  alt="Predicted binary solar mask"
                  className={`visual-img mask-img ${maskEnhanced ? 'mask-highlight' : ''}`}
                  onError={() => handleImageError('mask')}
                  onLoad={() => handleImageLoad('mask')}
                />
              )}
              {!imageErrors.mask && <div className="zoom-hint">Click to enlarge</div>}
            </div>
            <div className="panel-footer">
              <span className="panel-desc">White pixels = Detected solar panels</span>
              <button
                type="button"
                className="btn-download-sm"
                onClick={() => downloadArtifact(maskUrl, `${result.prediction_id}_mask.png`)}
                title="Download mask PNG"
              >
                Download
              </button>
            </div>
          </div>

          {/* 3. Visual Detection Overlay */}
          <div className="visual-panel">
            <div className="panel-header">
              <span className="panel-tag tag-overlay">Predicted Overlay</span>
              <span className="panel-dims">
                {result.image_width} × {result.image_height}
              </span>
            </div>
            <div
              className="panel-image-container"
              style={containerStyle}
              onClick={() =>
                !imageErrors.overlay &&
                setModalImage({ src: overlayUrl, title: 'Solar Detection Overlay' })
              }
            >
              {imageErrors.overlay ? (
                <div className="panel-error-box">
                  <span className="error-icon">⚠</span>
                  <p>Failed to load overlay artifact</p>
                  <button
                    type="button"
                    className="btn-retry"
                    onClick={(e) => {
                      e.stopPropagation();
                      retryImageLoad('overlay');
                    }}
                  >
                    Retry Artifact Fetch
                  </button>
                </div>
              ) : (
                <img
                  src={overlayUrl}
                  alt="Solar overlay segmentation visual"
                  className="visual-img"
                  onError={() => handleImageError('overlay')}
                  onLoad={() => handleImageLoad('overlay')}
                />
              )}
              {!imageErrors.overlay && <div className="zoom-hint">Click to enlarge</div>}
            </div>
            <div className="panel-footer">
              <span className="panel-desc">Solar panels highlighted in red-orange</span>
              <button
                type="button"
                className="btn-download-sm"
                onClick={() => downloadArtifact(overlayUrl, `${result.prediction_id}_overlay.png`)}
                title="Download overlay PNG"
              >
                Download
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Mode 2: Interactive Before/After Compare Slider */}
      {activeTab === 'slider' && (
        <div className="visual-slider-container">
          <div className="slider-instruction">
            Drag the handle, click anywhere on the image, or use the controls below to compare:
          </div>
          <div
            ref={sliderViewportRef}
            className="slider-viewport"
            style={containerStyle}
            onPointerDown={handleSliderPointerDown}
            onPointerMove={handleSliderPointerMove}
            onPointerUp={handleSliderPointerUp}
            onPointerCancel={handleSliderPointerUp}
          >
            {/* Background Layer: Overlay */}
            <img
              src={overlayUrl}
              alt="Solar overlay comparison"
              className="slider-image-base"
              onError={() => handleImageError('overlay')}
            />
            {/* Foreground Layer: Original clipped via clip-path */}
            <img
              src={originalUrl}
              alt="Original aerial source"
              className="slider-image-clipped"
              style={{ clipPath: `inset(0 calc(100% - ${sliderPos}%) 0 0)` }}
            />
            {/* Divider line & handle */}
            <div className="slider-divider" style={{ left: `${sliderPos}%` }}>
              <div className="slider-handle">
                <span>◀ ▶</span>
              </div>
            </div>

            <div className="slider-tag tag-left">Original Aerial</div>
            <div className="slider-tag tag-right">Solar Overlay</div>
          </div>

          <div className="slider-controls-bar">
            <div className="slider-bar-wrapper">
              <span className="slider-bar-label">Position:</span>
              <input
                type="range"
                min="0"
                max="100"
                step="0.5"
                value={sliderPos}
                onChange={(e) => setSliderPos(parseFloat(e.target.value))}
                className="slider-bar-input slider-range-input"
                aria-label="Comparison slider position"
              />
              <span className="slider-bar-value">{Math.round(sliderPos)}%</span>
            </div>
            <div className="slider-presets">
              <button
                type="button"
                className={`preset-btn ${sliderPos === 25 ? 'active' : ''}`}
                onClick={() => setSliderPos(25)}
              >
                25% Original
              </button>
              <button
                type="button"
                className={`preset-btn ${sliderPos === 50 ? 'active' : ''}`}
                onClick={() => setSliderPos(50)}
              >
                50% Split
              </button>
              <button
                type="button"
                className={`preset-btn ${sliderPos === 75 ? 'active' : ''}`}
                onClick={() => setSliderPos(75)}
              >
                75% Overlay
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Mode 3: Large Focus View */}
      {activeTab === 'focus' && (
        <div className="visual-focus-container">
          <div className="focus-subtabs">
            <button
              type="button"
              className={`focus-pill ${focusView === 'original' ? 'focus-pill-active' : ''}`}
              onClick={() => setFocusView('original')}
            >
              Original Aerial
            </button>
            <button
              type="button"
              className={`focus-pill ${focusView === 'overlay' ? 'focus-pill-active' : ''}`}
              onClick={() => setFocusView('overlay')}
            >
              Predicted Overlay
            </button>
            <button
              type="button"
              className={`focus-pill ${focusView === 'mask' ? 'focus-pill-active' : ''}`}
              onClick={() => setFocusView('mask')}
            >
              Binary Mask
            </button>
          </div>

          <div
            className="focus-image-frame"
            style={containerStyle}
            onClick={() => {
              const currentSrc =
                focusView === 'original'
                  ? originalUrl
                  : focusView === 'mask'
                  ? maskUrl
                  : overlayUrl;
              setModalImage({ src: currentSrc, title: focusView });
            }}
          >
            {focusView === 'original' && (
              <img src={originalUrl} alt="Original Aerial" className="focus-main-img" />
            )}
            {focusView === 'overlay' && (
              <img
                src={overlayUrl}
                alt="Solar Overlay"
                className="focus-main-img"
                onError={() => handleImageError('overlay')}
              />
            )}
            {focusView === 'mask' && (
              <img
                src={maskUrl}
                alt="Binary Mask"
                className={`focus-main-img mask-img ${maskEnhanced ? 'mask-highlight' : ''}`}
                onError={() => handleImageError('mask')}
              />
            )}
            <div className="zoom-hint">Click to enlarge</div>
          </div>
        </div>
      )}

      {/* Zoom Modal */}
      {modalImage && (
        <div className="modal-backdrop" onClick={() => setModalImage(null)}>
          <div className="modal-dialog" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>{modalImage.title}</h3>
              <button
                type="button"
                className="modal-close"
                onClick={() => setModalImage(null)}
                aria-label="Close image inspection"
              >
                ✕
              </button>
            </div>
            <div className="modal-body">
              <img src={modalImage.src} alt={modalImage.title} className="modal-full-img" />
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
