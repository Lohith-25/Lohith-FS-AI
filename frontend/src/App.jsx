import React, { useState, useEffect, useRef } from 'react';
import Header from './components/Header';
import HeroSection from './components/HeroSection';
import ImageUpload from './components/ImageUpload';
import ProcessingState from './components/ProcessingState';
import MetricsDashboard from './components/MetricsDashboard';
import VisualizationSection from './components/VisualizationSection';
import GeospatialMap from './components/GeospatialMap';
import ErrorAlert from './components/ErrorAlert';
import PredictionHistoryDrawer from './components/PredictionHistoryDrawer';
import {
  checkBackendHealth,
  sendPrediction,
  loadBundledSampleFile,
  getArtifactUrl,
  API_BASE_URL,
} from './services/api';
import './App.css';

export default function App() {
  const [health, setHealth] = useState(null);
  const [isCheckingHealth, setIsCheckingHealth] = useState(true);

  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [imageDimensions, setImageDimensions] = useState({ width: null, height: null });

  const [threshold, setThreshold] = useState(0.5);
  const [minComponentArea, setMinComponentArea] = useState(20);
  const [strategy, setStrategy] = useState('auto');

  // Phase 14.6: Spatial Calibration & Analytics State
  const [customGsd, setCustomGsd] = useState('');
  const [calibrationSource, setCalibrationSource] = useState('');
  const [useCeaBenchmark, setUseCeaBenchmark] = useState(true);

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [activeView, setActiveView] = useState('inspection'); // 'inspection' | 'geospatial'

  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [viewingHistoryRecord, setViewingHistoryRecord] = useState(null);

  const resultsRef = useRef(null);

  const handleRefreshHealth = async () => {
    setIsCheckingHealth(true);
    try {
      const res = await checkBackendHealth();
      setHealth(res);
    } catch (err) {
      setHealth({ healthy: false, data: null, error: err.message });
    } finally {
      setIsCheckingHealth(false);
    }
  };

  // Check health on mount asynchronously
  useEffect(() => {
    let ignore = false;
    checkBackendHealth()
      .then((res) => {
        if (!ignore) {
          setHealth(res);
          setIsCheckingHealth(false);
        }
      })
      .catch((err) => {
        if (!ignore) {
          setHealth({ healthy: false, data: null, error: err.message });
          setIsCheckingHealth(false);
        }
      });
    return () => {
      ignore = true;
    };
  }, []);

  // Cleanup object URL on unmount
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const handleFileSelect = (file) => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    const url = URL.createObjectURL(file);
    setSelectedFile(file);
    setPreviewUrl(url);
    setError(null);
    setPredictionResult(null);

    const img = new Image();
    img.onload = () => {
      setImageDimensions({ width: img.naturalWidth, height: img.naturalHeight });
    };
    img.src = url;
  };

  const handleClearFile = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setImageDimensions({ width: null, height: null });
    setPredictionResult(null);
    setError(null);
  };

  const handleLoadSample = async () => {
    try {
      setError(null);
      const sampleFile = await loadBundledSampleFile();
      handleFileSelect(sampleFile);
    } catch (err) {
      setError(`Failed to load benchmark sample: ${err.message}`);
    }
  };

  const handleToggleHistory = () => {
    setIsHistoryOpen((prev) => !prev);
  };

  const handleSelectHistoryRecord = (record) => {
    setViewingHistoryRecord(record);
    setPredictionResult(record);
    if (previewUrl && !previewUrl.startsWith('http')) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(
      record.original_image_url
        ? getArtifactUrl(record.original_image_url)
        : (record.overlay_url ? getArtifactUrl(record.overlay_url) : null)
    );
    setImageDimensions({ width: record.image_width, height: record.image_height });
    setSelectedFile({ name: record.filename || `prediction-${record.prediction_id.slice(0, 8)}` });
    setError(null);
    setTimeout(() => {
      resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
  };

  const handleExitHistoryView = () => {
    setViewingHistoryRecord(null);
    setPredictionResult(null);
    setSelectedFile(null);
    if (previewUrl && !previewUrl.startsWith('http')) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setImageDimensions({ width: null, height: null });
  };

  const handleSubmit = async () => {
    if (isLoading) return;
    if (!selectedFile) {
      setError('Please select or drop an aerial image first.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setViewingHistoryRecord(null);
    setPredictionResult(null);

    try {
      const result = await sendPrediction(selectedFile, {
        threshold,
        minComponentArea,
        strategy,
        custom_gsd: customGsd,
        calibration_source: calibrationSource,
        use_cea_benchmark: useCeaBenchmark,
      });

      setPredictionResult({
        ...result,
        applied_threshold: threshold,
        applied_min_component_area: minComponentArea,
        applied_strategy: strategy,
      });

      // Smooth scroll to results
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);
    } catch (err) {
      setError(err);
      setPredictionResult(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-layout">
      <div className="app-container">
        <Header
          health={health}
          onRefreshHealth={handleRefreshHealth}
          isCheckingHealth={isCheckingHealth}
          onToggleHistory={handleToggleHistory}
        />

        <PredictionHistoryDrawer
          isOpen={isHistoryOpen}
          onClose={() => setIsHistoryOpen(false)}
          onSelectRecord={handleSelectHistoryRecord}
          activePredictionId={predictionResult?.prediction_id}
        />

        <main className="main-content">
          {/* Historical Record Active Notice */}
          {viewingHistoryRecord && (
            <div className="history-banner" role="status">
              <div className="history-banner-info">
                <span className="history-banner-badge">Historical Record</span>
                <span>
                  Viewing saved prediction recorded on{' '}
                  <strong>{new Date(viewingHistoryRecord.created_at_utc).toLocaleString()}</strong>
                  {' '}(ID: <code>{viewingHistoryRecord.prediction_id.slice(0, 8)}...</code>)
                </span>
              </div>
              <button
                type="button"
                className="history-banner-btn"
                onClick={handleExitHistoryView}
              >
                Return to Live Upload
              </button>
            </div>
          )}

          {/* Compact Hero Section */}
          <HeroSection />

          {/* Upload and Ingestion */}
          <ImageUpload
            selectedFile={selectedFile}
            previewUrl={previewUrl}
            imageDimensions={imageDimensions}
            onFileSelect={(file) => {
              setViewingHistoryRecord(null);
              handleFileSelect(file);
            }}
            onClearFile={handleClearFile}
            onSubmit={handleSubmit}
            onLoadSample={() => {
              setViewingHistoryRecord(null);
              handleLoadSample();
            }}
            isLoading={isLoading}
            threshold={threshold}
            setThreshold={setThreshold}
            minComponentArea={minComponentArea}
            setMinComponentArea={setMinComponentArea}
            strategy={strategy}
            setStrategy={setStrategy}
            backendOnline={health?.healthy}
            customGsd={customGsd}
            setCustomGsd={setCustomGsd}
            calibrationSource={calibrationSource}
            setCalibrationSource={setCalibrationSource}
            useCeaBenchmark={useCeaBenchmark}
            setUseCeaBenchmark={setUseCeaBenchmark}
          />

          {/* Error Notification */}
          {error && <ErrorAlert error={error} onDismiss={() => setError(null)} />}

          {/* Loading Radar Scanner */}
          {isLoading && <ProcessingState />}

          {/* Results Anchor */}
          <div ref={resultsRef} />

          {/* Results Presentation */}
          {predictionResult && !isLoading && (
            <div className="results-container">
              {/* Task 3: Analysis Dashboard Metrics */}
              <MetricsDashboard result={predictionResult} />

              {/* View Selector Tabs (Phase 12: Visual Inspection vs Interactive Geospatial Map) */}
              <div className="view-mode-tabs" role="tablist" aria-label="Analysis View Selector">
                <button
                  type="button"
                  role="tab"
                  id="tab-inspection"
                  aria-selected={activeView === 'inspection'}
                  aria-controls="panel-inspection"
                  className={`view-tab-btn ${activeView === 'inspection' ? 'active' : ''}`}
                  onClick={() => setActiveView('inspection')}
                >
                  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                    <circle cx="12" cy="12" r="3" />
                  </svg>
                  Visual Inspection
                </button>
                <button
                  type="button"
                  role="tab"
                  id="tab-geospatial"
                  aria-selected={activeView === 'geospatial'}
                  aria-controls="panel-geospatial"
                  className={`view-tab-btn ${activeView === 'geospatial' ? 'active' : ''}`}
                  onClick={() => setActiveView('geospatial')}
                >
                  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                    <circle cx="12" cy="12" r="10" />
                    <line x1="2" y1="12" x2="22" y2="12" />
                    <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
                  </svg>
                  Interactive Geospatial Map
                  {predictionResult.georeferencing_level === 'LEVEL_B' && (
                    <span className="tab-pill">Level B</span>
                  )}
                </button>
              </div>

              {/* View Content Panels */}
              {activeView === 'inspection' ? (
                <div id="panel-inspection" role="tabpanel" aria-labelledby="tab-inspection">
                  <VisualizationSection
                    result={predictionResult}
                    originalPreviewUrl={previewUrl}
                  />
                </div>
              ) : (
                <div id="panel-geospatial" role="tabpanel" aria-labelledby="tab-geospatial">
                  <GeospatialMap
                    predictionResult={predictionResult}
                    selectedFile={selectedFile}
                  />
                </div>
              )}
            </div>
          )}
        </main>

        <footer className="site-footer">
          <div className="footer-content">
            <div className="footer-left">
              <strong>SolarMap-India</strong> • Satellite & Aerial Rooftop Solar Intelligence
            </div>
            <div className="footer-links">
              <span>API Base: <code>{API_BASE_URL || 'Local Origin'}</code></span>
              <a
                href={API_BASE_URL ? `${API_BASE_URL}/docs` : '/docs'}
                target="_blank"
                rel="noreferrer"
              >
                OpenAPI Docs
              </a>
              <a
                href={API_BASE_URL ? `${API_BASE_URL}/health` : '/health'}
                target="_blank"
                rel="noreferrer"
              >
                Health Check
              </a>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
