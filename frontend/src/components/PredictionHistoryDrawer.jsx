import React, { useState, useEffect, useCallback } from 'react';
import { getPredictionHistory, checkDatabaseHealth } from '../services/api';
import './PredictionHistoryDrawer.css';

export default function PredictionHistoryDrawer({
  isOpen,
  onClose,
  onSelectRecord,
  activePredictionId,
}) {
  const [records, setRecords] = useState([]);
  const [totalRecords, setTotalRecords] = useState(0);
  const [page, setPage] = useState(1);
  const limit = 5;

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dbStatus, setDbStatus] = useState({ connected: false, configured: false, checked: false });

  const fetchHistory = useCallback(async (targetPage = 1) => {
    setIsLoading(true);
    setError(null);

    // Check DB health first
    const health = await checkDatabaseHealth();
    setDbStatus({
      connected: health.connected,
      configured: health.configured,
      checked: true,
    });

    if (!health.connected) {
      setIsLoading(false);
      setRecords([]);
      setTotalRecords(0);
      return;
    }

    const skip = (targetPage - 1) * limit;
    const res = await getPredictionHistory({ skip, limit });

    if (res.offline) {
      setDbStatus((prev) => ({ ...prev, connected: false }));
      setError(res.error || 'MongoDB is offline or unconfigured.');
      setRecords([]);
      setTotalRecords(0);
    } else if (!res.success) {
      setError(res.error || 'Failed to load prediction history.');
    } else {
      setRecords(res.items || []);
      setTotalRecords(res.total || 0);
      setPage(targetPage);
    }
    setIsLoading(false);
  }, [limit]);

  useEffect(() => {
    if (isOpen) {
      const timer = setTimeout(() => {
        fetchHistory(1);
      }, 0);
      return () => clearTimeout(timer);
    }
  }, [isOpen, fetchHistory]);

  if (!isOpen) return null;

  const totalPages = Math.max(1, Math.ceil(totalRecords / limit));

  const handlePrevPage = () => {
    if (page > 1 && !isLoading) {
      fetchHistory(page - 1);
    }
  };

  const handleNextPage = () => {
    if (page < totalPages && !isLoading) {
      fetchHistory(page + 1);
    }
  };

  const formatTimestamp = (isoStr) => {
    if (!isoStr) return 'Unknown Date';
    try {
      const d = new Date(isoStr);
      return d.toLocaleString(undefined, {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      });
    } catch {
      return isoStr;
    }
  };

  return (
    <aside
      className="history-drawer-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      aria-label="Prediction History Drawer"
    >
      <div className="history-drawer-panel" role="dialog" aria-modal="true" aria-labelledby="history-drawer-title">
        {/* Drawer Header */}
        <div className="history-drawer-header">
          <div className="history-header-titles">
            <h2 id="history-drawer-title" className="history-title">
              <span className="history-icon" aria-hidden="true">⏱️</span>
              Prediction History
            </h2>
            <div className="history-db-indicator">
              <span
                className={`db-indicator-dot ${dbStatus.connected ? 'online' : 'offline'}`}
                aria-hidden="true"
              />
              <span className="db-indicator-text">
                {dbStatus.connected
                  ? 'MongoDB Connected'
                  : dbStatus.configured
                  ? 'Database Unreachable'
                  : 'Database Offline (Unconfigured)'}
              </span>
            </div>
          </div>

          <div className="history-header-actions">
            <button
              type="button"
              className="history-btn-refresh"
              onClick={() => fetchHistory(page)}
              disabled={isLoading}
              title="Refresh History"
              aria-label="Refresh prediction history"
            >
              <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                <path d="M23 4v6h-6M1 20v-6h6" />
                <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
              </svg>
            </button>
            <button
              type="button"
              className="history-btn-close"
              onClick={onClose}
              aria-label="Close prediction history drawer"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Informative DB Warning Banner when offline */}
        {!dbStatus.connected && dbStatus.checked && (
          <div className="history-db-banner" role="status">
            <div className="history-db-banner-icon" aria-hidden="true">⚠️</div>
            <div className="history-db-banner-content">
              <strong>Database Service Offline</strong>
              <p>
                MongoDB is unconfigured or not reachable. Historical records cannot be persisted or retrieved.
                Live image segmentation and geospatial preview remain <strong>fully operational</strong>.
              </p>
            </div>
          </div>
        )}

        {/* Error notification if any */}
        {error && (
          <div className="history-error-banner" role="alert">
            <span>{error}</span>
          </div>
        )}

        {/* Content Body */}
        <div className="history-drawer-body">
          {isLoading ? (
            <div className="history-loading-container" role="status" aria-live="polite">
              <div className="history-spinner" aria-hidden="true" />
              <span>Querying prediction records...</span>
            </div>
          ) : !dbStatus.connected ? (
            <div className="history-empty-state">
              <div className="history-empty-icon" aria-hidden="true">📂</div>
              <h3>History Unavailable</h3>
              <p>
                Configure <code>MONGODB_URI</code> in <code>config/.env</code> to automatically store and review all inference runs.
              </p>
            </div>
          ) : records.length === 0 ? (
            <div className="history-empty-state">
              <div className="history-empty-icon" aria-hidden="true">☀️</div>
              <h3>No History Recorded Yet</h3>
              <p>
                Upload an aerial or satellite image to run your first solar panel detection.
              </p>
            </div>
          ) : (
            <div className="history-records-list" role="list">
              {records.map((item) => {
                const isSelected = item.prediction_id === activePredictionId;
                const hasCoords =
                  typeof item.latitude === 'number' && typeof item.longitude === 'number';

                return (
                  <article
                    key={item.prediction_id}
                    className={`history-card ${isSelected ? 'active-selection' : ''}`}
                    role="listitem"
                  >
                    <div className="history-card-top">
                      <div className="history-card-meta">
                        <span className="history-card-time">{formatTimestamp(item.created_at_utc)}</span>
                        <span className="history-card-filename" title={item.filename || 'Direct Upload'}>
                          {item.filename || 'Direct Upload'}
                        </span>
                      </div>
                      <span className="history-card-id" title={item.prediction_id}>
                        {item.prediction_id.slice(0, 8)}...
                      </span>
                    </div>

                    {/* Metrics grid */}
                    <div className="history-metrics-grid">
                      <div className="history-metric-col">
                        <span className="history-metric-lbl">Solar Pixels</span>
                        <span className="history-metric-val">
                          {item.solar_area_pixels?.toLocaleString() || 0}
                        </span>
                      </div>

                      <div className="history-metric-col">
                        <span className="history-metric-lbl">Coverage</span>
                        <span className="history-metric-val">
                          {item.solar_coverage_percent?.toFixed(2) || '0.00'}%
                        </span>
                      </div>

                      <div className="history-metric-col">
                        <span className="history-metric-lbl">Regions</span>
                        <span className="history-metric-val">
                          {item.detected_region_count || 0}
                        </span>
                      </div>

                      <div className="history-metric-col">
                        <span className="history-metric-lbl">Georeferencing</span>
                        <span className="history-metric-val geo-badge">
                          {hasCoords ? item.location_name || 'Level B' : 'Unreferenced'}
                        </span>
                      </div>
                    </div>

                    {/* Model & Strategy Badges */}
                    <div className="history-analytics-tags">
                      <span className="history-tag">
                        Model: {item.model_name || 'HalfUNet'}
                      </span>
                      <span className="history-tag">
                        Strategy: {(item.processing_strategy || 'direct').toUpperCase()}
                      </span>
                      {item.applied_threshold !== undefined && item.applied_threshold !== null && (
                        <span className="history-tag">
                          Threshold: {item.applied_threshold}
                        </span>
                      )}
                    </div>

                    {/* Action button */}
                    <div className="history-card-actions">
                      <button
                        type="button"
                        className={`history-select-btn ${isSelected ? 'selected' : ''}`}
                        onClick={() => {
                          onSelectRecord(item);
                          onClose();
                        }}
                      >
                        {isSelected ? 'Currently Viewing' : 'Load into Dashboard'}
                      </button>
                    </div>
                  </article>
                );
              })}
            </div>
          )}
        </div>

        {/* Footer with Pagination */}
        {dbStatus.connected && totalRecords > 0 && (
          <div className="history-drawer-footer">
            <span className="pagination-info">
              Showing {(page - 1) * limit + 1}–{Math.min(page * limit, totalRecords)} of {totalRecords} records
            </span>
            <div className="pagination-controls">
              <button
                type="button"
                className="pagination-btn"
                onClick={handlePrevPage}
                disabled={page <= 1 || isLoading}
                aria-label="Previous page"
              >
                Previous
              </button>
              <span className="pagination-page-display">
                Page {page} of {totalPages}
              </span>
              <button
                type="button"
                className="pagination-btn"
                onClick={handleNextPage}
                disabled={page >= totalPages || isLoading}
                aria-label="Next page"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}
