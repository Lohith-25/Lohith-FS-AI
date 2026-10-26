import React from 'react';

export default function Header({ health, onRefreshHealth, isCheckingHealth, onToggleHistory }) {
  const isHealthy = health?.healthy;

  return (
    <header className="site-header">
      <div className="header-brand">
        <div className="brand-logo">
          <svg className="solar-sun-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <circle cx="12" cy="12" r="5" strokeWidth="2" fill="currentColor" fillOpacity="0.2" />
            <line x1="12" y1="1" x2="12" y2="3" strokeWidth="2" strokeLinecap="round" />
            <line x1="12" y1="21" x2="12" y2="23" strokeWidth="2" strokeLinecap="round" />
            <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" strokeWidth="2" strokeLinecap="round" />
            <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" strokeWidth="2" strokeLinecap="round" />
            <line x1="1" y1="12" x2="3" y2="12" strokeWidth="2" strokeLinecap="round" />
            <line x1="21" y1="12" x2="23" y2="12" strokeWidth="2" strokeLinecap="round" />
            <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" strokeWidth="2" strokeLinecap="round" />
            <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" strokeWidth="2" strokeLinecap="round" />
          </svg>
        </div>
        <div className="brand-text">
          <div className="brand-title-row">
            <h1 className="brand-title">SolarMap-India</h1>
            <span className="badge badge-phase">Phase 13</span>
            <span className="badge badge-model">Half U-Net</span>
          </div>
          <p className="brand-subtitle">
            Rooftop Solar Panel Segmentation & Pixel-Level Intelligence Dashboard
          </p>
        </div>
      </div>

      <div className="header-status">
        <button
          type="button"
          className="header-history-btn"
          onClick={onToggleHistory}
          title="Open Prediction History"
          aria-label="Open Prediction History"
        >
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
            <circle cx="12" cy="12" r="10" />
            <polyline points="12 6 12 12 16 14" />
          </svg>
          <span>History</span>
        </button>

        <div className={`status-pill ${isHealthy ? 'status-online' : 'status-offline'}`}>
          <span className={`status-dot ${isHealthy ? 'dot-online' : 'dot-offline'}`} />
          <div className="status-info">
            <span className="status-label">
              {isCheckingHealth
                ? 'Pinging API...'
                : isHealthy
                ? 'Backend Ready'
                : 'Backend Disconnected'}
            </span>
            {health?.data && (
              <span className="status-sub">
                {health.data.model} • {health.data.device?.toUpperCase()}
              </span>
            )}
            {!isHealthy && !isCheckingHealth && (
              <span className="status-sub status-sub-err">FastAPI (8000)</span>
            )}
          </div>
          <button
            type="button"
            className="btn-icon"
            onClick={onRefreshHealth}
            disabled={isCheckingHealth}
            title="Refresh backend status"
            aria-label="Refresh health status"
          >
            <svg
              className={`refresh-icon ${isCheckingHealth ? 'spin' : ''}`}
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2" />
            </svg>
          </button>
        </div>
      </div>
    </header>
  );
}
