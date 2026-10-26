import React from 'react';

export default function ErrorAlert({ error, onDismiss }) {
  if (!error) return null;

  return (
    <div className="card alert-card" role="alert">
      <div className="alert-content">
        <div className="alert-icon-circle">
          <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
        </div>

        <div className="alert-details">
          <h4 className="alert-title">Inference Request Failed</h4>
          <p className="alert-message">{typeof error === 'string' ? error : error.message}</p>
          {error.status && (
            <span className="badge badge-error-code">HTTP Status {error.status}</span>
          )}
        </div>

        {onDismiss && (
          <button
            type="button"
            className="btn-dismiss"
            onClick={onDismiss}
            aria-label="Dismiss error notification"
          >
            ✕
          </button>
        )}
      </div>
    </div>
  );
}
