import React from 'react';
import './HeroSection.css';

export default function HeroSection() {
  return (
    <section className="card hero-card" aria-labelledby="hero-heading">
      <div className="hero-content">
        <div className="hero-eyebrow">
          <svg
            className="hero-eyebrow-icon"
            viewBox="0 0 24 24"
            width="13"
            height="13"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            aria-hidden="true"
          >
            <circle cx="12" cy="12" r="4" fill="currentColor" fillOpacity="0.25" />
            <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41" strokeLinecap="round" />
          </svg>
          <span>DISCOVER SOLAR POTENTIAL</span>
        </div>

        <h1 id="hero-heading" className="hero-title">
          Turn Aerial Imagery into <span className="hero-title-accent">Solar Intelligence</span>
        </h1>

        <p className="hero-description">
          Harness AI-powered image segmentation to identify rooftop solar installations,
          visualize detected panels, and explore solar potential from aerial and satellite imagery.
        </p>

        <div className="hero-features" role="list" aria-label="Key Capabilities">
          <div className="hero-feature-item" role="listitem">
            <span className="hero-feature-icon-wrapper" aria-hidden="true">
              <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </span>
            <span className="hero-feature-text">AI-Powered Detection</span>
          </div>

          <div className="hero-feature-item" role="listitem">
            <span className="hero-feature-icon-wrapper" aria-hidden="true">
              <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="3" width="7" height="7" rx="1.5" />
                <rect x="14" y="3" width="7" height="7" rx="1.5" />
                <rect x="14" y="14" width="7" height="7" rx="1.5" />
                <rect x="3" y="14" width="7" height="7" rx="1.5" />
              </svg>
            </span>
            <span className="hero-feature-text">Pixel-Level Segmentation</span>
          </div>

          <div className="hero-feature-item" role="listitem">
            <span className="hero-feature-icon-wrapper" aria-hidden="true">
              <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <line x1="2" y1="12" x2="22" y2="12" />
                <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
              </svg>
            </span>
            <span className="hero-feature-text">Geospatial Intelligence</span>
          </div>
        </div>
      </div>
    </section>
  );
}
