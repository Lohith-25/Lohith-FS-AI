import React from 'react';

export default function ProcessingState() {
  return (
    <section className="card processing-card" aria-live="polite">
      <div className="processing-content">
        <div className="radar-scanner">
          <div className="radar-circle radar-circle-1" />
          <div className="radar-circle radar-circle-2" />
          <div className="radar-circle radar-circle-3" />
          <div className="radar-sweep" />
          <div className="radar-core">
            <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2">
              <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
            </svg>
          </div>
        </div>

        <div className="processing-text">
          <h3 className="processing-title">Running Half U-Net Segmentation</h3>
          <p className="processing-subtitle">
            Executing deep convolutional inference on aerial imagery, extracting rooftop solar arrays,
            and computing connected-region pixel metrics...
          </p>

          <div className="pipeline-steps">
            <div className="step-item step-complete">
              <span className="step-icon">✓</span>
              <span>1. Image Ingestion & Preprocessing</span>
            </div>
            <div className="step-item step-active">
              <span className="step-spinner" />
              <span>2. Half U-Net Forward Inference</span>
            </div>
            <div className="step-item step-pending">
              <span className="step-icon">○</span>
              <span>3. Connected Region Geometry</span>
            </div>
            <div className="step-item step-pending">
              <span className="step-icon">○</span>
              <span>4. Mask & Overlay Generation</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
