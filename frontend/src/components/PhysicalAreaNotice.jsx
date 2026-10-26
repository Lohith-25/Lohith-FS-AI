import React from 'react';

export default function PhysicalAreaNotice({
  physicalAreaStatus = 'insufficient_data',
  physicalAreaM2 = null,
  physicalAreaHectares = null,
  gsdStatus = 'UNAVAILABLE',
  calibrationType = 'NONE',
  calibrationStatus = 'UNAVAILABLE',
  calibrationSource = null,
  customGsd = null,
  calibrationRejectionReason = null,
  sourceResolution = null,
  sourceCrs = null,
  metadataProvenance = null,
  physicalAreaInsufficientReason = null,
}) {
  const isCalculated = physicalAreaStatus === 'calculated' && physicalAreaM2 !== null;
  const isUserSupplied = calibrationStatus === 'USER_SUPPLIED' || gsdStatus === 'USER_SUPPLIED';
  const isAutomatic = calibrationStatus === 'EXTRACTED_METADATA' || (calibrationType === 'AUTOMATIC' && isCalculated);

  const displayM2 =
    physicalAreaM2 !== null
      ? `${Number(physicalAreaM2).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} m²`
      : 'null (Unavailable)';

  const displayHectares =
    physicalAreaHectares !== null
      ? `${Number(physicalAreaHectares).toLocaleString(undefined, { minimumFractionDigits: 4, maximumFractionDigits: 4 })} ha`
      : isCalculated
      ? `${(physicalAreaM2 / 10000).toLocaleString(undefined, { minimumFractionDigits: 4, maximumFractionDigits: 4 })} ha`
      : 'null (Unavailable)';

  return (
    <section
      className={`card notice-card ${isCalculated ? (isUserSupplied ? 'notice-card-user' : 'notice-card-success') : ''}`}
      aria-labelledby="physical-area-heading"
    >
      <div className="notice-content">
        <div className={`notice-icon-box ${isCalculated ? (isUserSupplied ? 'notice-icon-user' : 'notice-icon-success') : ''}`}>
          {isCalculated ? (
            <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
              <polyline points="22 4 12 14.01 9 11.01" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
          )}
        </div>

        <div className="notice-body">
          <div className="notice-tag-row">
            {isAutomatic ? (
              <span className="badge badge-success">
                Automatic GeoTIFF Calibration
              </span>
            ) : isUserSupplied ? (
              <span className="badge badge-user-supplied">
                User-Supplied GSD (Unverified)
              </span>
            ) : (
              <span className="badge badge-warning">
                Uncalibrated Raster
              </span>
            )}

            <span className="badge badge-dim">Area Status: {physicalAreaStatus}</span>

            <span
              className={`badge ${
                isAutomatic
                  ? 'badge-info'
                  : isUserSupplied
                  ? 'badge-user-gsd'
                  : 'badge-mono'
              }`}
            >
              GSD Status: {calibrationStatus || gsdStatus}
            </span>

            {sourceResolution && (
              <span className="badge badge-mono">
                Res: {sourceResolution}
              </span>
            )}

            {sourceCrs && (
              <span className="badge badge-mono" title={sourceCrs}>
                CRS: {sourceCrs}
              </span>
            )}
          </div>

          <h3 id="physical-area-heading" className="notice-title">
            {isCalculated
              ? `Physical solar surface area: ${displayM2} (${displayHectares})`
              : 'Physical area unavailable: verified image resolution is required.'}
          </h3>

          <p className="notice-description">
            {isAutomatic ? (
              <>
                Ground Sampling Distance (GSD) was validated automatically from embedded GeoTIFF raster metadata
                {sourceCrs ? ` (${sourceCrs})` : ''}. Physical solar panel area is derived from segmented pixel counts
                and calibrated pixel Jacobian surface area.
              </>
            ) : isUserSupplied ? (
              <>
                Ground Sampling Distance was provided via user input (<strong>{sourceResolution || `${customGsd} m/px`}</strong>).
                In strict adherence to scientific defensibility standards, this value is tagged as{' '}
                <strong>USER_SUPPLIED</strong> and has not been independently verified against an on-site survey.
              </>
            ) : (
              <>
                Ground Sampling Distance (GSD) calibration is required to convert image pixel dimensions
                into physical metric surface areas (m² or hectares). To prevent scientifically ungrounded
                extrapolations, metric area conversion is disabled unless georeferenced metadata is verified
                or a known GSD is provided in the calibration settings.
              </>
            )}
          </p>

          {calibrationRejectionReason && (
            <div className="calibration-rejection-alert">
              <strong>Calibration Notice:</strong> {calibrationRejectionReason}
            </div>
          )}

          {physicalAreaInsufficientReason && !isCalculated && (
            <div className="calibration-reason-note">
              <em>Scientific Safeguard: {physicalAreaInsufficientReason}</em>
            </div>
          )}

          <div className="notice-footer-facts">
            <div className="fact-item">
              <span className="fact-label">Physical Area (m²):</span>
              <span className={`fact-value ${isCalculated ? 'fact-value-calculated' : 'fact-value-null'}`}>
                {displayM2}
              </span>
            </div>
            <div className="fact-item">
              <span className="fact-label">Physical Area (Hectares):</span>
              <span className={`fact-value ${isCalculated ? 'fact-value-calculated' : 'fact-value-null'}`}>
                {displayHectares}
              </span>
            </div>
            <div className="fact-item">
              <span className="fact-label">Calibration Method:</span>
              <span className="fact-value">
                {isAutomatic
                  ? 'Automatic GeoTIFF Metadata'
                  : isUserSupplied
                  ? 'Manual User-Supplied GSD'
                  : 'None (Uncalibrated)'}
              </span>
            </div>
            <div className="fact-item">
              <span className="fact-label">Calibration Source:</span>
              <span className="fact-value" title={calibrationSource || metadataProvenance || 'N/A'}>
                {calibrationSource || metadataProvenance || 'unreferenced_standard_raster'}
              </span>
            </div>
            <div className="fact-item">
              <span className="fact-label">Calibration Status:</span>
              <span className="fact-value">
                {calibrationStatus || gsdStatus}
              </span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
