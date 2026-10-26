/**
 * SolarMap-India — Frontend API Service.
 *
 * Interfaces strictly with FastAPI endpoints:
 * - GET  /health
 * - POST /predict
 * - GET  /outputs/{prediction_id}/mask
 * - GET  /outputs/{prediction_id}/overlay
 */

export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL !== undefined
    ? import.meta.env.VITE_API_BASE_URL
    : 'http://localhost:8000'
).replace(/\/$/, '');

/**
 * Resolves an artifact relative URL against the configured API Base URL.
 *
 * @param {string} relativePath - Path such as "/outputs/{id}/mask"
 * @returns {string} Fully resolved URL
 */
export function getArtifactUrl(relativePath) {
  if (!relativePath) return '';
  if (relativePath.startsWith('http://') || relativePath.startsWith('https://')) {
    return relativePath;
  }
  const cleanPath = relativePath.startsWith('/') ? relativePath : `/${relativePath}`;
  // If API_BASE_URL is empty (e.g. dev proxy or same origin), use cleanPath
  return API_BASE_URL ? `${API_BASE_URL}${cleanPath}` : cleanPath;
}

/**
 * Checks backend health and model readiness.
 *
 * @returns {Promise<{ healthy: boolean, data: object|null, error: string|null }>}
 */
export async function checkBackendHealth() {
  const targetUrl = API_BASE_URL ? `${API_BASE_URL}/health` : '/health';
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 6000);

    const res = await fetch(targetUrl, {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    const data = await res.json().catch(() => null);

    if (res.ok && data?.status === 'healthy') {
      return { healthy: true, data, error: null };
    }

    const errorMsg =
      data?.error ||
      (res.status === 503
        ? 'Model checkpoint is not loaded or unavailable on backend.'
        : `Backend responded with HTTP ${res.status}`);

    return { healthy: false, data, error: errorMsg };
  } catch (err) {
    if (err.name === 'AbortError') {
      return { healthy: false, data: null, error: 'Health check timed out after 6s.' };
    }
    return {
      healthy: false,
      data: null,
      error: `Cannot connect to backend at ${API_BASE_URL || 'local server'}. Please ensure FastAPI is running.`,
    };
  }
}

/**
 * Dispatches an aerial or satellite image to POST /predict for Half U-Net segmentation.
 *
 * @param {File|Blob} file - Target image
 * @param {object} options
 * @param {number} [options.threshold=0.5] - Foreground probability threshold
 * @param {number} [options.minComponentArea=20] - Minimum connected pixel area
 * @param {string} [options.filename='sample.png'] - Custom filename if Blob
 * @returns {Promise<object>} PredictResponse data
 */
export async function sendPrediction(file, options = {}) {
  const {
    threshold = 0.5,
    minComponentArea = 20,
    strategy = 'auto',
    custom_gsd,
    calibration_source,
    use_cea_benchmark,
    annual_ghi,
    filename,
  } = options;

  const queryParams = new URLSearchParams({
    threshold: String(threshold),
    min_component_area: String(minComponentArea),
    strategy: String(strategy),
  });

  if (custom_gsd !== undefined && custom_gsd !== null && String(custom_gsd).trim() !== '') {
    queryParams.append('custom_gsd', String(custom_gsd).trim());
  }
  if (calibration_source !== undefined && calibration_source !== null && String(calibration_source).trim() !== '') {
    queryParams.append('calibration_source', String(calibration_source).trim());
  }
  if (use_cea_benchmark) {
    queryParams.append('use_cea_benchmark', 'true');
  }
  if (annual_ghi !== undefined && annual_ghi !== null && String(annual_ghi).trim() !== '') {
    queryParams.append('annual_ghi', String(annual_ghi).trim());
  }

  const targetUrl = API_BASE_URL
    ? `${API_BASE_URL}/predict?${queryParams.toString()}`
    : `/predict?${queryParams.toString()}`;

  const formData = new FormData();
  if (filename && file instanceof Blob && !(file instanceof File)) {
    formData.append('file', file, filename);
  } else {
    formData.append('file', file);
  }

  let res;
  try {
    res = await fetch(targetUrl, {
      method: 'POST',
      body: formData,
    });
  } catch (networkErr) {
    throw new Error(
      `Network error connecting to backend API (${targetUrl}): ${networkErr.message}. Ensure backend is running.`
    );
  }

  let body;
  try {
    body = await res.json();
  } catch {
    throw new Error(`Invalid response format from server (HTTP ${res.status}).`);
  }

  if (!res.ok) {
    const errorSummary = body?.error || body?.detail || `Inference request failed with HTTP ${res.status}`;
    const detailedInfo =
      Array.isArray(body?.detail)
        ? body.detail.map((d) => d.msg || JSON.stringify(d)).join(', ')
        : typeof body?.detail === 'string'
        ? body.detail
        : null;

    const fullMessage = detailedInfo ? `${errorSummary}: ${detailedInfo}` : errorSummary;
    const errorObj = new Error(fullMessage);
    errorObj.status = res.status;
    errorObj.backendData = body;
    throw errorObj;
  }

  return body;
}

/**
 * Loads the bundled benchmark sample image as a File object.
 *
 * @returns {Promise<File>}
 */
export async function loadBundledSampleFile() {
  const res = await fetch('/sample_aerial.png');
  if (!res.ok) {
    throw new Error('Bundled sample aerial image could not be loaded.');
  }
  const blob = await res.blob();
  return new File([blob], '768.0_1.0.png', { type: 'image/png' });
}

/**
 * Checks MongoDB database connectivity and status.
 *
 * @returns {Promise<{ configured: boolean, connected: boolean, status: string, error: string|null }>}
 */
export async function checkDatabaseHealth() {
  const targetUrl = API_BASE_URL ? `${API_BASE_URL}/health/db` : '/health/db';
  try {
    const res = await fetch(targetUrl, {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
    const data = await res.json().catch(() => null);
    if (res.ok && data) {
      return {
        configured: Boolean(data.configured),
        connected: Boolean(data.connected),
        status: data.status || 'unknown',
        error: data.error || null,
      };
    }
    return {
      configured: false,
      connected: false,
      status: 'error',
      error: data?.detail || `HTTP ${res.status}`,
    };
  } catch (err) {
    return {
      configured: false,
      connected: false,
      status: 'offline',
      error: err.message,
    };
  }
}

/**
 * Retrieves paginated prediction history sorted newest first.
 *
 * @param {object} [params]
 * @param {number} [params.skip=0]
 * @param {number} [params.limit=10]
 * @returns {Promise<{ success: boolean, offline: boolean, total: number, skip: number, limit: number, items: Array, error: string|null }>}
 */
export async function getPredictionHistory({ skip = 0, limit = 10 } = {}) {
  const queryParams = new URLSearchParams({
    skip: String(skip),
    limit: String(limit),
  });
  const targetUrl = API_BASE_URL
    ? `${API_BASE_URL}/api/predictions/history?${queryParams.toString()}`
    : `/api/predictions/history?${queryParams.toString()}`;

  try {
    const res = await fetch(targetUrl, {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
    const data = await res.json().catch(() => null);

    if (res.status === 503) {
      return {
        success: false,
        offline: true,
        total: 0,
        skip,
        limit,
        items: [],
        error: data?.detail || 'MongoDB is unconfigured or offline.',
      };
    }

    if (!res.ok) {
      return {
        success: false,
        offline: false,
        total: 0,
        skip,
        limit,
        items: [],
        error: data?.detail || data?.error || `Failed to fetch history (HTTP ${res.status})`,
      };
    }

    return {
      success: true,
      offline: false,
      total: data?.total || 0,
      skip: data?.skip || 0,
      limit: data?.limit || limit,
      items: data?.items || [],
      error: null,
    };
  } catch (err) {
    return {
      success: false,
      offline: true,
      total: 0,
      skip,
      limit,
      items: [],
      error: `Network error reaching history service: ${err.message}`,
    };
  }
}

/**
 * Fetches an individual prediction history record by ID.
 *
 * @param {string} predictionId - UUID4 prediction ID
 * @returns {Promise<{ success: boolean, data: object|null, error: string|null }>}
 */
export async function getPredictionHistoryItem(predictionId) {
  if (!predictionId) {
    return { success: false, data: null, error: 'Prediction ID required.' };
  }
  const targetUrl = API_BASE_URL
    ? `${API_BASE_URL}/api/predictions/history/${predictionId}`
    : `/api/predictions/history/${predictionId}`;

  try {
    const res = await fetch(targetUrl, {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
    const data = await res.json().catch(() => null);

    if (!res.ok) {
      return {
        success: false,
        data: null,
        error: data?.detail || `Prediction record not found (HTTP ${res.status})`,
      };
    }

    return {
      success: true,
      data,
      error: null,
    };
  } catch (err) {
    return {
      success: false,
      data: null,
      error: `Network error retrieving record: ${err.message}`,
    };
  }
}
