import { useEffect, useRef, useState } from 'react';
import { api } from './api.js';

function fmt(x, d = 3) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return '—';
  return Number(x).toFixed(d);
}

export default function EchoSeg() {
  const [status, setStatus] = useState(null);
  const [statusErr, setStatusErr] = useState('');
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [out, setOut] = useState(null);
  const [threshold, setThreshold] = useState(0.5);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState('');
  const inputRef = useRef(null);

  useEffect(() => {
    api.echoStatus()
      .then(setStatus)
      .catch((e) => setStatusErr(String(e.message || e)));
  }, []);

  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview);
  }, [preview]);

  const onFile = (f) => {
    if (!f) return;
    if (preview) URL.revokeObjectURL(preview);
    setFile(f);
    setOut(null);
    setErr('');
    setPreview(URL.createObjectURL(f));
  };

  const clear = () => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview('');
    setOut(null);
    setErr('');
    if (inputRef.current) inputRef.current.value = '';
  };

  const run = async () => {
    if (!file) return;
    setBusy(true);
    setErr('');
    try {
      setOut(await api.echoSegment(file, threshold));
    } catch (e) {
      setErr(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  const isUnet = status?.mode === 'unet' || status?.checkpoint;

  return (
    <div className="grid c2">
      <div className="card">
        <div className="card-header">
          <div>
            <h3>Echocardiogram Left Ventricle Segmentation</h3>
            <div className="sub">
              Apical-4-Chamber (A4C) View. Automated myocardial endocardial border delineation.
            </div>
          </div>
          <span className={`status-pill ${isUnet ? 'pass' : 'neutral'}`}>
            {isUnet ? 'U-Net Model Active' : 'Otsu Morphological Fallback'}
          </span>
        </div>

        <div className="drop-zone" onClick={() => inputRef.current && inputRef.current.click()}>
          {preview ? (
            <div>
              <img src={preview} alt="Ultrasound still" />
              <div style={{ marginTop: 8, fontSize: 12, fontWeight: 600 }}>{file?.name}</div>
            </div>
          ) : (
            <div>
              <div style={{ fontSize: 24, marginBottom: 6 }}>🫀</div>
              <div style={{ fontWeight: 600, color: '#1e293b' }}>Select or Drag Apical-4-Chamber Frame</div>
              <div style={{ fontSize: 11.5, color: '#64748b', marginTop: 4 }}>Standard ultrasound cine still (PNG / JPG)</div>
            </div>
          )}
          <input
            ref={inputRef}
            type="file"
            accept=".png,.jpg,.jpeg"
            hidden
            onChange={(e) => onFile(e.target.files[0])}
          />
        </div>

        {file && (
          <div className="form-row" style={{ marginTop: 14 }}>
            <button className="btn" disabled={busy} onClick={run}>
              {busy ? 'Delineating Cavity…' : 'Segment Left Ventricle'}
            </button>
            <button className="btn ghost" onClick={clear}>Remove</button>
          </div>
        )}

        <div style={{ marginTop: 16 }}>
          <label style={{ display: 'block', fontSize: 12, fontWeight: 600, color: '#475569', marginBottom: 4 }}>
            Probability Cutoff Threshold: <b>{fmt(threshold, 2)}</b>
          </label>
          <input
            type="range"
            min="0.05"
            max="0.95"
            step="0.05"
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
            style={{ width: '100%', accentColor: '#0284c7' }}
          />
        </div>

        {err && <div className="err-box" style={{ marginTop: 14 }}>{err}</div>}
      </div>

      <div className="card">
        <div className="card-header">
          <div>
            <h3>Ventricular Boundary Delineation</h3>
            <div className="sub">Calibrated binary mask and anatomical transparency overlay</div>
          </div>
        </div>

        {out ? (
          <div>
            <div className="grid c3" style={{ textAlign: 'center', marginBottom: 16 }}>
              {[
                ['Source Frame', out.original_png],
                ['Predicted LV Mask', out.mask_png],
                ['Myocardial Overlay', out.overlay_png],
              ].map(([label, b64]) => (
                <div key={label}>
                  <div style={{ fontSize: 11.5, fontWeight: 600, marginBottom: 6, color: '#334155' }}>{label}</div>
                  <img
                    src={b64}
                    alt={label}
                    style={{ width: '100%', maxWidth: 170, borderRadius: 6, border: '1px solid #e2e8f0' }}
                  />
                </div>
              ))}
            </div>

            <div className="kv-grid">
              <span>Segmented LV Cavity Area:</span>
              <b>{out.area_px?.toLocaleString() || 0} pixels</b>
              <span>Fraction of Field:</span>
              <b>{fmt((out.fraction || 0) * 100, 2)}%</b>
              <span>Bounding Box:</span>
              <b>{out.bbox ? `[${out.bbox.join(', ')}]` : '—'}</b>
              <span>Algorithm Method:</span>
              <b>{out.method_used?.toUpperCase() || 'U-NET'}</b>
            </div>
          </div>
        ) : (
          <div style={{ textAlign: 'center', padding: '36px 16px', color: '#64748b' }}>
            <div style={{ fontSize: 28, marginBottom: 8 }}>🎯</div>
            <div style={{ fontWeight: 600 }}>Awaiting Ultrasound Scan</div>
            <div style={{ fontSize: 12, marginTop: 4 }}>Upload an echocardiogram frame and click "Segment Left Ventricle".</div>
          </div>
        )}
      </div>
    </div>
  );
}
