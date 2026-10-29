import { useEffect, useMemo, useState, useRef } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar
} from 'recharts';
import { api } from './api.js';
import EchoSeg from './EchoSeg.jsx';

const MODELS = [
  { id: 'jepa', label: 'JEPA (Latent Predictive ViT)' },
  { id: 'mae', label: 'MAE (Masked Reconstruction)' },
  { id: 'supervised', label: 'Supervised ViT Baseline' },
];

const METHODS = [
  { id: 'mahalanobis', label: 'Mahalanobis Distance (Latent Covariance)' },
  { id: 'energy', label: 'Logit Energy Score (Softmax Temperature)' },
];

const BASE = import.meta.env.VITE_API_URL || '';

function fmt(x, d = 4) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return '—';
  return Number(x).toFixed(d);
}

function StatusBadge({ decision }) {
  if (!decision) return <span className="status-pill neutral">AWAITING SCAN</span>;
  const isPass = decision === 'ID';
  return (
    <span className={`status-pill ${isPass ? 'pass' : 'flag'}`}>
      {isPass ? 'IN-DISTRIBUTION (PASS)' : 'OUT-OF-DISTRIBUTION (FLAGGED)'}
    </span>
  );
}

function useBackend() {
  const [models, setModels] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [datasets, setDatasets] = useState(null);
  const [error, setError] = useState('');

  const reload = async () => {
    try {
      const [m, met, ds] = await Promise.all([
        api.models().catch(() => null),
        api.metrics().catch(() => null),
        api.datasets().catch(() => null),
      ]);
      setModels(m);
      setMetrics(met);
      setDatasets(ds);
    } catch (e) {
      setError(String(e.message || e));
    }
  };

  useEffect(() => { reload(); }, []);
  return { models, metrics, datasets, error, reload };
}

function ClinicalInferenceView({ onActivity }) {
  const [model, setModel] = useState('jepa');
  const [method, setMethod] = useState('mahalanobis');
  const [emethod, setEmethod] = useState('rollout');
  const [etarget, setEtarget] = useState('energy');
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [out, setOut] = useState(null);
  const [compareRows, setCompareRows] = useState(null);
  const [attnData, setAttnData] = useState(null);
  const [dist, setDist] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState('');
  const fileInputRef = useRef(null);

  const handleFile = (f) => {
    if (!f) return;
    if (preview) URL.revokeObjectURL(preview);
    setFile(f);
    setOut(null);
    setCompareRows(null);
    setAttnData(null);
    setErr('');
    setPreview(URL.createObjectURL(f));
  };

  const clearScan = () => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview('');
    setOut(null);
    setCompareRows(null);
    setAttnData(null);
    setDist(null);
    setErr('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // Run clinical safety scoring
  const executeScoring = async () => {
    if (!file) return;
    setBusy(true);
    setErr('');
    try {
      const result = await api.predict(file, model, method);
      setOut(result);
      try {
        const d = await api.scoreDist(model, method);
        setDist(d);
      } catch { /* optional distribution */ }
    } catch (e) {
      setErr(String(e.message || e));
    } finally {
      setBusy(false);
      if (onActivity) onActivity();
    }
  };

  // Run tri-model comparative analysis
  const executeComparison = async () => {
    if (!file) return;
    setBusy(true);
    setErr('');
    try {
      const res = await api.compare(file, method);
      setCompareRows(res.rows || []);
    } catch (e) {
      setErr(String(e.message || e));
    } finally {
      setBusy(false);
      if (onActivity) onActivity();
    }
  };

  // Generate visual attention maps
  const generateAttention = async () => {
    if (!file) return;
    setBusy(true);
    setErr('');
    try {
      const fd = new FormData();
      fd.append('image', file);
      const res = await fetch(`${BASE}/api/attention?model=${model}&view=split&method=${emethod}&target=${etarget}`, {
        method: 'POST',
        body: fd
      });
      if (!res.ok) throw new Error(await res.text());
      setAttnData(await res.json());
    } catch (e) {
      setErr(String(e.message || e));
    } finally {
      setBusy(false);
      if (onActivity) onActivity();
    }
  };

  const isOod = out?.decision === 'OOD';

  return (
    <div className="grid c2">
      {/* Upload and Screening Panel */}
      <div className="card">
        <div className="card-header">
          <div>
            <h3>Diagnostic Scan Evaluation</h3>
            <div className="sub">Input patient chest radiograph (PA view) for real-time distribution audit</div>
          </div>
        </div>

        <div className="form-row">
          <label style={{ fontSize: 12, fontWeight: 600, color: '#475569' }}>Backbone:</label>
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            {MODELS.map(m => <option key={m.id} value={m.id}>{m.label}</option>)}
          </select>

          <label style={{ fontSize: 12, fontWeight: 600, color: '#475569' }}>Metric:</label>
          <select value={method} onChange={(e) => setMethod(e.target.value)}>
            {METHODS.map(m => <option key={m.id} value={m.id}>{m.label}</option>)}
          </select>
        </div>

        <div className="drop-zone" onClick={() => fileInputRef.current && fileInputRef.current.click()}>
          {preview ? (
            <div>
              <img src={preview} alt="Uploaded radiograph" />
              <div style={{ marginTop: 8, fontSize: 12, fontWeight: 600, color: '#334155' }}>{file?.name}</div>
            </div>
          ) : (
            <div>
              <div style={{ fontSize: 24, marginBottom: 6 }}>🩻</div>
              <div style={{ fontWeight: 600, color: '#1e293b' }}>Select or Drag Chest Radiograph (DICOM / PNG / JPEG)</div>
              <div style={{ fontSize: 11.5, color: '#64748b', marginTop: 4 }}>Grayscale radiograph automatically formatted to 224×224 tensor</div>
            </div>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept=".png,.jpg,.jpeg"
            hidden
            onChange={(e) => handleFile(e.target.files[0])}
          />
        </div>

        {file && (
          <div className="form-row" style={{ marginTop: 14 }}>
            <button className="btn" disabled={busy} onClick={executeScoring}>
              {busy ? 'Running Inference…' : 'Evaluate Scan Safety'}
            </button>
            <button className="btn ghost" disabled={busy} onClick={executeComparison}>
              Compare All 3 Encoders
            </button>
            <button className="btn ghost" disabled={busy} onClick={generateAttention}>
              Generate Heatmap
            </button>
            <button className="btn ghost" onClick={clearScan} style={{ marginLeft: 'auto' }}>
              Clear
            </button>
          </div>
        )}

        {err && <div className="err-box" style={{ marginTop: 14 }}>{err}</div>}
      </div>

      {/* Safety Assessment & Decision Panel */}
      <div className="card">
        <div className="card-header">
          <div>
            <h3>Safety Decision Matrix</h3>
            <div className="sub">Calibrated against In-Distribution validation boundary (TPR 95% threshold specification)</div>
          </div>
          {out && <StatusBadge decision={out.decision} />}
        </div>

        {out ? (
          <div>
            <div className={`decision-banner ${isOod ? 'review' : 'pass'}`}>
              <div className="stat-block">
                <span className="stat-label">Clinical Decision</span>
                <span className="stat-value status-text">
                  {out.decision === 'ID' ? 'ACCEPTED (ID)' : 'FLAGGED (OOD)'}
                </span>
              </div>
              <div className="stat-block">
                <span className="stat-label">Anomaly Percentile</span>
                <span className="stat-value">{fmt(out.percentile, 1)}%</span>
              </div>
              <div className="stat-block">
                <span className="stat-label">Normalized Metric</span>
                <span className="stat-value">{fmt(out.normalized, 3)}</span>
              </div>
            </div>

            <div className="kv-grid" style={{ marginBottom: 12 }}>
              <span>Raw Score (Latent Distance):</span>
              <b>{fmt(out.raw, 4)}</b>
              <span>Safety Threshold (95th Percentile):</span>
              <b>{fmt(out.threshold, 4)}</b>
              <span>Target Clinical Specificity:</span>
              <b>{(out.tpr_target * 100).toFixed(0)}% (ID preservation)</b>
              <span>Selected Architecture:</span>
              <b>{out.model} ({out.method})</b>
            </div>

            <div className="sub" style={{ marginBottom: 4 }}>Anomaly Risk Meter:</div>
            <div className="metric-bar-bg">
              <div
                className={`metric-bar-fill ${out.percentile < 70 ? 'safe' : (out.percentile < 95 ? 'warn' : 'alert')}`}
                style={{ width: `${Math.min(100, Math.max(5, out.percentile))}%` }}
              />
            </div>

            <div style={{ fontSize: 11.5, color: '#64748b', lineHeight: 1.4 }}>
              {isOod ? (
                <span style={{ color: '#b91c1c', fontWeight: 500 }}>
                  ⚠️ Scan falls outside the calibrated institutional training manifold. Recommendation: Route to senior thoracic radiologist for manual protocol review.
                </span>
              ) : (
                <span style={{ color: '#047857', fontWeight: 500 }}>
                  ✓ Scan aligns with calibrated institutional training distribution. Safe for downstream automated classification.
                </span>
              )}
            </div>
          </div>
        ) : (
          <div style={{ textAlign: 'center', padding: '36px 16px', color: '#64748b' }}>
            <div style={{ fontSize: 28, marginBottom: 8 }}>📊</div>
            <div style={{ fontWeight: 600 }}>No active scan evaluated</div>
            <div style={{ fontSize: 12, marginTop: 4 }}>Upload a radiograph and click "Evaluate Scan Safety" to run inference.</div>
          </div>
        )}

        {/* 3-Model Comparison Table if requested */}
        {compareRows && (
          <div style={{ marginTop: 20, paddingTop: 16, borderTop: '1px solid #e2e8f0' }}>
            <h4 style={{ margin: '0 0 10px', fontSize: 13, fontWeight: 700 }}>Tri-Model Comparative Assessment</h4>
            <table>
              <thead>
                <tr>
                  <th>Model Architecture</th>
                  <th>Raw Metric</th>
                  <th>Percentile</th>
                  <th>Classification</th>
                </tr>
              </thead>
              <tbody>
                {compareRows.map((r, i) => (
                  <tr key={i}>
                    <td><b>{r.model}</b></td>
                    <td>{fmt(r.raw, 4)}</td>
                    <td>{fmt(r.percentile, 1)}%</td>
                    <td><StatusBadge decision={r.decision} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Visual Explainability Row */}
      {attnData && (
        <div className="card" style={{ gridColumn: 'span 2' }}>
          <div className="card-header">
            <div>
              <h3>Transformer Saliency & Attention Maps</h3>
              <div className="sub">{attnData.method} · Multi-Head Self-Attention Saliency across 12 Transformer Blocks</div>
            </div>
            <div className="form-row" style={{ margin: 0 }}>
              <select value={emethod} onChange={(e) => setEmethod(e.target.value)}>
                <option value="rollout">Attention Rollout (ViT Multi-Head Saliency)</option>
                <option value="gradcam">Grad-CAM (Energy Gradient Saliency)</option>
              </select>
              <button className="btn" disabled={busy} onClick={generateAttention}>Update Heatmap</button>
            </div>
          </div>

          <div className="grid c3" style={{ textAlign: 'center' }}>
            <div>
              <div className="sub" style={{ marginBottom: 6, fontWeight: 600 }}>Original Chest Radiograph</div>
              <img src={attnData.original} alt="Original" style={{ width: '100%', maxWidth: 260, borderRadius: 8, border: '1px solid #e2e8f0' }} />
            </div>
            <div>
              <div className="sub" style={{ marginBottom: 6, fontWeight: 600 }}>Transformer Saliency Heatmap</div>
              <img src={attnData.heatmap} alt="Heatmap" style={{ width: '100%', maxWidth: 260, borderRadius: 8, border: '1px solid #e2e8f0' }} />
            </div>
            <div>
              <div className="sub" style={{ marginBottom: 6, fontWeight: 600 }}>Anatomical Co-Registration</div>
              <img src={attnData.overlay} alt="Overlay" style={{ width: '100%', maxWidth: 260, borderRadius: 8, border: '1px solid #e2e8f0' }} />
            </div>
          </div>
        </div>
      )}

      {/* Distribution Calibration Plot */}
      {dist && (
        <div className="card" style={{ gridColumn: 'span 2' }}>
          <div className="card-header">
            <div>
              <h3>Empirical In-Distribution Reference Curve</h3>
              <div className="sub">{dist.note} · Specificity Threshold: {fmt(dist.threshold, 3)}</div>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={dist.id_scores.map((v, i) => ({ idx: i, score: v, threshold: dist.threshold, scan: out ? out.raw : null }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="idx" label={{ value: 'Sorted Training Samples', position: 'insideBottom', offset: -5, fontSize: 11 }} tick={{ fontSize: 10 }} />
              <YAxis label={{ value: 'Latent Distance', angle: -90, position: 'insideLeft', fontSize: 11 }} tick={{ fontSize: 10 }} />
              <Tooltip />
              <Legend wrapperStyle={{ fontSize: 11.5 }} />
              <Line type="monotone" dataKey="score" name="ID Reference Distribution" stroke="#0284c7" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="threshold" name="95% Acceptance Threshold" stroke="#ef4444" strokeDasharray="4 4" strokeWidth={1.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

function ClinicalMetricsView({ metrics }) {
  if (!metrics) return <div className="card"><h3>Loading Benchmark Registry…</h3></div>;
  const b = metrics.baseline?.data;
  const o = metrics.ood_detection?.data;
  const t = metrics.threshold?.data;
  const r = metrics.robustness?.data;

  const idRows = useMemo(() => {
    if (!b?.methods) return [];
    return Object.entries(b.methods).map(([k, v]) => ({ name: k, ...v.id, best: v.best_epoch }));
  }, [b]);

  const oodRows = useMemo(() => {
    if (!b?.methods) return [];
    const out = [];
    for (const [m, v] of Object.entries(b.methods)) {
      for (const ds of ['Montgomery', 'Shenzhen']) {
        const d = v.ood?.[ds];
        if (d) out.push({ model: m, ds, ...d });
      }
    }
    return out;
  }, [b]);

  const robustChart = useMemo(() => {
    if (!r?.robustness) return [];
    return [0, 1, 2, 3, 4, 5].map(s => ({
      sev: `Severity ${s}`,
      JEPA: r.robustness.JEPA?.contrast?.[s]?.auc,
      MAE: r.robustness.MAE?.contrast?.[s]?.auc,
      Supervised: r.robustness.Supervised?.contrast?.[s]?.auc,
    }));
  }, [r]);

  return (
    <div className="grid">
      {/* Overview Cards */}
      <div className="grid c3">
        <div className="card">
          <h3>Validation Protocol</h3>
          <div className="sub">Cohort Partitioning</div>
          <div className="kv-grid">
            <span>ID Training Cohort:</span><b>{b?.experiment_info?.id_train_samples || 3360} CXR</b>
            <span>ID Validation Cohort:</span><b>{b?.experiment_info?.id_val_samples || 840} CXR</b>
            <span>Montgomery OOD Test:</span><b>{b?.experiment_info?.ood_montgomery_samples || 138} scans</b>
            <span>Shenzhen OOD Test:</span><b>{b?.experiment_info?.ood_shenzhen_samples || 662} scans</b>
          </div>
        </div>

        <div className="card">
          <h3>Architecture Baseline</h3>
          <div className="sub">Standardized ViT-Small Backbone</div>
          <div className="kv-grid">
            <span>Transformer Depth:</span><b>12 Blocks</b>
            <span>Latent Dimension:</span><b>384 Embed Dim</b>
            <span>Attention Heads:</span><b>6 Heads</b>
            <span>Patch Resolution:</span><b>16×16 Tokens</b>
          </div>
        </div>

        <div className="card">
          <h3>Safety Findings</h3>
          <div className="sub">Key Empirical Conclusions</div>
          <div className="kv-grid">
            <span>Best Energy OOD:</span><b>MAE (0.866 AUROC)</b>
            <span>Best Mahalanobis:</span><b>Supervised (0.997 AUROC)</b>
            <span>Contrast Robustness:</span><b>MAE holds 0.983 at L5</b>
            <span>Clinical Standard:</span><b>Mahalanobis + MAE</b>
          </div>
        </div>
      </div>

      {/* ID Classification Benchmark */}
      <div className="card">
        <h3>In-Distribution Diagnostic Accuracy (TB Classification)</h3>
        <div className="sub">Linear probe evaluation with frozen Vision Transformer backbone</div>
        <ResponsiveContainer width="100%" height={210}>
          <BarChart data={idRows.map(x => ({ name: x.name, AUROC: +x.auroc.toFixed(3), Accuracy: +x.accuracy.toFixed(3) }))}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis domain={[0.8, 1.0]} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Bar dataKey="AUROC" name="Area Under ROC" fill="#0284c7" radius={[4, 4, 0, 0]} />
            <Bar dataKey="Accuracy" name="Diagnostic Accuracy" fill="#059669" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
        <table>
          <thead>
            <tr>
              <th>Method</th>
              <th>AUROC</th>
              <th>Accuracy</th>
              <th>Expected Calibration Error (ECE)</th>
              <th>Negative Log-Likelihood</th>
              <th>Optimal Epoch</th>
            </tr>
          </thead>
          <tbody>
            {idRows.map(x => (
              <tr key={x.name}>
                <td><b>{x.name}</b></td>
                <td>{fmt(x.auroc)}</td>
                <td>{fmt(x.accuracy)}</td>
                <td>{fmt(x.ece)}</td>
                <td>{fmt(x.nll)}</td>
                <td>Epoch {x.best}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Multi-Center OOD Detection Benchmark */}
      <div className="card">
        <h3>Cross-Institutional OOD Detection Performance</h3>
        <div className="sub">Energy Score vs Mahalanobis Distance across Montgomery County & Shenzhen Hospital datasets</div>
        <div className="grid c2" style={{ marginBottom: 16 }}>
          {['Montgomery', 'Shenzhen'].map(ds => (
            <div key={ds}>
              <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 6, color: '#334155' }}>
                {ds} Hospital Cohort — AUROC by Metric
              </div>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={oodRows.filter(x => x.ds === ds).map(x => ({
                  name: x.model,
                  Energy: +x.energy_auroc.toFixed(3),
                  Mahalanobis: +x.mahal_auroc.toFixed(3)
                }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                  <YAxis domain={[0.5, 1.0]} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Legend wrapperStyle={{ fontSize: 11.5 }} />
                  <Bar dataKey="Energy" name="Energy Metric" fill="#d97706" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Mahalanobis" name="Mahalanobis Metric" fill="#0284c7" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ))}
        </div>
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Evaluation Site</th>
              <th>Energy AUROC</th>
              <th>Energy FPR@95 (Safety Risk)</th>
              <th>Mahalanobis AUROC</th>
              <th>Mahalanobis FPR@95</th>
            </tr>
          </thead>
          <tbody>
            {oodRows.map((x, i) => (
              <tr key={i}>
                <td><b>{x.model}</b></td>
                <td>{x.ds}</td>
                <td>{fmt(x.energy_auroc)}</td>
                <td style={{ color: x.energy_fpr95 > 0.5 ? '#b91c1c' : '#334155' }}>{fmt(x.energy_fpr95)}</td>
                <td><b>{fmt(x.mahal_auroc)}</b></td>
                <td style={{ color: '#059669', fontWeight: 600 }}>{fmt(x.mahal_fpr95)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Robustness & Corruption Ablation */}
      <div className="card">
        <h3>Clinical Robustness to Imaging Variation (Contrast Perturbation 0→5)</h3>
        <div className="sub">Simulates scanner windowing, exposure shifts, and institutional protocol discrepancies</div>
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={robustChart}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="sev" tick={{ fontSize: 11 }} />
            <YAxis domain={[0.5, 1.0]} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line type="monotone" dataKey="JEPA" stroke="#0284c7" strokeWidth={2.5} />
            <Line type="monotone" dataKey="MAE" stroke="#059669" strokeWidth={2.5} />
            <Line type="monotone" dataKey="Supervised" stroke="#dc2626" strokeWidth={2.5} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState('infer');
  const { models, metrics, datasets, error } = useBackend();
  const [alertCount, setAlertCount] = useState(0);

  const refreshAlerts = async () => {
    try {
      const res = await api.alerts();
      setAlertCount((res.alerts || []).length);
    } catch { /* offline */ }
  };

  useEffect(() => { refreshAlerts(); }, []);

  return (
    <div className="app">
      {/* Sidebar Navigation */}
      <aside className="side">
        <div className="brand">
          <span className="brand-badge">Clinical AI Safety</span>
          <h2>JEPA-Med-OOD</h2>
          <p>Institutional OOD Detection & Cardiac Segmentation</p>
        </div>

        <nav className="nav">
          {[
            ['infer', 'Diagnostic Screening'],
            ['metrics', 'Benchmark Registry'],
            ['echo', 'Echo LV Segmentation'],
            ['specs', 'System Architecture'],
          ].map(([id, label]) => (
            <button
              key={id}
              className={tab === id ? 'active' : ''}
              onClick={() => setTab(id)}
            >
              {label}
            </button>
          ))}
        </nav>

        <div className="sidebar-telemetry">
          <div className="telemetry-row">
            <span>Inference API</span>
            <span>
              <span className={`status-dot ${models ? 'on' : 'off'}`} />
              {models ? 'Connected' : 'Offline'}
            </span>
          </div>
          <div className="telemetry-row">
            <span>Port</span>
            <code>127.0.0.1:8000</code>
          </div>
          <div className="telemetry-row">
            <span>Backbone</span>
            <span>ViT-Small/16</span>
          </div>
        </div>

        <div className="foot">
          v1.4.2 · Clinical Research Platform
        </div>
      </aside>

      {/* Main Viewport */}
      <div className="main">
        <header className="top-bar">
          <div className="page-title">
            <h1>
              {tab === 'infer' && 'Institutional Chest Radiograph Screening'}
              {tab === 'metrics' && 'Multi-Center OOD & Calibration Benchmarks'}
              {tab === 'echo' && 'Echocardiogram Left Ventricular Segmentation'}
              {tab === 'specs' && 'System Architecture & Data Specifications'}
            </h1>
            <p>
              {tab === 'infer' && 'Real-time Out-of-Distribution safety evaluation and multi-head attention explainability'}
              {tab === 'metrics' && 'Comparative results: JEPA vs MAE vs Supervised learning across 4,200 ID and 800 OOD scans'}
              {tab === 'echo' && 'Automated LV cavity delineation and volumetric ejection fraction estimation'}
              {tab === 'specs' && 'Pretraining configurations, tokenization parameters, and cohort boundaries'}
            </p>
          </div>

          <div className="top-telemetry">
            <span className="pill-badge active-spec">
              <span>●</span> SPEC-95: TPR 0.95 Acceptance
            </span>
            <span className="pill-badge">
              ViT-Small (21.6M Params)
            </span>
          </div>
        </header>

        <div className="content">
          {error && <div className="err-box">{error}</div>}

          {tab === 'infer' && <ClinicalInferenceView onActivity={refreshAlerts} />}
          {tab === 'metrics' && <ClinicalMetricsView metrics={metrics} />}
          {tab === 'echo' && <EchoSeg />}

          {tab === 'specs' && (
            <div className="grid c2">
              <div className="card">
                <h3>Cohort Boundaries & Dataset Roles</h3>
                <div className="sub">Guaranteed partitioning: Training data strictly isolated from evaluation cohorts</div>
                <table>
                  <thead>
                    <tr>
                      <th>Cohort Name</th>
                      <th>Clinical Role</th>
                      <th>Sample Breakdown</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td><b>TB Chest Radiography</b></td>
                      <td>In-Distribution (ID) Training & Val</td>
                      <td>3,500 Normal / 700 TB = 4,200 total</td>
                    </tr>
                    <tr>
                      <td><b>Montgomery County CXR</b></td>
                      <td>External OOD Test Set 1</td>
                      <td>138 High-Resolution Scans (Eureka DR)</td>
                    </tr>
                    <tr>
                      <td><b>Shenzhen Hospital CXR</b></td>
                      <td>External OOD Test Set 2</td>
                      <td>662 Clinical Scans (Philips DR)</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div className="card">
                <h3>Model Encoder Inventory</h3>
                <div className="sub">Active Vision Transformer Checkpoints</div>
                <table>
                  <thead>
                    <tr>
                      <th>Encoder Architecture</th>
                      <th>Backbone Status</th>
                      <th>Head Status</th>
                      <th>Status Details</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(models?.models || []).map(m => (
                      <tr key={m.id}>
                        <td><b>{m.label}</b></td>
                        <td><span className="status-pill pass">Active</span></td>
                        <td><span className="status-pill pass">Calibrated</span></td>
                        <td>{m.detail}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <div className="footer-notice">
            JEPA-Med-OOD Clinical Safety Workstation · Investigational Research System · Not a certified diagnostic device
          </div>
        </div>
      </div>
    </div>
  );
}
