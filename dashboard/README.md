# JEPA-Med-OOD dashboard (read-only front end + inference API)

FastAPI backend + React front end for CXR OOD scoring, explainability, and
echocardiogram LV segmentation.

## Run

Backend (repo root):

```powershell
# real mode: serves checkpoints/metrics if present, otherwise 404 "not found"
.venv\Scripts\python.exe -m uvicorn dashboard.backend.main:app --port 8000

# demo mode: fake scores + real algorithms on an untrained ViT, with a permanent
# "DEMO DATA - NOT REAL RESULTS" banner
$env:DEMO = "1"
.venv\Scripts\python.exe -m uvicorn dashboard.backend.main:app --port 8000
# equivalent: .venv\Scripts\python.exe dashboard/backend/main.py --demo
```

Frontend: `cd dashboard/frontend && npm install && npm run dev` (port 5173,
proxies `/api` and `/results` to `localhost:8000`).

The Echo tab works on a fresh checkout via the Otsu fallback. To use the demo
U-Net, first run `python echo_seg/segmentation/train_synthetic.py` (writes
`experiments/echo_seg_demo/best_model.pth`, gitignored).

## Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/models` | Checkpoint/head availability per encoder |
| `GET /api/metrics` | Existing result files (`found: false` when missing) |
| `GET /api/dataset-info` | Dataset counts/roles |
| `POST /api/predict?model=&method=` | OOD score (energy / Mahalanobis) |
| `POST /api/compare?method=` | All three encoders |
| `POST /api/attention?method=rollout\|gradcam&target=energy\|logit` | Rollout or Grad-CAM PNGs |
| `GET /api/score-distribution` | Cached ID-validation scores + threshold |
| `GET /api/echo/status` | Active echo backend (`unet` / `otsu`), checkpoint metadata |
| `POST /api/echo/segment?threshold=0.5` | LV mask + overlay PNGs, area, bbox, probabilities |
| `GET /api/alerts`, `POST /api/alerts/{id}/ack` | Research flags |

## Data honesty

- Reference mockup numbers (0.82, 0.92, ...) are placeholders; never hardcoded.
- Every number comes from `CHECKPOINT_DIR` / `RESULTS_DIR` / live inference.
- Missing checkpoints -> "Checkpoint not found"; missing metrics -> "Not computed".
- AUPR and OOD-Accuracy are not in any result file; their charts show "Not computed".
- 0-1 OOD score formula: `normalized = P(ID_validation_score <= raw)`
  (empirical percentile within cached ID-validation scores; higher = more OOD).
- Threshold: ID 95th percentile (`TPR_TARGET`), ID data only, never OOD.
- **Attention rollout** = forward hooks, Abnarain-style rollout across 12 blocks;
  response header and UI label say "attention-rollout (not Grad-CAM)".
- **Grad-CAM** = real gradient of the Energy OOD score or predicted-class logit
  w.r.t. the last block's patch tokens, channel-averaged, ReLU, 14x14, bilinear
  224x224. Mahalanobis has no gradients and returns 400, never a substitute.
- **Demo explainability** (no trained checkpoint): the rollout/Grad-CAM algorithms
  run on a seeded, untrained ViT-Small + linear head. Responses carry
  `"demo": true`; the UI states the heatmap is not clinically meaningful.
- **Echo segmentation**: U-Net forward pass when `ECHO_CHECKPOINT` exists (default
  `experiments/echo_seg_demo/best_model.pth`), otherwise Otsu + central prior +
  largest component. The shipped checkpoint is trained on procedural synthetic
  frames; its Dice is not an EchoNet result. `demo: true` + `banner` are set in
  demo mode, and the payload always includes a synthetic-data note.
- Uploads: max 10 MB, PNG/JPG/JPEG, in-memory only, grayscale->RGB as the repo does.
- Paths from `CHECKPOINT_DIR`, `RESULTS_DIR`, `DATA_ROOT`, `ECHO_CHECKPOINT` env
  vars (repo-relative defaults).

Footer/About: "Research prototype for OOD detection. Not a medical device and not
for diagnosis."
