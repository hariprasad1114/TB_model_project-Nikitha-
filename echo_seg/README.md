# Echocardiogram Left-Ventricular Segmentation

LV segmentation on echocardiogram frames. Two paths:

1. **Real pipeline (EchoNet)** — ViT encoder (JEPA / MAE / supervised) + lightweight
   decoder, fine-tuned on EchoNet-Dynamic and evaluated for OOD transfer on
   EchoNet-Pediatric A4C / PSAX.
2. **Runnable demo (no credentialed data)** — compact U-Net trained on procedural
   synthetic echo frames, served by the dashboard's **Echo Seg** tab.

The demo checkpoint is a real network doing real inference, but it is trained on
synthetic frames and is **not** a clinical model. Its synthetic validation Dice
is not comparable to EchoNet results.

## Project structure

```
echo_seg/
├── pretraining/                  # JEPA / MAE / supervised pretraining
│   ├── jepa_pretrain.py
│   ├── mae_pretrain.py
│   └── supervised_pretrain.py
├── segmentation/
│   ├── seg_model.py              # ViT encoder + decoder, losses, Dice
│   ├── finetune.py               # EchoNet-Dynamic fine-tuning
│   ├── evaluate_ood.py           # Dynamic TEST / Pediatric A4C / PSAX
│   ├── unet_model.py             # compact CPU-friendly U-Net (demo)
│   └── train_synthetic.py        # trains the demo checkpoint
├── utils/
│   ├── echo_dataset.py           # EchoNet video/frame loaders
│   ├── synthetic_echo.py         # procedural echo frames + GT masks
│   └── common.py                 # paths, IJEPA imports
├── configs/echo_vit_small.yaml
├── run_experiments.py            # fine-tune + evaluate driver
└── run_all_pretrain.py
```

## Demo: train and serve without EchoNet

```powershell
# ~1-2 min on CPU; writes experiments/echo_seg_demo/{best,final}_model.pth,
# results.json, training_curves.png, samples.png
.venv\Scripts\python.exe echo_seg/segmentation/train_synthetic.py `
    --output_dir experiments/echo_seg_demo

# Dashboard backend (demo mode: fake OOD scores, real demo algorithms) + frontend
$env:DEMO = "1"
.venv\Scripts\python.exe -m uvicorn dashboard.backend.main:app --port 8000 --reload
cd dashboard/frontend; npm run dev
```

Open the **Echo Seg** tab, upload an apical-4-chamber still (or a frame saved
from `experiments/echo_seg_demo/samples.png`), and press **Segment LV**.

If `experiments/echo_seg_demo/best_model.pth` is missing, the dashboard falls
back to a classical Otsu + central-prior + largest-component segmentation, so
the view still works on a fresh checkout.

### Dashboard API

| Endpoint | Description |
|---|---|
| `GET /api/echo/status` | Backend mode (`unet` / `otsu`), checkpoint path, synthetic val Dice |
| `POST /api/echo/segment?threshold=0.5` | Multipart `image`; returns base64 original/mask/overlay PNGs, area fraction, bbox, probabilities |
| `GET /api/attention?method=gradcam&target=energy` | CXR Grad-CAM (existing, real algorithm) |

Set `ECHO_CHECKPOINT` to use a different checkpoint. Checkpoints are dicts:
`{"model": state_dict, "base": int, "img_size": int, "normalize": bool}`.

## Real EchoNet pipeline

Datasets (not redistributed here; request access from the EchoNet authors):

- **EchoNet-Dynamic** → `Datasets/Echo/EchoNet_Dynamic`
- **EchoNet-Pediatric** → `Datasets/Echo/Echonet_Pediatric/{A4C,PSAX}`

```powershell
# 1) Pretrain (ViT-Small; configure batch size in configs/echo_vit_small.yaml)
.venv\Scripts\python.exe echo_seg/pretraining/jepa_pretrain.py `
    --config echo_seg/configs/echo_vit_small.yaml --device cuda:0

# 2) Fine-tune segmentation on EchoNet-Dynamic
.venv\Scripts\python.exe echo_seg/segmentation/finetune.py `
    --encoder_checkpoint experiments/echo_seg_pilot/jepa/checkpoint_ep30.pth `
    --model_name jepa --output_dir experiments/echo_seg_pilot/jepa_seg

# 3) Evaluate in-distribution + OOD
.venv\Scripts\python.exe echo_seg/segmentation/evaluate_ood.py `
    --jepa_model experiments/echo_seg_pilot/jepa_seg/best_model.pth `
    --output_dir experiments/echo_seg_pilot/jepa_seg/ood_eval
```

`run_experiments.py` orchestrates fine-tuning + evaluation for all three encoders.

## Evaluation protocol

- **In-distribution:** EchoNet-Dynamic TEST split (patient-disjoint).
- **OOD:** EchoNet-Pediatric A4C (same view, different population) and PSAX
  (different view). Report Dice and HD95; no OOD data is used for training or
  threshold selection.
- **Demo:** synthetic validation Dice only — an easy procedural task. Never quote
  it as an EchoNet result.

## Limitations

- The demo U-Net is small, CPU-trained, and sees only synthetic fan/speckle
  frames. Domain shift to real ultrasound will be severe.
- The ViT segmentation path requires pretrained encoders and EchoNet frames;
  neither is included in the repository.
- Research prototype only. Not a medical device and not for diagnosis.
