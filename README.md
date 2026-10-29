# JEPA-Med-OOD: Self-Supervised Learning for Out-of-Distribution Detection in Medical Imaging

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.3+-61DAFB.svg?style=flat&logo=react&logoColor=black)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **A Comparative Study of JEPA, MAE, and Supervised Learning for Chest X-Ray OOD Detection and Cardiac Ultrasound Segmentation.**

---

## 📌 Executive Summary

Medical artificial intelligence systems deployed in real-world hospitals regularly face **distribution shift**: variations in imaging scanners, patient demographics, windowing protocols, and regional disease prevalence. When traditional supervised models encounter unfamiliar data, they suffer from **silent, overconfident failures** — assigning high confidence to erroneous diagnoses.

**JEPA-Med-OOD** is a clinical AI safety platform and experimental framework designed to detect when a model is operating outside its training distribution. By benchmarking self-supervised representations (**JEPA**, **MAE**) against standard **Supervised Learning**, this framework introduces an institutional safety filter that intercepts anomalous scans before automated diagnostics are delivered.

---

## 🔬 Core Concepts & Methodology

```
                                [ Input Chest Radiograph ]
                                            │
                       ┌────────────────────┼────────────────────┐
                       ▼                    ▼                    ▼
                1. JEPA Model        2. MAE Model        3. Supervised Model
             (Predicts concepts)  (Reconstructs pixels)  (Memorizes labels)
                       │                    │                    │
                       └────────────────────┬────────────────────┘
                                            │
                                            ▼
                                 [ 384-D Latent Vector ]
                                            │
                       ┌────────────────────┴────────────────────┐
                       ▼                                         ▼
             Mahalanobis Distance                           Energy Score
         (Geometric distance to ID cloud)            (Softmax logit uncertainty)
                       │                                         │
                       └────────────────────┬────────────────────┘
                                            ▼
                              [ Clinical Safety Gate ]
                               IN-DISTRIBUTION (PASS)
                                         OR
                           OUT-OF-DISTRIBUTION (FLAGGED)
```

### The 3 Learning Paradigms Compared
* **JEPA (Joint Embedding Predictive Architecture)**: Predicts abstract latent representations of hidden image regions from surrounding context using an Exponential Moving Average (EMA) target encoder. Learns macroscopic anatomical semantics (rib structures, cardiac borders) without generating pixels.
* **MAE (Masked Autoencoder)**: Masks 75% of input tokens and reconstructs raw pixel values via an asymmetric decoder. Learns fine-grained textures, local contrast, and tissue densities.
* **Supervised Learning**: Standard Vision Transformer baseline directly optimizing cross-entropy loss on Normal vs. Tuberculosis labels.

### OOD Scoring Mechanisms
1. **Mahalanobis Distance**: Models class-conditional Gaussian distributions on the training manifold. Measures exact geometric distance to the nearest class centroid in 384-dimensional latent space.
2. **Energy Score**: Uses unnormalized logit densities ($E(x) = -T \cdot \log \sum e^{f_y(x)/T}$). Distinguishes true diagnostic confidence from uniform uncalibrated softmax outputs.

---

## 📊 Real Empirical Results & Benchmarks

All metrics reported below are calculated on standardized test cohorts using a **Vision Transformer Small (ViT-S/16, 21.6M parameters)** backbone.

### 1. In-Distribution (ID) Diagnostic Classification
*Trained & evaluated on the TB Chest Radiography Database (4,200 images: 3,500 Normal / 700 TB).*

| Model Paradigm | AUROC ↑ | Diagnostic Accuracy ↑ | ECE (Calibration Error) ↓ | Negative Log-Likelihood (NLL) ↓ | Optimal Epoch |
|---|---|---|---|---|---|
| **JEPA** (Self-supervised) | 0.939 | 91.7% | 0.029 | 0.206 | Epoch 48 |
| **MAE** (Self-supervised) | 0.993 | 97.1% | 0.013 | 0.085 | Epoch 50 |
| **Supervised** (Baseline) | **0.996** | **98.0%** | **0.007** | **0.064** | Epoch 45 |

---

### 2. Multi-Center Cross-Institutional OOD Detection
*Evaluated against external hospital cohorts never seen during training: **Montgomery County (USA, 138 scans)** and **Shenzhen Hospital (China, 662 scans)**.*

| Model Architecture | Montgomery Energy AUROC | Montgomery Energy FPR@95 ↓ | Montgomery Mahal AUROC ↑ | Shenzhen Energy AUROC | Shenzhen Energy FPR@95 ↓ | Shenzhen Mahal AUROC ↑ |
|---|---|---|---|---|---|---|
| **JEPA** | 0.771 | 54.0% | 0.956 | 0.718 | 63.3% | 0.985 |
| **MAE** | **0.769** | **57.7%** | **0.994** | **0.866** | **35.8%** | 0.984 |
| **Supervised** | 0.607 | 81.4% | 0.991 | 0.582 | 93.1% | **0.997** |

> **Key Takeaway**: Under confidence-based scoring (Energy), self-supervised pretraining (JEPA & MAE) outperforms supervised models by **16–28 percentage points in AUROC**. When applying Mahalanobis geometric distance, all ViT representations achieve **>95% to 99.7% detection reliability**.

---

### 3. Robustness Against Imaging Protocol Shifts (Severity 0→5)
*Evaluating model degradation under clinical exposure, scanner noise, and contrast variations.*

| Pretraining Paradigm | Clean (AUC) | Gaussian Noise (L5) | Motion / Defocus Blur (L5) | Contrast Perturbation (L5) |
|---|---|---|---|---|
| **JEPA** | 0.990 | 0.983 | 0.981 | 0.672 *(Degrades)* |
| **MAE** | 0.992 | 0.984 | **0.992** | **0.983** *(Robust)* |
| **Supervised** | **0.996** | **0.992** | 0.994 | 0.758 *(Degrades)* |

> **Clinical Recommendation**: **MAE pretraining paired with Mahalanobis distance** provides the highest stability across different medical imaging hardware, maintaining 98.3% diagnostic capacity under severe contrast shifts where JEPA and Supervised models degrade significantly.

---

## 🖥️ Clinical Research Workstation (Dashboard)

The project includes a full-stack clinical workstation built with **FastAPI** (Python backend) and **React 18 / Vite** (modern UI):

* **Diagnostic Screening Panel**: Live single-scan evaluation providing instant Anomaly Percentile, Latent Distance, and Pass/Flag safety decisions.
* **Tri-Model Comparative Matrix**: Side-by-side assessment running the same patient scan through JEPA, MAE, and Supervised ViT backbones simultaneously.
* **Dual Visual Explainability**:
  - *Attention Rollout*: Aggregated multi-head self-attention across 12 transformer layers.
  - *Grad-CAM Saliency*: Gradient-weighted activation map targeting energy score derivatives.
* **Empirical Calibration Plot**: Live interactive distribution curve showing where the patient scan sits relative to the institutional 95% specificity cutoff.
* **Cardiac Ultrasound (Echo Seg)**: Automated Left Ventricular (LV) cavity segmentation on Apical-4-Chamber views with real-time area calculation and myocardial overlay.

---

## 🚀 Quick Setup & Installation

### 1. Prerequisites
* Python 3.10+
* Node.js 18+ and npm
* Git

### 2. Clone Repository
```bash
git clone https://github.com/hariprasad1114/TB_model_project-Nikitha-.git
cd TB_model_project-Nikitha-
```

### 3. Backend Setup
```bash
# Create and activate Python virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# Install backend dependencies
pip install -r dashboard/backend/requirements.txt
```

### 4. Frontend Setup
```bash
cd dashboard/frontend
npm install
cd ../..
```

---

## ⚡ Running the Application

### Option A: Launch Interactive Dashboard

**Terminal 1 — FastAPI Backend (Port 8000):**
```powershell
.venv\Scripts\python.exe -m uvicorn dashboard.backend.main:app --port 8000
```

**Terminal 2 — React Vite Frontend (Port 5173):**
```bash
cd dashboard/frontend
npm run dev
```

Open **[http://localhost:5173](http://localhost:5173)** in your browser.

---

### Option B: Run Experimental Pipeline CLI

To execute training and evaluation stages individually:

```bash
# Linear Probe evaluation
python cxr_ood/evaluation/linear_probe.py \
  --checkpoint experiments/cxr_5pct/checkpoint_latest.pth \
  --data-path Datasets/CXR/TB_Chest_Radiography_Database \
  --output-dir experiments/cxr_5pct/linear_probe

# Cross-Institutional OOD Detection
python cxr_ood/evaluation/ood_detection.py \
  --jepa-checkpoint experiments/cxr_5pct/checkpoint_latest.pth \
  --probe-checkpoint experiments/cxr_5pct/linear_probe/best_linear_probe.pth \
  --data-root Datasets/CXR \
  --split-file experiments/cxr_5pct/linear_probe/dataset_split.json \
  --output-dir experiments/cxr_5pct/ood_detection

# 3-Way Baseline Comparison
python cxr_ood/analysis/baseline_comparison.py \
  --jepa-checkpoint experiments/cxr_5pct/checkpoint_latest.pth \
  --mae-checkpoint experiments/cxr_5pct/baseline_comparison/mae_pretrain/encoder_final.pth \
  --supervised-checkpoint experiments/cxr_5pct/baseline_comparison/supervised_pretrain/best_model.pth \
  --data-root Datasets/CXR \
  --output-dir experiments/cxr_5pct/baseline_comparison
```

---

## 📁 Repository Directory Structure

```
medical-imaging/
├── cxr_ood/                        # Chest X-Ray OOD detection core
│   ├── pretraining/                # JEPA, MAE, and Supervised ViT pretraining
│   ├── evaluation/                 # Linear probe & OOD detection scoring
│   ├── analysis/                   # Baseline comparison & robustness ablation
│   ├── utils/                      # Dataset loaders & image normalization
│   └── configs/                    # ViT-Small YAML configurations
│
├── dashboard/                      # Clinical Research Workstation
│   ├── backend/                    # FastAPI inference & explainability engine
│   │   ├── main.py                 # REST API endpoints & route handlers
│   │   ├── inference.py            # PyTorch ViT feature extraction & Mahalanobis scoring
│   │   ├── echo_inference.py       # Cardiac ultrasound U-Net engine
│   │   └── config.py               # Path configurations & calibration thresholds
│   └── frontend/                   # React 18 + Vite clinical dashboard
│       ├── src/                    # App shell, views, and styling
│       └── dist/                   # Production-ready client bundle
│
├── echo_seg/                       # Echocardiogram Left Ventricular segmentation
│   ├── segmentation/               # U-Net architecture & training routines
│   └── utils/                      # EchoNet loaders & synthetic ultrasound generator
│
├── experiments/                    # Registered experiment results & benchmarks
│   ├── cxr_5pct/                   # Real benchmark JSON metrics & splits
│   └── echo_seg_demo/              # Ultrasound validation metrics
│
├── IJEPA_Meta/                     # Meta I-JEPA core architecture implementation
├── requirements.txt                # Python environment specifications
├── run_full_pipeline.py            # End-to-end 7-stage orchestrator
├── setup_datasets.py               # Dataset ingestion & folder layout tool
└── README.md                       # Comprehensive documentation
```

---

## 🗄️ Dataset Organization

The data loaders expect the following folder structure under `Datasets/`:

```
Datasets/
└── CXR/
    ├── TB_Chest_Radiography_Database/    # In-Distribution (4,200 images)
    │   ├── Normal/                       # 3,500 normal scans
    │   └── Tuberculosis/                 # 700 TB positive scans
    ├── Montgomery TB CXR/                # OOD Cohort 1 (138 scans)
    │   └── images/
    └── Shenzhen TB CXR/                  # OOD Cohort 2 (662 scans)
        └── images/
            └── images/
```

*Datasets can be retrieved from public archives: [TB Radiography Database (Kaggle)](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) and [NLM Open-i Chest Radiographs](https://openi.nlm.nih.gov/).*

---

## ⚖️ Research Disclaimer

This software is an investigational research prototype developed for machine learning robustness evaluation. It is **not a certified medical device** and is not approved for clinical diagnostic procedures.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
