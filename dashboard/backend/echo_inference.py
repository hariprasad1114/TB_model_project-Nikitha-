"""Echocardiogram LV segmentation for the dashboard.

Two execution paths, both honest about provenance:

* ``unet``     - loads ``ECHO_CHECKPOINT`` (a compact U-Net trained by
                 ``echo_seg/segmentation/train_synthetic.py`` on procedural
                 echo frames).  Real forward pass, but the shipped checkpoint
                 is a synthetic-data demo, not a clinical model.
* ``otsu``     - no checkpoint on disk: classical Otsu threshold + central
                 prior + largest connected component.  Always available, so
                 the Echo view works on a fresh checkout.

The real clinical path remains ``echo_seg/segmentation/finetune.py`` on
EchoNet-Dynamic (ViT encoder + decoder).  This module does not replace it.
"""
from __future__ import annotations

import base64
import io
import json
import sys
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BACKEND = Path(__file__).resolve().parent
REPO_ROOT = BACKEND.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402
from echo_seg.segmentation.unet_model import UNet  # noqa: E402

NORM_MEAN = 0.5
NORM_STD = 0.5
DISPLAY_MAX = 320
DEFAULT_THRESHOLD = 0.5

_lock = threading.Lock()
_cache: dict = {}


# ---------------------------------------------------------------------------
# Checkpoint / status
# ---------------------------------------------------------------------------
def _results_json(ckpt_path: Path):
    p = ckpt_path.parent / "results.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _load_unet():
    """Returns (model, meta) or (None, None) when no checkpoint exists."""
    p = config.ECHO_CHECKPOINT
    if not p.exists():
        return None, None
    key = str(p)
    if key in _cache:
        return _cache[key]
    with _lock:
        if key in _cache:
            return _cache[key]
        ckpt = torch.load(p, map_location="cpu", weights_only=True)
        base = int(ckpt.get("base", 16))
        img_size = int(ckpt.get("img_size", 96))
        model = UNet(in_ch=3, out_ch=1, base=base)
        model.load_state_dict(ckpt["model"])
        model.eval()
        res = _results_json(p) or {}
        meta = {
            "checkpoint": str(p),
            "img_size": img_size,
            "base": base,
            "epoch": ckpt.get("epoch"),
            "val_dice": ckpt.get("val_dice", res.get("best_val_dice")),
            "trained_on": res.get("model", "synthetic frames"),
            "note": res.get("note",
                            "Demo checkpoint trained on procedural frames; "
                            "not a clinical model."),
        }
        _cache[key] = (model, meta)
    return _cache[key]


def status() -> dict:
    model, meta = _load_unet()
    p = config.ECHO_CHECKPOINT
    return {
        "available": model is not None,
        "mode": "unet" if model is not None else "otsu",
        "path": str(p),
        "checkpoint": meta,
        "threshold": DEFAULT_THRESHOLD,
        "note": ("U-Net checkpoint loaded (synthetic-data demo)."
                 if model is not None else
                 "No segmentation checkpoint; using classical Otsu fallback."),
    }


# ---------------------------------------------------------------------------
# Segmentation backends (return a float probability map in [0, 1])
# ---------------------------------------------------------------------------
def _otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 127
    sum_all = float(np.dot(np.arange(256), hist))
    sum_b = 0.0
    w_b = 0.0
    best_var = -1.0
    thresh = 127
    for t in range(256):
        w_b += hist[t]
        if w_b <= 0:
            continue
        w_f = total - w_b
        if w_f <= 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var = var
            thresh = t
    return int(thresh)


def _central_prior(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = 0.58 * h, 0.50 * w
    ry, rx = 0.36 * h, 0.30 * w
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0


def _otsu_mask(gray: np.ndarray) -> np.ndarray:
    """Dark-cavity Otsu mask, restricted to the central LV region."""
    thresh = _otsu_threshold(gray)
    dark = (gray <= max(thresh - 2, 1)) & _central_prior(*gray.shape)
    if not dark.any():
        return np.zeros_like(dark, dtype=bool)
    try:
        from scipy import ndimage
        labels, n = ndimage.label(dark)
        if n == 0:
            return dark
        sizes = ndimage.sum(dark, labels, index=np.arange(1, n + 1))
        keep = int(np.argmax(sizes)) + 1
        comp = labels == keep
        comp = ndimage.binary_fill_holes(comp)
        comp = ndimage.binary_dilation(comp, iterations=2)
        return comp
    except Exception:  # scipy missing: keep the raw central dark region
        return dark


def _predict_unet(model, meta, rgb: Image.Image, threshold: float):
    size = int(meta.get("img_size", 96))
    small = rgb.resize((size, size), Image.BILINEAR)
    arr = np.asarray(small, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    tensor = (tensor - NORM_MEAN) / NORM_STD
    with _lock:
        with torch.no_grad():
            logits = model(tensor)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy().astype(np.float32)
    return prob, "unet"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _fit_display(rgb: Image.Image) -> Image.Image:
    w, h = rgb.size
    scale = DISPLAY_MAX / max(w, h)
    if scale < 1.0:
        rgb = rgb.resize((max(1, int(round(w * scale))),
                          max(1, int(round(h * scale)))), Image.BILINEAR)
    return rgb


def _render(rgb: Image.Image, mask: np.ndarray, prob: np.ndarray):
    """Build original / binary-mask / overlay PNGs at display resolution."""
    disp = _fit_display(rgb)
    size = disp.size
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    mask_img = mask_img.resize(size, Image.NEAREST)
    mask_disp = np.asarray(mask_img) > 127

    base = np.asarray(disp, dtype=np.float32)
    over = base.copy()
    over[mask_disp] = 0.55 * over[mask_disp] + 0.45 * np.array([220, 52, 52],
                                                               dtype=np.float32)
    edge = np.zeros_like(mask_disp)
    try:
        from scipy import ndimage
        edge = mask_disp & ~ndimage.binary_erosion(mask_disp, iterations=2)
    except Exception:
        edge = mask_disp & ~np.roll(mask_disp, 1, axis=0)
    over[edge] = np.array([255, 255, 255], dtype=np.float32)

    return (disp,
            mask_img,
            Image.fromarray(np.clip(over, 0, 255).astype(np.uint8), mode="RGB"))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def segment(img: Image.Image, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Segment the left-ventricle cavity and return PNGs + measurements."""
    threshold = float(min(max(threshold, 0.05), 0.95))
    rgb = img.convert("RGB")
    model, meta = _load_unet()

    if model is not None:
        prob_small, method = _predict_unet(model, meta, rgb, threshold)
        w, h = rgb.size
        prob_img = Image.fromarray((prob_small * 255).astype(np.uint8), mode="L")
        prob = np.asarray(prob_img.resize((w, h), Image.BILINEAR),
                          dtype=np.float32) / 255.0
        mask = prob >= threshold
        model_label = "U-Net (synthetic-data demo)"
    else:
        w, h = rgb.size
        gray = np.asarray(rgb.convert("L"), dtype=np.uint8)
        mask = _otsu_mask(gray)
        # Pseudo-probability so downstream stats are on the same 0-1 scale.
        prob = mask.astype(np.float32)
        method = "otsu"
        model_label = "Otsu fallback (no checkpoint)"

    if mask.any():
        ys, xs = np.where(mask)
        bbox = {"x0": float(xs.min() / w), "y0": float(ys.min() / h),
                "x1": float((xs.max() + 1) / w), "y1": float((ys.max() + 1) / h)}
        mean_prob = float(prob[mask].mean())
    else:
        bbox = None
        mean_prob = 0.0

    original, mask_img, overlay = _render(rgb, mask, prob)
    results = _results_json(config.ECHO_CHECKPOINT)
    return {
        "original_png": _b64(original),
        "mask_png": _b64(mask_img.convert("RGB")),
        "overlay_png": _b64(overlay),
        "width": int(original.size[0]),
        "height": int(original.size[1]),
        "source_width": int(w),
        "source_height": int(h),
        "area_fraction": float(mask.mean()),
        "mask_pixels": int(mask.sum()),
        "bbox": bbox,
        "mean_prob": mean_prob,
        "max_prob": float(prob.max()),
        "threshold": threshold,
        "method": method,
        "model": model_label,
        "checkpoint": meta["checkpoint"] if meta else None,
        "val_dice": meta.get("val_dice") if meta else None,
        "example_val_dice": (results or {}).get("best_val_dice"),
        "synthetic": method == "unet",
        "note": ("LV cavity segmentation. Demo model trained on procedural "
                 "synthetic frames - not for diagnosis."
                 if method == "unet" else
                 "LV cavity segmentation via Otsu threshold; no trained "
                 "checkpoint found."),
    }
