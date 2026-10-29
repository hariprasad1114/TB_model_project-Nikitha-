"""Procedural echocardiogram-like frames with ground-truth LV masks.

Why this exists
---------------
The EchoNet-Dynamic / Pediatric datasets are not redistributed with this repo
(and need credentialed download).  To keep the LV-segmentation pipeline and the
dashboard demo runnable end-to-end, this module synthesises apical-4-chamber
style frames: a fan-shaped ultrasound sector with speckle, a bright myocardial
rim and a dark left-ventricular cavity whose elliptical mask is known exactly.

The synthetic data is only for smoke-tests / demos.  The real training path is
`echo_seg/segmentation/finetune.py` on EchoNet-Dynamic.

Everything here is deterministic given `seed`.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Normalisation used by training and inference (must stay in sync).
NORM_MEAN = 0.5
NORM_STD = 0.5


def _ellipse_mask(size: int, cx: float, cy: float,
                  rx: float, ry: float, angle_deg: float) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    a = np.deg2rad(angle_deg)
    x = xx - cx
    y = yy - cy
    xr = x * np.cos(a) + y * np.sin(a)
    yr = -x * np.sin(a) + y * np.cos(a)
    return ((xr / rx) ** 2 + (yr / ry) ** 2) <= 1.0


def _dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask.copy()
    for _ in range(iterations):
        out = (out
               | np.roll(out, 1, axis=0) | np.roll(out, -1, axis=0)
               | np.roll(out, 1, axis=1) | np.roll(out, -1, axis=1))
    return out


def _blur(img: np.ndarray) -> np.ndarray:
    return (img * 4.0
            + np.roll(img, 1, axis=0) + np.roll(img, -1, axis=0)
            + np.roll(img, 1, axis=1) + np.roll(img, -1, axis=1)) / 8.0


def make_echo_frame(seed: int = 0, size: int = 128):
    """Return (PIL RGB image, uint8 mask {0,1}) of a synthetic echo frame."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)

    # --- ultrasound sector ("fan") from an apex near the top centre ---------
    apex_x, apex_y = size * 0.5, size * 0.04
    dx = xx - apex_x
    dy = yy - apex_y
    angle = np.degrees(np.arctan2(np.abs(dx), dy + 1e-6))
    fan = (angle <= rng.uniform(28.0, 36.0)) & (dy > 0) & (dy <= size * 0.92)

    # --- speckle texture ----------------------------------------------------
    tex = rng.normal(0.0, 1.0, (size, size)).astype(np.float32)
    tex = _blur(_blur(tex))
    tex = np.clip(tex, -2.0, 2.0)

    img = np.full((size, size), 10.0, np.float32)
    img += fan * (46.0 + 26.0 * tex)
    # depth attenuation
    img -= fan * np.clip(dy, 0.0, None) / max(size, 1) * 14.0

    # --- LV cavity + myocardium rim ----------------------------------------
    cx = size * (0.50 + rng.uniform(-0.05, 0.05))
    cy = size * (0.58 + rng.uniform(-0.04, 0.04))
    rx = size * rng.uniform(0.13, 0.18)
    ry = size * rng.uniform(0.19, 0.25)
    rot = rng.uniform(-30.0, 30.0)
    lv = _ellipse_mask(size, cx, cy, rx, ry, rot)

    rim = _dilate(lv, 3) & ~lv
    img[rim] += 40.0
    img[lv] = 8.0 + 14.0 * rng.random(int(lv.sum())).astype(np.float32) + 5.0 * tex[lv]

    # A faint second chamber keeps models from collapsing to "one dark blob".
    la = _ellipse_mask(size, cx + size * rng.uniform(-0.02, 0.02),
                       cy - ry * 1.9, rx * 0.75, ry * 0.55, rot * 0.5)
    img[la & fan] = 14.0 + 10.0 * rng.random(int((la & fan).sum())).astype(np.float32)

    img = _blur(img)
    img = np.clip(img, 0.0, 255.0).astype(np.uint8)
    rgb = np.stack([img, img, img], axis=-1)
    return Image.fromarray(rgb), lv.astype(np.uint8)


class SyntheticEchoDataset(Dataset):
    """Deterministic synthetic echo frames + LV masks for segmentation tests."""

    def __init__(self, n: int = 192, img_size: int = 96, seed: int = 0,
                 normalize: bool = True):
        self.n = int(n)
        self.img_size = int(img_size)
        self.seed = int(seed)
        self.normalize = bool(normalize)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        img, mask = make_echo_frame(self.seed * 100003 + idx, size=self.img_size)
        arr = np.asarray(img, dtype=np.uint8)
        # PIL image is (H, W, 3) already
        t = torch.from_numpy(np.array(arr.transpose(2, 0, 1), copy=True)).float() / 255.0
        if self.normalize:
            t = (t - NORM_MEAN) / NORM_STD
        m = torch.from_numpy(mask).float().unsqueeze(0)
        return {"image": t, "mask": m}


if __name__ == "__main__":
    # Quick visual sanity dump: python -m echo_seg.utils.synthetic_echo out.png
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "synthetic_echo_example.png"
    grid = Image.new("RGB", (96 * 4, 96), 0)
    for i in range(4):
        frame, m = make_echo_frame(seed=i, size=96)
        heat = Image.fromarray((m * 255).astype(np.uint8)).convert("RGB")
        blend = Image.blend(frame, heat, 0.45)
        grid.paste(blend, (i * 96, 0))
    grid.save(out)
    print(f"wrote {out}")
