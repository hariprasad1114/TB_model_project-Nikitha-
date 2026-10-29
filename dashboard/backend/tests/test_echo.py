"""Tests for the echo segmentation backends. No training; CPU only."""
import base64
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
import config  # noqa: E402
import echo_inference as E  # noqa: E402
from echo_seg.utils.synthetic_echo import make_echo_frame  # noqa: E402


def _decode(b64, size):
    raw = base64.b64decode(b64)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    return Image.open(io.BytesIO(raw)).convert("L").resize(size, Image.NEAREST)


def _with_otsu(fn):
    """Run fn with the checkpoint forced missing, then restore state."""
    real = config.ECHO_CHECKPOINT
    config.ECHO_CHECKPOINT = BACKEND / "does_not_exist.pth"
    E._cache.clear()
    try:
        return fn()
    finally:
        config.ECHO_CHECKPOINT = real
        E._cache.clear()


def test_status_shape():
    st = E.status()
    assert st["mode"] in ("unet", "otsu")
    assert st["threshold"] == 0.5
    assert isinstance(st["path"], str)


def test_otsu_fallback_returns_mask_and_pngs():
    def run():
        img, gt = make_echo_frame(seed=5, size=160)
        return E.segment(img), gt

    out, gt = _with_otsu(run)
    assert out["method"] == "otsu"
    pred = np.asarray(_decode(out["mask_png"], (160, 160))) > 127
    assert pred.any(), "empty mask"
    dice = 2 * (pred & gt.astype(bool)).sum() / max(pred.sum() + gt.sum(), 1)
    assert dice > 0.7, f"Otsu sanity check failed (dice={dice:.3f})"
    assert 0.01 < out["area_fraction"] < 0.6
    assert out["bbox"] is not None
    assert out["checkpoint"] is None


def test_segment_outputs_are_decodable_pngs():
    def run():
        img, _ = make_echo_frame(seed=17, size=160)
        return E.segment(img)

    out = _with_otsu(run)
    for key in ("original_png", "mask_png", "overlay_png"):
        raw = base64.b64decode(out[key])
        assert raw[:8] == b"\x89PNG\r\n\x1a\n", key
    assert out["width"] <= 320 and out["height"] <= 320
    assert out["threshold"] == 0.5


def test_threshold_is_clamped():
    def run():
        img, _ = make_echo_frame(seed=42, size=96)
        return E.segment(img, threshold=5.0)

    out = _with_otsu(run)
    assert out["threshold"] == 0.95


def test_unet_path_when_checkpoint_exists():
    if not config.ECHO_CHECKPOINT.exists():
        return  # fresh checkout: Otsu fallback is covered above
    img, gt = make_echo_frame(seed=5, size=160)
    out = E.segment(img)
    assert out["method"] == "unet"
    pred = np.asarray(_decode(out["mask_png"], (160, 160))) > 127
    dice = 2 * (pred & gt.astype(bool)).sum() / max(pred.sum() + gt.sum(), 1)
    assert dice > 0.8, f"UNet sanity check failed (dice={dice:.3f})"
    assert out["synthetic"] is True


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed.")
