"""Model registry for the web app.

Every task maps to one function:

    fn(image: np.ndarray uint8 RGB (H, W, 3), **options) -> np.ndarray uint8 RGB

and a short label that is sent back in the X-Model response header.

HOW TEAMMATES PLUG IN THEIR MODELS
----------------------------------
Person 2 (super_resolution, colorization) and Person 3 (inpainting) write an
inference function with the signature above and register it:

    from app.registry import register
    register("super_resolution", my_sr_fn, "swin2sr-finetuned")

Until a real model is registered, the app uses the classical stand-ins below
so the pipeline can be demoed end to end. Colorization has no stand-in.
"""
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("denoise", "super_resolution", "colorization", "inpainting")
_REGISTRY = {}
_denoiser_cache = {}


class ModelNotIntegrated(Exception):
    """Raised when a task has no model yet."""


def register(task, fn, label):
    if task not in TASKS:
        raise ValueError(f"unknown task {task}")
    _REGISTRY[task] = (fn, label)


def get(task):
    if task not in TASKS:
        raise KeyError(task)
    if task in _REGISTRY:
        return _REGISTRY[task]
    raise ModelNotIntegrated(task)


def status():
    """Which tasks have a model and what it is."""
    out = {}
    for t in TASKS:
        try:
            out[t] = get(t)[1]
        except ModelNotIntegrated:
            out[t] = None
    return out


# ---- denoising -----------------------------------------------------------
def _ddpm_denoise(img, sigma=25, **_):
    from imgrestore.denoise.ddpm_denoiser import load_checkpoint
    ckpt = os.environ["DENOISER_CKPT"]
    if ckpt not in _denoiser_cache:
        _denoiser_cache[ckpt] = load_checkpoint(ckpt)
    f = img.astype(np.float32) / 255.0
    out = _denoiser_cache[ckpt].denoise_image(f, float(sigma) / 255.0)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _nlm_denoise(img, sigma=25, **_):
    h = max(3.0, float(sigma) * 0.9)
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)


# ---- classical stand-ins ---------------------------------------------------
def _bicubic_sr(img, factor=2, **_):
    h, w = img.shape[:2]
    f = int(factor)
    return cv2.resize(img, (w * f, h * f), interpolation=cv2.INTER_CUBIC)


def _telea_inpaint(img, mask=None, **_):
    if mask is None:
        raise ValueError("inpainting needs a mask image (white = missing)")
    m = (mask > 127).astype(np.uint8) * 255
    return cv2.inpaint(img, m, 3, cv2.INPAINT_TELEA)


def install_defaults():
    """Register the best available model per task."""
    if os.environ.get("DENOISER_CKPT") and Path(os.environ["DENOISER_CKPT"]).exists():
        register("denoise", _ddpm_denoise, "ddpm-finetuned")
    else:
        register("denoise", _nlm_denoise, "stand-in:non-local-means")
    register("super_resolution", _bicubic_sr, "stand-in:bicubic")
    register("inpainting", _telea_inpaint, "stand-in:opencv-telea")


install_defaults()
