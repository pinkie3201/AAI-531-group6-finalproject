"""Functions that damage clean images so we get (input, target) pairs.

All images are float32 numpy arrays shaped (H, W, 3) with values in [0, 1].
Every random function takes a numpy Generator so results can be repeated.
Person 2 and Person 3 reuse these for super-resolution, colorization and
inpainting.
"""
import cv2
import numpy as np


def add_gaussian_noise(img, sigma, rng):
    """Add white Gaussian noise. sigma is in [0, 1] units (25/255 is common)."""
    noisy = img + rng.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def downscale(img, factor):
    """Shrink the image by an integer factor (used for super-resolution)."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // factor, h // factor), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32)


def upscale_bicubic(img, factor):
    """Bicubic upscale. This is the baseline for super-resolution."""
    h, w = img.shape[:2]
    big = cv2.resize(img, (w * factor, h * factor), interpolation=cv2.INTER_CUBIC)
    return np.clip(big, 0.0, 1.0).astype(np.float32)


def to_grayscale(img):
    """Convert to gray but keep 3 channels so the shapes still match."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return np.repeat(gray[..., None], 3, axis=2).astype(np.float32)


def random_mask(h, w, rng, kind="mixed"):
    """Make a damage mask. 1 means the pixel is missing, 0 means it is kept."""
    mask = np.zeros((h, w), dtype=np.uint8)
    if kind in ("box", "mixed"):
        for _ in range(int(rng.integers(1, 4))):
            bh = int(rng.integers(h // 8, h // 3))
            bw = int(rng.integers(w // 8, w // 3))
            y = int(rng.integers(0, h - bh))
            x = int(rng.integers(0, w - bw))
            mask[y:y + bh, x:x + bw] = 1
    if kind in ("scratch", "mixed"):
        for _ in range(int(rng.integers(3, 8))):
            p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            cv2.line(mask, p1, p2, 1, int(rng.integers(2, 6)))
    return mask.astype(np.float32)


def apply_mask(img, mask):
    """Blank out the masked pixels (set them to 0)."""
    return (img * (1.0 - mask[..., None])).astype(np.float32)


def degrade(img, task, rng, sigma=25 / 255, factor=2):
    """Return (degraded_input, extra) for one of the four tasks.

    extra is the mask for inpainting, otherwise None.
    """
    if task == "denoise":
        return add_gaussian_noise(img, sigma, rng), None
    if task == "super_resolution":
        return downscale(img, factor), None
    if task == "colorization":
        return to_grayscale(img), None
    if task == "inpainting":
        mask = random_mask(img.shape[0], img.shape[1], rng)
        return apply_mask(img, mask), mask
    raise ValueError(f"unknown task: {task}")
