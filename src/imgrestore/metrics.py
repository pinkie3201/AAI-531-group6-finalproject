"""Image quality metrics. Inputs are float arrays in [0, 1], shape (H, W, 3)."""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(clean, test):
    return float(peak_signal_noise_ratio(clean, test, data_range=1.0))


def ssim(clean, test):
    return float(structural_similarity(clean, test, data_range=1.0, channel_axis=2))


def mean_metrics(pairs):
    """pairs is an iterable of (clean, test). Returns (mean PSNR, mean SSIM)."""
    p, s = [], []
    for clean, test in pairs:
        p.append(psnr(clean, test))
        s.append(ssim(clean, test))
    return float(np.mean(p)), float(np.mean(s))
