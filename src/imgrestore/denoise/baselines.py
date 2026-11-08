"""Classical denoisers. These are the bar the Hugging Face model must beat."""
import cv2
import numpy as np


def _u8(img):
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def gaussian(img, sigma=1.0):
    return cv2.GaussianBlur(img, (0, 0), sigma)


def median(img, k=3):
    return cv2.medianBlur(_u8(img), k).astype(np.float32) / 255.0


def bilateral(img, d=7, sigma_color=0.1, sigma_space=3.0):
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def nlm(img, sigma):
    """Non-local means. h scales with the noise level."""
    h = max(3.0, sigma * 255.0 * 0.9)
    out = cv2.fastNlMeansDenoisingColored(_u8(img), None, h, h, 7, 21)
    return out.astype(np.float32) / 255.0


BASELINES = {
    "gaussian": lambda img, sigma: gaussian(img, 0.5 + 25 * sigma / 5),
    "median": lambda img, sigma: median(img, 3),
    "bilateral": lambda img, sigma: bilateral(img),
    "nlm": nlm,
}
