"""Dataset loading for BSDS500 (train 200 / val 100 / test 200)."""
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "bsds500"
SPLITS = ("train", "val", "test")


def list_images(split, root=DEFAULT_ROOT):
    """Sorted list of image paths for a split."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    folder = Path(root) / split
    paths = sorted(folder.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(
            f"No images in {folder}. Run: python scripts/download_data.py")
    return paths


def load_image(path):
    """Load an image as float32 RGB in [0, 1]."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def random_crop(img, size, rng):
    h, w = img.shape[:2]
    y = int(rng.integers(0, h - size + 1))
    x = int(rng.integers(0, w - size + 1))
    return img[y:y + size, x:x + size]


class PatchDataset:
    """Random clean patches from one split. Used for training.

    Returns channel-first float32 arrays so they can go straight into torch.
    """

    def __init__(self, split="train", patch_size=64, root=DEFAULT_ROOT,
                 length=None, seed=0):
        self.paths = list_images(split, root)
        self.images = [load_image(p) for p in self.paths]
        self.patch_size = patch_size
        self.length = length or len(self.paths) * 16
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = self.images[int(self.rng.integers(0, len(self.images)))]
        patch = random_crop(img, self.patch_size, self.rng)
        if self.rng.random() < 0.5:
            patch = patch[:, ::-1]
        if self.rng.random() < 0.5:
            patch = patch[::-1]
        return np.ascontiguousarray(patch.transpose(2, 0, 1))
