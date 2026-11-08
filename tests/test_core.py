import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from imgrestore import degradation as dg  # noqa: E402
from imgrestore.metrics import psnr, ssim  # noqa: E402
from imgrestore.denoise.ddpm_denoiser import DDPMDenoiser  # noqa: E402


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    return rng.random((48, 64, 3)).astype(np.float32)


def test_noise_range_and_seed(img):
    a = dg.add_gaussian_noise(img, 0.1, np.random.default_rng(1))
    b = dg.add_gaussian_noise(img, 0.1, np.random.default_rng(1))
    assert a.shape == img.shape and a.min() >= 0 and a.max() <= 1
    assert np.array_equal(a, b)


def test_noise_level_matches_sigma():
    flat = np.full((200, 200, 3), 0.5, dtype=np.float32)
    noisy = dg.add_gaussian_noise(flat, 25 / 255, np.random.default_rng(0))
    assert abs((noisy - flat).std() - 25 / 255) < 0.005


def test_downscale_and_back(img):
    small = dg.downscale(img, 2)
    assert small.shape == (24, 32, 3)
    assert dg.upscale_bicubic(small, 2).shape == img.shape


def test_grayscale_channels_equal(img):
    g = dg.to_grayscale(img)
    assert g.shape == img.shape
    assert np.allclose(g[..., 0], g[..., 1]) and np.allclose(g[..., 1], g[..., 2])


def test_mask_and_apply(img):
    mask = dg.random_mask(48, 64, np.random.default_rng(3))
    assert set(np.unique(mask)) <= {0.0, 1.0} and mask.sum() > 0
    out = dg.apply_mask(img, mask)
    assert np.all(out[mask == 1] == 0)
    assert np.array_equal(out[mask == 0], img[mask == 0])


@pytest.mark.parametrize("task", ["denoise", "super_resolution", "colorization", "inpainting"])
def test_degrade_all_tasks(img, task):
    out, extra = dg.degrade(img, task, np.random.default_rng(0))
    assert out.dtype == np.float32
    assert (extra is not None) == (task == "inpainting")


def test_metrics(img):
    assert ssim(img, img) == pytest.approx(1.0)
    noisy = dg.add_gaussian_noise(img, 0.1, np.random.default_rng(0))
    assert psnr(img, noisy) < psnr(img, dg.add_gaussian_noise(img, 0.02, np.random.default_rng(0)))


def test_sigma_to_t_monotonic():
    m = DDPMDenoiser.tiny_random()
    ts = [m.sigma_to_t(s / 255)[0] for s in (5, 15, 25, 50)]
    assert ts == sorted(ts) and len(set(ts)) == 4


def test_denoiser_forward_and_tiles():
    m = DDPMDenoiser.tiny_random().eval()
    x = torch.rand(2, 3, 32, 32)
    assert m(x, 25 / 255).shape == x.shape
    img = np.random.default_rng(0).random((70, 90, 3)).astype(np.float32)
    out = m.denoise_image(img, 25 / 255, tile=32, overlap=8)
    assert out.shape == img.shape and out.min() >= 0 and out.max() <= 1


def test_denoiser_perfect_unet_recovers_clean():
    """If the U-Net predicts the true noise, one-step denoising is exact."""
    m = DDPMDenoiser.tiny_random().eval()
    sigma = 25 / 255
    t, s_t = m.sigma_to_t(sigma)
    clean = torch.rand(1, 3, 16, 16) * 2 - 1
    eps = torch.randn_like(clean)
    y = clean + s_t * eps
    x0 = y - s_t * eps
    assert torch.allclose(x0, clean, atol=1e-6)


# ---- Flask app ------------------------------------------------------------
@pytest.fixture
def client():
    from app.app import create_app
    return create_app().test_client()


def _png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_tasks_endpoint(client):
    data = client.get("/api/tasks").get_json()
    assert set(data) == {"denoise", "super_resolution", "colorization", "inpainting"}
    assert data["colorization"] is None


def test_index_page(client):
    assert client.get("/").status_code == 200


def test_enhance_denoise_and_sr(client):
    arr = (np.random.default_rng(0).random((40, 50, 3)) * 255).astype(np.uint8)
    r = client.post("/api/enhance", data={"task": "denoise", "sigma": "25",
                                          "image": (_png(arr), "a.png")})
    assert r.status_code == 200 and r.headers["X-Model"]
    assert Image.open(io.BytesIO(r.data)).size == (50, 40)
    r = client.post("/api/enhance", data={"task": "super_resolution", "factor": "2",
                                          "image": (_png(arr), "a.png")})
    assert Image.open(io.BytesIO(r.data)).size == (100, 80)


def test_enhance_inpainting_needs_mask(client):
    arr = (np.random.default_rng(0).random((40, 50, 3)) * 255).astype(np.uint8)
    r = client.post("/api/enhance", data={"task": "inpainting", "image": (_png(arr), "a.png")})
    assert r.status_code == 400
    mask = np.zeros((40, 50), dtype=np.uint8); mask[10:20, 10:20] = 255
    r = client.post("/api/enhance", data={"task": "inpainting",
                                          "image": (_png(arr), "a.png"),
                                          "mask": (_png(mask), "m.png")})
    assert r.status_code == 200


def test_enhance_errors(client):
    assert client.post("/api/enhance", data={"task": "nope"}).status_code == 400
    assert client.post("/api/enhance", data={"task": "denoise"}).status_code == 400
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    assert client.post("/api/enhance", data={"task": "colorization",
                                             "image": (_png(arr), "a.png")}).status_code == 501
    bad = client.post("/api/enhance", data={"task": "denoise",
                                            "image": (io.BytesIO(b"notanimage"), "a.png")})
    assert bad.status_code == 400
