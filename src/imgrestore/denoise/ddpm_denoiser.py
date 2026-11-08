"""Denoiser built on a pre-trained DDPM U-Net from Hugging Face.

Idea
----
A DDPM U-Net predicts the noise in x_t = sqrt(a) * x0 + sqrt(1 - a) * eps,
where a is alphas_cumprod[t]. Our noisy image is y = x0 + s * eps (s is the
noise level, in the [-1, 1] image range). Multiplying y by sqrt(a) gives an
x_t if s = sqrt((1 - a) / a). So for a known noise level we:

1. pick the timestep t whose s_t is closest to the real noise level,
2. feed x_t = sqrt(a_t) * y to the U-Net to get eps_hat,
3. estimate the clean image as x0_hat = y - s_t * eps_hat (one step).

Fine-tuning then trains the U-Net so x0_hat matches the clean image
(L2 loss on x0), which adapts the model from generation to restoration.

Model ids are loaded with diffusers.UNet2DModel.from_pretrained. The default,
google/ddpm-cifar10-32, is the smallest official DDPM checkpoint.
"""
import math

import numpy as np
import torch
import torch.nn as nn
from diffusers import DDPMScheduler, UNet2DModel

DEFAULT_MODEL_ID = "google/ddpm-cifar10-32"


class DDPMDenoiser(nn.Module):
    def __init__(self, unet, alphas_cumprod):
        super().__init__()
        self.unet = unet
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())

    # ---- construction ----------------------------------------------------
    @classmethod
    def from_pretrained(cls, model_id=DEFAULT_MODEL_ID):
        unet = UNet2DModel.from_pretrained(model_id)
        try:
            sched = DDPMScheduler.from_pretrained(model_id)
        except Exception:
            sched = DDPMScheduler(num_train_timesteps=1000)
        return cls(unet, sched.alphas_cumprod)

    @classmethod
    def tiny_random(cls):
        """Small random U-Net. Only for tests and CPU smoke runs."""
        unet = UNet2DModel(
            sample_size=32, in_channels=3, out_channels=3, layers_per_block=1,
            block_out_channels=(16, 32, 32),
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
            norm_num_groups=8,
        )
        sched = DDPMScheduler(num_train_timesteps=1000)
        return cls(unet, sched.alphas_cumprod)

    # ---- noise level <-> timestep ---------------------------------------
    def sigma_to_t(self, sigma):
        """sigma is in [0, 1] image units. Returns (t, s_t) in [-1, 1] units."""
        s = 2.0 * sigma
        s_all = torch.sqrt((1 - self.alphas_cumprod) / self.alphas_cumprod)
        t = int(torch.argmin(torch.abs(s_all - s)))
        return t, float(s_all[t])

    # ---- forward ---------------------------------------------------------
    def forward(self, noisy, sigma):
        """noisy: (B, 3, H, W) in [0, 1]. Returns the clean estimate in [0, 1].

        H and W should be multiples of 4 (the U-Net halves the size twice or
        more). Use denoise_image() for arbitrary sizes.
        """
        t, s_t = self.sigma_to_t(sigma)
        y = noisy * 2.0 - 1.0
        a = self.alphas_cumprod[t]
        x_t = torch.sqrt(a) * y
        timesteps = torch.full((noisy.shape[0],), t, device=noisy.device, dtype=torch.long)
        eps_hat = self.unet(x_t, timesteps).sample
        x0 = y - s_t * eps_hat
        return ((x0 + 1.0) / 2.0).clamp(0.0, 1.0)

    # ---- full image inference -------------------------------------------
    @torch.no_grad()
    def denoise_image(self, img, sigma, tile=64, overlap=16, multiple=8, device="cpu"):
        """img: float32 (H, W, 3) in [0, 1]. Tiles the image so attention
        layers stay small, then averages the overlapping regions."""
        self.eval().to(device)
        h, w = img.shape[:2]
        pad_h = (-h) % multiple
        pad_w = (-w) % multiple
        padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        ph, pw = padded.shape[:2]
        x = torch.from_numpy(padded.transpose(2, 0, 1))[None].to(device)
        out = torch.zeros_like(x)
        weight = torch.zeros(1, 1, ph, pw, device=device)
        stride = tile - overlap
        ys = list(range(0, max(ph - tile, 0) + 1, stride))
        xs = list(range(0, max(pw - tile, 0) + 1, stride))
        if ys[-1] != max(ph - tile, 0):
            ys.append(max(ph - tile, 0))
        if xs[-1] != max(pw - tile, 0):
            xs.append(max(pw - tile, 0))
        for y0 in ys:
            for x0 in xs:
                patch = x[:, :, y0:y0 + tile, x0:x0 + tile]
                out[:, :, y0:y0 + tile, x0:x0 + tile] += self(patch, sigma)
                weight[:, :, y0:y0 + tile, x0:x0 + tile] += 1
        out = out / weight
        result = out[0].cpu().numpy().transpose(1, 2, 0)[:h, :w]
        return np.clip(result, 0.0, 1.0).astype(np.float32)


def sample_sigma(rng, lo=10 / 255, hi=50 / 255):
    """Training noise level. Varying it teaches the model several levels."""
    return float(rng.uniform(lo, hi))


def save_checkpoint(model, path):
    torch.save({"unet": model.unet.state_dict(),
                "alphas_cumprod": model.alphas_cumprod,
                "config": dict(model.unet.config)}, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    unet = UNet2DModel.from_config(ckpt["config"])
    unet.load_state_dict(ckpt["unet"])
    model = DDPMDenoiser(unet, ckpt["alphas_cumprod"])
    return model.eval().to(device)
