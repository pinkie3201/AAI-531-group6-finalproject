"""Fine-tune a Hugging Face DDPM U-Net as an image denoiser on BSDS500.

Run this in Colab (GPU) or on any machine with internet and a GPU:

    python scripts/finetune_denoiser.py --model-id google/ddpm-cifar10-32 \
        --epochs 20 --out checkpoints/denoiser.pt

Quick CPU check that the loop works (random tiny model, no download):

    python scripts/finetune_denoiser.py --tiny --steps 20 --out /tmp/tiny.pt
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from imgrestore.data import PatchDataset  # noqa: E402
from imgrestore.denoise.ddpm_denoiser import (  # noqa: E402
    DEFAULT_MODEL_ID, DDPMDenoiser, sample_sigma, save_checkpoint)
from imgrestore.metrics import psnr  # noqa: E402


def make_batch(ds, batch_size, rng, device):
    clean = torch.from_numpy(np.stack([ds[i] for i in range(batch_size)])).to(device)
    sigma = sample_sigma(rng)
    noisy = (clean + sigma * torch.randn_like(clean)).clamp(0, 1)
    return noisy, clean, sigma


@torch.no_grad()
def validate(model, ds, device, sigma=25 / 255, n=32):
    model.eval()
    clean = torch.from_numpy(np.stack([ds[i] for i in range(n)])).to(device)
    g = torch.Generator(device="cpu").manual_seed(123)
    noise = torch.randn(clean.shape, generator=g).to(device)
    noisy = (clean + sigma * noise).clamp(0, 1)
    out = model(noisy, sigma)
    to_np = lambda x: x.permute(0, 2, 3, 1).cpu().numpy()  # noqa: E731
    c, o, nz = to_np(clean), to_np(out), to_np(noisy)
    model.train()
    return (float(np.mean([psnr(a, b) for a, b in zip(c, nz)])),
            float(np.mean([psnr(a, b) for a, b in zip(c, o)])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--tiny", action="store_true", help="random tiny model (CPU test)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps", type=int, default=None, help="overrides epochs")
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "denoiser.pt"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = DDPMDenoiser.tiny_random() if args.tiny else DDPMDenoiser.from_pretrained(args.model_id)
    model.to(device).train()
    train_ds = PatchDataset("train", args.patch, seed=args.seed)
    val_ds = PatchDataset("val", args.patch, seed=99)
    opt = torch.optim.AdamW(model.unet.parameters(), lr=args.lr, weight_decay=1e-4)

    total = args.steps or args.epochs * args.steps_per_epoch
    log_every = max(1, min(args.steps_per_epoch, total // 5))
    best = -1.0
    t0 = time.time()
    for step in range(1, total + 1):
        noisy, clean, sigma = make_batch(train_ds, args.batch_size, rng, device)
        loss = torch.nn.functional.mse_loss(model(noisy, sigma), clean)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.unet.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == total:
            noisy_db, out_db = validate(model, val_ds, device)
            print(f"step {step}/{total} loss {loss.item():.5f} "
                  f"val PSNR noisy {noisy_db:.2f} -> denoised {out_db:.2f} dB "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if out_db > best:
                best = out_db
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                save_checkpoint(model, args.out)
    print(f"best val PSNR {best:.2f} dB, saved to {args.out}")


if __name__ == "__main__":
    main()
