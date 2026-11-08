"""Score denoisers on a split at several noise levels.

Examples
--------
Classical baselines only (works offline):
    python scripts/evaluate_denoisers.py --split test

Add a fine-tuned DDPM checkpoint:
    python scripts/evaluate_denoisers.py --split test --ckpt checkpoints/denoiser.pt

Add the un-tuned Hugging Face model for the before/after fine-tuning row:
    python scripts/evaluate_denoisers.py --split test --pretrained google/ddpm-cifar10-32
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from imgrestore import degradation as dg  # noqa: E402
from imgrestore.data import list_images, load_image  # noqa: E402
from imgrestore.denoise.baselines import BASELINES  # noqa: E402
from imgrestore.metrics import psnr, ssim  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--sigmas", type=int, nargs="+", default=[15, 25, 50])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ckpt")
    ap.add_argument("--pretrained")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "reports" / "denoise_results.csv"))
    args = ap.parse_args()

    methods = dict(BASELINES)
    if args.ckpt or args.pretrained:
        from imgrestore.denoise.ddpm_denoiser import DDPMDenoiser, load_checkpoint
        model = (load_checkpoint(args.ckpt) if args.ckpt
                 else DDPMDenoiser.from_pretrained(args.pretrained))
        name = "ddpm_finetuned" if args.ckpt else "ddpm_pretrained"
        methods[name] = lambda img, sigma: model.denoise_image(img, sigma)

    paths = list_images(args.split)[: args.limit]
    rows = []
    for sigma_int in args.sigmas:
        sigma = sigma_int / 255
        rng = np.random.default_rng(args.seed + sigma_int)
        scores = {"noisy": ([], [])}
        scores.update({m: ([], []) for m in methods})
        for p in paths:
            clean = load_image(p)
            noisy = dg.add_gaussian_noise(clean, sigma, rng)
            scores["noisy"][0].append(psnr(clean, noisy))
            scores["noisy"][1].append(ssim(clean, noisy))
            for name, fn in methods.items():
                out = fn(noisy, sigma)
                scores[name][0].append(psnr(clean, out))
                scores[name][1].append(ssim(clean, out))
        for name, (ps, ss) in scores.items():
            rows.append({"split": args.split, "sigma": sigma_int, "method": name,
                         "psnr_db": round(float(np.mean(ps)), 2),
                         "ssim": round(float(np.mean(ss)), 4), "n_images": len(paths)})
        print(f"sigma={sigma_int} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df.pivot(index="method", columns="sigma", values="psnr_db"))
    print(df.pivot(index="method", columns="sigma", values="ssim"))


if __name__ == "__main__":
    main()
