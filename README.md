# AAI-521 Final Team Project: Image Restoration and Enhancement

Team project for AAI-521 (Computer Vision), University of San Diego. We build a
system that restores old, damaged, or low-resolution images with pre-trained
Hugging Face models. It handles four tasks: denoising, super-resolution,
colorization, and inpainting. A Flask web app lets a user upload an image and
pick a task.

## Dataset

[BSDS500](https://github.com/BIDS/BSDS500) (Berkeley Segmentation Dataset).
500 color photos of natural scenes, split by the dataset authors into
200 train, 100 validation, and 200 test images. Each image is 481x321 or
321x481 pixels. We use the images only, not the segmentation labels. We make
the damaged versions ourselves from the clean photos (see `src/imgrestore/degradation.py`).

## Repo layout

```
app/                  Flask web app and model registry
  app.py              routes: /, /api/tasks, /api/enhance
  registry.py         where each task's model is plugged in
  templates/index.html
src/imgrestore/
  data.py             dataset loading and random patches
  degradation.py      noise, downscale, grayscale, damage masks
  metrics.py          PSNR and SSIM
  denoise/            denoising baselines and the DDPM-based denoiser
scripts/
  download_data.py    get BSDS500 into data/bsds500/
  run_eda.py          EDA figures and stats -> reports/eda/
  evaluate_denoisers.py  score denoisers on a split
  finetune_denoiser.py   fine-tune the Hugging Face model (use a GPU)
tests/                pytest tests
reports/              EDA output and result tables
Final Report Draft.md draft of the written report
```

## Setup

```bash
pip install -r requirements.txt
python scripts/download_data.py        # clones BSDS500 and builds data/bsds500/
python -m pytest -q tests              # should pass
```

## Run

EDA (writes figures to `reports/eda/`):

```bash
python scripts/run_eda.py
```

Score the classical denoising baselines on the test set:

```bash
python scripts/evaluate_denoisers.py --split test
```

Fine-tune the denoiser (needs internet for Hugging Face and a GPU, for example
Colab):

```bash
python scripts/finetune_denoiser.py --model-id google/ddpm-cifar10-32 --epochs 20 --out checkpoints/denoiser.pt
python scripts/evaluate_denoisers.py --split test --pretrained google/ddpm-cifar10-32 --ckpt checkpoints/denoiser.pt
```

Start the web app:

```bash
python -m app.app                                          # http://127.0.0.1:5000
DENOISER_CKPT=checkpoints/denoiser.pt python -m app.app    # use the fine-tuned denoiser
```

## How the denoiser works

A DDPM U-Net predicts the noise in a noisy image. For a known noise level we
map it to a diffusion timestep, run the U-Net once, and subtract the predicted
noise to get a clean estimate. Fine-tuning trains the U-Net so that estimate
matches the clean image. Details are in `src/imgrestore/denoise/ddpm_denoiser.py`.

## Adding your model to the app (Person 2 and Person 3)

Write a function that takes an RGB `uint8` array of shape (H, W, 3) and returns
an RGB `uint8` array, then register it in `app/registry.py`:

```python
from app.registry import register
register("super_resolution", my_sr_fn, "swin2sr-finetuned")
```

Until then the app uses simple stand-ins (non-local means, bicubic, OpenCV
Telea). Colorization has no stand-in and returns a 501 error.

## Current status

- Done: data pipeline, EDA, degradation functions, denoising baselines,
  fine-tuning script, web app, tests.
- Not done: the fine-tuning run on a GPU, and the super-resolution,
  colorization, and inpainting models.
- Baseline results are in `reports/denoise_results.csv`.

## AI tool disclosure

Claude (Anthropic) helped write the code in the `csadiq` branch. This is
disclosed per the course AI-use policy.

## Branches

`main` holds the shared code. `csadiq` holds Person 1's work.
