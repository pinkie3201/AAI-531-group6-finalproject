# Image Restoration and Enhancement with Pre-trained Generative Models

AAI-521 Final Team Project (Extra Credit Option)
University of San Diego

Team members: [fill in]

> Draft status: Sections 1 to 3 are written from work that is done. Section 4 has
> baseline numbers only. Fine-tuned model results are marked TODO until the GPU
> run finishes. Sections for super-resolution, colorization, and inpainting are
> left for their owners.

## Abstract

[TODO: write last, 150 to 250 words.]

## 1. Project Setup

### 1.1 Problem and goals

Old photos get scratched, faded, grainy, and small. This project builds one
system that fixes four common problems in an image: noise, low resolution,
missing color, and missing or damaged regions. We use pre-trained models from
Hugging Face and fine-tune them on our own data. A web app ties the four models
together so a user can upload a picture and pick a fix.

The four tasks are:

1. Denoising: remove noise and keep the detail.
2. Super-resolution: raise the resolution of a small image.
3. Colorization: add color to a grayscale image.
4. Inpainting: fill in missing or damaged parts.

### 1.2 Approach

Each task follows the same steps. We start with a pre-trained model, fine-tune
it on pairs of damaged and clean images, then score it on images it has not
seen. We damage clean photos ourselves, so the clean photo is always the answer
key. This is the main reason one dataset can serve all four tasks.

We compare every model against a simple method that needs no training. If the
fine-tuned model cannot beat the simple method, it is not worth using. For
denoising, the simple methods are Gaussian blur, median filter, bilateral
filter, and non-local means (Buades et al., 2005).

### 1.3 Scope

The project is feasible for three people in one course module because we use
small pre-trained models and a small dataset. We test on synthetic damage,
meaning damage we create by code. Real old photos would need more work and are
outside the scope. The system runs as a Flask web app on a local machine.

## 2. Dataset and Exploratory Data Analysis

### 2.1 Choice of dataset

The assignment does not name a dataset. It suggests ImageNet, COCO, or a custom
set. We picked the Berkeley Segmentation Dataset 500 (BSDS500; Arbeláez et al.,
2011) for three reasons. First, it is small, at 35 MB, so it downloads fast and
trains on a free Colab GPU. Second, it is made of natural color photos, which fit
all four tasks. Third, it already has a train, validation, and test split, and
it is a known benchmark for denoising and super-resolution work. Using the
authors' split also keeps our test images separate from training images.

### 2.2 What is in the data

| Item | Value |
| --- | --- |
| Images | 500 color (RGB) JPEG photos |
| Split | 200 train, 100 validation, 200 test |
| Sizes | 481 x 321 (348 images) or 321 x 481 (152 images) |
| Pixels per image | 154,401 |
| Total size | 35.3 MB (72 KB per image on average) |

We use the images only. The segmentation labels that come with BSDS500 are not
part of this project.

### 2.3 How raw the data is

The photos come as JPEG files at a fixed size, so the dataset authors cropped or
resized them before release. The files are compressed, which means our "clean"
images contain small JPEG artifacts. We treat them as clean anyway, as most
denoising papers do. The dataset does not document which cameras took the
photos. BSDS500 has been used a lot for edge detection and segmentation, and
subsets of it (BSD68 and BSD100) are common test sets for denoising and
super-resolution. That history gives us published numbers to compare against.

### 2.4 EDA results

We ran `scripts/run_eda.py` on all 500 images. Figures are in `reports/eda/`.

- Orientation: 348 images are landscape and 152 are portrait. We keep both and
  train on square patches, so this does not cause a problem.
- Brightness: the mean pixel value is 0.412 on a 0 to 1 scale, with a standard
  deviation of 0.108 across images. Eight images are dark (mean below 0.2) and
  one is bright (mean above 0.8). Most images sit in the middle.
- Color: mean saturation is 0.369, and none of the 500 images is close to
  grayscale. This matters for colorization. We make grayscale inputs ourselves,
  and every image has real color to use as the target.
- Sharpness: the variance of the Laplacian runs from 0.001 to 0.162, with a
  median of 0.022. Smooth images and textured images are both present. Noise
  is easier to see and remove in smooth areas, so this mix is useful for testing.
- Bad data: we found no corrupt files, no duplicate sizes outside the two
  listed, and no grayscale images. We did not remove anything.

Figure references: `sample_grid.png` (random training photos),
`distributions.png` (orientation, brightness, and sharpness), and
`degradations.png` (all four kinds of damage on one photo).

### 2.5 Preprocessing and damage pipeline

Images are scaled to the 0 to 1 range as 32-bit floats. For training we cut
random 64 x 64 patches and flip them at random. Patches let us train in small
batches and give the model many different views of 200 images.

The damage functions live in `src/imgrestore/degradation.py` and take a seed so
runs can be repeated:

- Noise: Gaussian noise added at standard deviation 15, 25, or 50 on the 0 to
  255 scale, then clipped to the valid range.
- Low resolution: downscale by 2 with area averaging. The baseline is bicubic
  upscaling.
- Grayscale: convert to one channel of luminance and copy it to three channels.
- Missing regions: a mask made of 1 to 3 random boxes and 3 to 7 scratch lines.
  Masked pixels are set to zero.

## 3. Denoising Method

### 3.1 Model

A denoising diffusion probabilistic model (DDPM; Ho et al., 2020) trains a
U-Net to predict the noise in an image that was corrupted to some level. We
reuse that idea for denoising. If a noisy image has a known noise level, we find
the diffusion timestep that matches it, ask the U-Net for the noise, and
subtract it. This gives a clean estimate in one step.

In symbols, let `a_t` be the cumulative noise schedule at timestep `t` and
`s_t = sqrt((1 - a_t) / a_t)` be the matching noise level. For a noisy image
`y`, the model input is `sqrt(a_t) * y`, the U-Net output is `eps_hat`, and the
clean estimate is `x0_hat = y - s_t * eps_hat`.

We load the checkpoint `google/ddpm-cifar10-32` from Hugging Face. It is the
smallest official DDPM model. It was trained on 32 x 32 images, but a U-Net is
fully convolutional, so it can run on larger patches. At test time we split an
image into overlapping tiles and average the overlap, which keeps memory low.

### 3.2 Fine-tuning

The pre-trained model was built to generate images, not to restore them. We
fine-tune it with a loss on the clean estimate:

`L = mean((x0_hat - x)^2)`

where `x` is the clean patch. We sample the noise level at random between 10 and
50 for each batch so the model learns more than one level. Settings so far:
AdamW, learning rate 2e-5, batch size 16, 64 x 64 patches, gradient clipping at
1.0. We keep the checkpoint with the best validation PSNR.

[TODO: fill in the final epoch count, training time, and GPU after the Colab run.]

### 3.3 Metrics

We report two numbers:

- PSNR (peak signal-to-noise ratio, in dB). Higher is better. It measures pixel
  error and is the standard score for denoising.
- SSIM (structural similarity, 0 to 1). Higher is better. It compares local
  structure, so it catches blur that PSNR can miss.

Both are computed against the clean image on the 200 test photos.

## 4. Results

### 4.1 Baselines (done)

Mean over the 200 test images. Noise levels are on the 0 to 255 scale. Seeds are
fixed, so the numbers repeat.

PSNR (dB):

| Method | sigma 15 | sigma 25 | sigma 50 |
| --- | --- | --- | --- |
| Noisy input | 24.83 | 20.54 | 15.01 |
| Gaussian blur | 28.06 | 26.01 | 23.24 |
| Median filter | 26.96 | 24.96 | 20.98 |
| Bilateral filter | 28.08 | 21.70 | 15.17 |
| Non-local means | 27.69 | 25.01 | 21.77 |

SSIM:

| Method | sigma 15 | sigma 25 | sigma 50 |
| --- | --- | --- | --- |
| Noisy input | 0.5831 | 0.4083 | 0.2124 |
| Gaussian blur | 0.7909 | 0.6949 | 0.5605 |
| Median filter | 0.7246 | 0.5950 | 0.3815 |
| Bilateral filter | 0.7281 | 0.4544 | 0.2166 |
| Non-local means | 0.7567 | 0.6644 | 0.5385 |

Gaussian blur scores best at all three levels except PSNR at sigma 15, where
bilateral is 0.02 dB higher. That is a low bar. Blur works on this dataset
because many BSDS500 photos are smooth, but it also erases texture, and the
scores do not show that. The bilateral filter used fixed settings, so it falls
apart at high noise. None of these filters was tuned per noise level, so a
tuned version would score higher. The fine-tuned model should be compared to the
best baseline in each column.

### 4.2 Fine-tuned model

[TODO: add the table for the pre-trained model before fine-tuning, the
fine-tuned model, and the best baseline. Add before and after image samples.]

### 4.3 What we can and cannot claim

So far we have shown that the data pipeline, the training loop, and the tiled
inference run, and that the baselines are scored. A 20-step test of the loop on
a small random model ran on CPU. That test only checks that the code works and
says nothing about quality. We have not run the pre-trained weights, because the
workspace used to write this code could not reach Hugging Face. All results for
the fine-tuned model are still to come.

All noise in our tests is synthetic Gaussian noise. Real camera noise and the
noise on scanned old photos look different, so a model that does well here may
do worse on real photos.

## 5. Web Application

The app uses Flask. A page lets the user upload an image, choose a task, and see
the before and after side by side with a download button. The API has three
routes: `GET /`, `GET /api/tasks`, and `POST /api/enhance`. Each task maps to one
function in `app/registry.py`, so each teammate can plug in a model without
changing the app. Until a fine-tuned model is registered, the app uses simple
stand-ins and says which one in the `X-Model` header. We tested the routes with
a pytest suite and by running the server and sending requests to it.

## 6. Super-Resolution and Colorization

[Person 2 to write.]

## 7. Inpainting, Validation, and User Testing

[Person 3 to write.]

## 8. Discussion and Conclusion

[TODO: write last.]

## AI Tool Disclosure

Claude (Anthropic) helped write the code in the `csadiq` branch and the first
draft of Sections 1 to 5. We tested the code and checked the numbers against the
files in `reports/`. [Team to review and edit this statement before submission.]

## References

Arbeláez, P., Maire, M., Fowlkes, C., & Malik, J. (2011). Contour detection and
hierarchical image segmentation. *IEEE Transactions on Pattern Analysis and
Machine Intelligence, 33*(5), 898-916.

Buades, A., Coll, B., & Morel, J. M. (2005). A non-local algorithm for image
denoising. *2005 IEEE Computer Society Conference on Computer Vision and
Pattern Recognition (CVPR)*, 2, 60-65.

Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models.
*Advances in Neural Information Processing Systems, 33*, 6840-6851.

## Appendix

- A. EDA figures: `reports/eda/`
- B. Baseline results table: `reports/denoise_results.csv`
- C. Code: `src/`, `scripts/`, `app/`, `tests/`
