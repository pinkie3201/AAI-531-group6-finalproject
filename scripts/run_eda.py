"""EDA for BSDS500. Writes figures and stats to reports/eda/."""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from imgrestore import degradation as dg  # noqa: E402
from imgrestore.data import SPLITS, list_images, load_image  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports" / "eda"
OUT.mkdir(parents=True, exist_ok=True)


def image_stats():
    rows = []
    for split in SPLITS:
        for p in list_images(split):
            img = load_image(p)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            rows.append({
                "split": split, "file": p.name,
                "height": img.shape[0], "width": img.shape[1],
                "orientation": "landscape" if img.shape[1] > img.shape[0] else "portrait",
                "file_kb": p.stat().st_size / 1024,
                "mean": float(img.mean()), "std": float(img.std()),
                "sharpness": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
                "sat": float(cv2.cvtColor((img * 255).astype(np.uint8),
                                          cv2.COLOR_RGB2HSV)[..., 1].mean() / 255),
                "near_gray": bool(np.abs(img[..., 0] - img[..., 1]).mean() < 0.01
                                  and np.abs(img[..., 1] - img[..., 2]).mean() < 0.01),
            })
    return pd.DataFrame(rows)


def fig_samples():
    paths = list_images("train")
    rng = np.random.default_rng(0)
    pick = rng.choice(len(paths), 8, replace=False)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.5))
    for ax, i in zip(axes.ravel(), pick):
        ax.imshow(load_image(paths[i])); ax.set_title(paths[i].name, fontsize=9)
        ax.axis("off")
    fig.suptitle("Random training images from BSDS500")
    fig.tight_layout(); fig.savefig(OUT / "sample_grid.png", dpi=110); plt.close(fig)


def fig_distributions(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    df.groupby(["split", "orientation"]).size().unstack().plot.bar(ax=axes[0])
    axes[0].set_title("Images per split and orientation"); axes[0].set_xlabel("")
    axes[1].hist(df["mean"], bins=25, color="#4472C4")
    axes[1].set_title("Mean brightness per image"); axes[1].set_xlabel("mean pixel (0-1)")
    axes[2].hist(np.log10(df["sharpness"]), bins=25, color="#70AD47")
    axes[2].set_title("Sharpness (log10 Laplacian variance)")
    fig.tight_layout(); fig.savefig(OUT / "distributions.png", dpi=110); plt.close(fig)


def fig_degradations():
    img = load_image(list_images("train")[3])
    rng = np.random.default_rng(1)
    panels = [("Clean", img)]
    for s in (15, 25, 50):
        panels.append((f"Noise sigma={s}", dg.add_gaussian_noise(img, s / 255, rng)))
    panels.append(("Downscale x2 (shown small)", dg.downscale(img, 2)))
    panels.append(("Grayscale", dg.to_grayscale(img)))
    mask = dg.random_mask(*img.shape[:2], rng)
    panels.append(("Damage mask", np.repeat(mask[..., None], 3, 2)))
    panels.append(("Masked input", dg.apply_mask(img, mask)))
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, (title, im) in zip(axes.ravel(), panels):
        ax.imshow(im); ax.set_title(title, fontsize=10); ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "degradations.png", dpi=110); plt.close(fig)


def main():
    df = image_stats()
    df.to_csv(OUT / "image_stats.csv", index=False)
    summary = {
        "images_per_split": df.groupby("split").size().to_dict(),
        "sizes": {f"{h}x{w}": int(n) for (h, w), n in
                  df.groupby(["height", "width"]).size().items()},
        "orientation": df["orientation"].value_counts().to_dict(),
        "total_file_mb": round(df["file_kb"].sum() / 1024, 1),
        "file_kb_mean": round(df["file_kb"].mean(), 1),
        "mean_brightness": round(df["mean"].mean(), 3),
        "std_brightness_across_images": round(df["mean"].std(), 3),
        "mean_saturation": round(df["sat"].mean(), 3),
        "near_grayscale_images": int(df["near_gray"].sum()),
        "sharpness_min": round(df["sharpness"].min(), 5),
        "sharpness_median": round(df["sharpness"].median(), 5),
        "sharpness_max": round(df["sharpness"].max(), 5),
        "dark_images_mean_lt_0.2": int((df["mean"] < 0.2).sum()),
        "bright_images_mean_gt_0.8": int((df["mean"] > 0.8).sum()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    fig_samples(); fig_distributions(df); fig_degradations()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
