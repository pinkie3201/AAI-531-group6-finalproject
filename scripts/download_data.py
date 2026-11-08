"""Download BSDS500 and lay it out as data/bsds500/{train,val,test}/*.jpg.

BSDS500 (Arbelaez et al., 2011) has 500 natural photos: 200 train, 100 val,
200 test. We use the images only, not the segmentation labels.

Usage:
    python scripts/download_data.py                 # clones from GitHub
    python scripts/download_data.py --src PATH      # reuse an existing clone
"""
import argparse
import shutil
import subprocess
from pathlib import Path

REPO = "https://github.com/BIDS/BSDS500"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "bsds500"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="path to an existing BSDS500 clone")
    args = ap.parse_args()

    if args.src:
        src = Path(args.src)
    else:
        src = ROOT / "data" / "_src"
        if not src.exists():
            subprocess.run(["git", "clone", "--depth", "1", REPO, str(src)], check=True)

    images = src / "BSDS500" / "data" / "images"
    if not images.exists():
        images = src / "data" / "images"
    for split in ("train", "val", "test"):
        dest = OUT / split
        dest.mkdir(parents=True, exist_ok=True)
        files = sorted((images / split).glob("*.jpg"))
        for f in files:
            shutil.copy2(f, dest / f.name)
        print(f"{split}: {len(files)} images -> {dest}")


if __name__ == "__main__":
    main()
