"""
prepare_data.py -- validate and standardize a folder of raw photos before training.

What it does:
  - reads every image in --input
  - fixes phone-photo rotation (EXIF orientation)
  - converts everything to RGB JPEG
  - resizes so the shorter side is --size pixels to make it square
  - skips and reports any file it can't read, instead of crashing

Usage:
    python src/prepare_data.py --input data/raw --output data/processed --size 300
"""

import argparse
import os
from PIL import Image, ImageOps
from tqdm import tqdm

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder of raw images")
    parser.add_argument("--output", required=True, help="Folder to write cleaned images to")
    parser.add_argument("--size", type=int, default=300,
                         help="Images are resized so their shorter side is this many pixels. "
                              "Keep this a bit above your training --img_size so train.py's "
                              "random crop has room to move (default 300 works for 256).")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    files = [f for f in sorted(os.listdir(args.input)) if f.lower().endswith(IMG_EXTS)]
    if not files:
        raise SystemExit(f"No images found in {args.input}")

    kept, skipped = 0, 0
    for fname in tqdm(files, desc="Processing images"):
        src_path = os.path.join(args.input, fname)
        try:
            img = Image.open(src_path)
            img = ImageOps.exif_transpose(img)  # fix phone photo rotation
            img = img.convert("RGB")
            w, h = img.size
            scale = args.size / min(w, h)
            new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)
            out_name = f"{os.path.splitext(fname)[0]}.jpg"
            img.save(os.path.join(args.output, out_name), quality=95)
            kept += 1
        except Exception as e:
            print(f"  Skipping {fname}: {e}")
            skipped += 1

    print(f"\nDone. {kept} images written to {args.output}, {skipped} skipped.")
    if kept < 500:
        print(f"Note: only {kept} images. This kit is designed for roughly 600-1000+; "
              f"fewer images means a higher risk of the model just memorizing the dataset.")


if __name__ == "__main__":
    main()
