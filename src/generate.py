"""
generate.py -- sample images from a trained checkpoint.

Example:
    python src/generate.py --checkpoint runs/my_run/checkpoints/latest.pt --num_images 16
"""

import os
import sys
import argparse

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from torchvision.utils import save_image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Generator
from utils import get_device, save_sample_grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num_images", type=int, default=16)
    p.add_argument("--out", type=str, default="samples_out")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--truncation", type=float, default=1.0,
                    help="Scale noise by this factor before generating. Values below 1.0 "
                         "(try 0.6-0.8) trade some variety for more consistent-looking output.")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--no_ema", dest="use_ema", action="store_false", default=True,
                   help="Use the raw (non-averaged) generator weights instead of the EMA copy")
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    train_args = ckpt.get("args", {})
    img_size = train_args.get("img_size", 256)
    latent_dim = train_args.get("latent_dim", 256)

    G = Generator(latent_dim=latent_dim, img_size=img_size).to(device)
    state_key = "G_ema" if args.use_ema and "G_ema" in ckpt else "G"
    G.load_state_dict(ckpt[state_key])
    G.eval()

    os.makedirs(args.out, exist_ok=True)
    with torch.no_grad():
        z = torch.randn(args.num_images, latent_dim, device=device) * args.truncation
        imgs = G(z)

    save_sample_grid(imgs, os.path.join(args.out, "grid.png"), nrow=max(1, int(args.num_images ** 0.5)))

    imgs01 = (imgs.clamp(-1, 1) + 1) / 2
    for i in range(imgs01.size(0)):
        save_image(imgs01[i], os.path.join(args.out, f"sample_{i:03d}.png"))

    print(f"Saved {args.num_images} images to {args.out} "
          f"(using {'EMA' if state_key == 'G_ema' else 'raw'} weights)")


if __name__ == "__main__":
    main()
