"""
export_onnx.py -- export a trained generator to ONNX.

This is an intermediate step: run this first, then feed the .onnx file
into export_tfjs.py to get a browser-ready TensorFlow.js model. (The .onnx
file is also directly usable with onnxruntime-web, if you'd rather skip
the TensorFlow.js conversion entirely.)

Example:
    python src/export_onnx.py --checkpoint runs/my_run/checkpoints/latest.pt --out exported/model.onnx
"""

import os
import sys
import argparse

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Generator


class ExportWrapper(nn.Module):
    """Rescales output from the model's internal [-1, 1] range to [0, 1],
    so browser code doesn't need to know about that detail."""

    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, z):
        img = self.generator(z)
        return (img + 1.0) / 2.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="exported/model.onnx")
    p.add_argument("--no_ema", dest="use_ema", action="store_false", default=True,
                   help="Export the raw (non-averaged) generator weights instead of the EMA copy")
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = ckpt.get("args", {})
    img_size = train_args.get("img_size", 256)
    latent_dim = train_args.get("latent_dim", 256)

    G = Generator(latent_dim=latent_dim, img_size=img_size)
    state_key = "G_ema" if args.use_ema and "G_ema" in ckpt else "G"
    G.load_state_dict(ckpt[state_key])
    G.eval()

    wrapper = ExportWrapper(G)
    dummy = torch.randn(1, latent_dim)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.onnx.export(
        wrapper, dummy, args.out,
        input_names=["z"], output_names=["image"],
        opset_version=args.opset,
        dynamic_axes=None,  # fixed batch size of 1 keeps browser inference simple
    )
    print(f"Exported ONNX model to {args.out}")
    print(f"  input:  'z'     shape (1, {latent_dim})  -- standard normal noise")
    print(f"  output: 'image' shape (1, 3, {img_size}, {img_size})  -- values in [0, 1], channel-first (NCHW)")


if __name__ == "__main__":
    main()
