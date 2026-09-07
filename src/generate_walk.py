"""
generate_walk.py -- render a smooth latent-space walk as an .mp4 video.

Picks several random "anchor" points in latent space and smoothly
interpolates between them (spherical interpolation, which looks noticeably
smoother than a straight line for Gaussian latent vectors), running the
generator on every interpolated point to produce one video frame. By
default the walk loops back to its starting point, so the video plays
seamlessly on repeat.

Example:
    python src/generate_walk.py --checkpoint runs/my_run/checkpoints/latest.pt \
        --out walk.mp4 --num_anchors 8 --steps_per_segment 30 --fps 24

Needs one extra dependency beyond requirements.txt:
    pip install imageio imageio-ffmpeg
"""

import os
import sys
import argparse

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Generator
from utils import get_device


def slerp(a, b, t):
    """Spherical interpolation between two latent vectors -- looks smoother
    than linear interpolation for vectors drawn from a Gaussian, which is
    what both G's training noise and these anchors are."""
    a_n = a / a.norm()
    b_n = b / b.norm()
    omega = torch.acos((a_n * b_n).sum().clamp(-1 + 1e-7, 1 - 1e-7))
    sin_omega = torch.sin(omega)
    if sin_omega.abs() < 1e-6:  # a and b point almost the same direction
        return (1.0 - t) * a + t * b
    return (torch.sin((1.0 - t) * omega) / sin_omega) * a + (torch.sin(t * omega) / sin_omega) * b


def build_frame_latents(num_anchors, steps_per_segment, latent_dim, truncation, loop, seed):
    if seed is not None:
        torch.manual_seed(seed)
    anchors = [torch.randn(latent_dim) * truncation for _ in range(num_anchors)]
    if loop:
        anchors.append(anchors[0])  # walk back to the start for a seamless loop

    zs = []
    for i in range(len(anchors) - 1):
        a, b = anchors[i], anchors[i + 1]
        for step in range(steps_per_segment):
            zs.append(slerp(a, b, step / steps_per_segment))
    if not loop:
        zs.append(anchors[-1])
    return torch.stack(zs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="walk.mp4")
    p.add_argument("--num_anchors", type=int, default=8, help="Number of random points to walk between")
    p.add_argument("--steps_per_segment", type=int, default=30,
                   help="Interpolated frames rendered between each pair of anchors")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--batch_size", type=int, default=16, help="Frames rendered through the generator at once")
    p.add_argument("--truncation", type=float, default=0.8,
                   help="Scale anchor points by this factor. Lower values (0.6-0.8) tend to give a "
                        "smoother, less erratic-looking walk than full-strength (1.0) noise.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no_loop", dest="loop", action="store_false", default=True,
                   help="Don't walk back to the starting point (video won't loop seamlessly)")
    p.add_argument("--no_ema", dest="use_ema", action="store_false", default=True)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"])
    args = p.parse_args()

    try:
        import imageio
    except ImportError:
        raise SystemExit("This script needs imageio and imageio-ffmpeg. Install with:\n"
                          "  pip install imageio imageio-ffmpeg")

    device = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    train_args = ckpt.get("args", {})
    img_size = train_args.get("img_size", 256)
    latent_dim = train_args.get("latent_dim", 256)

    G = Generator(latent_dim=latent_dim, img_size=img_size).to(device)
    state_key = "G_ema" if args.use_ema and "G_ema" in ckpt else "G"
    G.load_state_dict(ckpt[state_key])
    G.eval()

    zs = build_frame_latents(
        args.num_anchors, args.steps_per_segment, latent_dim,
        args.truncation, args.loop, args.seed,
    ).to(device)
    n_frames = zs.size(0)
    print(f"Rendering {n_frames} frames ({n_frames / args.fps:.1f}s at {args.fps} fps)...")

    writer = imageio.get_writer(args.out, fps=args.fps, macro_block_size=None)
    with torch.no_grad():
        for start in range(0, n_frames, args.batch_size):
            batch_z = zs[start:start + args.batch_size]
            imgs = G(batch_z)
            imgs = ((imgs.clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8)
            imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, 3)
            for frame in imgs:
                writer.append_data(frame)
            done = min(start + args.batch_size, n_frames)
            print(f"  {done}/{n_frames} frames rendered", end="\r")
    writer.close()
    print(f"\nSaved video to {args.out}")


if __name__ == "__main__":
    main()
