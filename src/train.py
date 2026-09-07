"""
train.py -- train the lightweight GAN from scratch on your own image folder.

Example (first, do a quick smoke test):
    python src/train.py --data data/processed --out runs/smoke_test \
        --img_size 64 --total_iters 100 --sample_every 20 --ckpt_every 50

Then the real run:
    python src/train.py --data data/processed --out runs/my_run --img_size 256

Stop any time with Ctrl+C -- a checkpoint is saved before the script exits,
so you can resume later with:
    python src/train.py --data data/processed --out runs/my_run \
        --img_size 256 --resume runs/my_run/checkpoints/interrupted.pt
"""

import os
import sys
import time
import argparse

# Must be set before any MPS op runs: if the Apple Silicon backend hits an
# operation it doesn't support, fall back to CPU for just that op instead
# of crashing the whole script.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Generator, Discriminator
from dataset import ImageFolderDataset, infinite_loader
from diffaug import diff_augment
from utils import set_seed, get_device, EMA, rotate_checkpoints, save_sample_grid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Folder of training images (after prepare_data.py)")
    p.add_argument("--out", required=True, help="Output folder for checkpoints/samples")
    p.add_argument("--img_size", type=int, default=256, choices=[64, 128, 256])
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--total_iters", type=int, default=40000)
    p.add_argument("--ckpt_every", type=int, default=1000)
    p.add_argument("--sample_every", type=int, default=200)
    p.add_argument("--keep_last", type=int, default=5, help="Number of numbered checkpoints to keep on disk")
    p.add_argument("--recon_weight", type=float, default=1.0)
    p.add_argument("--diffaug_policy", type=str, default="color,translation,cutout")
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default=None, help="Path to a checkpoint .pt file to resume from")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("Warning: training on CPU will be slow. If you're on an M-series Mac, "
              "make sure you installed a recent torch build (MPS should be detected).")

    ckpt_dir = os.path.join(args.out, "checkpoints")
    sample_dir = os.path.join(args.out, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    dataset = ImageFolderDataset(args.data, img_size=args.img_size)
    print(f"Found {len(dataset)} training images in {args.data}")
    if len(dataset) < args.batch_size:
        raise SystemExit(f"Only {len(dataset)} images found, which is smaller than "
                          f"--batch_size {args.batch_size}. Add more images or lower --batch_size.")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True, pin_memory=False,
    )
    data_iter = infinite_loader(loader)

    G = Generator(latent_dim=args.latent_dim, img_size=args.img_size).to(device)
    D = Discriminator(img_size=args.img_size).to(device)
    ema = EMA(G, decay=args.ema_decay)

    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    start_iter = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        ema.shadow.load_state_dict(ckpt["G_ema"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        start_iter = ckpt["iteration"] + 1

    fixed_z = torch.randn(16, args.latent_dim, device=device)

    def save_now(iteration, tag=None):
        ckpt = dict(
            G=G.state_dict(), D=D.state_dict(), G_ema=ema.shadow.state_dict(),
            opt_G=opt_G.state_dict(), opt_D=opt_D.state_dict(),
            iteration=iteration, args=vars(args),
        )
        name = f"ckpt_{iteration:07d}.pt" if tag is None else f"{tag}.pt"
        path = os.path.join(ckpt_dir, name)
        torch.save(ckpt, path)
        torch.save(ckpt, os.path.join(ckpt_dir, "latest.pt"))
        if tag is None:
            rotate_checkpoints(ckpt_dir, keep_last=args.keep_last)
        print(f"  saved checkpoint: {path}")

    print(f"Starting training for {args.total_iters} iterations (already completed: {start_iter})")
    t0 = time.time()
    it = start_iter
    try:
        for it in range(start_iter, args.total_iters):
            real = next(data_iter).to(device)
            b = real.size(0)

            # ---- Discriminator step ----
            opt_D.zero_grad(set_to_none=True)
            z = torch.randn(b, args.latent_dim, device=device)
            with torch.no_grad():
                fake = G(z)

            real_aug = diff_augment(real, args.diffaug_policy)
            fake_aug = diff_augment(fake, args.diffaug_policy)

            real_logits, real_recon = D(real_aug, want_recon=True)
            fake_logits, _ = D(fake_aug, want_recon=False)

            d_real_loss = F.relu(1.0 - real_logits).mean()
            d_fake_loss = F.relu(1.0 + fake_logits).mean()
            recon_target = F.interpolate(real, size=real_recon.shape[-2:], mode="bilinear", align_corners=False)
            d_recon_loss = F.l1_loss(real_recon, recon_target)
            d_loss = d_real_loss + d_fake_loss + args.recon_weight * d_recon_loss
            d_loss.backward()
            opt_D.step()

            # ---- Generator step ----
            opt_G.zero_grad(set_to_none=True)
            z2 = torch.randn(b, args.latent_dim, device=device)
            fake2 = G(z2)
            fake2_aug = diff_augment(fake2, args.diffaug_policy)
            g_logits, _ = D(fake2_aug, want_recon=False)
            g_loss = -g_logits.mean()
            g_loss.backward()
            opt_G.step()
            ema.update(G)

            if it % 20 == 0:
                elapsed = time.time() - t0
                print(f"[{it:6d}/{args.total_iters}] "
                      f"D: {d_loss.item():.3f} (recon {d_recon_loss.item():.3f})  "
                      f"G: {g_loss.item():.3f}  {elapsed:.0f}s elapsed")

            if it % args.sample_every == 0 and it > 0:
                ema.shadow.eval()
                with torch.no_grad():
                    samples = ema.shadow(fixed_z)
                save_sample_grid(samples, os.path.join(sample_dir, f"sample_{it:07d}.png"))

            if it % args.ckpt_every == 0 and it > 0:
                save_now(it)

        save_now(args.total_iters, tag="final")
        print("Training complete.")

    except KeyboardInterrupt:
        print("\nInterrupted -- saving a checkpoint before exiting...")
        save_now(it, tag="interrupted")
        print(f"Safe to stop. Resume with --resume {os.path.join(ckpt_dir, 'interrupted.pt')}")


if __name__ == "__main__":
    main()
