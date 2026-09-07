"""
utils.py -- small helpers shared by the training/generation scripts.
"""

import os
import glob
import random
import copy
import numpy as np
import torch
import torchvision.utils as vutils


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(preference="auto"):
    if preference != "auto":
        return torch.device(preference)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class EMA:
    """Keeps an exponential moving average copy of a model's weights.

    GAN generators are noisy from one training step to the next; sampling
    from an EMA copy instead of the live weights gives noticeably cleaner
    images without changing the training dynamics at all.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for shadow_p, p in zip(self.shadow.parameters(), model.parameters()):
            shadow_p.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for shadow_b, b in zip(self.shadow.buffers(), model.buffers()):
            shadow_b.copy_(b)


def rotate_checkpoints(ckpt_dir, keep_last=5):
    """Delete older numbered checkpoints, keeping only the most recent `keep_last`."""
    ckpts = sorted(
        glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[1]),
    )
    for old in ckpts[:-keep_last]:
        try:
            os.remove(old)
        except OSError:
            pass


def save_sample_grid(images, path, nrow=4):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    images = (images.clamp(-1, 1) + 1) / 2  # [-1, 1] -> [0, 1]
    vutils.save_image(images, path, nrow=nrow)
