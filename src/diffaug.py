"""
diffaug.py -- differentiable augmentation for GAN training.

Applying random augmentations to both real and fake images before they hit
the discriminator is one of the simplest, most effective tricks for
training GANs on small datasets: it effectively multiplies how much
"variety" the discriminator sees, which delays it memorizing the training
set (a common failure mode below a few thousand images).

Independent reimplementation of the general idea from Zhao et al.,
"Differentiable Augmentation for Data-Efficient GAN Training" (2020).
"""

import torch
import torch.nn.functional as F


def rand_brightness(x):
    factor = torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5
    return x + factor


def rand_saturation(x):
    factor = torch.rand(x.size(0), 1, 1, 1, device=x.device) * 2
    x_mean = x.mean(dim=1, keepdim=True)
    return (x - x_mean) * factor + x_mean


def rand_contrast(x):
    factor = torch.rand(x.size(0), 1, 1, 1, device=x.device) + 0.5
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    return (x - x_mean) * factor + x_mean


def rand_translation(x, ratio=0.125):
    b, c, h, w = x.shape
    max_dx, max_dy = int(h * ratio), int(w * ratio)
    shift_x = torch.randint(-max_dx, max_dx + 1, (b,))
    shift_y = torch.randint(-max_dy, max_dy + 1, (b,))
    padded = F.pad(x, [max_dy, max_dy, max_dx, max_dx], mode="reflect")
    out = torch.empty_like(x)
    for i in range(b):
        out[i] = padded[i, :,
                         max_dx + int(shift_x[i]):max_dx + int(shift_x[i]) + h,
                         max_dy + int(shift_y[i]):max_dy + int(shift_y[i]) + w]
    return out


def rand_cutout(x, ratio=0.3):
    b, c, h, w = x.shape
    cut_h, cut_w = int(h * ratio), int(w * ratio)
    mask = torch.ones(b, 1, h, w, device=x.device)
    for i in range(b):
        cy = int(torch.randint(0, h, (1,)))
        cx = int(torch.randint(0, w, (1,)))
        y0, y1 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
        x0, x1 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
        mask[i, :, y0:y1, x0:x1] = 0
    return x * mask


AUGMENT_FNS = {
    "color": [rand_brightness, rand_saturation, rand_contrast],
    "translation": [rand_translation],
    "cutout": [rand_cutout],
}


def diff_augment(x, policy="color,translation,cutout"):
    for p in policy.split(","):
        p = p.strip()
        if not p:
            continue
        for fn in AUGMENT_FNS[p]:
            x = fn(x)
    return x
