"""
model.py -- Lightweight GAN (FastGAN-style) generator and discriminator.

Designed for unconditional image generation at up to 256x256 resolution,
trained from scratch on small datasets (a few hundred to a couple thousand
images). Two ideas make this practical on a laptop with limited data:

1. Skip-Layer Excitation (SLE) in the generator: low-resolution features
   gate high-resolution features (a cheap squeeze-and-excite). This
   improves gradient flow and lets the network learn a sensible layout
   faster, which matters a lot when you don't have much data.

2. A small "reconstruction" decoder attached to the discriminator: it
   reconstructs the input image from an intermediate discriminator
   feature map. This extra self-supervised task keeps the discriminator's
   features meaningful (tied to actual image content) instead of just
   memorizing "real vs fake" -- which is the single biggest failure mode
   when training GANs on small datasets.

This is a simplified, independent reimplementation for teaching purposes,
inspired by the general ideas in Liu et al., "Towards Faster and
Stabilized GAN Training for High-Fidelity Few-Shot Image Synthesis"
(FastGAN, ICLR 2021) -- not a copy of the official code.
"""

import torch
import torch.nn as nn

# Channel count used at each spatial resolution. The same table is used
# by both the generator (small -> large) and the discriminator
# (large -> small), so the two networks are roughly mirror images.
CHANNELS = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 32, 256: 16}

SUPPORTED_SIZES = (64, 128, 256)


def _resolutions_up_to(img_size):
    res = 4
    out = [res]
    while res < img_size:
        res *= 2
        out.append(res)
    return out


class GLUConvBlock(nn.Module):
    """Upsample 2x, then conv -> BatchNorm -> GLU.

    GLU (Gated Linear Unit) splits the channels in half and uses one half
    to gate the other: out = a * sigmoid(b). It tends to help GAN
    generators converge faster than a plain ReLU stack. Upsampling with
    nearest-neighbor interpolation (instead of a transposed conv) avoids
    the checkerboard artifacts transposed convs are prone to.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, out_ch * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch * 2),
            nn.GLU(dim=1),
        )

    def forward(self, x):
        return self.block(x)


class SLEBlock(nn.Module):
    """Skip-Layer Excitation: gate a high-res feature map using a low-res one.

    The low-res feature map is pooled to 4x4, squeezed through two 1x1
    convolutions, and turned into a per-channel gate (values in [0, 1])
    broadcast over the high-res feature map. This is much cheaper than a
    normal skip connection (no large feature map is copied across the
    network) but still lets coarse, early "layout" information influence
    the fine details generated much later.
    """

    def __init__(self, low_ch, high_ch):
        super().__init__()
        reduced = max(low_ch // 4, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(low_ch, reduced, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(reduced, high_ch, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, low_res_feat, high_res_feat):
        gate = self.gate(low_res_feat)  # (B, high_ch, 1, 1)
        return high_res_feat * gate


class Generator(nn.Module):
    def __init__(self, latent_dim=256, img_size=256):
        super().__init__()
        assert img_size in SUPPORTED_SIZES, f"img_size must be one of {SUPPORTED_SIZES}"
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.resolutions = _resolutions_up_to(img_size)  # e.g. [4,8,16,32,64,128,256]

        # Project the latent vector -> a 4x4 feature map.
        self.init = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, CHANNELS[4] * 2, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(CHANNELS[4] * 2),
            nn.GLU(dim=1),
        )

        self.up_blocks = nn.ModuleDict()
        for i in range(1, len(self.resolutions)):
            in_res, out_res = self.resolutions[i - 1], self.resolutions[i]
            self.up_blocks[str(out_res)] = GLUConvBlock(CHANNELS[in_res], CHANNELS[out_res])

        # SLE connections: resolution R gates resolution 16*R, whenever both exist
        # in this network (e.g. 4->64, 8->128, 16->256).
        self.sle_blocks = nn.ModuleDict()
        for res in self.resolutions:
            target = res * 16
            if target in self.resolutions:
                self.sle_blocks[f"{res}->{target}"] = SLEBlock(CHANNELS[res], CHANNELS[target])

        self.to_rgb = nn.Conv2d(CHANNELS[img_size], 3, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        z = z.view(z.size(0), self.latent_dim, 1, 1)
        feats = {4: self.init(z)}
        for i in range(1, len(self.resolutions)):
            in_res, out_res = self.resolutions[i - 1], self.resolutions[i]
            feat = self.up_blocks[str(out_res)](feats[in_res])
            for res in self.resolutions:
                key = f"{res}->{out_res}"
                if key in self.sle_blocks:
                    feat = self.sle_blocks[key](feats[res], feat)
            feats[out_res] = feat
        img = torch.tanh(self.to_rgb(feats[self.img_size]))
        return img


class DownBlock(nn.Module):
    """Strided conv that halves spatial resolution."""

    def __init__(self, in_ch, out_ch, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SimpleDecoder(nn.Module):
    """Small decoder used only for the discriminator's self-supervision task.

    Takes a mid-resolution discriminator feature map and upsamples it back
    into a small RGB image, which is compared (L1 loss) against a
    downsampled version of the real input during training. See the module
    docstring for why this matters on small datasets.
    """

    def __init__(self, in_ch, in_res, out_res):
        super().__init__()
        blocks = []
        ch, res = in_ch, in_res
        while res < out_res:
            next_ch = max(ch // 2, 32)
            blocks.append(GLUConvBlock(ch, next_ch))
            ch, res = next_ch, res * 2
        self.body = nn.Sequential(*blocks)
        self.to_rgb = nn.Conv2d(ch, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.body(x)
        return torch.tanh(self.to_rgb(x))


class Discriminator(nn.Module):
    def __init__(self, img_size=256, recon_res=128):
        super().__init__()
        assert img_size in SUPPORTED_SIZES, f"img_size must be one of {SUPPORTED_SIZES}"
        self.img_size = img_size
        self.resolutions = _resolutions_up_to(img_size)
        assert 8 in self.resolutions, "img_size too small for this architecture"

        self.from_rgb = nn.Sequential(
            nn.Conv2d(3, CHANNELS[img_size], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Downsample all the way to 8x8, then a final conv gives a small
        # "patch" of real/fake logits (PatchGAN-style, as in pix2pix)
        # instead of a single scalar.
        self.down_blocks = nn.ModuleDict()
        cur_res, i = img_size, 0
        while cur_res > 8:
            next_res = cur_res // 2
            self.down_blocks[str(next_res)] = DownBlock(CHANNELS[cur_res], CHANNELS[next_res], use_bn=(i != 0))
            cur_res, i = next_res, i + 1

        self.final = nn.Conv2d(CHANNELS[8], 1, kernel_size=4, stride=1, padding=0)

        # Reconstruction decoder, attached at the 16x16 feature map when available.
        self.recon_source_res = 16 if 16 in self.resolutions else 8
        self.recon_res = min(recon_res, img_size)
        self.decoder = SimpleDecoder(CHANNELS[self.recon_source_res], self.recon_source_res, self.recon_res)

    def forward(self, x, want_recon=False):
        feat = self.from_rgb(x)
        cur_res = self.img_size
        recon_feat = feat if cur_res == self.recon_source_res else None
        while cur_res > 8:
            next_res = cur_res // 2
            feat = self.down_blocks[str(next_res)](feat)
            cur_res = next_res
            if cur_res == self.recon_source_res:
                recon_feat = feat
        patch_logits = self.final(feat)
        recon = self.decoder(recon_feat) if (want_recon and recon_feat is not None) else None
        return patch_logits, recon
