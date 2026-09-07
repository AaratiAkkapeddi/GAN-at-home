"""
dataset.py -- loads a folder of images for unconditional GAN training.
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class ImageFolderDataset(Dataset):
    def __init__(self, root, img_size=256):
        self.paths = [
            os.path.join(root, f)
            for f in sorted(os.listdir(root))
            if f.lower().endswith(IMG_EXTS)
        ]
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {root}. Supported extensions: {IMG_EXTS}")

        # Resize a bit larger than the target and random-crop: gives the
        # model slightly different framing each time it sees an image,
        # which matters a lot when you only have a few hundred of them.
        resize_to = int(img_size * 1.15)
        self.transform = T.Compose([
            T.Resize(resize_to),
            T.CenterCrop(resize_to),
            #T.RandomCrop(img_size),
            #T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> pixel range [-1, 1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Could not read image {path}: {e}")
        return self.transform(img)


def infinite_loader(dataloader):
    """Yield batches forever, reshuffling whenever the dataset is exhausted."""
    while True:
        for batch in dataloader:
            yield batch
