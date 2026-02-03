import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from utils import is_flagged_image

class SDODataset(Dataset):
    """
    Example-dataset compatible loader.
    Builds samples from folders on disk and pads missing images.
    """

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []

        # Scan folders on disk
        for ar in os.listdir(root_dir):
            ar_path = os.path.join(root_dir, ar)
            if not os.path.isdir(ar_path):
                continue

            for sample in os.listdir(ar_path):
                sample_path = os.path.join(ar_path, sample)
                if os.path.isdir(sample_path):
                    self.samples.append((ar, sample))

        if len(self.samples) == 0:
            raise RuntimeError("❌ No sample folders found.")

        print(f"✅ Found {len(self.samples)} samples on disk")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ar, sample = self.samples[idx]
        sample_dir = os.path.join(self.root_dir, ar, sample)

        # Dummy label (example dataset only)
        peak_flux = 1e-6
        y = torch.tensor(np.log10(peak_flux), dtype=torch.float32)

        image_files = sorted(os.listdir(sample_dir))
        images = []

        for fname in image_files:
            img_path = os.path.join(sample_dir, fname)

            if is_flagged_image(img_path):
                img = torch.zeros((256, 256))
            else:
                img = Image.open(img_path).convert("L")
                img = torch.tensor(
                    np.array(img), dtype=torch.float32
                ) / 255.0

            images.append(img)

        # Pad / truncate to 40 images
        while len(images) < 40:
            images.append(torch.zeros((256, 256)))
        images = images[:40]

        # Reshape → (4,10,256,256)
        x = torch.stack(images).view(4, 10, 256, 256)

        return x, y
