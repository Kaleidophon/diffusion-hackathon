"""
Utilities provided by Aasa, from her course on advanced medical image analysis.
"""

from typing import Callable
from pathlib import Path

import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .helpers import set_seed, im_normalize

ROOT_DIR = Path(__file__).parent.parent.resolve()


# Dataset from: https://github.com/Ryota-Kawamura/How-Diffusion-Models-Work
class SpritesDataset(Dataset):
    def __init__(
        self,
        transform: Callable[[torch.Tensor], torch.Tensor] = None,
        img_file=ROOT_DIR / "data" / "sprites.npy",
        num_samples=40000,
        seed=1,
    ):
        self.images = np.load(img_file)
        self.num_samples = num_samples
        self.seed = seed
        # Reduce dataset size
        if num_samples:
            set_seed(seed=self.seed)
            sampled_indeces = random.sample(range(len(self.images)), self.num_samples)
            self.images = self.images[sampled_indeces]

        self.transform = transform
        self.images_shape = self.images.shape

    # Return the number of images in the dataset
    def __len__(self):
        return len(self.images)

    # Get the image and label at a given index
    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image
