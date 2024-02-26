"""
Main experimental script.
"""

# STD
from argparse import ArgumentParser

# EXT
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# PROJECT
from dataset.sprites_dataset import SpritesDataset
from unet import UNet


class DiffusionModel(nn.Module):
    """
    Implementation of a simple diffusion model.
    """

    def __init__(self, num_channels: int, image_size: int, num_timesteps: int):
        super().__init__()
        self.num_channels = num_channels
        self.image_size = image_size
        self.unet = UNet(n_channels=num_channels, num_timesteps=num_timesteps)

    def forward(self, x: torch.FloatTensor, t: int):
        # Implement a minimal U-net

        out = self.unet(x, t)
        pass






def run_diffusion_model(
    batch_size: int,
    shuffle: bool,
    num_timesteps: int,
):
    # This is a torch.utils.data.Dataset object
    dataset = SpritesDataset()

    # Using it, we could make a simple dataloader:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # Initialize diffusion model
    model = DiffusionModel(num_channels=3, image_size=16, num_timesteps=num_timesteps)

    # Now we can iterate over the dataloader to get batches of data
    for batch in dataloader:
        print(batch.shape)  # torch.Size([32, 3, 16, 16])
        model.forward(batch, 0)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument("--num-timesteps", type=int, default=100)

    args = parser.parse_args()

    run_diffusion_model(
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_timesteps=args.num_timesteps,
    )
