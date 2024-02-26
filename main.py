"""
Main experimental script.
"""

# STD
from argparse import ArgumentParser

# EXT
import torch
import torch.nn as nn
import torch.optim as optim
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

    def forward(self, x: torch.FloatTensor, t: torch.LongTensor):
        # Implement a minimal U-net
        out = self.unet(x, t)

        return out


def create_linear_noise_schedule(beta_min: float, beta_max: float, num_timesteps: int) -> torch.FloatTensor:
    schedule = torch.linspace(beta_min, beta_max, num_timesteps)

    return schedule



def run_diffusion_model(
    batch_size: int,
    shuffle: bool,
    num_timesteps: int,
    beta_min: float,
    beta_max: float,
    lr: float,
    num_training_steps: int
):
    # This is a torch.utils.data.Dataset object
    dataset = SpritesDataset()

    # Using it, we could make a simple dataloader:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # Initialize diffusion model
    model = DiffusionModel(num_channels=3, image_size=16, num_timesteps=num_timesteps)
    noise_schedule = create_linear_noise_schedule(beta_min, beta_max, num_timesteps)

    # Training code
    loss_func = nn.MSELoss()
    optimizer = optim.RMSprop(lr=lr, alpha=0.9999, params=model.parameters())

    # TODO: Training code

    # Now we can iterate over the dataloader to get batches of data
    for step, batch in enumerate(dataloader):

        if step > num_training_steps:
            break

        # Rescale inputs to [-1, 1]
        batch = batch * 2 - 1

        # Noise schedule
        sampled_timesteps = torch.randint(num_timesteps, size=(batch.shape[0],))
        alpha = (1 - noise_schedule.unsqueeze(0).repeat(batch_size, 1))
        alpha_bar = torch.stack([
            torch.prod(alpha[i, :timestep], dim=0)
            for i, timestep in enumerate(sampled_timesteps)
        ], dim=0)
        alpha_bar = alpha_bar.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        noise = torch.randn(batch.shape)
        input_ = torch.sqrt(alpha_bar) * batch + torch.sqrt(1 - alpha_bar) * noise
        predicted_noise = model.forward(input_, sampled_timesteps)
        loss = loss_func(noise, predicted_noise)

        print(f"[Step {step +1}] Loss: {loss.item():.4f}")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()



    # TODO: Implement sampling
    # TODO: Implement saving results



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument("--num-timesteps", type=int, default=100)
    parser.add_argument("--beta-min", type=float, default=0.0001)
    parser.add_argument("--beta-max", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--num-training-steps", type=int, default=100)

    args = parser.parse_args()

    run_diffusion_model(
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_timesteps=args.num_timesteps,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        lr=args.lr,
        num_training_steps=args.num_training_steps
    )
