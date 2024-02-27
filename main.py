"""
Main experimental script.
"""

# STD
from argparse import ArgumentParser
import copy
from typing import Optional

# EXT
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# PROJECT
from dataset.sprites_dataset import SpritesDataset
from utils.plotting import plot_process
from unets import UNet, UNet_conditional
from ema import EMA


class DiffusionModel(nn.Module):
    """
    Implementation of a simple diffusion model.
    """

    def __init__(
        self,
        model_type: str,
        num_channels: int,
        image_size: int,
        num_timesteps: int
    ):
        super().__init__()
        self.num_channels = num_channels
        self.image_size = image_size

        if model_type == "convolution":
            self.unet = UNet(n_channels=num_channels, num_timesteps=num_timesteps)

        elif model_type == "attention":
            self.unet = UNet_conditional(c_in=num_channels, time_dim=num_timesteps)

    def forward(self, x: torch.FloatTensor, t: torch.LongTensor):
        # Implement a minimal U-net
        out = self.unet(x, t)

        return out


def create_linear_noise_schedule(beta_min: float, beta_max: float, num_timesteps: int) -> torch.FloatTensor:
    schedule = torch.linspace(beta_min, beta_max, num_timesteps)

    return schedule



def run_diffusion_model(
    model_type: str,
    batch_size: int,
    shuffle: bool,
    num_timesteps: int,
    beta_min: float,
    beta_max: float,
    lr: float,
    num_training_steps: int,
    num_samples: int,
    device: str,
    max_batches: Optional[int] = None,
    wandb_run = None
):
    # This is a torch.utils.data.Dataset object
    dataset = SpritesDataset()

    # Using it, we could make a simple dataloader:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # Initialize diffusion model
    model = DiffusionModel(
        model_type=model_type, num_channels=3, image_size=16, num_timesteps=num_timesteps
    ).to(device)
    noise_schedule = create_linear_noise_schedule(beta_min, beta_max, num_timesteps).to(device)

    # Training code
    loss_func = nn.MSELoss()
    optimizer = optim.RMSprop(lr=lr, alpha=0.9999, params=model.parameters())

    # Initialize exponential moving average
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)


    # ### TRAINING ###
    # Now we can iterate over the dataloader to get batches of data
    img_size = None
    alpha = (1 - noise_schedule)
    alpha_bar = torch.cumprod(alpha, dim=0)

    progress_bar = tqdm(total=num_training_steps)

    def loop_dataloader(dataloader, max_batches: Optional[int] = None):
        while True:
            for i, batch in enumerate(dataloader):
                if max_batches is not None:
                    if i > max_batches - 1:
                        break

                yield batch

    for step, batch in enumerate(loop_dataloader(dataloader, max_batches=max_batches)):

        batch = batch.to(device)

        if step > num_training_steps:
            break

        # Remember image size
        if img_size is None:
            img_size = (1, *batch[0].shape)

        # Rescale inputs to [-1, 1]
        batch = batch * 2 - 1

        # Noise schedule
        sampled_timesteps = torch.randint(num_timesteps, size=(batch.shape[0],)).to(device)
        # sampled_timesteps = torch.randint(num_timesteps, size=(1, )).repeat(batch_size)
        alpha_batched = alpha.unsqueeze(0).repeat(batch_size, 1)
        alpha_bar_batched = torch.stack([
            torch.prod(alpha_batched[i, :timestep], dim=0)
            for i, timestep in enumerate(sampled_timesteps)
        ], dim=0).to(device)
        alpha_bar_batched = alpha_bar_batched.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        noise = torch.randn(batch.shape).to(device)
        input_ = torch.sqrt(alpha_bar_batched) * batch + torch.sqrt(1 - alpha_bar_batched) * noise
        predicted_noise = model.forward(input_, sampled_timesteps)
        loss = loss_func(noise, predicted_noise)

        if wandb_run is not None:
            wandb_run.log({
                "loss": loss.item(),
                "predicted_noise": predicted_noise.mean(),
                "alpha_bar": alpha_bar_batched.mean()
            })

        progress_bar.set_description(f"[Step {step +1}] Loss: {loss.item():.4f}")
        progress_bar.update(1)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        ema.step_ema(ema_model, model)

    # ### SAMPLING ###
    images = []
    final_images = []
    model.eval()

    # When using mps, sampling is somehow faster on CPU
    device = "cpu"
    alpha = alpha.to(device)
    alpha_bar = alpha_bar.to(device)
    noise_schedule = noise_schedule.to(device)
    model = model.to(device)

    with torch.no_grad():
        for _ in tqdm(range(num_samples)):

            x = torch.randn(img_size).to(device)

            x_samples = []

            for t in list(range(1, num_timesteps))[::-1]:
                t = torch.LongTensor([t]).to(device)
                z = torch.randn(img_size).to(device) if t > 1 else torch.zeros(img_size).to(device)
                factor = (1 - alpha[t]) / torch.sqrt(1 - alpha_bar[t])
                sigma_t = torch.sqrt((1 - alpha_bar[t - 1]) / (1 - alpha_bar[t]) * noise_schedule[t])
                predicted_noise = model(x, t)
                x = 1 / torch.sqrt(alpha[t]) * (x - factor * predicted_noise) + sigma_t * z

                if t % 100 == 0 or t == 1:  # Save every 100 steps and during last step
                    x_samples.append((torch.clip(x, min=-1, max=1) + 1) / 2)

            final_images.append((torch.clip(x, min=-1, max=1) + 1) / 2)

            # Rescale image values back
            images.append(torch.cat(x_samples, dim=0))

        images = torch.stack(images, dim=0).cpu()
        #images = (images - torch.min(images, dim=-1)[0].unsqueeze(-1)) / (torch.max(images, dim=-1)[0].unsqueeze(-1) - torch.min(images, dim=-1)[0].unsqueeze(-1))
        #images = torch.clip(images, 0, 1) * 255
        plot_process(images, "Garbage 1", save_path="plots.png")

        if wandb_run is not None:
            images = wandb.Image(torch.cat(final_images, dim=0), caption="Samples sprites.")
            wandb.log({"examples": images})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument("--num-timesteps", type=int, default=100)
    parser.add_argument("--beta-min", type=float, default=0.0001)
    parser.add_argument("--beta-max", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--num-training-steps", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=25)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--model-type", type=str, choices=["convolution", "attention"], default="convolution")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-batches", type=int, default=None)

    args = parser.parse_args()

    wandb_run = None
    if args.wandb:
        wandb_run = wandb.init(
            project="diffusion-hackathon",
            config={
                "batch_size": args.batch_size,
                "num_timesteps": args.num_timesteps,
                "beta_min": args.beta_min,
                "beta_max": args.beta_max,
                "lr": args.lr
            },
        )

    run_diffusion_model(
        model_type=args.model_type,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_timesteps=args.num_timesteps,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        lr=args.lr,
        num_training_steps=args.num_training_steps,
        num_samples=args.num_samples,
        device=args.device,
        max_batches=args.max_batches,
        wandb_run=wandb_run
    )
