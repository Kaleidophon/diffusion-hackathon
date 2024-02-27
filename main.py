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
from tqdm import tqdm
import wandb

# PROJECT
from dataset.sprites_dataset import SpritesDataset
from utils.plotting import plot_process
from unets import UNet, UNet_conditional

# TODO: Check tomorrow
# - Vary noise schedule hyperparameters
# - Other regularization?


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
    wandb_run = None
):
    # This is a torch.utils.data.Dataset object
    dataset = SpritesDataset()

    # Using it, we could make a simple dataloader:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # Initialize diffusion model
    model = DiffusionModel(model_type=model_type, num_channels=3, image_size=16, num_timesteps=num_timesteps)
    noise_schedule = create_linear_noise_schedule(beta_min, beta_max, num_timesteps)

    # Training code
    loss_func = nn.MSELoss()
    optimizer = optim.RMSprop(lr=lr, alpha=0.9999, params=model.parameters())

    # ### TRAINING ###
    # Now we can iterate over the dataloader to get batches of data
    img_size = None
    alpha = (1 - noise_schedule)
    alpha_bar = torch.cumprod(alpha, dim=0)

    progress_bar = tqdm(total=num_training_steps)

    def loop_dataloader(dataloader):
        while True:
            for batch in dataloader:
                yield batch

    for step, batch in enumerate(loop_dataloader(dataloader)):

        if step > num_training_steps:
            break

        # Remember image size
        if img_size is None:
            img_size = (1, *batch[0].shape)

        # Rescale inputs to [-1, 1]
        batch = batch * 2 - 1

        # Noise schedule
        sampled_timesteps = torch.randint(num_timesteps, size=(batch.shape[0],))
        # sampled_timesteps = torch.randint(num_timesteps, size=(1, )).repeat(batch_size)
        alpha_batched = alpha.unsqueeze(0).repeat(batch_size, 1)
        alpha_bar_batched = torch.stack([
            torch.prod(alpha_batched[i, :timestep], dim=0)
            for i, timestep in enumerate(sampled_timesteps)
        ], dim=0)
        alpha_bar_batched = alpha_bar_batched.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        noise = torch.randn(batch.shape)
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

    # ### SAMPLING ###

    images = []
    final_images = []
    model.eval()

    with torch.no_grad():
        for _ in tqdm(range(num_samples)):

            x = torch.randn(img_size)

            x_samples = []

            for t in list(range(1, num_timesteps))[::-1]:
                z = torch.randn(img_size) if t > 1 else torch.zeros(img_size)
                factor = (1 - alpha[t]) / torch.sqrt(1 - alpha_bar[t])
                sigma_t = torch.sqrt((1 - alpha_bar[t - 1]) / (1 - alpha_bar[t]) * noise_schedule[t])
                predicted_noise = model(x, t)
                x = 1 / torch.sqrt(alpha[t]) * (x - factor * predicted_noise) + sigma_t * z

                if t % 100 == 0 or t == 1:  # Save every 100 steps and during last step
                    x_samples.append((x + 1) / 2)

            final_images.append((x + 1) / 2)

            # Rescale image values back
            images.append(torch.cat(x_samples, dim=0))

        images = torch.stack(images, dim=0)
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
        wandb_run=wandb_run
    )
