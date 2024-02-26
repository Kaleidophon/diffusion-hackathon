"""
This script trains a VAEMario using early stopping.
"""

from time import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import numpy as np

from baselines.vae_torch import VAESprites

from dataset.sprites_dataset import SpritesDataset
from utils.data.dataloaders import get_train_and_test_dataloaders

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()


def fit(
    model: VAESprites,
    optimizer: Optimizer,
    data_loader: DataLoader,
    device: str,
) -> torch.Tensor:
    """
    Runs a training epoch: evaluating the model in
    the data provided by the data_loader, computing
    the ELBO loss inside the model, and propagating
    the error backwards to the parameters.
    """
    model.train()
    running_loss = 0.0
    for sprites in data_loader:
        sprites = sprites.to(device)
        optimizer.zero_grad()
        q_z_given_x, p_x_given_z = model.forward(sprites)
        loss = model.elbo_loss_function(sprites, q_z_given_x, p_x_given_z)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    return running_loss / len(data_loader)


def test(
    model: VAESprites,
    test_loader: DataLoader,
    device: str,
    epoch: int = 0,
) -> torch.Tensor:
    """
    Evaluates the current model on the test set,
    returning the average loss.
    """
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for sprites in test_loader:
            sprites.to(device)
            q_z_given_x, p_x_given_z = model.forward(sprites)
            loss = model.elbo_loss_function(sprites, q_z_given_x, p_x_given_z)
            running_loss += loss.item()

    return running_loss / len(test_loader)


def run(
    max_epochs: int = 500,
    batch_size: int = 64,
    test_split: float = 0.2,
    seed_for_data_splitting: int = 1,
    lr: int = 1e-3,
    overfit: bool = False,
):
    """
    Trains a VAESprites on the dataset for the provided hyperparameters.
    """
    # Defining the name of the experiment
    timestamp = str(int(time()))
    comment = f"{timestamp}_vae_sprites"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Creating the dataloader.
    train_data_loader, test_data_loader = get_train_and_test_dataloaders(
        dataset=SpritesDataset(num_samples=1000),
        batch_size=batch_size,
        test_split=test_split,
        seed=seed_for_data_splitting,
    )

    # Loading the model and optimizer
    print("Model:")
    vae = VAESprites()
    print(vae)
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    # Creating the models folder if it doesn't exist
    saving_path = ROOT_DIR / "models" / "vae_torch"
    saving_path.mkdir(parents=True, exist_ok=True)

    # Training and testing.
    print(f"Training experiment {comment}")
    best_loss = np.Inf
    n_without_improvement = 0
    for epoch in range(max_epochs):
        train_loss = fit(vae, optimizer, train_data_loader, device)
        test_loss = test(vae, test_data_loader, device, epoch)
        print(
            f"Epoch {epoch + 1} of {max_epochs}. Test loss: {test_loss}. Train loss: {train_loss}"
        )
        if test_loss < best_loss:
            best_loss = test_loss
            n_without_improvement = 0

            # Saving the best model so far.
            torch.save(vae.state_dict(), saving_path / f"{comment}.pt")
        else:
            if not overfit:
                n_without_improvement += 1

        # Early stopping:
        if n_without_improvement == 25:
            print("Stopping early")
            break


if __name__ == "__main__":
    run()
