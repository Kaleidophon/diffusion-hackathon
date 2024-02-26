import math
from typing import Tuple

import optax
import torch
import vae
from jax import random
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import models
from dataset.sprites_dataset import SpritesDataset


def _get_train_and_test_dataloaders(
    dataset: SpritesDataset = SpritesDataset(),
    batch_size: int = 32,
    test_split: float = 0.2,
    seed: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    """
    Given a dataset, returns the training and test dataloaders.
    """
    # Split the dataset into training and test sets
    train_size = int((1 - test_split) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed)
    )

    # Create the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader


def _return_transform_dataset(cfg, *, key: random.PRNGKey):
    return transforms.Compose([transforms.ToTensor(), lambda x: x.reshape(-1)])


def _return_dataset(cfg, *, key: random.PRNGKey):
    transform = _return_transform_dataset(cfg, key=key)
    return SpritesDataset(
        transform=transform,
        num_samples=cfg.dataset.num_samples,
        seed=cfg.train.seed,
    )


def return_dataloader(cfg, *, key: random.PRNGKey):
    dataset = _return_dataset(cfg, key=key)
    train_dataloader, test_dataloader = _get_train_and_test_dataloaders(
        dataset,
        cfg.train.batch_size,
        test_split=cfg.dataset.test_split,
        seed=cfg.train.seed,
    )
    return train_dataloader, test_dataloader


def return_optim(cfg, *, key: random.PRNGKey):
    return optax.adam(cfg.optim.lr)


def return_vae(cfg, *, key: random.PRNGKey):
    data_dim = math.prod(cfg.dataset.data_dim)
    encoder_key, decoder_key = random.split(key, num=2)
    return vae.VAE(
        encoder=models.Encoder(data_dim, cfg.vae.latent_dim, key=encoder_key),
        decoder=models.Decoder(data_dim, cfg.vae.latent_dim, key=decoder_key),
    )


def return_loss(cfg, *, key: random.PRNGKey):
    return vae.Loss()
