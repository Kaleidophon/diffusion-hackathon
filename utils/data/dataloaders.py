"""
This module contains a function to create dataloaders for training and testing
"""

from typing import Tuple

import torch
from torch.utils.data import random_split, DataLoader

from dataset.sprites_dataset import SpritesDataset


def get_train_and_test_dataloaders(
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
