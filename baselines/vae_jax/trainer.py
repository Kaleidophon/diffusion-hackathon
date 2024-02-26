import logging

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import torch
from jax import random
from optax import GradientTransformation
from torch.utils.data import DataLoader
from vae import VAE, Loss

from utils import plotting

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

    def train(
        self,
        vae: VAE,
        loss_module: Loss,
        optimizer: GradientTransformation,
        opt_state,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader = None,
        *,
        key,
    ):
        train_key, test_key = random.split(key, num=2)
        train_loss_list, test_loss_list = [], []
        for epoch in range(self.cfg.train.epochs):
            train_loss = 0
            for train_batch in train_dataloader:
                train_batch = jnp.asarray(train_batch)
                loss, grad = loss_module(vae, train_batch, key=train_key)
                updates, opt_state = optimizer.update(grad, opt_state, vae)
                vae = eqx.apply_updates(vae, updates)
                train_key, _ = random.split(train_key)
                train_loss += loss.item()
            train_loss /= len(train_dataloader)
            train_loss_list.append(train_loss)

            test_loss = 0
            for test_batch in test_dataloader:
                test_batch = jnp.asarray(test_batch)
                loss = loss_module(vae, test_batch, key=test_key)[0].item()
                test_loss += loss
                test_key, _ = random.split(test_key)
            test_loss /= len(test_dataloader)
            test_loss_list.append(test_loss)

            if epoch % 10 == 0:
                samples = self.sample(
                    vae, self.cfg.train.num_samples_plot, key=test_key
                )
                plotting.plot_grid(
                    torch.from_numpy(np.array(samples)),
                    "Samples",
                    self.cfg.train.sample_path / f"epoch_{epoch}.png",
                    nrow=10,
                )

            logger.info(f"{epoch = }, {test_loss = :_}, {train_loss = :_}")

        plotting.plot_losses(
            train_loss_list,
            test_loss_list,
            save_path=self.cfg.train.plot_path / "losses.png",
        )
        return vae

    def sample(self, vae: VAE, num_samples: int, *, key: random.PRNGKey):
        keys = random.split(key, num_samples)
        return eqx.filter_vmap(self._sample, in_axes=dict(vae=None, key=0))(vae, keys)

    def _sample(self, vae: VAE, key: random.PRNGKey):
        z = random.normal(key, (self.cfg.vae.latent_dim,))
        dist = vae.decode(z)
        return dist.probs.reshape(self.cfg.dataset.data_dim)
        # return dist.sample(key=key).reshape(self.cfg.dataset.data_dim)
