"""
A Bernoulli VAE that can train on the sprites dataset.

Taken and adapted from
https://github.com/miguelgondu/minimal_VAE_on_Mario
"""

from itertools import product
from typing import Tuple

import numpy as np
import torch
from torch.distributions import Distribution, Normal, ContinuousBernoulli, kl_divergence
import torch.nn as nn


# Taken and adapted from
# https://github.com/miguelgondu/minimal_VAE_on_Mario
class VAESprites(nn.Module):
    """
    A VAE that decodes to the ContinuousBernoulli distribution
    of images of shape (n_channels, h, w).
    """

    def __init__(
        self,
        w: int = 16,
        h: int = 16,
        z_dim: int = 2,
        n_channels: int = 3,
        device: str = None,
    ):
        super(VAESprites, self).__init__()
        self.w = w
        self.h = h
        self.n_channels = n_channels
        self.input_dim = w * h * n_channels  # for flattening
        self.z_dim = z_dim
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
        ).to(self.device)
        self.enc_mu = nn.Sequential(nn.Linear(128, z_dim)).to(self.device)
        self.enc_var = nn.Sequential(nn.Linear(128, z_dim)).to(self.device)

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, self.input_dim),
        ).to(self.device)

        # The VAE prior on latent codes. Only used for the KL term in
        # the ELBO loss.
        self.p_z = Normal(
            torch.zeros(self.z_dim, device=self.device),
            torch.ones(self.z_dim, device=self.device),
        )

    def encode(self, x: torch.Tensor) -> Normal:
        """
        An encoding function that returns the normal distribution
        q(z|x) for some data x.

        It flattens x after the first dimension, passes it through
        the encoder networks which parametrize the mean and log-variance
        of the Normal, and returns the distribution.
        """
        x = x.view(-1, self.input_dim)
        result = self.encoder(x)
        mu = self.enc_mu(result)
        log_var = self.enc_var(result)

        return Normal(mu, torch.exp(0.5 * log_var))

    def decode(self, z: torch.Tensor) -> ContinuousBernoulli:
        """
        A decoding function that returns the ContinuousBernoulli distribution
        p(x|z) for some latent codes z.

        It passes it through the decoder network, which parametrizes
        the logits of the ContinuousBernoulli distribution of shape (h, w).
        """
        logits = self.decoder(z)
        p_x_given_z = ContinuousBernoulli(
            logits=logits.reshape(-1, self.n_channels, self.h, self.w)
        )

        return p_x_given_z

    def forward(self, x: torch.Tensor) -> Tuple[Normal, ContinuousBernoulli]:
        """
        A forward pass for some data x, returning the tuple
        [q(z|x), p(x|z)] where the latent codes in the second
        distribution are sampled from the first one.
        """
        q_z_given_x = self.encode(x.to(self.device))

        z = q_z_given_x.rsample()

        p_x_given_z = self.decode(z.to(self.device))

        return [q_z_given_x, p_x_given_z]

    def elbo_loss_function(
        self, x: torch.Tensor, q_z_given_x: Distribution, p_x_given_z: Distribution
    ) -> torch.Tensor:
        """
        The ELBO (Evidence Lower Bound) loss for the VAE,
        which is a linear combination of the reconconstruction
        loss (i.e. the negative log likelihood of the data), plus
        a Kullback-Leibler regularization term which shapes the
        approximate posterior q(z|x) to be close to the prior p(z),
        which we take as the unit Gaussian in latent space.
        """
        rec_loss = -p_x_given_z.log_prob(x).sum(dim=(1, 2, 3))  # b
        kld = kl_divergence(q_z_given_x, self.p_z).sum(dim=1)  # b

        return (rec_loss + kld).mean()

    def plot_grid_in_latent_space(
        self,
        x_lims=(-5, 5),
        y_lims=(-5, 5),
        n_rows=10,
        n_cols=10,
        sample=False,
        ax=None,
    ) -> np.ndarray:
        """
        A helper function which plots, as images, the levels in a
        fine grid in latent space, specified by the provided limits,
        number of rows and number of columns.

        The figure can be plotted in a given axis; if none is passed,
        a new figure is created.

        This function also returns the final image (which is the result
        of concatenating all the individual decoded images) as a numpy
        array.
        """
        z1 = np.linspace(*x_lims, n_cols)
        z2 = np.linspace(*y_lims, n_rows)

        zs = np.array([[a, b] for a, b in product(z1, z2)])

        images_dist = self.decode(torch.from_numpy(zs).type(torch.float))
        if sample:
            images = images_dist.sample()
        else:
            images = images_dist.probs

        images = images.numpy(force=True)
        img_dict = {(z[0], z[1]): img for z, img in zip(zs, images)}

        positions = {
            (x, y): (i, j) for j, x in enumerate(z1) for i, y in enumerate(reversed(z2))
        }

        pixels = 16
        final_img = np.zeros((3, n_cols * pixels, n_rows * pixels))
        for z, (i, j) in positions.items():
            final_img[
                :, i * pixels : (i + 1) * pixels, j * pixels : (j + 1) * pixels
            ] = img_dict[z]

        # Moving the channel to the last dimension
        final_img = np.moveaxis(final_img, 0, -1)

        if ax is not None:
            ax.imshow(final_img, extent=[*x_lims, *y_lims])

        return final_img


if __name__ == "__main__":
    vae = VAESprites()
    print(vae)
