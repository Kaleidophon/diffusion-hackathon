import equinox as eqx
import jax
import jax.numpy as jnp
import mytypes
from jax import random


class Encoder(eqx.Module):
    encoder: eqx.nn.Sequential
    encoder_mean: eqx.nn.Linear
    encoder_logvar: eqx.nn.Linear

    def __init__(self, data_dim: int, latent_dim: int, *, key: random.PRNGKey):
        keys = random.split(key, 5)
        self.encoder = eqx.nn.Sequential(
            [
                eqx.nn.Linear(data_dim, 512, key=keys[0]),
                eqx.nn.Lambda(jax.nn.tanh),
                eqx.nn.Linear(512, 256, key=keys[1]),
                eqx.nn.Lambda(jax.nn.tanh),
                eqx.nn.Linear(256, 128, key=keys[2]),
                eqx.nn.Lambda(jax.nn.tanh),
            ]
        )
        self.encoder_mean = eqx.nn.Linear(128, latent_dim, key=keys[3])
        self.encoder_logvar = eqx.nn.Linear(128, latent_dim, key=keys[4])

    def __call__(self, x: mytypes.FlatImage) -> mytypes.VariationalParams:
        h = self.encoder(x)
        return self.encoder_mean(h), jnp.exp(0.5 * self.encoder_logvar(h))


class Decoder(eqx.Module):
    decoder: eqx.nn.Sequential

    def __init__(self, data_dim: int, latent_dim: int, *, key: random.PRNGKey):
        keys = random.split(key, 3)
        self.decoder = eqx.nn.Sequential(
            [
                eqx.nn.Linear(latent_dim, 256, key=keys[0]),
                eqx.nn.Lambda(jax.nn.tanh),
                eqx.nn.Linear(256, 512, key=keys[1]),
                eqx.nn.Lambda(jax.nn.tanh),
                eqx.nn.Linear(512, data_dim, key=keys[2]),
                eqx.nn.Lambda(jax.nn.sigmoid),
            ]
        )

    def __call__(self, z: mytypes.LatentCode) -> mytypes.FlatImage:
        return self.decoder(z)
