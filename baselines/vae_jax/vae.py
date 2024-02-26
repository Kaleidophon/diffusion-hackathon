import distribution
import divergence
import equinox as eqx
import mytypes
from jax import random
from jaxtyping import Array, Float


class VAE(eqx.Module):
    """
    Variational Autoencoder that assumes:
        - a standard normal prior
        - a multivariate normal diagonal posterior
        - a continuous bernoulli likelihood
    """

    encoder: eqx.Module
    decoder: eqx.Module

    def encode(self, x: mytypes.Image) -> distribution.MultivariateNormalDiag:
        z_mean, z_std = self.encoder(x)
        return distribution.MultivariateNormalDiag(mean=z_mean, std=z_std)

    def decode(self, z: mytypes.LatentCode) -> distribution.ContinuousBernoulli:
        probs = self.decoder(z)
        return distribution.ContinuousBernoulli(probs=probs)

    def elbo(
        self,
        x: mytypes.Image,
        q_z_given_x: distribution.MultivariateNormalDiag,
        p_x_given_z: distribution.ContinuousBernoulli,
    ) -> Float:
        kl = divergence.kl_divergence_normal_diag(q_z_given_x)
        log_prob = p_x_given_z.log_prob(x).sum()
        return -log_prob + kl

    def __call__(self, x: mytypes.Image, *, key: random.PRNGKey):
        q_z_given_x = self.encode(x)
        z = q_z_given_x.rsample(key=key)
        p_x_given_z = self.decode(z)
        return q_z_given_x, p_x_given_z, z


class Loss(eqx.Module):

    def loss(self, vae: VAE, x: mytypes.Image, key: random.PRNGKey):
        q_z_given_x, p_x_given_z, z = vae(x, key=key)
        loss = vae.elbo(x, q_z_given_x, p_x_given_z)
        return loss

    def vmap_loss_fn(
        self, vae: eqx.Module, x: mytypes.BatchImage, keys: random.PRNGKey
    ):
        return eqx.filter_vmap(self.loss, in_axes=(None, 0, 0))(vae, x, keys).mean()

    def loss_and_grad(
        self, vae: eqx.Module, x: mytypes.BatchImage, key: random.PRNGKey
    ):
        keys = random.split(key, x.shape[0])
        loss, grad = eqx.filter_value_and_grad(self.vmap_loss_fn)(vae, x, keys)
        return loss, grad

    @eqx.filter_jit
    def __call__(self, vae: eqx.Module, x: mytypes.BatchImage, *, key: random.PRNGKey):
        return self.loss_and_grad(vae, x, key)
