from typing import Tuple

import equinox as eqx
import jax.numpy as jnp
from jax import lax, random
from jaxtyping import Array, Float


def clamp_probs(probs: Float[Array, "*dim"]) -> Float[Array, "*dim"]:
    eps = jnp.finfo(probs.dtype).eps
    return jnp.clip(probs, eps, 1 - eps)


def binary_cross_entropy(probs, y):
    logits = jnp.log(probs / (1 - probs))
    max_logit = jnp.clip(logits, 0, None)
    bces = (
        logits
        - logits * y
        + max_logit
        + jnp.log(jnp.exp(-max_logit) + jnp.exp(-logits - max_logit))
    )
    return bces


class MultivariateNormalDiag(eqx.Module):
    mean: Float[Array, "dim"]
    std: Float[Array, "dim"]

    def log_prob(self, x: Float[Array, "dim"]) -> Float:
        d = x.shape[0]
        return -0.5 * (
            -d * jnp.log(2 * jnp.pi)
            - jnp.sum(jnp.log(self.std**2))
            - jnp.sum(((x - self.mean) / self.std) ** 2)
        )

    def rsample(self, *, key: random.PRNGKey) -> Float[Array, "dim"]:
        return random.normal(key, shape=self.mean.shape) * self.std + self.mean


class StandardNormal(MultivariateNormalDiag):
    def __init__(self, dim: int):
        mean = jnp.zeros(dim)
        std = jnp.ones(dim)
        super().__init__(mean=mean, std=std)


class ContinuousBernoulli(eqx.Module):
    probs: Float[Array, "*dim"]

    def __init__(self, probs: Float[Array, "*dim"]):
        self.probs = clamp_probs(probs)

    def _outside_unstable_region(self):
        return jnp.logical_or(
            jnp.less_equal(self.probs, 0.499), jnp.greater(self.probs, 0.501)
        )

    def _cut_probs(self):
        return jnp.where(
            self._outside_unstable_region(),
            self.probs,
            0.499 * jnp.ones_like(self.probs),
        )

    def _constant(self):
        """computes the log normalizing constant as a function of the 'probs' parameter"""
        cut_probs = self._cut_probs()
        cut_probs_below_half = jnp.where(
            jnp.less(cut_probs, 0.5), cut_probs, jnp.zeros_like(cut_probs)
        )
        cut_probs_above_half = jnp.where(
            jnp.greater_equal(cut_probs, 0.5), cut_probs, jnp.ones_like(cut_probs)
        )
        log_norm = jnp.log(
            jnp.abs(jnp.log1p(-cut_probs) - jnp.log(cut_probs))
        ) - jnp.where(
            jnp.less(cut_probs, 0.5),
            jnp.log1p(-2.0 * cut_probs_below_half),
            jnp.log(2.0 * cut_probs_above_half - 1.0),
        )
        x = jnp.pow(self.probs - 0.5, 2)
        taylor = jnp.log(2.0) + (4.0 / 3.0 + 104.0 / 45.0 * x) * x
        return jnp.where(self._outside_unstable_region(), log_norm, taylor)

    def log_prob(self, x: Float[Array, "*dim"]) -> Float[Array, "*dim"]:
        cut_probs = self._cut_probs()
        bce = binary_cross_entropy(cut_probs, x)
        return -bce + self._constant()

    def sample(self, *, key: random.PRNGKey) -> Float[Array, "dim"]:
        return self.rsample(key=key)

    def rsample(self, *, key: random.PRNGKey) -> Float[Array, "dim"]:
        u = random.uniform(key, shape=self.probs.shape)
        return self.icdf(u)

    def icdf(self, value: Float[Array, "dim"]) -> Float[Array, "dim"]:
        cut_probs = self._cut_probs()
        return jnp.where(
            self._outside_unstable_region(),
            (
                jnp.log1p(-cut_probs + (2.0 * cut_probs - 1.0) * value)
                - jnp.log1p(-cut_probs)
            )
            / (jnp.log(cut_probs) - jnp.log1p(-cut_probs)),
            value,
        )
