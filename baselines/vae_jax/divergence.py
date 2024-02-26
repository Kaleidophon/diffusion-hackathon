import distribution
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


def kl_divergence_normal_diag(
    multi: distribution.MultivariateNormalDiag,
    standard: distribution.StandardNormal = None,
) -> Float:
    d = multi.mean.shape[0]
    return 0.5 * (
        jnp.sum(multi.std**2)
        + jnp.sum(multi.mean**2)
        - jnp.log(jnp.prod(multi.std**2))
        - d
    )
