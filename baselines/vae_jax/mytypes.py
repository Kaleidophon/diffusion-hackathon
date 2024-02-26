from typing import Tuple

from jaxtyping import Array, Float

Image = Float[Array, "channels height width"]
BatchImage = Float[Array, "batch channels height width"]
FlatImage = Float[Array, "channels_x_height_x_width"]
VariationalMean = Float[Array, "latent_dim"]
VariationalStd = Float[Array, "latent_dim"]
LatentCode = Float[Array, "latent_dim"]
VariationalParams = Tuple[VariationalMean, VariationalStd]
