"""
Define positional embeddings.
"""

# STD
import math

# EXT
import torch


class SinusoidEncoding(torch.nn.Module):
    """
    Mostly copied from
    https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    """

    def __init__(self, hidden_dim, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pos_embed = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        # pos_embed = pos_embed.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def forward(self, x: torch.FloatTensor, t: torch.LongTensor):
        """
        Adds positional embeddings to token embeddings.
        N = batch size
        L = sequence length
        E = embedding dim

        :param x: token embeddings. Shape: (N, L, E)
        :return: token_embeddings + positional embeddings. Shape: (N, L, E)
        """
        batch_pos_embed = self.pos_embed[t, :]
        batch_pos_embed = batch_pos_embed.unsqueeze(-1).unsqueeze(-1)
        x = x + batch_pos_embed
        return x