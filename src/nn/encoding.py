import numpy as np
import torch
import torch.nn as nn
from src.utils import positional_encoding
from src.nn.fusion import ResidualFusion


__all__ = ['FourierEncoding', 'LearnableFourierPositionalEncoding']


class BasePositionalEncoding(nn.Module):
    def __init__(self, dim, x_dim=None):
        """Base class for positional encoding. Takes care of additive
        fusion with a potential embedding vector.

        Child classes are expected to overwrite the `_encode` method.

        :param dim: positional encoding dimension
        :param x_dim:
        """
        self.dim = dim
        self.fusion = ResidualFusion()
        self.proj = nn.Identity() if x_dim is None else nn.Linear(x_dim, dim)

    def _encode(self):
        raise NotImplementedError

    def forward(self, pos, x):
        if x is not None:
            x = self.proj(x)
        return self.fusion(self.encode(pos), x)


class FourierEncoding(BasePositionalEncoding):
    def __init__(self, dim, x_dim=None, f_min=1e-1, f_max=1e1):
        """Convert [N, M] M-dimensional positions into [N, dim] encodings
        using sine and cosine decomposition along each axis. Expects dim
        to be a multiple of 2*M, for each of the M-dimensions to have
        access to the same number of encoding dimensions.

        Input positions are expected to be normalized in [-1, 1] before
        encoding. This operation is important, since passing positions
        outside of this range will result in ambiguities where two
        distinct positions have the same encoding.

        :param dim: positional encoding dimension
        """
        super().__init__(dim, x_dim=x_dim)
        self.f_min = f_min
        self.f_max = f_max

    def _encode(self, pos):
        return positional_encoding(
            pos, self.dim, f_min=self.f_min, f_max=self.f_max)


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, F_dim: int, H_dim: int, D: int, gamma: float):
        """Learnable Fourier Features from:
            https://arxiv.org/pdf/2106.02795.pdf (Algorithm 1)

        Implementation of Algorithm 1: Compute the Fourier feature positional encoding of a multi-dimensional position
        Computes the positional encoding of a tensor of shape [N, M]
        :param M: each point has a M-dimensional positional values
        :param F_dim: depth of the Fourier feature dimension
        :param H_dim: hidden layer dimension
        :param D: positional encoding dimension
        :param gamma: parameter to initialize Wr
        """
        super().__init__()
        self.M = M
        self.F_dim = F_dim
        self.H_dim = H_dim
        self.D = D
        self.gamma = gamma

        # Projection matrix on learned lines (used in eq. 2)
        self.Wr = nn.Linear(self.M, self.F_dim // 2, bias=False)
        # MLP (GeLU(F @ W1 + B1) @ W2 + B2 (eq. 6)
        self.mlp = nn.Sequential(
            nn.Linear(self.F_dim, self.H_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.H_dim, self.D)
        )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma ** -2)

    def forward(self, x):
        """
        Produce positional encodings from x
        :param x: tensor of shape [N, G, M] that represents N positions where each position is in the shape of [G, M],
                  where G is the positional group and each group has M-dimensional positional values.
                  Positions in different positional groups are independent
        :return: positional encoding for X
        """
        N, G, M = x.shape
        # Step 1. Compute Fourier features (eq. 2)
        projected = self.Wr(x)
        cosines = torch.cos(projected)
        sines = torch.sin(projected)
        F = 1 / np.sqrt(self.F_dim) * torch.cat([cosines, sines], dim=-1)
        # Step 2. Compute projected Fourier features (eq. 6)
        Y = self.mlp(F)
        # Step 3. Reshape to x's shape
        PEx = Y.reshape((N, self.D))
        return PEx
