from torch import nn
from src.nn.norm import FastBatchNorm1d


__all__ = ['MLP', 'FFN', 'RPEFFN', 'Classifier']


def mlp(
        dims, activation=nn.LeakyReLU(), last_activation=True,
        norm=FastBatchNorm1d, drop=None):
    """Helper to build MLP-like structures.

    :param dims: List[int]
        List of channel sizes. Expects `len(dims) >= 2`
    :param activation: nn.Module instance
        Non-linearity
    :param last_activation: bool
        Whether the last layer should have an activation
    :param norm: nn.Module
        Normalization. Can be None, for FFN for instance
    :param drop: float in [0, 1]
        Dropout on the output features. No dropout layer will be
        created if `drop=None` or `drop < 0`
    :return:
    """
    assert len(dims) >= 2

    # Only use bias if no normalization is applied
    bias = norm is None

    # Iteratively build the layers based on dims
    modules = []
    for i in range(1, len(dims)):
        modules.append(nn.Linear(dims[i - 1], dims[i], bias=bias))
        if norm is not None:
            modules.append(norm(dims[i]))
        if activation is not None and (last_activation or i < len(dims) - 1):
            modules.append(activation)

    # Add final dropout if required
    if drop is not None and drop > 0:
        modules.append(nn.Dropout(drop, inplace=True))

    return nn.Sequential(*modules)


class MLP(nn.Module):
    """MLP operating on features [N, D] tensors. You can think of
    it as a series of 1x1 conv -> 1D batch norm -> activation.
    """

    def __init__(
            self, dims, activation=nn.LeakyReLU(), norm=FastBatchNorm1d,
            drop=None):
        super().__init__()
        self.mlp = mlp(
            dims, activation=activation, last_activation=True, norm=norm,
            drop=drop)
        self.out_dim = dims[-1]

    def forward(self, x):
        return self.mlp(x)


class FFN(nn.Module):
    """Feed-Forward Network as used in Transformers. By convention,
    these MLPs have 2 Linear layers and no normalization, the last layer
    has no activation and an optional dropout may be applied on the
    output features.
    """

    def __init__(
            self, dim, hidden_dim=None, out_dim=None, activation=nn.LeakyReLU(),
            drop=None):
        super().__init__()

        # Build the channel sizes for the 2 linear layers
        hidden_dim = hidden_dim or dim
        out_dim = out_dim or dim
        channels = [dim, hidden_dim, out_dim]

        self.ffn = mlp(
            channels, activation=activation, last_activation=False, norm=None,
            drop=drop)
        self.out_dim = out_dim

    def forward(self, x):
        return self.ffn(x)


class RPEFFN(FFN):
    """Feed-Forward Network for Relative Position Encoding. By
    convention, these MLPs have 2 Linear layers and no normalization,
    the last layer has no activation and an optional dropout may be
    applied on the output features. Besides, if not provided, the hidden
    layer is chosen to be the max between input dim, output dim and 16
    (this subtlety is the only difference with FFN.
    """

    def __init__(
            self, dim, hidden_dim=None, out_dim=None, activation=nn.LeakyReLU(),
            drop=None):
        super().__init__(
            dim, hidden_dim=hidden_dim or max(dim, out_dim, 16), out_dim=out_dim,
            activation=activation, drop=drop)


class Classifier(nn.Module):
    """A simple fully-connected head with no activation and no
    normalization.
    """

    def __init__(self, in_dim, num_classes, bias=True):
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_classes, bias=bias)

    def forward(self, x):
        return self.classifier(x)
