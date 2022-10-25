from torch import nn
from src.nn.norm import FastBatchNorm1d


__all__ = ['PointMLP', 'PointClassifier']


def mlp(
        channels, activation=nn.LeakyReLU(0.2, inplace=True), bn_momentum=0.1,
        bias=False):
    """Credits: https://github.com/torch-points3d/torch-points3d"""
    return nn.Sequential(
        *[
            nn.Sequential(
                nn.Linear(channels[i - 1], channels[i], bias=bias),
                FastBatchNorm1d(channels[i], momentum=bn_momentum),
                activation)
            for i in range(1, len(channels))])


class PointMLP(nn.Module):
    """MLP operating on point features à la PointNet. You can think of
    it as a series of 1x1 conv -> 1D batch norm -> activation.
    """
    def __init__(
            self, channels, activation=nn.LeakyReLU(0.2, inplace=True),
            bn_momentum=0.1, bias=False):
        super().__init__()
        self.mlp = mlp(
            channels, activation=activation, bn_momentum=bn_momentum, bias=bias)

    def forward(self, x):
        return self.mlp(x)


class PointClassifier(nn.Module):
    """A simple fully-connected head with no activation and no
    normalization.
    """
    def __init__(
            self, in_channels, num_classes, bias=True):
        super().__init__()
        self.classifier = nn.Linear(in_channels, num_classes, bias=bias)

    def forward(self, x):
        return self.classifier(x)
