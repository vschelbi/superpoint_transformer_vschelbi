from torch import nn
from src.nn import FastBatchNorm1d
from src.models.components import NeST


__all__ = ['PointNet']


class PointNet(NeST):
    """Simple architecture encoding Level-1 segments à la PointNet.

    :param point_mlp:
    :param down_mlp:
    :param point_drop:
    :param down_drop:
    :param activation:
    :param norm:
    :param pool:
    """

    def __init__(
            self, point_mlp, down_mlp, point_drop=None, down_drop=None,
            activation=nn.LeakyReLU(), norm=FastBatchNorm1d, pool='max'):
        super().__init__(
            point_mlp, point_drop=point_drop, down_dim=down_mlp[-1],
            down_in_mlp=down_mlp, down_mlp_drop=down_drop, down_num_blocks=0,
            mlp_activation=activation, mlp_norm=norm, pool=pool, unpool='index',
            fusion='cat')
