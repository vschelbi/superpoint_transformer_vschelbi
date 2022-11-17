from torch import nn
from src.nn import FastBatchNorm1d, CatInjection
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
            activation=nn.LeakyReLU(), norm=FastBatchNorm1d, pool='max',
            point_pos_injection=CatInjection, point_pos_injection_x_dim=None,
            point_cat_diameter=False, pos_injection=CatInjection):
        super().__init__(
            point_mlp, point_drop=point_drop, down_dim=down_mlp[-1],
            down_in_mlp=down_mlp, down_mlp_drop=down_drop, down_num_blocks=0,
            point_pos_injection=point_pos_injection,
            point_pos_injection_x_dim=point_pos_injection_x_dim,
            point_cat_diameter=point_cat_diameter, pos_injection=pos_injection,
            mlp_activation=activation, mlp_norm=norm, pool=pool, unpool='index',
            fusion='cat')
