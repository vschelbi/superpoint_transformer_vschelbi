from torch import nn
from src.nn import BatchNorm, CatInjection
from src.models.components import NeST


__all__ = ['PointNet']


class PointNet(NeST):
    """Simple architecture encoding Level-1 segments à la PointNet.

    :param point_mlp:
    :param down_mlp:
    :param point_drop:
    :param point_pos_injection:
    :param point_pos_injection_x_dim:
    :param point_cat_diameter:
    :param small:
    :param small_point_mlp:
    :param small_down_mlp:
    :param down_drop:
    :param activation:
    :param norm:
    :param pool:
    :param pos_injection:
    """

    def __init__(
            self,

            point_mlp,
            down_mlp,

            point_drop=None,
            point_pos_injection=CatInjection,
            point_pos_injection_x_dim=None,
            point_cat_diameter=False,
            point_log_diameter=False,

            small=None,
            small_point_mlp=None,
            small_down_mlp=None,

            down_drop=None,

            activation=nn.LeakyReLU(),
            norm=BatchNorm,
            pool='max',
            pos_injection=CatInjection):

        super().__init__(
            point_mlp,
            point_drop=point_drop,
            point_pos_injection=point_pos_injection,
            point_pos_injection_x_dim=point_pos_injection_x_dim,
            point_cat_diameter=point_cat_diameter,
            point_log_diameter=point_log_diameter,

            small=small,
            small_point_mlp=small_point_mlp,
            small_down_mlp=small_down_mlp,

            down_dim=down_mlp[-1],
            down_in_mlp=down_mlp,
            down_mlp_drop=down_drop,
            down_num_blocks=0,

            pos_injection=pos_injection,
            mlp_activation=activation,
            mlp_norm=norm,
            pool=pool,
            unpool='index',
            fusion='cat')
