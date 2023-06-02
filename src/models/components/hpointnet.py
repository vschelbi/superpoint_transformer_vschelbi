from torch import nn
from src.nn import BatchNorm, CatInjection
from src.models.components import SPT


__all__ = ['HPointNet']


class HPointNet(SPT):
    """Hierarchical PointNet. This UNet-like architecture encodes nested
    segments with a simple PointNet approach: the features of a
    point/segment are only aggregated and passed to the parent segment.
    Here, there is no convolution / message passing / self-attention
    module to propagate the information between elements of a same
    segment, nor between segments that do not share a parent.

    :param point_mlp:
    :param down_mlp:
    :param up_mlp:
    :param point_drop:
    :param point_pos_injection:
    :param point_pos_injection_x_dim:
    :param point_cat_diameter:
    :param point_log_diameter:
    :param down_drop:
    :param down_inject_pos:
    :param down_inject_x:
    :param up_drop:
    :param up_inject_pos:
    :param up_inject_x:
    :param activation:
    :param pos_injection:
    :param cat_diameter:
    :param log_diameter:
    :param norm:
    :param pool:
    :param unpool:
    :param fusion:
    """

    def __init__(
            self,

            point_mlp,
            down_mlp,
            up_mlp,
            point_drop=None,
            point_pos_injection=CatInjection,
            point_pos_injection_x_dim=None,
            point_cat_diameter=False,
            point_log_diameter=False,
            down_drop=None,
            down_inject_pos=True,
            down_inject_x=False,
            up_drop=None,
            up_inject_pos=True,
            up_inject_x=False,
            activation=nn.LeakyReLU(),
            pos_injection=CatInjection,
            cat_diameter=False,
            log_diameter=False,
            norm=BatchNorm,
            pool='max',
            unpool='index',
            fusion='cat'):

        down_dim = [mlp[-1] for mlp in down_mlp]
        up_dim = [mlp[-1] for mlp in up_mlp]

        super().__init__(
            point_mlp,
            point_drop=point_drop,
            point_pos_injection=point_pos_injection,
            point_pos_injection_x_dim=point_pos_injection_x_dim,
            point_cat_diameter=point_cat_diameter,
            point_log_diameter=point_log_diameter,
            down_dim=down_dim,
            down_in_mlp=down_mlp,
            down_mlp_drop=down_drop,
            down_num_blocks=0,
            down_inject_pos=down_inject_pos,
            down_inject_x=down_inject_x,
            up_dim=up_dim,
            up_in_mlp=up_mlp,
            up_mlp_drop=up_drop,
            up_num_blocks=0,
            up_inject_pos=up_inject_pos,
            up_inject_x=up_inject_x,
            pos_injection=pos_injection,
            cat_diameter=cat_diameter,
            log_diameter=log_diameter,
            mlp_activation=activation,
            mlp_norm=norm,
            pool=pool,
            unpool=unpool,
            fusion=fusion)
