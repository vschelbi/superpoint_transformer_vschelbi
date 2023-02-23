from torch import nn
from src.nn import BatchNorm, CatInjection
from src.models.components import NeST


__all__ = ['HPointNet']


class HPointNet(NeST):
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
    :param small:
    :param small_point_mlp:
    :param small_down_mlp:
    :param down_drop:
    :param down_inject_pos:
    :param down_inject_x:
    :param up_drop:
    :param up_inject_pos:
    :param up_inject_x:
    :param last_dim:
    :param last_in_mlp:
    :param last_out_mlp:
    :param last_mlp_drop:
    :param last_num_heads:
    :param last_num_blocks:
    :param last_ffn_ratio:
    :param last_residual_drop:
    :param last_attn_drop:
    :param last_drop_path:
    :param last_inject_pos:
    :param last_pos_injection_x_dim:
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

            small=None,
            small_point_mlp=None,
            small_down_mlp=None,

            down_drop=None,
            down_inject_pos=True,
            down_inject_x=False,

            up_drop=None,
            up_inject_pos=True,
            up_inject_x=False,

            last_dim=None,
            last_in_mlp=None,
            last_out_mlp=None,
            last_mlp_drop=None,
            last_num_heads=1,
            last_num_blocks=1,
            last_ffn_ratio=4,
            last_residual_drop=None,
            last_attn_drop=None,
            last_drop_path=None,
            last_inject_pos=True,
            last_pos_injection_x_dim=None,

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

            small=small,
            small_point_mlp=small_point_mlp,
            small_down_mlp=small_down_mlp,

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

            last_dim=last_dim,
            last_in_mlp=last_in_mlp,
            last_out_mlp=last_out_mlp,
            last_mlp_drop=last_mlp_drop,
            last_num_heads=last_num_heads,
            last_num_blocks=last_num_blocks,
            last_ffn_ratio=last_ffn_ratio,
            last_residual_drop=last_residual_drop,
            last_attn_drop=last_attn_drop,
            last_drop_path=last_drop_path,
            last_inject_pos=last_inject_pos,
            last_pos_injection_x_dim=last_pos_injection_x_dim,

            pos_injection=pos_injection,
            cat_diameter=cat_diameter,
            log_diameter=log_diameter,
            mlp_activation=activation,
            mlp_norm=norm,
            pool=pool,
            unpool=unpool,
            fusion=fusion)
