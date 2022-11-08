import torch
from torch import nn
from src.data import NAG
from src.utils import listify_with_reference
from src.nn import PointStage, DownNFuseStage, UpNFuseStage, FastBatchNorm1d


__all__ = ['NeST']


class NeST(nn.Module):
    """Nested Set Transformer. A UNet-like architecture processing NAG.
    """

    def __init__(
            self,

            point_mlp,
            point_drop=None,

            down_dim=None,
            down_in_mlp=None,
            down_out_mlp=None,
            down_mlp_drop=None,
            down_num_heads=1,
            down_num_blocks=1,
            down_ffn_ratio=4,
            down_residual_drop=None,
            down_attn_drop=None,
            down_drop_path=None,

            up_dim=None,
            up_in_mlp=None,
            up_out_mlp=None,
            up_mlp_drop=None,
            up_num_heads=1,
            up_num_blocks=1,
            up_ffn_ratio=4,
            up_residual_drop=None,
            up_attn_drop=None,
            up_drop_path=None,

            mlp_activation=nn.LeakyReLU(0.2),
            mlp_norm=FastBatchNorm1d,
            qkv_bias=True,
            qk_scale=None,
            activation=nn.GELU(),
            pre_ln=True,
            no_sa=False,
            no_ffn=False,

            pool='max',
            unpool='index',
            fusion='cat',
            norm_mode='graph'):

        super().__init__()

        self.norm_mode = norm_mode

        # Convert input arguments to nested lists
        (down_dim, down_in_mlp, down_out_mlp, down_mlp_drop, down_num_heads,
         down_num_blocks, down_ffn_ratio, down_residual_drop, down_attn_drop,
         down_drop_path) = listify_with_reference(
            down_dim, down_in_mlp, down_out_mlp, down_mlp_drop, down_num_heads,
            down_num_blocks, down_ffn_ratio, down_residual_drop, down_attn_drop,
            down_drop_path)

        (up_dim, up_in_mlp, up_out_mlp, up_mlp_drop, up_num_heads,
         up_num_blocks, up_ffn_ratio, up_residual_drop, up_attn_drop,
         up_drop_path) = listify_with_reference(
            up_dim, up_in_mlp, up_out_mlp, up_mlp_drop, up_num_heads,
            up_num_blocks, up_ffn_ratio, up_residual_drop, up_attn_drop,
            up_drop_path)

        # PointNet-like module operating on Level-0 data
        self.point_stage = PointStage(
            point_mlp, mlp_activation=mlp_activation, mlp_norm=mlp_norm,
            mlp_drop=point_drop)

        # Transformer encoder (down) Stages operating on Level-i data
        if len(down_dim) > 0:
            self.down_stages = nn.ModuleList([
                DownNFuseStage(
                    dim, num_blocks=num_blocks, in_mlp=in_mlp, out_mlp=out_mlp,
                    mlp_activation=mlp_activation, mlp_norm=mlp_norm,
                    mlp_drop=mlp_drop, num_heads=num_heads, qkv_bias=qkv_bias,
                    qk_scale=qk_scale, ffn_ratio=ffn_ratio,
                    residual_drop=residual_drop, attn_drop=attn_drop,
                    drop_path=drop_path, activation=activation, pre_ln=pre_ln,
                    no_sa=no_sa, no_ffn=no_ffn, pool=pool, fusion=fusion)
                for dim, num_blocks, in_mlp, out_mlp, mlp_drop, num_heads,
                    ffn_ratio, residual_drop, attn_drop, drop_path
                in zip(
                    down_dim, down_num_blocks, down_in_mlp, down_out_mlp,
                    down_mlp_drop, down_num_heads, down_ffn_ratio,
                    down_residual_drop, down_attn_drop, down_drop_path)])
        else:
            self.down_stages = None

        # Transformer decoder (up) Stages operating on Level-i data
        if len(down_dim) > 0:
            self.up_stages = nn.ModuleList([
                UpNFuseStage(
                    dim, num_blocks=num_blocks, in_mlp=in_mlp, out_mlp=out_mlp,
                    mlp_activation=mlp_activation, mlp_norm=mlp_norm,
                    mlp_drop=mlp_drop, num_heads=num_heads, qkv_bias=qkv_bias,
                    qk_scale=qk_scale, ffn_ratio=ffn_ratio,
                    residual_drop=residual_drop, attn_drop=attn_drop,
                    drop_path=drop_path, activation=activation, pre_ln=pre_ln,
                    no_sa=no_sa, no_ffn=no_ffn, unpool=unpool, fusion=fusion)
                for dim, num_blocks, in_mlp, out_mlp, mlp_drop, num_heads,
                    ffn_ratio, residual_drop, attn_drop, drop_path
                in zip(
                    up_dim, up_num_blocks, up_in_mlp, up_out_mlp,
                    up_mlp_drop, up_num_heads, up_ffn_ratio,
                    up_residual_drop, up_attn_drop, up_drop_path)])
        else:
            self.up_stages = None

        assert bool(self.down_stages) != bool(self.up_stages) \
               or self.num_down_stages > self.num_up_stages, \
            'The number of Up stages should be lower than the number of Down ' \
            'stages. That is to say, we do not want to output Level-0 ' \
            'features but at least Level-1.'

    @property
    def num_down_stages(self):
        return len(self.down_stages) if self.down_stages is not None else 0

    @property
    def num_up_stages(self):
        return len(self.up_stages) if self.up_stages is not None else 0

    @property
    def out_dim(self):
        if self.up_stages is not None:
            return self.up_stages[-1].out_dim
        if self.down_stages is not None:
            return self.down_stages[-1].out_dim
        return self.point_stage.out_dim

    def forward(self, nag, return_down_outputs=False, return_up_outputs=False):
        assert isinstance(nag, NAG)
        assert nag.num_levels >= 2
        assert nag.num_levels > self.num_down_stages

        # Encode Level-0 data
        x, diameter = self.point_stage(
            nag[0].pos, nag[0].x, nag[0].super_index,
            num_super=nag[1].num_nodes)

        # Append the diameter to the level-1 features
        nag[1].x = torch.cat((nag[1].x, diameter), dim=1)

        # Iteratively encode level-1 and above
        down_outputs = []
        if self.down_stages is not None:
            for i_stage, stage in enumerate(self.down_stages):
                i_level = i_stage + 1
                x_super = nag[i_level].x
                norm_index = nag[i_level].norm_index(mode=self.norm_mode)
                pool_index = nag[i_level - 1].super_index
                edge_index = nag[i_level].edge_index
                num_super = nag[i_level].num_nodes
                x = stage(
                    x_super, x, norm_index, pool_index, edge_index=edge_index,
                    num_super=num_super)
                down_outputs.append(x)

        # Iteratively decode level-num_down_stages and below
        up_outputs = []
        if self.up_stages is not None:
            for i_stage, stage in enumerate(self.up_stages):
                i_level = self.num_down_stages - i_stage - 1
                norm_index = nag[i_level].norm_index(mode=self.norm_mode)
                unpool_index = nag[i_level].super_index
                edge_index = nag[i_level].edge_index
                x_skip = down_outputs[-(2 + i_stage)]
                x = stage(
                    x_skip, x, norm_index, unpool_index, edge_index=edge_index)
                up_outputs.append(x)

        # Different types of outputs, depending
        if not return_down_outputs and not return_up_outputs:
            return x
        if not return_down_outputs:
            return x, up_outputs
        if not return_up_outputs:
            return x, down_outputs
        return x, down_outputs, up_outputs
