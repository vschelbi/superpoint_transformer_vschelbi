import torch
from torch import nn
from src.data import NAG
from src.utils import listify_with_reference
from src.nn import Stage, PointStage, DownNFuseStage, UpNFuseStage, \
    FastBatchNorm1d, CatFusion, CatInjection, RPEFFN

__all__ = ['NeST']


class NeST(nn.Module):
    """Nested Set Transformer. A UNet-like architecture processing NAG.
    """

    def __init__(
            self,

            point_mlp,
            point_drop=None,
            point_pos_injection=CatInjection,
            point_pos_injection_x_dim=None,
            point_cat_diameter=False,

            small=None,
            small_point_mlp=None,
            small_down_mlp=None,

            down_dim=None,
            down_in_mlp=None,
            down_out_mlp=None,
            down_mlp_drop=None,
            down_num_heads=1,
            down_num_blocks=0,
            down_ffn_ratio=4,
            down_residual_drop=None,
            down_attn_drop=None,
            down_drop_path=None,
            down_inject_pos=True,
            down_inject_x=False,
            down_pos_injection_x_dim=None,

            up_dim=None,
            up_in_mlp=None,
            up_out_mlp=None,
            up_mlp_drop=None,
            up_num_heads=1,
            up_num_blocks=0,
            up_ffn_ratio=4,
            up_residual_drop=None,
            up_attn_drop=None,
            up_drop_path=None,
            up_inject_pos=True,
            up_inject_x=False,
            up_pos_injection_x_dim=None,

            last_dim=None,
            last_in_mlp=None,
            last_out_mlp=None,
            last_mlp_drop=None,
            last_num_heads=1,
            last_num_blocks=0,
            last_ffn_ratio=4,
            last_residual_drop=None,
            last_attn_drop=None,
            last_drop_path=None,
            last_inject_pos=True,
            last_pos_injection_x_dim=None,

            mlp_activation=nn.LeakyReLU(),
            mlp_norm=FastBatchNorm1d,
            qk_dim=8,
            qkv_bias=True,
            qk_scale=None,
            activation=nn.GELU(),
            pre_ln=True,
            no_sa=False,
            no_ffn=False,
            k_rpe=False,
            q_rpe=False,
            c_rpe=False,
            v_rpe=False,
            stages_share_rpe=False,
            blocks_share_rpe=False,
            heads_share_rpe=False,

            pos_injection=CatInjection,
            cat_diameter=False,
            pool='max',
            unpool='index',
            fusion='cat',
            norm_mode='graph'):

        super().__init__()

        self.down_inject_pos = down_inject_pos
        self.down_inject_x = down_inject_x
        self.up_inject_pos = up_inject_pos
        self.up_inject_x = up_inject_x
        self.last_inject_pos = last_inject_pos
        self.norm_mode = norm_mode
        self.stages_share_rpe = stages_share_rpe
        self.blocks_share_rpe = blocks_share_rpe
        self.heads_share_rpe = heads_share_rpe

        # Convert input arguments to nested lists
        (down_dim, down_in_mlp, down_out_mlp, down_mlp_drop, down_num_heads,
         down_num_blocks, down_ffn_ratio, down_residual_drop, down_attn_drop,
         down_drop_path, down_pos_injection_x_dim) = listify_with_reference(
            down_dim, down_in_mlp, down_out_mlp, down_mlp_drop, down_num_heads,
            down_num_blocks, down_ffn_ratio, down_residual_drop, down_attn_drop,
            down_drop_path, down_pos_injection_x_dim)

        (up_dim, up_in_mlp, up_out_mlp, up_mlp_drop, up_num_heads,
         up_num_blocks, up_ffn_ratio, up_residual_drop, up_attn_drop,
         up_drop_path, up_pos_injection_x_dim) = listify_with_reference(
            up_dim, up_in_mlp, up_out_mlp, up_mlp_drop, up_num_heads,
            up_num_blocks, up_ffn_ratio, up_residual_drop, up_attn_drop,
            up_drop_path, up_pos_injection_x_dim)

        # Module operating on Level-0 points in isolation
        self.point_stage = PointStage(
            point_mlp, mlp_activation=mlp_activation, mlp_norm=mlp_norm,
            mlp_drop=point_drop, pos_injection=point_pos_injection,
            pos_injection_x_dim=point_pos_injection_x_dim,
            cat_diameter=point_cat_diameter)

        # Operator to append the features such as the diameter or other 
        # handcrafted features to the NAG's features
        self.feature_fusion = CatFusion()

        # Transformer encoder (down) Stages operating on Level-i data
        if len(down_dim) > 0:

            # Build the RPE encoders here if shared across all stages
            down_k_rpe = _build_shared_rpe_encoders(
                k_rpe, len(down_dim), 13, qk_dim, stages_share_rpe)

            down_q_rpe = _build_shared_rpe_encoders(
                q_rpe, len(down_dim), 13, qk_dim, stages_share_rpe)

            self.down_stages = nn.ModuleList([
                DownNFuseStage(
                    dim, num_blocks=num_blocks, in_mlp=in_mlp, out_mlp=out_mlp,
                    mlp_activation=mlp_activation, mlp_norm=mlp_norm,
                    mlp_drop=mlp_drop, num_heads=num_heads, qk_dim=qk_dim,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, ffn_ratio=ffn_ratio,
                    residual_drop=residual_drop, attn_drop=attn_drop,
                    drop_path=drop_path, activation=activation, pre_ln=pre_ln,
                    no_sa=no_sa, no_ffn=no_ffn, k_rpe=stage_k_rpe,
                    q_rpe=stage_q_rpe, c_rpe=c_rpe, v_rpe=v_rpe, pool=pool,
                    fusion=fusion, pos_injection=pos_injection,
                    pos_injection_x_dim=pos_injection_x_dim,
                    cat_diameter=cat_diameter, blocks_share_rpe=blocks_share_rpe,
                    heads_share_rpe=heads_share_rpe)
                for dim, num_blocks, in_mlp, out_mlp, mlp_drop, num_heads,
                    ffn_ratio, residual_drop, attn_drop, drop_path,
                    stage_k_rpe, stage_q_rpe, pos_injection_x_dim
                in zip(
                    down_dim, down_num_blocks, down_in_mlp, down_out_mlp,
                    down_mlp_drop, down_num_heads, down_ffn_ratio,
                    down_residual_drop, down_attn_drop, down_drop_path,
                    down_k_rpe, down_q_rpe, down_pos_injection_x_dim)])
        else:
            self.down_stages = None

        # Transformer decoder (up) Stages operating on Level-i data
        if len(up_dim) > 0:

            # Build the RPE encoder here if shared across all stages
            up_k_rpe = _build_shared_rpe_encoders(
                k_rpe, len(up_dim), 13, qk_dim, stages_share_rpe)

            up_q_rpe = _build_shared_rpe_encoders(
                q_rpe, len(up_dim), 13, qk_dim, stages_share_rpe)

            self.up_stages = nn.ModuleList([
                UpNFuseStage(
                    dim, num_blocks=num_blocks, in_mlp=in_mlp, out_mlp=out_mlp,
                    mlp_activation=mlp_activation, mlp_norm=mlp_norm,
                    mlp_drop=mlp_drop, num_heads=num_heads, qk_dim=qk_dim,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, ffn_ratio=ffn_ratio,
                    residual_drop=residual_drop, attn_drop=attn_drop,
                    drop_path=drop_path, activation=activation, pre_ln=pre_ln,
                    no_sa=no_sa, no_ffn=no_ffn, k_rpe=stage_k_rpe,
                    q_rpe=stage_q_rpe, c_rpe=c_rpe, v_rpe=v_rpe, unpool=unpool,
                    fusion=fusion, pos_injection=pos_injection,
                    pos_injection_x_dim=pos_injection_x_dim,
                    blocks_share_rpe=blocks_share_rpe,
                    heads_share_rpe=heads_share_rpe)
                for dim, num_blocks, in_mlp, out_mlp, mlp_drop, num_heads,
                    ffn_ratio, residual_drop, attn_drop, drop_path,
                    stage_k_rpe, stage_q_rpe, pos_injection_x_dim
                in zip(
                    up_dim, up_num_blocks, up_in_mlp, up_out_mlp,
                    up_mlp_drop, up_num_heads, up_ffn_ratio,
                    up_residual_drop, up_attn_drop, up_drop_path,
                    up_k_rpe, up_q_rpe, up_pos_injection_x_dim)])
        else:
            self.up_stages = None

        assert bool(self.down_stages) != bool(self.up_stages) \
               or self.num_down_stages > self.num_up_stages, \
            "The number of Up stages should be lower than the number of Down " \
            "stages. That is to say, we do not want to output Level-0 " \
            "features but at least Level-1."

        # Optional pointNet-like module operating on Level-0 points in
        # for points belonging to small L1 nodes
        assert small is None or small > 0 and small_point_mlp is not None \
               and small_down_mlp is not None, \
            "If specified, `small` should be an integer larger than 0 and " \
            "`small_point_mlp` and `small_down_mlp` should also be specified"

        assert small is None or self.num_down_stages == self.num_up_stages + 1, \
            "If `small` is set, the model should have of one more Down " \
            "stages than Up stages (ie the output will be Level-1 features."

        self.small = small

        self.point_stage_small = PointStage(
            small_point_mlp, mlp_activation=mlp_activation, mlp_norm=mlp_norm,
            mlp_drop=point_drop, pos_injection=point_pos_injection,
            pos_injection_x_dim=point_pos_injection_x_dim,
            cat_diameter=point_cat_diameter) \
            if small is not None else None

        self.down_stage_small = DownNFuseStage(
            small_down_mlp[-1], num_blocks=0, in_mlp=small_down_mlp,
            mlp_activation=mlp_activation, mlp_norm=mlp_norm,
            mlp_drop=down_mlp_drop[0], pool=pool, fusion='cat',
            pos_injection=pos_injection, cat_diameter=cat_diameter,
            pos_injection_x_dim=down_pos_injection_x_dim[0]) \
            if small is not None else None

        # Optional last transformer stage operating on Level-1 nodes.
        # In particular, allows feature propagation between large and
        # small nodes when `self.small` is not None
        if last_dim is not None:

            # Build the RPE encoder here if shared across all stages
            last_k_rpe = _build_shared_rpe_encoders(
                k_rpe, 1, 13, qk_dim, stages_share_rpe)[0]

            last_q_rpe = _build_shared_rpe_encoders(
                q_rpe, 1, 13, qk_dim, stages_share_rpe)[0]

            self.last_stage = Stage(
                last_dim, num_blocks=last_num_blocks, in_mlp=last_in_mlp,
                out_mlp=last_out_mlp, mlp_activation=mlp_activation,
                mlp_norm=mlp_norm, mlp_drop=last_mlp_drop,
                num_heads=last_num_heads, qk_dim=qk_dim, qkv_bias=qkv_bias,
                qk_scale=qk_scale, ffn_ratio=last_ffn_ratio,
                residual_drop=last_residual_drop,
                attn_drop=last_attn_drop, drop_path=last_drop_path,
                activation=activation, pre_ln=pre_ln, no_sa=no_sa,
                no_ffn=no_ffn, k_rpe=last_k_rpe, q_rpe=last_q_rpe, c_rpe=c_rpe,
                v_rpe=v_rpe, pool=pool, pos_injection=pos_injection,
                cat_diameter=cat_diameter, blocks_share_rpe=blocks_share_rpe,
                heads_share_rpe=heads_share_rpe,
                pos_injection_x_dim=last_pos_injection_x_dim)
        else:
            self.last_stage = None

    @property
    def num_down_stages(self):
        return len(self.down_stages) if self.down_stages is not None else 0

    @property
    def num_up_stages(self):
        return len(self.up_stages) if self.up_stages is not None else 0

    @property
    def out_dim(self):
        if self.last_stage is not None:
            return self.last_stage.out_dim
        if self.up_stages is not None:
            return self.up_stages[-1].out_dim
        if self.down_stages is not None:
            return self.down_stages[-1].out_dim
        return self.point_stage.out_dim

    def forward(self, nag, return_down_outputs=False, return_up_outputs=False):
        assert isinstance(nag, NAG)
        assert nag.num_levels >= 2
        assert nag.num_levels > self.num_down_stages

        # Separate small L1 nodes from the rest of the NAG. Only
        # large-enough nodes will be encoded using the UNet. Small nodes
        # will be encoded separately, and merged before the `last_stage`
        if self.small is not None:

            # Separate the input NAG into two sub-NAGs: one holding the
            # small L1 nodes and one carrying the large ones and their
            # hierarchy
            mask_small = self.small < nag[1].node_size
            nag_full = nag
            nag = nag_full.select(1, torch.where(mask_small)[0])
            nag_small = nag_full.select(1, torch.where(~mask_small)[0])

            # Encode level-0 data for small L1 nodes
            x_small, diameter_small = self.point_stage_small(
                nag_small[0].x, nag_small[0].pos,
                super_index=nag_small[0].super_index)

            # Append the diameter to the level-1 features
            i_level = 1
            nag_small[i_level].x = self.feature_fusion(
                nag_small[i_level].x, diameter_small)

            # Forward on the down stage and the corresponding NAG
            # level
            x_small, diameter_small = self._forward_down_stage(
                self.down_stage_small, nag_small, i_level, x_small)

            # Append the diameter to the next level's features
            if i_level < nag_small.num_levels - 1:
                nag_small[i_level + 1].x = self.feature_fusion(
                    nag_small[i_level + 1].x, diameter_small)

        # Encode level-0 data
        # NB: no node_size for the level-0 points
        x, diameter = self.point_stage(
            nag[0].x, nag[0].pos, super_index=nag[0].super_index)

        # Append the diameter to the level-1 features
        nag[1].x = self.feature_fusion(nag[1].x, diameter)

        # Iteratively encode level-1 and above
        down_outputs = []
        if self.down_stages is not None:
            for i_stage, stage in enumerate(self.down_stages):

                # Forward on the down stage and the corresponding NAG
                # level
                i_level = i_stage + 1
                x, diameter = self._forward_down_stage(stage, nag, i_level, x)
                down_outputs.append(x)

                # End here if we reached the last NAG level
                if i_level == nag.num_levels - 1:
                    continue

                # Append the diameter to the next level's features
                nag[i_level + 1].x = self.feature_fusion(
                    nag[i_level + 1].x, diameter)

        # Iteratively decode level-num_down_stages and below
        up_outputs = []
        if self.up_stages is not None:
            for i_stage, stage in enumerate(self.up_stages):
                i_level = self.num_down_stages - i_stage - 1
                x_skip = down_outputs[-(2 + i_stage)]
                x, _ = self._forward_up_stage(stage, nag, i_level, x, x_skip)
                up_outputs.append(x)

        # Fuse L1-level features back together, if need be
        if self.small is not None:
            nag = nag_full
            nag[1].x = torch.empty(
                (nag[1].num_nodes, x.shape[1]), device=x.device)
            nag[1].x[torch.where(mask_small)] = x
            nag[1].x[torch.where(~mask_small)] = x_small
            x = nag[1].x
            if len(up_outputs) > 0:
                up_outputs[-1] = x

            # TODO: if the per-stage output is returned, need to define how to
            # deal with small L1 nodes
            if return_down_outputs or return_up_outputs:
                raise NotImplementedError(
                    "Returning per-stage output is not supported yet when "
                    "`small` is specified")

        # Last spatial propagation of features
        if self.last_stage is not None:
            x, _ = self._forward_last_stage(nag, x)

        # Different types of output signatures
        if not return_down_outputs and not return_up_outputs:
            return x
        if not return_down_outputs:
            return x, up_outputs
        if not return_up_outputs:
            return x, down_outputs
        return x, down_outputs, up_outputs

    def _forward_down_stage(self, stage, nag, i_level, x):
        # Convert stage index to NAG index
        is_last_level = (i_level == nag.num_levels - 1)

        # Recover segment-level attributes
        x_handcrafted = nag[i_level].x if self.down_inject_x else None
        pos = nag[i_level].pos if self.down_inject_pos else None
        node_size = nag[i_level].node_size if self.down_inject_pos \
            else None
        num_nodes = nag[i_level].num_nodes

        # Recover indices for normalization, pooling, position
        # normalization and horizontal attention
        norm_index = nag[i_level].norm_index(mode=self.norm_mode)
        pool_index = nag[i_level - 1].super_index
        super_index = nag[i_level].super_index if not is_last_level \
            else None
        edge_index = nag[i_level].edge_index
        edge_attr = nag[i_level].edge_attr

        # Forward pass on the stage and store output x
        x_out, diameter = stage(
            x_handcrafted, x, norm_index, pool_index,
            pos=pos, node_size=node_size, super_index=super_index,
            edge_index=edge_index, edge_attr=edge_attr, num_super=num_nodes)

        return x_out, diameter

    def _forward_up_stage(self, stage, nag, i_level, x, x_skip):
        # Recover segment-level attributes
        x_handcrafted = nag[i_level].x if self.up_inject_x else None
        pos = nag[i_level].pos if self.up_inject_pos else None
        node_size = nag[i_level].node_size if self.up_inject_pos \
            else None

        # Append the handcrafted features to the x_skip. Has no
        # effect if 'x_handcrafted' is None
        x_skip = self.feature_fusion(x_skip, x_handcrafted)

        # Recover indices for normalization, unpooling, position
        # normalization and horizontal attention
        norm_index = nag[i_level].norm_index(mode=self.norm_mode)
        unpool_index = nag[i_level].super_index
        super_index = nag[i_level].super_index
        edge_index = nag[i_level].edge_index
        edge_attr = nag[i_level].edge_attr

        x_out, diameter = stage(
            x_skip, x, norm_index, unpool_index, pos=pos,
            node_size=node_size, super_index=super_index,
            edge_index=edge_index, edge_attr=edge_attr)

        return x_out, diameter

    def _forward_last_stage(self, nag, x):
        # Convert stage index to NAG index
        i_level = 1
        is_last_level = (i_level == nag.num_levels - 1)

        # Recover segment-level attributes
        pos = nag[i_level].pos if self.last_inject_pos else None
        node_size = nag[i_level].node_size if self.last_inject_pos \
            else None

        # Recover indices for normalization, pooling, position
        # normalization and horizontal attention
        norm_index = nag[i_level].norm_index(mode=self.norm_mode)
        super_index = nag[i_level].super_index if not is_last_level \
            else None
        edge_index = nag[i_level].edge_index
        edge_attr = nag[i_level].edge_attr

        # Forward pass on the stage and store output x
        x_out, diameter = self.last_stage(
            x, norm_index, pos=pos, node_size=node_size,
            super_index=super_index, edge_index=edge_index, edge_attr=edge_attr)

        return x_out, diameter


def _build_shared_rpe_encoders(
        rpe, num_stages, in_dim, out_dim, stages_share):
    """Local helper to build RPE encoders for NEST. The main goal is to
    make shared encoders construction easier.

    Note that setting stages_share=True will make all stages, blocks and
    heads use the same RPE encoder.
    """
    if not isinstance(rpe, bool):
        assert stages_share, \
            "If anything else but a boolean is passed for the RPE encoder, " \
            "this value will be passed to all Stages and stages_share should " \
            "be set to True."
        return [rpe] * num_stages

    # If all stages share the same RPE encoder, all blocks and all heads
    # too. We copy the same module instance to be shared across all
    # stages and blocks
    if stages_share and rpe:
        return [RPEFFN(in_dim, out_dim=out_dim)] * num_stages

    return [rpe] * num_stages
