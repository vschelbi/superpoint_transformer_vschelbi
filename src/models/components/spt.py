import torch
from torch import nn
from src.utils import listify_with_reference
from src.nn import Stage, PointStage, DownNFuseStage, UpNFuseStage, \
    BatchNorm, CatFusion, CatInjection, MLP, LayerNorm
from src.nn.pool import BaseAttentivePool
from src.nn.pool import pool_factory

__all__ = ['SPT']


class SPT(nn.Module):
    """Superpoint Transformer. A UNet-like architecture processing NAG.
    """

    def __init__(
            self,

            point_mlp=None,
            point_drop=None,
            point_pos_injection=CatInjection,
            point_pos_injection_x_dim=None,
            point_cat_diameter=False,
            point_log_diameter=False,

            small=None,
            small_point_mlp=None,
            small_down_mlp=None,

            nano=False,

            down_dim=None,
            down_pool_dim=None,
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

            node_mlp=None,
            h_edge_mlp=None,
            v_edge_mlp=None,
            mlp_activation=nn.LeakyReLU(),
            mlp_norm=BatchNorm,
            qk_dim=8,
            qkv_bias=True,
            qk_scale=None,
            in_rpe_dim=18,
            activation=nn.LeakyReLU(),
            norm=LayerNorm,
            pre_norm=True,
            no_sa=False,
            no_ffn=False,
            k_rpe=False,
            q_rpe=False,
            v_rpe=False,
            k_delta_rpe=False,
            q_delta_rpe=False,
            qk_share_rpe=False,
            q_on_minus_rpe=False,
            share_hf_mlps=False,
            stages_share_rpe=False,
            blocks_share_rpe=False,
            heads_share_rpe=False,

            pos_injection=CatInjection,
            cat_diameter=False,
            log_diameter=False,
            pool='max',
            unpool='index',
            fusion='cat',
            norm_mode='graph',
            output_stage_wise=False):

        super().__init__()

        self.nano = nano
        self.down_inject_pos = down_inject_pos
        self.down_inject_x = down_inject_x
        self.up_inject_pos = up_inject_pos
        self.up_inject_x = up_inject_x
        self.last_inject_pos = last_inject_pos
        self.norm_mode = norm_mode
        self.stages_share_rpe = stages_share_rpe
        self.blocks_share_rpe = blocks_share_rpe
        self.heads_share_rpe = heads_share_rpe
        self.output_stage_wise = output_stage_wise

        # Convert input arguments to nested lists
        (
            down_dim,
            down_pool_dim,
            down_in_mlp,
            down_out_mlp,
            down_mlp_drop,
            down_num_heads,
            down_num_blocks,
            down_ffn_ratio,
            down_residual_drop,
            down_attn_drop,
            down_drop_path,
            down_pos_injection_x_dim
        ) = listify_with_reference(
            down_dim,
            down_pool_dim,
            down_in_mlp,
            down_out_mlp,
            down_mlp_drop,
            down_num_heads,
            down_num_blocks,
            down_ffn_ratio,
            down_residual_drop,
            down_attn_drop,
            down_drop_path,
            down_pos_injection_x_dim)

        (
            up_dim,
            up_in_mlp,
            up_out_mlp,
            up_mlp_drop,
            up_num_heads,
            up_num_blocks,
            up_ffn_ratio,
            up_residual_drop,
            up_attn_drop,
            up_drop_path,
            up_pos_injection_x_dim
        ) = listify_with_reference(
            up_dim,
            up_in_mlp,
            up_out_mlp,
            up_mlp_drop,
            up_num_heads,
            up_num_blocks,
            up_ffn_ratio,
            up_residual_drop,
            up_attn_drop,
            up_drop_path,
            up_pos_injection_x_dim)

        # Local helper variables describing the architecture
        num_down = len(down_dim) - self.nano
        num_up = len(up_dim)
        needs_node_hf = down_inject_x or up_inject_x
        needs_h_edge_hf = any(
            x > 0 for x in down_num_blocks + up_num_blocks + [last_num_blocks])
        needs_v_edge_hf = num_down > 0 and isinstance(
            pool_factory(pool, down_pool_dim[0]), BaseAttentivePool)

        # Build MLPs that will be used to process handcrafted segment
        # and edge features. These will be called before each
        # DownNFuseStage and their output will be passed to
        # DownNFuseStage and UpNFuseStage. For the special case of nano
        # models, the first mlps will be run before the first Stage too
        node_mlp = node_mlp if needs_node_hf else None
        self.node_mlps = _build_mlps(
            node_mlp,
            num_down + self.nano,
            mlp_activation,
            mlp_norm,
            share_hf_mlps)

        h_edge_mlp = h_edge_mlp if needs_h_edge_hf else None
        self.h_edge_mlps = _build_mlps(
            h_edge_mlp,
            num_down + self.nano,
            mlp_activation,
            mlp_norm,
            share_hf_mlps)

        v_edge_mlp = v_edge_mlp if needs_v_edge_hf else None
        self.v_edge_mlps = _build_mlps(
            v_edge_mlp,
            num_down,
            mlp_activation,
            mlp_norm,
            share_hf_mlps)

        # Module operating on Level-0 points in isolation
        if self.nano:
            self.first_stage = Stage(
                down_dim[0],
                num_blocks=down_num_blocks[0],
                in_mlp=down_in_mlp[0],
                out_mlp=down_out_mlp[0],
                mlp_activation=mlp_activation,
                mlp_norm=mlp_norm,
                mlp_drop=down_mlp_drop[0],
                num_heads=down_num_heads[0],
                qk_dim=qk_dim,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                in_rpe_dim=in_rpe_dim,
                ffn_ratio=down_ffn_ratio[0],
                residual_drop=down_residual_drop[0],
                attn_drop=down_attn_drop[0],
                drop_path=down_drop_path[0],
                activation=activation,
                norm=norm,
                pre_norm=pre_norm,
                no_sa=no_sa,
                no_ffn=no_ffn,
                k_rpe=k_rpe,
                q_rpe=q_rpe,
                v_rpe=v_rpe,
                k_delta_rpe=k_delta_rpe,
                q_delta_rpe=q_delta_rpe,
                qk_share_rpe=qk_share_rpe,
                q_on_minus_rpe=q_on_minus_rpe,
                pos_injection=pos_injection,
                pos_injection_x_dim=down_pos_injection_x_dim[0],
                cat_diameter=cat_diameter,
                log_diameter=log_diameter,
                blocks_share_rpe=blocks_share_rpe,
                heads_share_rpe=heads_share_rpe)
        else:
            self.first_stage = PointStage(
                point_mlp,
                mlp_activation=mlp_activation,
                mlp_norm=mlp_norm,
                mlp_drop=point_drop,
                pos_injection=point_pos_injection,
                pos_injection_x_dim=point_pos_injection_x_dim,
                cat_diameter=point_cat_diameter,
                log_diameter=point_log_diameter)

        # Operator to append the features such as the diameter or other 
        # handcrafted features to the NAG's features
        self.feature_fusion = CatFusion()

        # Transformer encoder (down) Stages operating on Level-i data
        if num_down > 0:

            # Build the RPE encoders here if shared across all stages
            down_k_rpe = _build_shared_rpe_encoders(
                k_rpe, num_down, 18, qk_dim, stages_share_rpe)

            # If key and query RPEs share the same MLP, only the key MLP
            # is preserved, to limit the number of model parameters
            down_q_rpe = _build_shared_rpe_encoders(
                q_rpe and not (k_rpe and qk_share_rpe), num_down, 18, qk_dim,
                stages_share_rpe)

            # Since the first value of each down_ parameter is used for
            # the nano Stage (if self.nano=True), we artificially
            # prepend None values to the rpe lists, so they have the
            # same length as other down_ parameters
            if self.nano:
                down_k_rpe = [None] + down_k_rpe
                down_q_rpe = [None] + down_q_rpe

            self.down_stages = nn.ModuleList([
                DownNFuseStage(
                    dim,
                    num_blocks=num_blocks,
                    in_mlp=in_mlp,
                    out_mlp=out_mlp,
                    mlp_activation=mlp_activation,
                    mlp_norm=mlp_norm,
                    mlp_drop=mlp_drop,
                    num_heads=num_heads,
                    qk_dim=qk_dim,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    in_rpe_dim=in_rpe_dim,
                    ffn_ratio=ffn_ratio,
                    residual_drop=residual_drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    activation=activation,
                    norm=norm,
                    pre_norm=pre_norm,
                    no_sa=no_sa,
                    no_ffn=no_ffn,
                    k_rpe=stage_k_rpe,
                    q_rpe=stage_q_rpe,
                    v_rpe=v_rpe,
                    k_delta_rpe=k_delta_rpe,
                    q_delta_rpe=q_delta_rpe,
                    qk_share_rpe=qk_share_rpe,
                    q_on_minus_rpe=q_on_minus_rpe,
                    pool=pool_factory(pool, pool_dim),
                    fusion=fusion,
                    pos_injection=pos_injection,
                    pos_injection_x_dim=pos_injection_x_dim,
                    cat_diameter=cat_diameter,
                    log_diameter=log_diameter,
                    blocks_share_rpe=blocks_share_rpe,
                    heads_share_rpe=heads_share_rpe)
                for
                    i_down,
                    (dim,
                    num_blocks,
                    in_mlp,
                    out_mlp,
                    mlp_drop,
                    num_heads,
                    ffn_ratio,
                    residual_drop,
                    attn_drop,
                    drop_path,
                    stage_k_rpe,
                    stage_q_rpe,
                    pool_dim,
                    pos_injection_x_dim)
                in enumerate(zip(
                    down_dim,
                    down_num_blocks,
                    down_in_mlp,
                    down_out_mlp,
                    down_mlp_drop,
                    down_num_heads,
                    down_ffn_ratio,
                    down_residual_drop,
                    down_attn_drop,
                    down_drop_path,
                    down_k_rpe,
                    down_q_rpe,
                    down_pool_dim,
                    down_pos_injection_x_dim))
                if i_down >= self.nano])
        else:
            self.down_stages = None

        # Transformer decoder (up) Stages operating on Level-i data
        if num_up > 0:

            # Build the RPE encoder here if shared across all stages
            up_k_rpe = _build_shared_rpe_encoders(
                k_rpe, num_up, 18, qk_dim, stages_share_rpe)

            # If key and query RPEs share the same MLP, only the key MLP
            # is preserved, to limit the number of model parameters
            up_q_rpe = _build_shared_rpe_encoders(
                q_rpe and not (k_rpe and qk_share_rpe), num_up, 18, qk_dim,
                stages_share_rpe)

            self.up_stages = nn.ModuleList([
                UpNFuseStage(
                    dim,
                    num_blocks=num_blocks,
                    in_mlp=in_mlp,
                    out_mlp=out_mlp,
                    mlp_activation=mlp_activation,
                    mlp_norm=mlp_norm,
                    mlp_drop=mlp_drop,
                    num_heads=num_heads,
                    qk_dim=qk_dim,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    in_rpe_dim=in_rpe_dim,
                    ffn_ratio=ffn_ratio,
                    residual_drop=residual_drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    activation=activation,
                    norm=norm,
                    pre_norm=pre_norm,
                    no_sa=no_sa,
                    no_ffn=no_ffn,
                    k_rpe=stage_k_rpe,
                    q_rpe=stage_q_rpe,
                    v_rpe=v_rpe,
                    k_delta_rpe=k_delta_rpe,
                    q_delta_rpe=q_delta_rpe,
                    qk_share_rpe=qk_share_rpe,
                    q_on_minus_rpe=q_on_minus_rpe,
                    unpool=unpool,
                    fusion=fusion,
                    pos_injection=pos_injection,
                    pos_injection_x_dim=pos_injection_x_dim,
                    blocks_share_rpe=blocks_share_rpe,
                    heads_share_rpe=heads_share_rpe)
                for dim,
                    num_blocks,
                    in_mlp,
                    out_mlp,
                    mlp_drop,
                    num_heads,
                    ffn_ratio,
                    residual_drop,
                    attn_drop,
                    drop_path,
                    stage_k_rpe,
                    stage_q_rpe,
                    pos_injection_x_dim
                in zip(
                    up_dim,
                    up_num_blocks,
                    up_in_mlp,
                    up_out_mlp,
                    up_mlp_drop,
                    up_num_heads,
                    up_ffn_ratio,
                    up_residual_drop,
                    up_attn_drop,
                    up_drop_path,
                    up_k_rpe,
                    up_q_rpe,
                    up_pos_injection_x_dim)])
        else:
            self.up_stages = None

        assert self.num_up_stages > 0 or not self.output_stage_wise, \
            "At least one up stage is needed for output_stage_wise=True"

        assert bool(self.down_stages) != bool(self.up_stages) \
               or self.num_down_stages >= self.num_up_stages, \
            "The number of Up stages should be <= the number of Down " \
            "stages."
        assert self.nano or self.num_down_stages > self.num_up_stages, \
            "The number of Up stages should be < the number of Down " \
            "stages. That is to say, we do not want to output Level-0 " \
            "features but at least Level-1."

        # Optional pointNet-like module operating on Level-0 points in
        # for points belonging to small L1 nodes
        # TODO: if the per-stage output is returned, need to define how to
        #  deal with small L1 nodes
        assert small is None or not self.output_stage_wise,\
            "Returning per-stage output with `small` is not supported"

        assert small is None or small > 0 and small_point_mlp is not None \
               and small_down_mlp is not None, \
            "If specified, `small` should be an integer larger than 0 and " \
            "`small_point_mlp` and `small_down_mlp` should also be specified"

        assert small is None or self.num_down_stages == self.num_up_stages + 1, \
            "If `small` is set, the model should have of one more Down " \
            "stages than Up stages (ie the output will be Level-1 features."

        if not self.nano and small is not None:
            self.small = small

            self.first_stage_small = PointStage(
                small_point_mlp,
                mlp_activation=mlp_activation,
                mlp_norm=mlp_norm,
                mlp_drop=point_drop,
                pos_injection=point_pos_injection,
                pos_injection_x_dim=point_pos_injection_x_dim,
                cat_diameter=point_cat_diameter,
                log_diameter=point_log_diameter)

            self.down_stage_small = DownNFuseStage(
                small_down_mlp[-1],
                num_blocks=0,
                in_mlp=small_down_mlp,
                mlp_activation=mlp_activation,
                mlp_norm=mlp_norm,
                mlp_drop=down_mlp_drop[0],
                pool=pool,
                fusion='cat',
                pos_injection=pos_injection,
                cat_diameter=cat_diameter,
                log_diameter=log_diameter,
                pos_injection_x_dim=down_pos_injection_x_dim[0])
        else:
            self.small = None
            self.first_stage_small = None
            self.down_stage_small = None

        # Optional last transformer stage operating on Level-1 nodes.
        # In particular, allows feature propagation between large and
        # small nodes when `self.small` is not None
        if last_dim is not None:

            # Build the RPE encoder here if shared across all stages
            last_k_rpe = _build_shared_rpe_encoders(
                k_rpe, 1, 18, qk_dim, stages_share_rpe)[0]

            # If key and query RPEs share the same MLP, only the key MLP
            # is preserved, to limit the number of model parameters
            last_q_rpe = _build_shared_rpe_encoders(
                q_rpe and not (k_rpe and qk_share_rpe), 1, 18, qk_dim,
                stages_share_rpe)[0]

            self.last_stage = Stage(
                last_dim,
                num_blocks=last_num_blocks,
                in_mlp=last_in_mlp,
                out_mlp=last_out_mlp,
                mlp_activation=mlp_activation,
                mlp_norm=mlp_norm,
                mlp_drop=last_mlp_drop,
                num_heads=last_num_heads,
                qk_dim=qk_dim,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                in_rpe_dim=in_rpe_dim,
                ffn_ratio=last_ffn_ratio,
                residual_drop=last_residual_drop,
                attn_drop=last_attn_drop,
                drop_path=last_drop_path,
                activation=activation,
                norm=norm,
                pre_norm=pre_norm,
                no_sa=no_sa,
                no_ffn=no_ffn,
                k_rpe=last_k_rpe,
                q_rpe=last_q_rpe,
                v_rpe=v_rpe,
                k_delta_rpe=k_delta_rpe,
                q_delta_rpe=q_delta_rpe,
                qk_share_rpe=qk_share_rpe,
                q_on_minus_rpe=q_on_minus_rpe,
                pos_injection=pos_injection,
                cat_diameter=cat_diameter,
                log_diameter=log_diameter,
                blocks_share_rpe=blocks_share_rpe,
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
        if self.output_stage_wise:
            out_dim = [stage.out_dim for stage in self.up_stages][::-1]
            if self.last_stage is not None:
                out_dim[0] = self.last_stage.out_dim
            out_dim += [self.down_stages[-1].out_dim]
            return out_dim
        if self.last_stage is not None:
            return self.last_stage.out_dim
        if self.up_stages is not None:
            return self.up_stages[-1].out_dim
        if self.down_stages is not None:
            return self.down_stages[-1].out_dim
        return self.first_stage.out_dim

    def forward(self, nag):
        # assert isinstance(nag, NAG)
        # assert nag.num_levels >= 2
        # assert nag.num_levels > self.num_down_stages

        # TODO: this will need to be changed if we want FAST NANO
        if self.nano:
            nag = nag[1:]

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
            x_small, diameter_small = self.first_stage_small(
                nag_small[0].x,
                nag_small[0].norm_index(mode=self.norm_mode),
                pos=nag_small[0].pos,
                node_size=getattr(nag_small[0], 'node_size', None),
                super_index=nag_small[0].super_index,
                edge_index=nag_small[0].edge_index,
                edge_attr=getattr(nag_small[0], 'edge_attr', None))

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

        # Apply the first MLPs on the handcrafted features
        if self.nano:
            if self.node_mlps is not None and self.node_mlps[0] is not None:
                norm_index = nag[0].norm_index(mode=self.norm_mode)
                x = nag[0].x

                # The first node_mlp may expect a diameter to be passed
                # from the UnitSphereNorm. In the specific case of Nano,
                # the first stage does not have a UnitSphereNorm to
                # produce such diameter, so we artificially create it
                # here
                if self.down_inject_pos:
                    diameter = torch.zeros(
                        (nag[0].num_nodes, 1), dtype=x.dtype, device=x.device)
                    x = self.feature_fusion(x, diameter)

                nag[0].x = self.node_mlps[0](x, batch=norm_index)
            if self.h_edge_mlps is not None:
                norm_index = nag[0].norm_index(mode=self.norm_mode)
                norm_index = norm_index[nag[0].edge_index[0]]
                nag[0].edge_attr = self.h_edge_mlps[0](
                    nag[0].edge_attr, batch=norm_index)

        # Encode level-0 data
        x, diameter = self._forward_first_stage(self.first_stage, nag)

        # Append the diameter to the level-1 features
        if self.down_inject_pos:
            nag[1].x = self.feature_fusion(nag[1].x, diameter)

        # Iteratively encode level-1 and above
        down_outputs = []
        if self.nano:
            down_outputs.append(x)
        if self.down_stages is not None:

            enum = enumerate(zip(
                self.down_stages,
                self.node_mlps[int(self.nano):],
                self.h_edge_mlps[int(self.nano):],
                self.v_edge_mlps))

            for i_stage, (stage, node_mlp, h_edge_mlp, v_edge_mlp) in enum:

                # Forward on the down stage and the corresponding NAG
                # level
                i_level = i_stage + 1

                # Process handcrafted node and edge features. We need to
                # do this here before those can be passed to the
                # DownNFuseStage and, later on, to the UpNFuseStage
                if node_mlp is not None:
                    norm_index = nag[i_level].norm_index(mode=self.norm_mode)
                    nag[i_level].x = node_mlp(nag[i_level].x, batch=norm_index)
                if h_edge_mlp is not None:
                    norm_index = nag[i_level].norm_index(mode=self.norm_mode)
                    norm_index = norm_index[nag[i_level].edge_index[0]]
                    edge_attr = getattr(nag[i_level], 'edge_attr', None)
                    if edge_attr is not None:
                        nag[i_level].edge_attr = h_edge_mlp(
                            edge_attr, batch=norm_index)
                if v_edge_mlp is not None:
                    norm_index = nag[i_level - 1].norm_index(mode=self.norm_mode)
                    v_edge_attr = getattr(nag[i_level], 'v_edge_attr', None)
                    if v_edge_attr is not None:
                        nag[i_level - 1].v_edge_attr = v_edge_mlp(
                            v_edge_attr, batch=norm_index)

                # Forward on the DownNFuseStage
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

        # Last spatial propagation of features
        if self.last_stage is not None:
            x, _ = self._forward_last_stage(nag, x)

        # Different types of output signatures. For stage-wise output,
        # return the output for each stage. For the L1 level, we must
        # take the 'last_stage' into account and not simply the output
        # of the last 'up_stage'. Besides, for the Lmax level, we take
        # the output of the innermost 'down_stage'. Finally, these
        # outputs are sorted by order of increasing NAG level (from low
        # to high)
        if self.output_stage_wise:
            out = [x] + up_outputs[::-1][1:] + [down_outputs[-1]]
            return out
        else:
            return x

    def _forward_first_stage(self, stage, nag):
        x = nag[0].x
        norm_index = nag[0].norm_index(mode=self.norm_mode)
        pos = nag[0].pos if not self.nano or self.down_inject_pos else None
        node_size = getattr(nag[0], 'node_size', None) if self.down_inject_pos \
            else None
        super_index = nag[0].super_index
        edge_index = nag[0].edge_index
        edge_attr = nag[0].edge_attr

        x_out, diameter = stage(
            x,
            norm_index,
            pos=pos,
            node_size=node_size,
            super_index=super_index,
            edge_index=edge_index,
            edge_attr=edge_attr)

        return x_out, diameter

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
        v_edge_attr = nag[i_level - 1].v_edge_attr

        # Forward pass on the stage and store output x
        x_out, diameter = stage(
            x_handcrafted,
            x,
            norm_index,
            pool_index,
            pos=pos,
            node_size=node_size,
            super_index=super_index,
            edge_index=edge_index,
            edge_attr=edge_attr,
            v_edge_attr=v_edge_attr,
            num_super=num_nodes)

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
            x_skip,
            x,
            norm_index,
            unpool_index,
            pos=pos,
            node_size=node_size,
            super_index=super_index,
            edge_index=edge_index,
            edge_attr=edge_attr)

        return x_out, diameter

    def _forward_last_stage(self, nag, x):
        # Convert stage index to NAG index
        i_level = int(not self.nano)
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
            x,
            norm_index,
            pos=pos,
            node_size=node_size,
            super_index=super_index,
            edge_index=edge_index,
            edge_attr=edge_attr)

        return x_out, diameter


def _build_shared_rpe_encoders(
        rpe, num_stages, in_dim, out_dim, stages_share):
    """Local helper to build RPE encoders for spt. The main goal is to
    make shared encoders construction easier.

    Note that setting stages_share=True will make all stages, blocks and
    heads use the same RPE encoder.
    """
    if not isinstance(rpe, bool):
        assert stages_share, \
            "If anything else but a boolean is passed for the RPE encoder, " \
            "this value will be passed to all Stages and `stages_share` " \
            "should be set to True."
        return [rpe] * num_stages

    # If all stages share the same RPE encoder, all blocks and all heads
    # too. We copy the same module instance to be shared across all
    # stages and blocks
    if stages_share and rpe:
        return [nn.Linear(in_dim, out_dim)] * num_stages

    return [rpe] * num_stages


def _build_mlps(layers, num_stage, activation, norm, shared):
    if layers is None:
        return [None] * num_stage

    if shared:
        return nn.ModuleList([
            MLP(layers, activation=activation, norm=norm)] * num_stage)

    return nn.ModuleList([
        MLP(layers, activation=activation, norm=norm)
        for _ in range(num_stage)])
