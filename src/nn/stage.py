import torch
from torch import nn
from src.nn import MLP, TransformerBlock, FastBatchNorm1d, UnitSphereNorm
from src.nn.pool import *
from src.nn.unpool import *
from src.nn.fusion import *


__all__ = ['Stage', 'DownStage', 'DownNFuseStage', 'UpNFuseStage', 'PointStage']


class Stage(nn.Module):
    """A Stage has the following structure:

        x -- in_MLP -- TransformerBlock -- out_MLP -->
           (optional)   (x num_blocks)   (optional)

    :param dim: int
        Number of channels for the TransformerBlock
    :param num_blocks: int
        Number of TransformerBlocks in the Stage
    :param in_mlp: List, optional
        Channels for the input MLP. The last channel must match
        `dim`
    :param out_mlp: List, optional
        Channels for the output MLP. The first channel must match
        `dim`
    :param mlp_activation: nn.Module
        Activation function for the input and output MLPs
    :param mlp_norm: nn.Module
        Normalization for the input and output MLPs
    :param mlp_drop: float, optional
        Dropout rate for the last layer of the input and output MLPs
    :param transformer_kwargs:
        Keyword arguments for the TransformerBlock
    """

    def __init__(
            self, dim, num_blocks=1, in_mlp=None, out_mlp=None,
            mlp_activation=nn.LeakyReLU(), mlp_norm=FastBatchNorm1d,
            mlp_drop=None, pos_encoding=False, **transformer_kwargs):

        super().__init__()

        self.dim = dim
        self.num_blocks = num_blocks

        # ScatterUnitNorm converts global node coordinates to
        # segment-level coordinates expressed in a unit-sphere. The
        # corresponding scaling factor (diameter) is returned, to be
        # used in potential subsequent stages
        self.pos_norm = UnitSphereNorm()

        # Fusion operator to combine node features with coordinates
        self.pos_fusion = CatFusion() if not pos_encoding else FourierEncoding(dim, x_dim=None, f_min=1e-1, f_max=1e1):

        # MLP to change input channel size
        if in_mlp is not None:
            assert in_mlp[-1] == dim
            self.in_mlp = MLP(
                in_mlp, activation=mlp_activation, norm=mlp_norm, momentum=0.1,
                drop=mlp_drop)
        else:
            self.in_mlp = None

        # MLP to change output channel size
        if out_mlp is not None:
            assert out_mlp[0] == dim
            self.out_mlp = MLP(
                out_mlp, activation=mlp_activation, norm=mlp_norm, momentum=0.1,
                drop=mlp_drop)
        else:
            self.out_mlp = None

        # Transformer blocks
        if num_blocks > 0:
            self.blocks = nn.Sequential(
                TransformerBlock(dim, **transformer_kwargs)
                for _ in range(num_blocks))
        else:
            self.blocks = None

    @property
    def out_dim(self):
        if self.out_mlp is not None:
            return self.out_mlp.out_dim
        if self.blocks is not None:
            return self.blocks[-1].dim
        if self.in_mlp is not None:
            return self.in_mlp.out_dim
        return self.dim

    def forward(
            self, x, norm_index, pos=None, node_size=None, super_index=None,
            edge_index=None):

        # Append normalized coordinates to the point features
        if pos is not None:
            pos, diameter = self.pos_norm(pos, super_index, w=node_size)
            x = self.pos_fusion(x, pos)
        else:
            diameter = None

        # MLP on input features to change channel size
        if self.in_mlp is not None:
            x = self.in_mlp(x)

        # Transformer blocks
        if self.blocks is not None:
            x = self.blocks(x, norm_index, edge_index=edge_index)

        # MLP on output features to change channel size
        if self.out_mlp is not None:
            x = self.out_mlp(x)

        return x, diameter


class DownStage(Stage):
    """A Stage preceded by a pooling operator to aggregate node
    features from level-i to level-i+1. A DownStage has the following
    structure:

        x -- Pool -- Stage -->
    """

    def __init__(self, *args, pool='max', **kwargs):
        super().__init__(*args, **kwargs)

        # Pooling operator
        if pool == 'max':
            self.pool = MaxPool()
        elif pool == 'min':
            self.pool = MinPool()
        elif pool == 'mean':
            self.pool = MeanPool()
        elif pool == 'sum':
            self.pool = SumPool()
        else:
            raise NotImplementedError(f'Unknown pool={pool} mode')

    def forward(
            self, x, norm_index, pool_index, pos=None, node_size=None,
            super_index=None, edge_index=None, num_super=None):
        x = self.pool(x, index=pool_index, dim_size=num_super)
        return super().forward(
            x, norm_index, pos=pos, node_size=node_size,
            super_index=super_index, edge_index=edge_index)


class DownNFuseStage(Stage):
    """A Stage preceded by a pooling operator and a fusion operator to
    aggregate node features from level-i to level-i+1 and fuse them
    with other features from level-i+1. A DownNFuseStage has the
    following structure:

        x1 ------- Fusion -- Stage -->
                     |
        x2 -- Pool --
    """

    def __init__(self, *args, pool='max', fusion='cat', **kwargs):
        super().__init__(*args, **kwargs)

        # Pooling operator
        if pool == 'max':
            self.pool = MaxPool()
        elif pool == 'min':
            self.pool = MinPool()
        elif pool == 'mean':
            self.pool = MeanPool()
        elif pool == 'sum':
            self.pool = SumPool()
        else:
            raise NotImplementedError(f'Unknown pool={pool} mode')

        # Fusion operator
        if fusion in ['cat', 'concatenate', 'concatenation']:
            self.fusion = CatFusion()
        elif fusion in ['residual', '+']:
            self.fusion = ResidualFusion()
        else:
            raise NotImplementedError(f'Unknown fusion={fusion} mode')

    def forward(
            self, x1, x2, norm_index, pool_index, pos=None, node_size=None,
            super_index=None, edge_index=None, num_super=None):
        x = self.fusion(x1, self.pool(x2, index=pool_index, dim_size=num_super))
        return super().forward(
            x, norm_index, pos=pos, node_size=node_size,
            super_index=super_index, edge_index=edge_index)


class UpNFuseStage(Stage):
    """A Stage preceded by an unpooling operator and a fusion operator
    to expand node features to from level-i+1 to level-i and fuse them
    with other features from level-i. An UpNFuseStage has the following
    structure:

        x1 --------- Fusion -- Stage -->
                       |
        x2 -- Unpool --

    The UpNFuseStage is typically used in a UNet-like decoder.
    """

    def __init__(self, *args, unpool='index', fusion='cat', **kwargs):
        super().__init__(*args, **kwargs)

        # Unpooling operator
        if unpool == 'index':
            self.unpool = IndexUnpool()
        else:
            raise NotImplementedError(f'Unknown unpool={unpool} mode')

        # Fusion operator
        if fusion in ['cat', 'concatenate', 'concatenation']:
            self.fusion = CatFusion()
        elif fusion in ['residual', '+']:
            self.fusion = ResidualFusion()
        else:
            raise NotImplementedError(f'Unknown fusion={fusion} mode')

    def forward(
            self, x1, x2, norm_index, unpool_index, pos=None, node_size=None,
            super_index=None, edge_index=None):
        x = self.fusion(x1, self.unpool(x2, unpool_index))
        return super().forward(
            x, norm_index, pos=pos, node_size=node_size,
            super_index=super_index, edge_index=edge_index)


class PointStage(Stage):
    """A Stage specifically designed for operating on raw points. This
    is similar to the point-wise part of PointNet, operating on Level-1
    segments. A PointStage has the following structure:

        x -- ScatterUnitNorm -- in_MLP -->

    :param in_mlp: List, optional
        Channels for the input MLP. The last channel must match
        `dim`
    :param mlp_activation: nn.Module
        Activation function for the input and output MLPs
    :param mlp_norm: nn.Module
        Normalization for the input and output MLPs
    :param mlp_drop: float, optional
        Dropout rate for the last layer of the input and output MLPs
    """

    def __init__(
            self, in_mlp, mlp_activation=nn.LeakyReLU(),
            mlp_norm=FastBatchNorm1d, mlp_drop=None):

        assert len(in_mlp) > 1, \
            'in_mlp should be a list of channels of length >= 2'

        super().__init__(
            in_mlp[-1], num_blocks=0, in_mlp=in_mlp, out_mlp=None,
            mlp_activation=mlp_activation, mlp_norm=mlp_norm,
            mlp_drop=mlp_drop)

        # ScatterUnitNorm converts global point coordinates to
        # cluster-level coordinates expressed in a unit-sphere. The
        # corresponding scaling factor (diameter) is returned, to be
        # used in potential subsequent network stages
        self.sphere_norm = UnitSphereNorm()

        # Fusion operator to combine point features with coordinates
        self.fusion = CatFusion()

    def forward(self, x, pos, node_size=None, super_index=None):
        return super().forward(
            x, None, pos=pos, node_size=node_size, super_index=super_index)
