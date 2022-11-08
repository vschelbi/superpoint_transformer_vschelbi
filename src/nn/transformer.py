from torch import nn
from src.nn import SelfAttentionBlock, FFN, DropPath, LayerNorm


__all__ = ['TransformerBlock']


# TODO: Careful with how we define the index for LayerNorm:
#  cluster-wise or cloud-wise ? Maybe cloud-wise, seems more stable...


class TransformerBlock(nn.Module):
    """Base block of the Transformer architecture:

        x ----------------- + ----------------- + -->
            \              |   \               |
             -- LN -- SA --     -- LN -- FFN --

    Where:
        - LN: LayerNorm
        - SA: Self-Attention
        - FFN: Feed-Forward Network

    Inspired by: https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
            self, dim, num_heads=1, qkv_bias=True, qk_scale=None, ffn_ratio=4,
            residual_drop=None, attn_drop=None, drop_path=None, activation=nn.GELU(),
            pre_ln=True, no_sa=False, no_ffn=False):
        super().__init__()

        self.dim = dim
        self.pre_ln = pre_ln

        # Self-Attention residual branch
        self.no_sa = no_sa
        if not no_sa:
            self.sa_norm = LayerNorm(dim)
            self.num_heads = num_heads
            self.sa = SelfAttentionBlock(
                dim, num_heads=num_heads, in_dim=None, out_dim=dim,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                drop=residual_drop)

        # Feed-Forward Network residual branch
        self.no_ffn = no_ffn
        if not no_ffn:
            self.ffn_norm = LayerNorm(dim)
            self.ffn_ratio = ffn_ratio
            self.ffn = FFN(
                dim, hidden_dim=int(dim * ffn_ratio), activation=activation,
                drop=residual_drop)

        # Optional DropPath module for stochastic depth
        self.drop_path = DropPath(drop_path) \
            if drop_path is not None and drop_path > 0 else nn.Identity()

    def forward(self, x, norm_index, edge_index=None):
        """
        :param x: FloatTensor or shape (N, C)
            Node features
        :param norm_index: LongTensor or shape (N)
            Node indices for the LayerNorm
        :param edge_index: LongTensor of shape (2, E)
            Edges in torch_geometric [[sources], [targets]] format for
            the self-attention module
        :return:
        """
        assert x.dim() == 2, 'x should be a 2D Tensor'
        assert x.is_floating_point(), 'x should be a 2D FloatTensor'
        assert norm_index.dim() == 1 and norm_index.shape[0] == x.shape[0], \
            'norm_index should be a 1D LongTensor'
        assert edge_index is None or edge_index.dim() == 2, \
            'edge_index should be a 2D LongTensor'
        assert edge_index is None or not edge_index.is_floating_point(), \
            'edge_index should be a 2D LongTensor'

        # Keep track of x for the residual connection
        shortcut = x

        # Self-Attention residual branch
        if not self.no_sa and self.pre_ln:
            x = self.sa_norm(x, norm_index)
            x = self.sa(x, edge_index)
            x = shortcut + self.drop_path(x)
        if not self.no_sa and not self.pre_ln:
            x = self.sa(x, edge_index)
            x = self.drop_path(x)
            x = self.sa_norm(shortcut + x, norm_index)

        # Feed-Forward Network residual branch
        if not self.no_ffn and self.pre_ln:
            x = self.ffn_norm(x, norm_index)
            x = self.ffn(x)
            x = shortcut + self.drop_path(x)
        if not self.no_ffn and not self.pre_ln:
            x = self.ffn(x)
            x = self.drop_path(x)
            x = self.ffn_norm(shortcut + x, norm_index)

        return x, norm_index, edge_index
