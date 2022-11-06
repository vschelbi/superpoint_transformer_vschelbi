from torch import nn
from src.nn import SelfAttentionBlock, FFN, DropPath, LayerNorm


__all__ = ['TransformerBlock']


# TODO: Careful with how we define the index for LayerNorm:
#  cluster-wise or cloud-wise ? Maybe cloud-wise, seems more stable...


class TransformerBlock(nn.Module):
    """TransformerBlock is composed of
    - residual SelfAttentionBlock
    - residual MLP conv1x1 (equivalent of FFN) (NO BATCHNORM !!!)
    - ScatterLayerNorm can be Pre-LN or Post-LN (default is pre-LN) (see in Swin-T) (see Pre-LN Transformer: https://arxiv.org/pdf/2002.04745.pdf)
        - pre-LN: LN inside the residual branches, right before the SelfAttentionBlock and MLP blocks
        - post-LN (setup from seminal transformer paper https://arxiv.org/abs/1706.03762 but https://arxiv.org/pdf/2002.04745.pdf proves bad gradients at initialization and requires warmup): LN after each residual sum
    - stochastic depth with drop_path
    - activation

    see SwinT SwinTransformerBlock for details:
    https://github.com/microsoft/Swin-Transformer/blob/d19503d7fbed704792a5e5a3a5ee36f9357d26c1/models/swin_transformer.py#L175
    """
    def __init__(
            self, dim, num_heads, qkv_bias=True, qk_scale=None, ffn_ratio=4,
            drop=None, attn_drop=None, drop_path=None, activation=nn.GELU(),
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
                drop=drop)

        # Feed-Forward Network residual branch
        self.no_ffn = no_ffn
        if not no_ffn:
            self.ffn_norm = LayerNorm(dim)
            self.ffn_ratio = ffn_ratio
            self.ffn = FFN(
                dim, hidden_dim=int(dim * ffn_ratio), activation=activation,
                drop=drop)

        # Optional DropPath module for stochastic depth
        self.drop_path = DropPath(drop_path) \
            if drop_path is not None and drop_path > 0 else nn.Identity()

    def forward(self, x, edge_index, norm_index, num_super=None):
        """"""
        assert x.dim() == 2, 'x should be a 2D Tensor'
        assert x.is_floating_point(), 'x should be a 2D FloatTensor'
        assert edge_index.dim() == 2, 'edge_index should be a 2D LongTensor'
        assert not edge_index.is_floating_point(), \
            'edge_index should be a 2D LongTensor'

        # Keep track of x for the residual connection
        shortcut = x

        # Self-Attention residual branch
        if not self.no_sa and self.pre_ln:
            x = self.sa_norm(x, norm_index)
            x = self.sa(x, edge_index, num_super=num_super)
            x = shortcut + self.drop_path(x)
        if not self.no_sa and not self.pre_ln:
            x = self.sa(x, edge_index, num_super=num_super)
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

        return x
