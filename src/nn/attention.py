import torch
from torch import nn
from torch_scatter import scatter_sum
from torch_geometric.utils import softmax


__all__ = ['SelfAttentionBlock']


class SelfAttentionBlock(nn.Module):
    """SelfAttentionBlock is intended to be used in a residual fashion
    (or not) in TransformerBlock.

    Inspired by: https://github.com/microsoft/Swin-Transformer

    :param dim:
    :param num_heads:
    :param in_dim:
    :param out_dim:
    :param qkv_bias:
    :param qk_scale:
    :param attn_drop:
    :param drop:
    """
    def __init__(
            self, dim, num_heads=1, in_dim=None, out_dim=None, qkv_bias=True,
            qk_scale=None, attn_drop=None, drop=None):
        super().__init__()

        assert dim % num_heads == 0, f"dim must be a multiple of num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.qk_scale = qk_scale or (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.in_proj = nn.Linear(in_dim, dim) if in_dim is not None else None
        self.out_proj = nn.Linear(out_dim, dim) if out_dim is not None else None

        self.attn_drop = nn.Dropout(attn_drop) \
            if attn_drop is not None and attn_drop > 0 else None
        self.out_drop = nn.Dropout(drop) \
            if drop is not None and drop > 0 else None

        # TODO: define relative positional encoding parameters and
        #  trunacted-normal initialize them (see Swin-T implementation)

    def forward(self, x, edge_index, num_super=None):
        """
        :param x: Tensor of shape (N, C)
            Features
        :param edge_index: LongTensor of shape (2, E)
            Source and target indices for the edges of the attention
            graph. Source indicates the querying element, while Target
            indicates the key elements
        """
        N = x.shape[0]
        E = edge_index.shape[1]
        H = self.num_heads

        # Optional linear projection of features
        if self.in_proj is not None:
            x = self.in_proj(x)

        # Compute queries, keys and values
        qkv = self.qkv(x).view(N, 3, self.num_heads, self.dim // self.num_heads)

        # Separate and expand queries, keys, values and indices to edge
        # shape
        # TODO: make sure edge_index is undirected ? has self-loops ?
        s = edge_index[0]  # [E]
        t = edge_index[1]  # [E]
        q = qkv[s, 0]      # [E, H, C // H]
        k = qkv[t, 1]      # [E, H, C // H]
        v = qkv[t, 2]      # [E, H, C // H]

        # Apply scaling on the queries
        q = q * self.qk_scale

        # Compute compatibility scores from the query-key products
        compat = torch.einsum('ehd, ehd -> eh', q, k)  # [E, H]

        # Compute the attention scores with scaled softmax
        # TODO: need to scale the softmax based on the number of
        #  elements in each group ?
        attn = softmax(compat, index=s, dim=0, num_nodes=num_super)  # [E, H]

        # TODO: add the relative positional encodings to the
        #  compatibilities here

        # Optional attention dropout
        if self.attn_drop is not None:
            attn = self.attn_drop(attn)

        # Apply the attention on the values
        x = (v * attn.unsqueeze(-1)).view(N, self.dim)    # [E, C]
        x = scatter_sum(x, s, dim=0, dim_size=num_super)  # [N, C]

        # Optional linear projection of features
        if self.out_proj is not None:
            x = self.out_proj(x)  # [N, out_dim]

        # Optional dropout on projection of features
        if self.out_drop is not None:
            x = self.out_drop(x)  # [N, C or out_dim]

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'
