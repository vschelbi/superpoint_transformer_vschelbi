import torch
from torch import nn
from src.nn import MLP, Classifier, ScatterPool, ScatterUnitNorm
from src.utils import listify
from torch_scatter import scatter_softmax, scatter_sum


def parse_list_args(args):
    if args is None or len(args) == 0:
        return

    # Check whether args contains list of lists
    is_list_list = hasattr(args[0], '__iter__')


def init_weights(module):
    """Manual weight initialization. Allows setting specific init modes
    for certain modules. In particular, it is recommended to initialize
    Linear layers with Xavier initialization:
    https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

    see Romain's: https://github.com/romainloiseau/Helix4D/blob/87a9d01922e22ae692851233e8614d2b4f7baeed/helix4d/model/voxel/transformer/utils.py#3
    """
    pass


class NeST(nn.Module):
    """NeST is composed of
    - input spherenorm on points and stuff
    - nn.ModuleList of Stage DOWN
    - nn.ModuleList of Stage UP
    - skip connection fusion mode (concatenation / residual)

    """

class Stage(nn.Module):
    """Stage is composed of
    - nn.ModuleList of TransformerBlock
    - input MLP to change dim (optional)
    - output MLP to change dim  (optional)
    - input DOWN Pool (optional) see UNet: https://www.researchgate.net/publication/331406702/figure/fig2/AS:731276273262594@1551361258173/Illustration-of-the-U-net-architecture-The-figure-illustrates-the-U-net-architecture.png
    - output UP Pool (optional) see UNet: https://www.researchgate.net/publication/331406702/figure/fig2/AS:731276273262594@1551361258173/Illustration-of-the-U-net-architecture-The-figure-illustrates-the-U-net-architecture.png
    """
    pass


class ScatterLayerNorm(nn.Module):
    """LayerNorm on COO-formatted data

    Careful with weight init: should be 1 for weights and 0 for bias

    Careful with how we define the index for scatter normalization -> LayerNorm cluster-wise or cloud-wise ? Maybe cloud-wise, seems more stable...

    definition see romain's:
    https://github.com/romainloiseau/Helix4D/blob/87a9d01922e22ae692851233e8614d2b4f7baeed/helix4d/model/sparseops/layernorm.py

    init see romain's:
    https://github.com/romainloiseau/Helix4D/blob/87a9d01922e22ae692851233e8614d2b4f7baeed/helix4d/model/voxel/transformer/utils.py#L12
    """
    pass


class TransformerBlock(nn.Module):
    """TransformerBlock is composed of
    - residual SelfAttentionBlock
    - residual MLP conv1x1 (equivalent of FFN) (NO BATCHNORM !!!)
    - ScatterLayerNorm can be Pre-LN or Post-LN (default is pre-LN) (see in Swin-T) (see Pre-LN Transformer: https://arxiv.org/pdf/2002.04745.pdf)
        - pre-LN: LN inside the residual branches, right before the SelfAttentionBlock and MLP blocks
        - post-LN (setup from seminal transformer paper https://arxiv.org/abs/1706.03762 but https://arxiv.org/pdf/2002.04745.pdf proves bad gradients at initialization and requires warmup): LN after each residual sum

    see SwinT SwinTransformerBlock for details:
    https://github.com/microsoft/Swin-Transformer/blob/d19503d7fbed704792a5e5a3a5ee36f9357d26c1/models/swin_transformer.py#L175
    """
    def __init__(
            self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
            mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
            act_layer=nn.GELU, fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process


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
    :param out_drop:
    """
    def __init__(
            self, dim, num_heads=1, in_dim=None, out_dim=None, qkv_bias=True,
            qk_scale=None, attn_drop=None, out_drop=None):
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
        self.out_drop = nn.Dropout(out_drop) \
            if out_drop is not None and out_drop > 0 else None

        # TODO: define relative positional encoding parameters and
        #  trunacted-normal initialize them (see Swin-T implementation)

    def forward(self, x, edge_index):
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

        # Optional linear projection of features
        if self.in_proj is not None:
            x = self.in_proj(x)

        # Compute queries, keys and values
        qkv = self.qkv(x).view(N, 3, self.num_heads, self.dim // self.num_heads)

        # Separate and expand queries, keys, values and indices to edge
        # shape
        s = edge_index[0]  # [E]
        t = edge_index[1]  # [E]
        q = qkv[s, 0]      # [E, num_heads, C // num_heads]
        k = qkv[t, 1]      # [E, num_heads, C // num_heads]
        v = qkv[t, 2]      # [E, num_heads, C // num_heads]

        # Apply scaling on the queries
        # TODO: need some more scaling for the scatter-softmax right ?
        #  Like based on the group size ?
        q = q * self.qk_scale

        # TODO: make sure edge_index is undirected ? has self-loops ?

        # Compute compatibility scores from the query-key products
        compat = torch.einsum('ehd, ehd -> eh', q, k)  # [E, num_heads]

        # Compute the attention scores with scaled softmax
        # TODO: need some more scaling for the scatter-softmax right ?
        attn = scatter_softmax(compat, s, dim=0)  # [E, num_heads]

        # TODO: add the relative positional encodings to the
        #  compatibilities here

        # Optional attention dropout
        if self.attn_drop is not None:
            attn = self.attn_drop(attn)

        # Apply the attention on the values
        x = (v * attn.unsqueeze(-1)).view(N, self.dim)  # [E, C]
        x = scatter_sum(x, s, dim=0, dim_size=N)        # [N, C]

        # Optional linear projection of features
        if self.out_proj is not None:
            x = self.out_proj(x)  # [N, out_dim]

        # Optional dropout on projection of features
        if self.out_drop is not None:
            x = self.out_drop(x)  # [N, C or out_dim]

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'



class NeST(nn.Module):
    def __init__(
            self, self_attention=None, down=None, up=None, num_classes=None,
            reduce='max'):
        super().__init__()

        # Convert input arguments to nested lists
        self_attention = listify(self_attention)
        down = listify(down)
        up = listify(up)

        assert self_attention is None or len(self_attention) > 0
        assert down is None or len(down) > 0
        assert up is None or len(up) > 0
        assert down is None and up is None or len(down) == len(up)





        assert local_channels is None or len(local_channels) > 1
        assert global_channels is None or len(global_channels) > 1
        assert global_channels is not None or local_channels is not None

        self.sphere_norm = ScatterUnitNorm()



    def forward(self, pos, x, idx):
        # Normalize each segment to a unit sphere
        pos, diameter = self.sphere_norm(pos, idx)

        # Add normalized coordinates to the point features
        x = torch.cat((x, pos), dim=1)

        # Compute pointwise features
        x = self.local_nn(x) if self.local_nn else x

        # Segment pooling
        x_global = self.pool(x, idx)

        # Add initial diameter to segment features
        x_global = torch.cat((x_global, diameter.view(-1, 1)), dim=1)

        # Compute segment-level features
        x_global = self.global_nn(x_global) if self.global_nn else x_global

        # Classifying head
        x_out = self.head(x_global) if self.head else x_global

        return x_out
