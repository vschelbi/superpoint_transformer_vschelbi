from torch import nn
from torch_geometric.nn.aggr import SumAggregation as SumPool
from torch_geometric.nn.aggr import MeanAggregation as MeanPool
from torch_geometric.nn.aggr import MaxAggregation as MaxPool
from torch_geometric.nn.aggr import MinAggregation as MinPool


__all__ = ['SumPool', 'MeanPool', 'MaxPool', 'MinPool']


class AttentivePool(nn.Module):

    # TODO: this module could be used for pooling from one segment level
    #  to the next. But requires defining how. With QKV paradigm ? Then
    #  how to define Q for superpoints ? from max-pooled/mean-pooled
    #  features ? from handcrafted features ? If not QKV, simply have a
    #  FFN predict (multi-headed) attention scores to be softmaxed ? How
    #  to guide pooling from the above level (same pb as for qkv) ?

    # TODO: see torch_geometric SoftmaxAggregation and
    #  AttentionalAggregation for possibilities. Among which, a
    #  learnable softmax temperature

    pass
