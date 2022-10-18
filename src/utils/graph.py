import torch
from torch_geometric.nn.pool.consecutive import consecutive_cluster


__all__ = ['is_pyg_edge_format', 'isolated_nodes', 'edge_to_superedge']


def is_pyg_edge_format(edge_index):
    """Check whether edge_index follows pytorch geometric graph edge
    format: a [2, N] torch.LongTensor.
    """
    return \
        isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2 \
        and edge_index.dtype == torch.long and edge_index.shape[0] == 2


def isolated_nodes(edge_index, num_nodes=None):
    """Return a boolean mask of size num_nodes indicating which node has
    no edge in edge_index.
    """
    assert is_pyg_edge_format(edge_index)
    num_nodes = edge_index.max() + 1 if num_nodes is None else num_nodes
    device = edge_index.device
    mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
    mask[edge_index.unique()] = False
    return mask


def edge_to_superedge(edges, super_index, edge_attr=None):
    """Convert point-level edges into superedges between clusters, based
    on point-to-cluster indexing 'super_index'. Optionally 'edge_attr'
    can be passed to describe edge attributes that will be returned
    filtered and ordered to describe the superedges.
    """
    # We are only interested in the edges connecting two different
    # clusters and not in the intra-cluster connections. So we first
    # identify the edges of interest. This step requires having access
    # to the 'super_index' to convert point indices into their
    # corresponding cluster indices
    idx_source = super_index[edges[0]]
    idx_target = super_index[edges[1]]
    inter_cluster = torch.where(idx_source != idx_target)[0]

    # Now only consider the edges of interest (ie inter-cluster edges)
    edges_inter = edges[:, inter_cluster]
    edge_attr = edge_attr[inter_cluster] if edge_attr is not None else None
    idx_source = idx_source[inter_cluster]
    idx_target = idx_target[inter_cluster]

    # So far we are manipulating inter-cluster edges, but there may be
    # multiple of those for a given source-target pair. Next, we want to
    # aggregate those into 'superedge" and compute corresponding
    # features (designated with 'se_'). Here, we create unique and
    # consecutive inter-cluster edge identifiers for torch_scatter
    # operations. We use 'se' to designate 'superedge' (ie an edge
    # between two clusters)
    se_id = \
        idx_source * (max(idx_source.max(), idx_target.max()) + 1) + idx_target
    se_id, perm = consecutive_cluster(se_id)
    se_id_source = idx_source[perm]
    se_id_target = idx_target[perm]
    se = torch.vstack((se_id_source, se_id_target))

    return se, se_id, edges_inter, edge_attr
